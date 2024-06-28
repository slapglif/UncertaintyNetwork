import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.attention.swa import MultiHeadAttention


class KANFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = "cuda"
        self.fc1 = nn.Linear(config.d_model, config.d_ff).to(self.device)
        self.fc2 = nn.Linear(config.d_ff, config.d_model).to(self.device)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # Ensure output has the same shape as input
        x = x.view(original_shape)

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff).to(config.device),
            nn.ReLU(),
            nn.Linear(config.d_ff, config.d_model).to(config.device),
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None):
        # Save original shape
        original_shape = x.shape

        # Ensure x is 3D: [batch_size, seq_len, d_model]
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Multi-head attention
        attn_output, _ = self.attention(x, attention_mask=attention_mask)

        # Add and norm
        x = self.norm1(x + self.dropout(attn_output))

        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        # Ensure output shape matches input shape
        if x.shape != original_shape:
            x = x.view(original_shape)

        return x


class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(
            torch.Tensor(out_features, in_features).normal_(0, 0.1)
        )
        self.weight_rho = nn.Parameter(
            torch.Tensor(out_features, in_features).normal_(-3, 0.1)
        )
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(-3, 0.1))

        self.prior_std = prior_std
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mu, -bound, bound)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))

        weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)

        out = F.linear(x, weight, bias)

        kl_weight = self._kl_divergence(self.weight_mu, weight_std, self.prior_std)
        kl_bias = self._kl_divergence(self.bias_mu, bias_std, self.prior_std)

        return out, kl_weight + kl_bias

    @staticmethod
    def _kl_divergence(
            mu: torch.Tensor, std: torch.Tensor, prior_std: float
    ) -> torch.Tensor:
        kl = 0.5 * (
                2 * torch.log(prior_std / std)
                - 1
                + (std / prior_std).pow(2)
                + (mu / prior_std).pow(2)
        )
        return kl.sum()


class EnsembleModel(nn.Module):
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = [model(x) for model in self.models]
        mean_output = torch.stack([out[0] for out in outputs]).mean(dim=0)
        variance = torch.stack([out[1] for out in outputs]).var(dim=0)
        return mean_output, variance
