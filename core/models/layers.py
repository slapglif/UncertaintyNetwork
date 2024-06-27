import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        self.window_size = config.sliding_window_size
        self.device = config.device

        self.W_q = nn.Linear(config.d_model, config.d_model).to(self.device)
        self.W_k = nn.Linear(config.d_model, config.d_model).to(self.device)
        self.W_v = nn.Linear(config.d_model, config.d_model).to(self.device)
        self.W_o = nn.Linear(config.d_model, config.d_model).to(self.device)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None, use_cache=False, **kwargs):
        # Reshape x if necessary
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0)

        batch_size, seq_len, _ = x.shape
        print(f"Input shape: {x.shape}")

        # Linear projections
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        print(f"Attention scores shape: {attn_scores.shape}")

        # Apply sliding window attention
        if self.window_size < seq_len:
            attn_scores = self._apply_sliding_window_attention(attn_scores)

        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            elif attention_mask.dim() == 4:
                # Ensure the mask matches the batch size of our input
                if attention_mask.size(0) != batch_size:
                    attention_mask = attention_mask[:batch_size]

            # Ensure the mask matches the attention scores shape
            if attention_mask.size(-1) != seq_len:
                attention_mask = F.pad(attention_mask, (0, seq_len - attention_mask.size(-1)), value=1)

            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        # Compute attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        print(f"Attention probabilities shape: {attn_probs.shape}")

        # Ensure attn_probs has the same batch size as the input
        if attn_probs.size(0) != batch_size:
            attn_probs = attn_probs[:batch_size]

        # Compute output
        attn_output = torch.matmul(attn_probs, v)
        print(f"Attention output shape: {attn_output.shape}")

        # Reshape attn_output to match expected dimensions
        attn_output = attn_output.transpose(1, 2).contiguous()
        print(f"Transposed attention output shape: {attn_output.shape}")

        output = attn_output.view(batch_size, seq_len, self.d_model)
        print(f"Reshaped output shape: {output.shape}")

        output = self.W_o(output)
        print(f"Final output shape: {output.shape}")

        # Ensure output has the same shape as input
        if output.shape != x.shape:
            output = F.pad(output, (0, 0, 0, x.size(1) - output.size(1), 0, 0))

        # Reshape back to original shape if necessary
        if original_shape != output.shape:
            output = output.view(original_shape)

        print(f"MultiHeadAttention input shape: {x.shape}, output shape: {output.shape}")

        return output, None  # (output, attn_probs) if use_cache else (output, None)

    def _apply_sliding_window_attention(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """
        Applies sliding window attention to the attention scores.

        Args:
            attn_scores (torch.Tensor): The attention scores.

        Returns:
            torch.Tensor: The attention scores with sliding window attention applied.
        """
        batch_size, n_heads, seq_len, _ = attn_scores.shape

        # Pad attention scores for easier windowing
        pad_len = (self.window_size - seq_len % self.window_size) % self.window_size
        padded_scores = F.pad(attn_scores, (0, pad_len, 0, pad_len))

        # Create causal mask within each window
        causal_mask = torch.tril(
            torch.ones(self.window_size, self.window_size, device=attn_scores.device)
        ).view(1, 1, self.window_size, self.window_size)

        # Apply causal mask to each window
        windowed_scores = padded_scores.unfold(2, self.window_size, self.window_size)
        windowed_scores = windowed_scores.unfold(3, self.window_size, self.window_size)
        windowed_scores = windowed_scores * causal_mask

        # Merge windows back
        attn_scores = windowed_scores.sum(dim=(-2, -1))

        # Remove padding
        attn_scores = attn_scores[:, :, :seq_len, :seq_len]

        return attn_scores


class KANFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = 'cuda'
        self.fc1 = nn.Linear(config.d_model, config.d_ff).to(self.device)
        self.fc2 = nn.Linear(config.d_ff, config.d_model).to(self.device)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


import torch
import torch.nn as nn


class CEMA(nn.Module):
    """
    Cumulative Exponential Moving Average (CEMA) module.

    This module computes a cumulative exponential moving average of the input tensor
    across the sequence dimension.

    Attributes:
        d_model (int): The dimension of the model (embedding dimension).
        alpha (float): The smoothing factor for the exponential moving average.
        ema (torch.Tensor): The exponential moving average tensor.
    """

    def __init__(self, d_model: int, alpha: float = 0.99):
        """
        Initialize the CEMA module.

        Args:
            d_model (int): The dimension of the model (embedding dimension).
            alpha (float, optional): The smoothing factor for the exponential moving average.
                Defaults to 0.99.
        """
        super().__init__()
        self.d_model = d_model
        self.alpha = alpha
        self.register_buffer("ema", torch.zeros(1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the cumulative exponential moving average of the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model) with CEMA applied.
        """
        batch_size, seq_len, d_model = x.shape

        # Ensure self.ema has the correct shape
        if self.ema.shape[1] != d_model:
            self.ema = self.ema.new_zeros(1, d_model)

        output = []

        for i in range(seq_len):
            # Compute the mean across the batch dimension for the current time step
            current_mean = x[:, i, :].mean(dim=0, keepdim=True)

            # Update the EMA
            self.ema = self.alpha * self.ema + (1 - self.alpha) * current_mean

            # Expand the EMA to match the batch size
            expanded_ema = self.ema.expand(batch_size, -1)

            output.append(expanded_ema)

        # Stack the output tensors along the sequence dimension
        return torch.stack(output, dim=1)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff).to(config.device),
            nn.ReLU(),
            nn.Linear(config.d_ff, config.d_model).to(config.device)
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None):
        # Reshape x if necessary
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0)

        attn_output, _ = self.attention(x, attention_mask=attention_mask)

        # Ensure attn_output has the same shape as x
        if attn_output.shape != x.shape:
            attn_output = F.pad(attn_output, (0, 0, 0, x.size(1) - attn_output.size(1), 0, 0))

        # Add and norm
        x = self.norm1(x + self.dropout(attn_output))

        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        # Reshape back to original shape if necessary
        if original_shape != x.shape:
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
