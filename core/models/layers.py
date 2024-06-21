import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.embedding import CEMA


class ReZero(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, sublayer):
        return x + self.alpha * sublayer(x)


class GELU_Approximation(nn.Module):
    def forward(self, x):
        return (
                0.5
                * x
                * (
                        1
                        + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
                )
        )


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ):
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads

        self.W_q = nn.Linear(config.d_model, config.d_model)
        self.W_k = nn.Linear(config.d_model, config.d_model)
        self.W_v = nn.Linear(config.d_model, config.d_model)
        self.W_o = nn.Linear(config.d_model, config.d_model)

        self.attention = ScaledDotProductAttention(
            config.d_model, config.n_heads, config.dropout
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Handle inputs with different dimensions
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
        elif x.dim() == 4:
            batch_size, _, seq_len, _ = x.shape
            x = x.squeeze(1)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        q = (
            self.W_q(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        k = (
            self.W_k(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        v = (
            self.W_v(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        attn_output = self.attention(q, k, v, mask)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        output = self.W_o(attn_output)
        output = self.dropout(output)

        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = (
            GELU_Approximation() if config.use_gelu_approximation else nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TimestepNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

    def forward(self, x):
        input_shape = x.size()
        b, h, w, c = input_shape

        # Reshape the input to (batch_size, num_channels, -1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, self.num_channels, -1)

        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)

        x = (x - mean) / torch.sqrt(var + self.eps)

        # Reshape back to the original shape
        x = x.view(b, self.num_channels, h, w)
        x = x.permute(0, 2, 3, 1).contiguous()

        return x * self.weight + self.bias


class NormalizedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cema = CEMA(config.d_model, config.cema_hidden_dim)
        self.z_proj = nn.Linear(config.d_model, config.z_dim)
        self.q_proj = nn.Linear(config.z_dim, config.z_dim)
        self.k_proj = nn.Linear(config.z_dim, config.z_dim)
        self.v_proj = nn.Linear(config.d_model, config.v_dim)
        self.o_proj = nn.Linear(config.v_dim, config.d_model)
        self.norm = nn.LayerNorm(config.z_dim)
        self.d_model = config.d_model
        self.z_dim = config.z_dim
        self.v_dim = config.v_dim

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Handle different input shapes
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
        elif x.dim() == 4:
            batch_size, _, seq_len, _ = x.shape
            x = x.squeeze(1)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # Apply CEMA
        x_flat = x.view(-1, self.d_model)
        x_cema = self.cema(x_flat)
        x_cema = x_cema.view(batch_size, seq_len, self.d_model)

        # Project and normalize
        z = self.z_proj(x_cema)
        z_norm = self.norm(z)

        # Compute query, key, and value
        q = self.q_proj(z_norm)
        k = self.k_proj(z_norm)
        v = self.v_proj(x)

        # Compute attention weights
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.z_dim ** 0.5)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Apply attention and project output
        output = torch.matmul(attn_weights, v)
        return self.o_proj(output)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with Normalized Attention and Feed Forward networks.
    """

    def __init__(self, config):
        """
        Initialize the TransformerEncoderLayer.

        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__()
        self.self_attn = NormalizedAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.norm1 = TimestepNorm(config.num_groups, config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the TransformerEncoderLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Optional[torch.Tensor]): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask)
        x = self.dropout1(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x = x + residual
        return x
