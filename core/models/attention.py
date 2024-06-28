from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PretrainedConfig


class TimestepNorm(nn.Module):
    """
    A simplified Timestep Normalization layer.

    This implementation uses a running average of the input features to normalize
    them over time. It does not include group normalization or a custom CUDA kernel.

    Args:
        dim (int): The dimension of the input features.
        eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-5.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0))
        self.momentum = 0.99  # You can adjust this momentum parameter

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies Timestep Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            padding_mask (torch.Tensor, optional): Mask to indicate padded tokens.

        Returns:
            torch.Tensor: Normalized output tensor of shape (batch_size, seq_len, dim).
        """
        batch_size, seq_len, _ = x.shape

        # Calculate the masked mean and variance (if padding_mask is provided)
        if padding_mask is not None:
            masked_x = x * padding_mask.unsqueeze(-1)  # Apply padding mask
            valid_count = padding_mask.sum(dim=1, keepdim=True).float()
            mean = masked_x.sum(dim=1, keepdim=True) / (valid_count + self.eps)
            variance = ((masked_x - mean) ** 2).sum(dim=1, keepdim=True) / (valid_count + self.eps)
        else:
            mean = x.mean(dim=1, keepdim=True)
            variance = x.var(dim=1, keepdim=True)

        # Update running statistics
        self.count += batch_size
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.mean(dim=0)
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * variance.mean(dim=0)

        return (x - self.running_mean.to(x.device)) / torch.sqrt(
            self.running_var.to(x.device) + self.eps
        )


import torch
import torch.nn as nn
from typing import Tuple

class CEMA(nn.Module):
    def __init__(self, embed_dim: int, ndim: int, device: str = 'cpu'):
        super(CEMA, self).__init__()
        self.embed_dim = embed_dim
        self.ndim = ndim
        self.omega = nn.Parameter(torch.randn(embed_dim))
        self.device = device

        # Initialize coefficients
        self.p_coeff = nn.Parameter(torch.randn(embed_dim, ndim))
        self.q_coeff = nn.Parameter(torch.randn(embed_dim, ndim))
        self.gamma = nn.Parameter(torch.randn(embed_dim, ndim, 1))  # Adjust shape

    def _calc_coeffs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Calculate the coefficients p, q, and gamma
        p = self.p_coeff
        q = self.q_coeff
        gamma = self.gamma
        return p, q, gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim, f"Input embed_dim ({embed_dim}) does not match self.embed_dim ({self.embed_dim})"
        residual = x * self.omega.view(1, 1, -1).to(self.device)  # Adjust omega to match input shape

        p, q, gamma = self._calc_coeffs()

        # Ensure p and q have the correct shape
        p = p.view(1, embed_dim, self.ndim, 1).to(self.device)
        q = q.view(1, embed_dim, self.ndim, 1).to(self.device)

        # Simplified EMA computation
        output = torch.zeros_like(x, device=self.device)
        hidden = torch.zeros(bsz, embed_dim, self.ndim, 1, dtype=torch.complex64, device=self.device)

        for t in range(seq_len):
            x_t = x[:, t, :].view(bsz, embed_dim, 1, 1).to(self.device)
            hidden = p * x_t + q * hidden
            output[:, t, :] = (gamma * hidden).sum(dim=-2).view(bsz, embed_dim).real

        return output



@dataclass
class MultiHeadAttentionConfig(PretrainedConfig):
    d_model: int = 64
    n_heads: int = 4
    dropout: float = 0.1
    max_position_embeddings: int = 1024
    sliding_window_size: int = 128
    embed_dim: int = 64  # Add this attribute
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultiHeadAttention(nn.Module):
    def __init__(self, config: MultiHeadAttentionConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        self.window_size = config.sliding_window_size
        self.device = config.device

        self.to_qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.to_out = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # CEMA integration
        self.cema = CEMA(config.d_model)

        # Adaptive weighting
        self.adaptive_weights = nn.Parameter(torch.ones(2))  # For attention and CEMA

        # Reduced dimensionality for CEMA (optional)
        self.cema_reduction_factor = getattr(config, 'cema_reduction_factor', 1)
        if self.cema_reduction_factor > 1:
            self.cema_down_proj = nn.Linear(config.d_model, config.d_model // self.cema_reduction_factor)
            self.cema_up_proj = nn.Linear(config.d_model // self.cema_reduction_factor, config.d_model)
        else:
            self.cema_down_proj = nn.Identity()
            self.cema_up_proj = nn.Identity()

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, use_cache=False, **kwargs):
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, f"Input embed_dim ({embed_dim}) does not match self.embed_dim ({self.embed_dim})"
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.n_heads), qkv)

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * (self.d_k ** -0.5)
        dots = self._apply_sliding_window_attention(dots)

        if attention_mask is not None:
            mask = rearrange(attention_mask, "b j -> b 1 1 j")
            dots = dots.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)

        out_attn = torch.einsum("bhij,bhjd->bhid", attn, v)
        out_attn = rearrange(out_attn, "b h n d -> b n (h d)")
        out_attn = self.to_out(out_attn)

        # CEMA processing
        x_cema = self.cema(x)

        # Adaptive weighting
        weights = F.softmax(self.adaptive_weights, dim=0)
        out = weights[0] * out_attn + weights[1] * x_cema

        return out, attn if use_cache else out

    def _apply_sliding_window_attention(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """
        Applies sliding window attention to the attention scores.

        Args:
            attn_scores (torch.Tensor): Attention scores of shape (batch_size, n_heads, seq_len, seq_len).

        Returns:
            torch.Tensor: Attention scores with sliding window applied, of shape (batch_size, n_heads, seq_len, seq_len).
        """
        batch_size, n_heads, seq_len, _ = attn_scores.shape

        pad_len = (self.window_size - seq_len % self.window_size) % self.window_size
        padded_scores = F.pad(attn_scores, (0, pad_len, 0, pad_len))

        causal_mask = torch.tril(
            torch.ones(self.window_size, self.window_size, device=attn_scores.device)
        ).view(1, 1, self.window_size, self.window_size)

        windowed_scores = padded_scores.unfold(2, self.window_size, self.window_size)
        windowed_scores = windowed_scores.unfold(3, self.window_size, self.window_size)
        windowed_scores = windowed_scores * causal_mask

        attn_scores = windowed_scores.sum(dim=(-2, -1))
        attn_scores = attn_scores[:, :, :seq_len, :seq_len]

        return attn_scores
