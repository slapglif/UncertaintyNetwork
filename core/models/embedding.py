# core/models/embedding.py
from typing import Optional
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from core.models.kan import SplineNetConv


class RotaryPositionEncoding(nn.Module):
    def __init__(self, dim, n_heads, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim // n_heads, 2).float() / (dim // n_heads))
        )
        self.register_buffer("inv_freq", inv_freq)

        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(max_position_embeddings, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        return cos.repeat(x.shape[0], self.n_heads, 1, 1), sin.repeat(
            x.shape[0], self.n_heads, 1, 1
        )

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])


def apply_rotary_pos_emb(x, cos, sin):
    """Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
        cos (torch.Tensor): The cosine component of the rotary embeddings.
        sin (torch.Tensor): The sine component of the rotary embeddings.

    Returns:
        torch.Tensor: The input tensor with rotary embeddings applied.
    """
    # Reshape for head-wise operation
    x = rearrange(x, "b l (h d) -> b h l d", h=cos.shape[1])

    # Apply rotary embeddings
    x = (x * cos) + (rotate_half(x) * sin)

    # Reshape back to original shape
    x = rearrange(x, "b h l d -> b l (h d)")
    return x


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


kan_config = {
    "layers_hidden": [1024, 2048],
    "grid_min": -1.2,
    "grid_max": 0.2,
    "num_grids": 8,
    "exponent": 2,
    "inv_denominator": 0.5,
    "train_grid": False,
    "train_inv_denominator": False,
    "spline_weight_init_scale": 1.0,
}


# class SentenceEncoder(nn.Module):
#     """
#     Sentence Encoder module using SplineNetConv for processing embedded sentences.
#     """
#
#     def __init__(self, vocab_size: int, hidden_dim: int, output_dim: int,
#                  kan_config: Dict[str, Any]) -> None:
#         """
#         Initialize the SentenceEncoder.
#
#         Args:
#             vocab_size (int): Size of the vocabulary.
#             hidden_dim (int): Dimension of the hidden layers.
#             output_dim (int): Dimension of the output.
#             kan_config (Dict[str, Any]): Configuration for the SplineNetConv.
#         """
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, hidden_dim)
#         self.faster_kan_volver = SplineNetConv(
#             kan_config['layers_hidden'],
#             input_channels=1,
#             hidden_dim=hidden_dim,
#             grid_min=kan_config.get('grid_min', -1.2),
#             grid_max=kan_config.get('grid_max', 0.2),
#             num_grids=kan_config.get('num_grids', 8),
#             exponent=kan_config.get('exponent', 2),
#             inv_denominator=kan_config.get('inv_denominator', 0.5),
#             train_grid=kan_config.get('train_grid', False),
#             train_inv_denominator=kan_config.get('train_inv_denominator', False),
#             uncertainty_output=kan_config.get('uncertainty_output', True),
#         )
#         self.output_proj = nn.Linear(kan_config['layers_hidden'][-1], output_dim)
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         x = self.embedding(x)  # Shape: (batch_size, seq_len, hidden_dim)
#         x = x.unsqueeze(1)  # Shape: (batch_size, 1, seq_len, hidden_dim)
#         x, uncertainty = self.faster_kan_volver(x)
#         x = x.squeeze(-1).transpose(1, 2)
#         uncertainty = uncertainty.squeeze(-1).transpose(1, 2)
#         x = self.output_proj(x)
#         uncertainty = self.output_proj(uncertainty)
#         return x, uncertainty


class SentenceGP(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, n_inducing: int, embedding_dim: int
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_inducing = n_inducing
        self.embedding_dim = embedding_dim

        self.inducing_points = nn.Parameter(torch.randn(n_inducing, input_dim))
        self.log_lengthscale = nn.Parameter(torch.zeros(1))
        self.log_variance = nn.Parameter(torch.zeros(1))
        self.output_proj = nn.Linear(n_inducing, output_dim)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.inducing_points = nn.Parameter(self.inducing_points.to(*args, **kwargs))
        return self

    def rbf_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(x1, x2, p=2).pow(2)
        return torch.exp(-0.5 * dist / torch.exp(self.log_lengthscale).pow(2))

    def forward(
        self, x: torch.Tensor, num_sentences: Optional[int] = None
    ) -> tuple[Tensor, Tensor]:
        batch_size, num_sentences_input, input_dim = x.shape

        if num_sentences is not None and num_sentences_input != num_sentences:
            raise ValueError(
                f"Input tensor has {num_sentences_input} sentences, but {num_sentences} sentences were expected."
            )
        if input_dim != self.input_dim:
            raise ValueError(
                f"Input tensor has {input_dim} input dimensions, but {self.input_dim} dimensions were expected."
            )

        if num_sentences is None:
            num_sentences = num_sentences_input

        K_xx = self.rbf_kernel(x, x)
        K_xi = self.rbf_kernel(x, self.inducing_points)
        K_ii = self.rbf_kernel(self.inducing_points, self.inducing_points)

        K_ii_inv = torch.inverse(
            K_ii
            + torch.exp(self.log_variance) * torch.eye(self.n_inducing, device=x.device)
        )
        mean = K_xi @ K_ii_inv @ self.output_proj.weight.T
        var = K_xx - K_xi @ K_ii_inv @ K_xi.transpose(-1, -2)

        mean = torch.einsum("bni,io->bno", mean, self.output_proj.weight)
        var = torch.diagonal(var, dim1=-2, dim2=-1)

        return mean, torch.nn.functional.softplus(var)  # Remove the third return value
