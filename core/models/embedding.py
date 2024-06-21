import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CEMA(nn.Module):
    def __init__(self, d: int, h: int, chunk_size: int = 128):
        super().__init__()
        self.d = d
        self.h = h
        self.chunk_size = chunk_size
        self.alpha = nn.Parameter(torch.rand(d, h))
        self.delta = nn.Parameter(torch.rand(d, h))
        self.omega = nn.Parameter(torch.rand(h))
        self.beta = nn.Parameter(torch.randn(d, h))
        self.eta = nn.Parameter(torch.randn(d, h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, d = x.shape
        assert d == self.d, f"Input dimension {d} must match CEMA's dimension {self.d}"

        # Compute theta
        theta = torch.outer(torch.arange(self.h, device=x.device), self.omega) * (2 * math.pi / self.h)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Compute complex alpha and delta
        alpha_complex = self.alpha[:, None, :] * (cos_theta[None, :, :] + 1j * sin_theta[None, :, :])
        delta_complex = self.delta[:, None, :] * (cos_theta[None, :, :] + 1j * sin_theta[None, :, :])

        # Process in chunks
        output = []
        for i in range(0, batch_size, self.chunk_size):
            chunk = x[i:i + self.chunk_size]

            # Compute u for the chunk
            u = torch.einsum('bd,dh->bdh', chunk, self.beta)

            # Apply CEMA for the chunk
            h = alpha_complex[None, :, :, :] * u[:, :, None, :]
            h += (1 - alpha_complex[None, :, :, :] * delta_complex[None, :, :, :]) * torch.zeros(chunk.size(0), d, 1,
                                                                                                 self.h,
                                                                                                 dtype=torch.complex64,
                                                                                                 device=x.device)

            # Compute output for the chunk
            y = torch.einsum('bdhk,dh->bd', h.real, self.eta)
            output.append(y)

        return torch.cat(output, dim=0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(
                self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer(
                "cos_cached", emb.cos()[None, None, :, :], persistent=False
            )
            self.register_buffer(
                "sin_cached", emb.sin()[None, None, :, :], persistent=False
            )
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def apply_rotary_pos_emb(q, k, cos, sin, offset=0):
    cos = cos[:, :, offset: q.shape[-2] + offset, :]
    sin = sin[:, :, offset: q.shape[-2] + offset, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class StableEmbedding(nn.Module):
    """
    A stable embedding layer that scales the output by the square root of the embedding dimension.

    This embedding layer is designed to provide more stable gradients during training.
    """

    def __init__(
            self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """
        Initialize the StableEmbedding layer.

        Args:
            num_embeddings (int): Size of the dictionary of embeddings.
            embedding_dim (int): The size of each embedding vector.
            padding_idx (Optional[int]): If specified, the entries at padding_idx do not contribute to the gradient.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters (weight) of the embedding layer.
        """
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the StableEmbedding layer.

        Args:
            input (torch.Tensor): Input tensor containing indices to extract embeddings for.

        Returns:
            torch.Tensor: The resulting embedding tensor scaled by sqrt(embedding_dim).
        """
        return F.embedding(
            input, self.weight, self.padding_idx, None, 2, False, False
        ) * (self.embedding_dim ** 0.5)
