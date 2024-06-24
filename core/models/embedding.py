# core/models/embedding.py

from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
from einops import rearrange

from core.kan import FasterKANvolver


class RotaryPositionEncoding(nn.Module):
    def __init__(self, dim, n_heads, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        inv_freq = 1.0 / (base ** (torch.arange(0, dim // n_heads, 2).float() / (dim // n_heads)))
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

        return cos.repeat(x.shape[0], self.n_heads, 1, 1), sin.repeat(x.shape[0], self.n_heads, 1, 1)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
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
    x = rearrange(x, 'b l (h d) -> b h l d', h=cos.shape[1])

    # Apply rotary embeddings
    x = (x * cos) + (rotate_half(x) * sin)

    # Reshape back to original shape
    x = rearrange(x, 'b h l d -> b l (h d)')
    return x


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


kan_config = {
    'layers_hidden': [1024, 2048],
    'grid_min': -1.2,
    'grid_max': 0.2,
    'num_grids': 8,
    'exponent': 2,
    'inv_denominator': 0.5,
    'train_grid': False,
    'train_inv_denominator': False,
    'spline_weight_init_scale': 1.0,
}


class SentenceEncoder(nn.Module):
    """
    Sentence Encoder module.

    This module encodes input sentences using the FasterKANvolver architecture,
    which combines a convolutional feature extractor with KAN layers.

    Args:
        vocab_size (int): The size of the vocabulary.
        hidden_dim (int): The dimensionality of the hidden states.
        output_dim (int): The dimensionality of the output embeddings.
        kan_config (Dict[str, Any]): The configuration dictionary for the KAN layers.

    Attributes:
        embedding (nn.Embedding): The word embedding layer.
        faster_kan_volver (FasterKANvolver): The FasterKANvolver module.

    """

    def __init__(self, vocab_size: int, hidden_dim: int, output_dim: int,
                 kan_config: Dict[str, Any] = kan_config) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Extract the necessary arguments for FasterKANvolver
        layers_hidden = kan_config['layers_hidden']
        grid_min = kan_config.get('grid_min', -1.2)
        grid_max = kan_config.get('grid_max', 0.2)
        num_grids = kan_config.get('num_grids', 8)
        exponent = kan_config.get('exponent', 2)
        inv_denominator = kan_config.get('inv_denominator', 0.5)
        train_grid = kan_config.get('train_grid', False)
        train_inv_denominator = kan_config.get('train_inv_denominator', False)

        # Ensure the last hidden dimension matches the output dimension
        layers_hidden[-1] = output_dim

        self.faster_kan_volver = FasterKANvolver(
            layers_hidden,
            input_channels=1,
            hidden_dim=hidden_dim,
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            exponent=exponent,
            inv_denominator=inv_denominator,
            train_grid=train_grid,
            train_inv_denominator=train_inv_denominator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.faster_kan_volver(x)
        return x


class SentenceGP(nn.Module):
    """
    Sentence-level Gaussian Process layer with RBF kernel for uncertainty estimation.

    This module applies a Gaussian Process to sentence embeddings to model uncertainty
    at the sentence level using an RBF kernel.

    Attributes:
        input_dim (int): The input feature dimension.
        output_dim (int): The output feature dimension.
        n_inducing (int): The number of inducing points for the Gaussian Process.
        embedding_dim (int): The embedding dimension used for reshaping inducing points.
        inducing_points (nn.Parameter): Learnable inducing points.
        log_lengthscale (nn.Parameter): Learnable length scale for RBF kernel.
        log_noise (nn.Parameter): Learnable noise parameter.
        output_proj (nn.Linear): Output projection layer.
    """

    def __init__(
            self, input_dim: int, output_dim: int, n_inducing: int, embedding_dim: int
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_inducing = n_inducing
        self.embedding_dim = embedding_dim

        # Learnable inducing points (initialize on CPU first)
        self.inducing_points = nn.Parameter(torch.randn(n_inducing, input_dim))

        # Learnable length scale for RBF kernel
        self.log_lengthscale = nn.Parameter(torch.zeros(1))

        # Learnable noise parameter
        self.log_noise = nn.Parameter(torch.zeros(1))

        # Output projection
        self.output_proj = nn.Linear(n_inducing, output_dim)

    def to(self, *args, **kwargs):
        """
        Moves and/or casts the parameters and buffers.

        This method overwrites the default `to` method to ensure that
        the `inducing_points` are moved to the correct device.
        """
        self = super().to(*args, **kwargs)
        self.inducing_points = nn.Parameter(self.inducing_points.to(*args, **kwargs))
        return self

    def rbf_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel between two sets of points.

        Args:
            x1 (torch.Tensor): First set of points of shape (..., input_dim).
            x2 (torch.Tensor): Second set of points of shape (..., input_dim).

        Returns:
            torch.Tensor: Kernel matrix of shape (..., x1.shape[:-1], x2.shape[:-1]).
        """
        dist = torch.cdist(x1, x2, p=2).pow(2)
        return torch.exp(-0.5 * dist / torch.exp(self.log_lengthscale).pow(2))

    def forward(
            self, x: torch.Tensor, num_sentences: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the SentenceGP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_sentences, input_dim).
            num_sentences (Optional[int]): Number of sentences in the input. If not provided, it will be inferred from the input tensor shape.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Mean tensor of shape (batch_size, num_sentences, output_dim).
                - Variance tensor of shape (batch_size, num_sentences, output_dim).
        """
        batch_size, num_sentences_input, input_dim = x.shape

        # Input validation: Check if the input dimensions are as expected
        if num_sentences is not None and num_sentences_input != num_sentences:
            raise ValueError(
                f"Input tensor has {num_sentences_input} sentences, but {num_sentences} sentences were expected.")
        if input_dim != self.input_dim:
            raise ValueError(
                f"Input tensor has {input_dim} input dimensions, but {self.input_dim} dimensions were expected.")

        # Use the inferred num_sentences if not provided explicitly
        if num_sentences is None:
            num_sentences = num_sentences_input

        # Compute kernel matrices
        K_xx = self.rbf_kernel(x, x)
        K_xi = self.rbf_kernel(x, self.inducing_points)
        K_ii = self.rbf_kernel(self.inducing_points, self.inducing_points)

        # Compute predictive distribution
        K_ii_inv = torch.inverse(
            K_ii + torch.exp(self.log_noise) * torch.eye(self.n_inducing, device=x.device)
        )
        mean = K_xi @ K_ii_inv @ self.output_proj.weight.T
        var = K_xx - K_xi @ K_ii_inv @ K_xi.transpose(-1, -2)

        # Reshape outputs using einsum
        mean = torch.einsum("bni,io->bno", mean, self.output_proj.weight)
        var = torch.diagonal(var, dim1=-2, dim2=-1)

        return mean, torch.nn.functional.softplus(var)
