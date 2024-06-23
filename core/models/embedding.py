import math
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.layers import GaussianProcessLayer


class NTKEmbedding(nn.Module):
    """
    NTK-inspired embedding layer with Gaussian process behavior.

    This embedding layer approximates the behavior of an infinitely wide network
    in its NTK regime, incorporating Gaussian process characteristics.

    Attributes:
        num_embeddings (int): Size of the dictionary of embeddings.
        embedding_dim (int): The size of each embedding vector.
        padding_idx (Optional[int]): If given, pads the output with zeros whenever it encounters this index.
        scale (float): Scaling factor for the embeddings.
        sigma (nn.Parameter): Learnable parameter for the GP kernel variance.
        length_scale (nn.Parameter): Learnable parameter for the GP kernel length scale.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: Optional[int] = None,
            scale: float = 1.0,
            sigma_init: float = 1.0,
            length_scale_init: float = 1.0,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.scale = scale

        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.sigma = nn.Parameter(torch.tensor(sigma_init))
        self.length_scale = nn.Parameter(torch.tensor(length_scale_init))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the embedding weights in the NTK regime."""
        nn.init.normal_(self.weight, mean=0, std=1.0 / math.sqrt(self.embedding_dim))
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute the NTK-inspired embedding for the input.

        Args:
            input (torch.Tensor): Input tensor containing indices into the embedding dictionary.

        Returns:
            torch.Tensor: The NTK-inspired embedding for the input indices.
        """
        embedded = F.embedding(input, self.weight, self.padding_idx)

        # Scale embeddings to approximate NTK behavior
        embedded = embedded * math.sqrt(self.embedding_dim)

        # Apply RBF kernel to approximate GP behavior
        if self.training:
            kernel = self.sigma ** 2 * torch.exp(
                -torch.sum(embedded ** 2, dim=-1) / (2 * self.length_scale ** 2)
            )
            noise = torch.randn_like(embedded) * kernel.unsqueeze(-1).sqrt()
            embedded = embedded + noise

        return embedded * self.scale


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.gp_layer = GaussianProcessLayer(1, d_model, n_inducing=min(max_len, 100))

        # Create position tensor
        self.register_buffer('position', torch.arange(0, max_len).unsqueeze(1).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor using Gaussian process.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns:
            torch.Tensor: Output tensor with positional encoding added, of shape (batch_size, seq_len, embedding_dim).
        """
        batch_size, seq_len, embedding_dim = x.shape
        positions = self.position[:seq_len].to(x.device)

        # Compute positional encoding using GP layer
        pos_encoding_mean, pos_encoding_var = self.gp_layer(positions)

        # Expand positional encoding to match input shape
        pos_encoding_mean = pos_encoding_mean.unsqueeze(0).expand(batch_size, -1, -1)

        # Truncate positional encoding if it exceeds the sequence length
        pos_encoding_mean = pos_encoding_mean[:, :seq_len, :]  # Add this line

        # Add positional encoding to input
        x = x + pos_encoding_mean

        # Add uncertainty during training
        if self.training:
            pos_encoding_var = pos_encoding_var.unsqueeze(0).expand(batch_size, -1, -1)
            pos_encoding_var = pos_encoding_var[:, :seq_len, :]  # Add this line
            x = x + torch.randn_like(x) * pos_encoding_var.sqrt()

        return self.dropout(x)


class StableEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()

        self.ntk_embedding = NTKEmbedding(num_embeddings, embedding_dim, padding_idx)
        self.gp_layer = GaussianProcessLayer(embedding_dim, embedding_dim)

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim ** -0.5)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input):
        embedded = self.ntk_embedding(input)
        mean, var = self.gp_layer(embedded)
        if self.training:
            return mean + torch.randn_like(mean) * torch.sqrt(var)
        else:
            return mean


class RotaryPositionEncoding(nn.Module):
    """
    Rotary Position Encoding as described in the RoFormer paper.

    This class implements rotary position encodings which can be particularly
    effective for capturing relative positions in transformer models.

    Attributes:
        dim (int): Dimension of the model.
        max_position_embeddings (int): Maximum number of positions to encode.
        base (int): Base for the angle calculation.
        inv_freq (torch.Tensor): Inverse frequency tensor for angle calculation.
    """

    def __init__(
            self, dim: int, max_position_embeddings: int = 2048, base: int = 10000
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", self.inv_freq)

        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        """Set up the cache for fast retrieval of position encodings."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def forward(
            self, x: torch.Tensor, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the rotary position encodings.

        Args:
            x (torch.Tensor): Input tensor.
            seq_len (Optional[int]): Sequence length. If None, use x.shape[1].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Cosine and sine of the position encodings.
        """
        if seq_len is None:
            seq_len = x.shape[1]

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.device),
            self.sin_cached[:, :, :seq_len, ...].to(x.device),
        )
