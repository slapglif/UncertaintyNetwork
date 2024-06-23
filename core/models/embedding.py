from typing import Optional, Tuple

import torch
import torch.nn as nn

from core.kan.fasterkan_basis import ReflectionalSwitchFunction, SplineLinear


class RotaryPositionEncoding(nn.Module):
    """
    Rotary Position Encoding as described in the RoFormer paper.

    This class implements rotary position encodings, which can be particularly
    effective for capturing relative positions in transformer models.

    Attributes:
        dim (int): Dimension of the model.
        max_position_embeddings (int): Maximum number of positions to encode.
        base (int): Base for the angle calculation.
        inv_freq (torch.Tensor): Inverse frequency tensor for angle calculation.

    Example:
        >>> rotary_pe = RotaryPositionEncoding(dim=512)
        >>> x = torch.randn(1, 10, 512)  # Example input tensor
        >>> cos, sin = rotary_pe(x)
        >>> print(cos.shape, sin.shape)
        torch.Size([1, 1, 10, 512]) torch.Size([1, 1, 10, 512])
    """

    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: int = 10000
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Pre-calculate inverse frequencies
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", self.inv_freq)

        # Initialize the cache for cosine and sine values
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        """Set up the cache for fast retrieval of position encodings."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Cache the cosine and sine values
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

        # Update the cache if the sequence length exceeds the cached length
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        # Retrieve cached cosine and sine values for the given sequence length
        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.device),
            self.sin_cached[:, :, :seq_len, ...].to(x.device),
        )


class SentenceEncoder(nn.Module):
    """
    Sentence Encoder implementing a Kolmogorov-Arnold Network (KAN) for transforming sentence representations.

    This module applies a series of RBF-based transformations followed by linear projections
    to create a semantically richer representation of the sentence embeddings.

    Attributes:
        input_dim (int): The input feature dimension (embedding dimension of the sentence).
        hidden_dim (int): The hidden dimension of the intermediate linear layers.
        output_dim (int): The output feature dimension.
        num_grids (int): The number of grids used in the RBF transformation.
        rbf (ReflectionalSwitchFunction): RBF transformation layer.
        spline_linear (SplineLinear): Linear projection after RBF.
        output_layer (nn.Linear): Final linear layer.

    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_grids: int = 8
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_grids = num_grids

        self.rbf = ReflectionalSwitchFunction(num_grids=num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SentenceEncoder (KAN).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_sentences, sentence_len, embedding_dim).

        Returns:
            torch.Tensor: Transformed sentence embedding of shape (batch_size, num_sentences, output_dim).
        """
        batch_size, num_sentences, sentence_len, _ = x.shape

        # Reshape for RBF transformation
        x = x.view(batch_size * num_sentences * sentence_len, -1)

        # Apply RBF transformation
        spline_basis = self.rbf(x).view(batch_size * num_sentences * sentence_len, -1)

        # Apply spline linear transformation
        x = self.spline_linear(spline_basis)

        # Reshape and pool across sentence length
        x = x.view(batch_size, num_sentences, sentence_len, -1).mean(dim=2)

        # Final linear projection
        x = self.output_layer(x)

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
    """

    def __init__(
        self, input_dim: int, output_dim: int, n_inducing: int, embedding_dim: int
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_inducing = n_inducing
        self.embedding_dim = embedding_dim

        # Learnable inducing points
        self.inducing_points = nn.Parameter(torch.randn(n_inducing, input_dim))

        # Learnable length scale for RBF kernel
        self.log_lengthscale = nn.Parameter(torch.zeros(1))

        # Learnable noise parameter
        self.log_noise = nn.Parameter(torch.zeros(1))

        # Output projection
        self.output_proj = nn.Linear(n_inducing, output_dim)

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
        self, x: torch.Tensor, num_sentences: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the SentenceGP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_sentences, input_dim).
            num_sentences (int): Number of sentences in the input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Mean tensor of shape (batch_size, num_sentences, output_dim).
                - Variance tensor of shape (batch_size, num_sentences, output_dim).
        """
        batch_size = x.shape[0]

        # Compute kernel matrices
        K_xx = self.rbf_kernel(x, x)
        K_xi = self.rbf_kernel(x, self.inducing_points)
        K_ii = self.rbf_kernel(self.inducing_points, self.inducing_points)

        # Compute predictive distribution
        K_ii_inv = torch.inverse(
            K_ii
            + torch.exp(self.log_noise) * torch.eye(self.n_inducing, device=x.device)
        )
        mean = K_xi @ K_ii_inv @ self.output_proj.weight.T
        var = K_xx - K_xi @ K_ii_inv @ K_xi.transpose(-1, -2)

        # Reshape outputs
        mean = mean.view(batch_size, num_sentences, self.output_dim)
        var = var.view(batch_size, num_sentences, num_sentences)
        var = (
            torch.diagonal(var, dim1=-2, dim2=-1)
            .unsqueeze(-1)
            .expand(-1, -1, self.output_dim)
        )

        return mean, torch.nn.functional.softplus(var)  # Ensure positive variance
