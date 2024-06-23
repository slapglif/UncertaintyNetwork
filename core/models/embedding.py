from typing import Optional, Tuple

import torch
import torch.nn as nn


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
