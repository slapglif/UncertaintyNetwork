import warnings
from typing import Tuple

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, embed_dim: int, max_positions: int = 10000, base: float = 10000):
        """
        Initialize the RotaryEmbedding module.

        Args:
            embed_dim (int): Dimension of the embedding.
            max_positions (int): Maximum number of positions.
            base (float): Base value for computation.
        """
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("The embedding dimension must be divisible by 2.")

        self.embed_dim = embed_dim
        self.max_positions = max_positions
        self.base = base

        # Precompute the frequency tensor
        self.freqs_cis = self._precompute_freqs_cis(max_positions, embed_dim, base)

    def get_freqs_cis(self, start: int, end: int) -> torch.Tensor:
        if self.freqs_cis is None:
            self.freqs_cis = self._precompute_until(self.max_positions)
        if end > self.freqs_cis.shape[0]:
            warnings.warn(f'Extending rotary range from {self.max_positions} to {end}')
            self.freqs_cis = self._precompute_until(end)
        return self.freqs_cis[start:end]  # type: ignore

    def _precompute_freqs_cis(self, max_positions, embed_dim, base):
        """Precomputes the complex frequencies."""
        position = torch.arange(max_positions).unsqueeze(1)  # [max_positions, 1]
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-torch.log(torch.tensor(base)) / embed_dim)
        )
        # Broadcast div_term across the sequence length dimension
        freqs = position * div_term[None, :]  # [max_positions, embed_dim // 2]
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [max_positions, embed_dim // 2]

        # Add the third dimension for real/imaginary parts
        freqs_cis = freqs_cis.unsqueeze(2).repeat(1, 1, 2)  # [max_positions, embed_dim // 2, 2]
        return freqs_cis.to("cuda")  # Move to device early

    def forward(self, xq, xk, start: int):
        """
        Applies rotary embeddings to the query and key tensors.

        Args:
            xq: Query tensor with shape (batch_size, seq_len, embed_dim).
            xk: Key tensor with shape (batch_size, seq_len, embed_dim).
            start: Starting position index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
        """
        seq_len = xq.shape[1]
        end = start + seq_len
        assert end <= self.max_positions, f"Sequence length too long, exceeds max positions: {end} > {self.max_positions}"

        freqs_cis = self.freqs_cis[start:end]
        return apply_rotary_emb(xq, xk, freqs_cis)


@torch.jit.script
def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary embeddings using efficient in-place operations.
    """
    xq_out = xq.clone()
    xk_out = xk.clone()

    dim = xq.shape[-1]
    for i in range(0, dim, 2):
        # The second dimension of freqs_cis should now be dim // 2
        a = freqs_cis[:, i // 2, 0]
        b = freqs_cis[:, i // 2, 1]

        xq_out[:, :, i] = xq[:, :, i] * a - xq[:, :, i + 1] * b
        xq_out[:, :, i + 1] = xq[:, :, i] * b + xq[:, :, i + 1] * a
        xk_out[:, :, i] = xk[:, :, i] * a - xk[:, :, i + 1] * b
        xk_out[:, :, i + 1] = xk[:, :, i] * b + xk[:, :, i + 1] * a

    return xq_out, xk_out
