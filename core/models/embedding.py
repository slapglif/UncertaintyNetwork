import torch
import torch.nn as nn
from typing import Tuple

class RotaryEmbedding(nn.Module):
    """
    Applies rotary position embeddings to a tensor.

    Args:
        embed_dim (int): The dimension of the embedding.
        max_positions (int, optional): The maximum sequence length. Defaults to 2048.
    """

    def __init__(self, embed_dim: int, max_positions: int = 2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_positions = max_positions

        inv_freq = 1.0 / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        self.register_buffer("inv_freq", inv_freq)

    def get_freqs_cis(self, start: int, seq_len: int) -> torch.Tensor:
        """
        Gets the frequencies and complex sinusoids.

        Args:
            start (int): The starting position.
            seq_len (int): The sequence length.

        Returns:
            torch.Tensor: The frequencies and complex sinusoids.
        """
        t = torch.arange(start, start + seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.stack((freqs.cos(), freqs.sin()), dim=-1)

    def forward(self, xq: torch.Tensor, xk: torch.Tensor, seq_len: int, start: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies rotary position embeddings to the input.

        Args:
            xq (torch.Tensor): The query tensor.
            xk (torch.Tensor): The key tensor.
            seq_len (int): The sequence length.
            start (int, optional): The starting position. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The tensors with rotary position embeddings applied.
        """
        # Ensure seq_len is an integer
        if isinstance(seq_len, torch.Tensor):
            seq_len = seq_len.item()

        freqs_cis = self.get_freqs_cis(start, seq_len)

        # Apply rotary embedding
        # Assuming `x` has shape (batch_size, seq_len, embed_dim)
        cos = freqs_cis[:, :, 0].unsqueeze(0).unsqueeze(-1)  # (1, seq_len, embed_dim//2, 1)
        sin = freqs_cis[:, :, 1].unsqueeze(0).unsqueeze(-1)  # (1, seq_len, embed_dim//2, 1)

        # Repeat along batch and sequence dimensions
        cos = cos.repeat(xq.shape[0], 1, 1, 1)  # (batch_size, seq_len, embed_dim//2, 1)
        sin = sin.repeat(xq.shape[0], 1, 1, 1)  # (batch_size, seq_len, embed_dim//2, 1)

        # Split the embedding dimension into two halves
        xq1, xq2 = torch.chunk(xq, 2, dim=-1)  # (batch_size, seq_len, embed_dim//2)
        xk1, xk2 = torch.chunk(xk, 2, dim=-1)  # (batch_size, seq_len, embed_dim//2)

        # Adjust dimensions of cos and sin to match xq1, xq2, xk1, and xk2
        cos = cos.repeat(1, 1, xq1.shape[-1], 1)  # (batch_size, seq_len, embed_dim//2, 1)
        sin = sin.repeat(1, 1, xk1.shape[-1], 1)  # (batch_size, seq_len, embed_dim//2, 1)

        # Apply rotation
        xq_cos = xq1 * cos - xq2 * sin
        xq_sin = xq1 * sin + xq2 * cos
        xk_cos = xk1 * cos - xk2 * sin
        xk_sin = xk1 * sin + xk2 * cos

        # Concatenate the results along the last dimension
        xq_out = torch.cat((xq_cos, xq_sin), dim=-1)
        xk_out = torch.cat((xk_cos, xk_sin), dim=-1)

        return xq_out, xk_out

    @staticmethod
    def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Applies rotary position embeddings given precomputed cos and sin."""
        cos = cos[:, :, : x.shape[-1]]
        sin = sin[:, :, : x.shape[-1]]
        x_embed = (x * cos) + (rotate_half(x) * sin)
        return x_embed.type_as(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)