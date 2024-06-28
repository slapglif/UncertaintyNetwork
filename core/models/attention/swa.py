from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import PretrainedConfig


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
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.n_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            # Expand mask to match the shape of attn_weights
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(batch_size, self.num_heads, seq_len, seq_len)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)

        # Apply the mask to the output
        if attention_mask is not None:
            # Use the original mask for output masking
            out = out * attention_mask[:, 0, 0, :].unsqueeze(-1)

            # Print for debugging
            logger.debug(f"attn_weights shape: {attn_weights.shape}")
            logger.debug(f"attention_mask shape: {attention_mask.shape}")

        logger.debug(f"output shape: {out.shape}")
        return out, attn_weights

    def _apply_sliding_window_attention(self, attn_scores: torch.Tensor,
                                        attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies sliding window attention to the attention scores.

        Args:
            attn_scores (torch.Tensor): Attention scores of shape (batch_size, n_heads, seq_len, seq_len).
            attention_mask (torch.Tensor, optional): Attention mask to apply.

        Returns:
            torch.Tensor: Attention scores with sliding window applied.
        """
        batch_size, n_heads, seq_len, _ = attn_scores.shape

        # Apply the attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Initialize the output tensor
        output = torch.zeros_like(attn_scores)

        # Apply sliding window attention
        for i in range(seq_len):
            window_start = max(0, i - self.window_size + 1)
            window_end = min(seq_len, i + 1)

            # Extract the window
            window = attn_scores[:, :, i, window_start:window_end]

            # Apply softmax to the window
            window_probs = F.softmax(window, dim=-1)

            # Place the softmaxed values back into the output tensor
            output[:, :, i, window_start:window_end] = window_probs

        return output
