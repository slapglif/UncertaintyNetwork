# .\core\models\layers.py
# core/models/layers.py
from typing import Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from core.models.kan import SplineNetLayer


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer.

    Args:
        config (UncertainTransformerConfig): The configuration for the model.
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        self.window_size = config.sliding_window_size

        self.W_q = nn.Linear(config.d_model, config.d_model)
        self.W_k = nn.Linear(config.d_model, config.d_model)
        self.W_v = nn.Linear(config.d_model, config.d_model)
        self.W_o = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, use_cache: bool = False,
                **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        q = rearrange(self.W_q(x), 'b l (h d) -> b h l d', h=self.n_heads)
        k = rearrange(self.W_k(x), 'b l (h d) -> b h l d', h=self.n_heads)
        v = rearrange(self.W_v(x), 'b l (h d) -> b h l d', h=self.n_heads)

        # Compute attention scores
        attn_scores = torch.einsum('bhid,bhjd->bhij', q, k) / (self.d_k ** 0.5)

        # Apply sliding window attention
        if self.window_size < seq_len:
            attn_scores = self._apply_sliding_window_attention(attn_scores)

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        # Compute attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute output
        attn_output = torch.einsum('bhij,bhjd->bhid', attn_probs, v)
        output = rearrange(attn_output, 'b h l d -> b l (h d)')
        output = self.W_o(output)

        return (output, attn_probs) if use_cache else (output, None)

    def _apply_sliding_window_attention(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """
        Applies sliding window attention to the attention scores.

        Args:
            attn_scores (torch.Tensor): The attention scores.

        Returns:
            torch.Tensor: The attention scores with sliding window attention applied.
        """
        batch_size, n_heads, seq_len, _ = attn_scores.shape

        # Pad attention scores for easier windowing
        pad_len = (self.window_size - seq_len % self.window_size) % self.window_size
        padded_scores = F.pad(attn_scores, (0, pad_len, 0, pad_len))

        # Reshape for windowed attention
        windowed_scores = rearrange(padded_scores,
                                    'b h (w1 d1) (w2 d2) -> b h w1 w2 d1 d2',
                                    d1=self.window_size, d2=self.window_size)

        # Create causal mask within each window
        causal_mask = torch.tril(torch.ones(self.window_size, self.window_size, device=attn_scores.device))
        causal_mask = repeat(causal_mask, 'd1 d2 -> b h w1 w2 d1 d2', b=batch_size, h=n_heads,
                             w1=windowed_scores.shape[2], w2=windowed_scores.shape[3])

        # Apply causal mask
        windowed_scores = windowed_scores.masked_fill(causal_mask == 0, float('-inf'))

        # Reshape back to original shape
        attn_scores = rearrange(windowed_scores,
                                'b h w1 w2 d1 d2 -> b h (w1 d1) (w2 d2)')

        # Remove padding
        attn_scores = attn_scores[:, :, :seq_len, :seq_len]

        return attn_scores


class KANFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.kan1 = SplineNetLayer(config.d_model, config.d_ff)
        self.kan2 = SplineNetLayer(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = self.kan1(x)
        x = self.dropout(x)
        x = self.kan2(x)
        # Ensure the output has the correct shape
        return x.view(*original_shape[:-1], self.config.d_model)


class CEMA(nn.Module):
    def __init__(self, d_model, alpha=0.99):
        super().__init__()
        self.d_model = d_model
        self.alpha = alpha
        self.register_buffer("ema", torch.zeros(1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        output = []
        for i in range(seq_len):
            self.ema = self.alpha * self.ema + (1 - self.alpha) * x[:, i, :].mean(dim=0, keepdim=True)
            output.append(self.ema.expand(batch_size, -1))
        return torch.stack(output, dim=1)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = KANFeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, use_cache: bool = False,
                **kwargs: Any) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Preserve input shape
        original_shape = x.shape
        batch_size, seq_length, d_model = original_shape

        # Self-attention
        attn_output, attn_weights = self.attention(self.norm1(x), attention_mask, use_cache=use_cache, **kwargs)
        x = x + self.dropout(attn_output)

        # Feed-forward
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)

        # Ensure output shape matches input shape
        assert x.shape == original_shape, f"Shape mismatch in TransformerEncoderLayer: input {original_shape}, output {x.shape}"

        return (x, (attn_weights, ff_output)) if use_cache else (x, None)
