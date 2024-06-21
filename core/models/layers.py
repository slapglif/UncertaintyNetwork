from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.einsum('bhij,bhjd->bhid', attn, v)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(d_model, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bs = x.size(0)
        qkv = self.W_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(bs, -1, self.n_heads, self.d_k).transpose(1, 2), qkv)

        if mask is not None:
            mask = mask.unsqueeze(1)

        attn_output = self.attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        out = self.W_o(attn_output)
        out = self.dropout(out)

        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, mask)
        x = residual + self.dropout1(x)

        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + self.dropout2(x)

        return x
