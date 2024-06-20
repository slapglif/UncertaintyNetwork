import math

import torch
import torch.nn as nn


class MatryoshkaEmbedding(nn.Module):
    def __init__(
            self, vocab_size: int, d_model: int, max_len: int = 5000, n_layers: int = 3
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.n_layers = n_layers

        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(vocab_size, d_model) for _ in range(n_layers)]
        )
        self.position_embeddings = nn.ModuleList(
            [self.create_position_embedding(d_model, max_len) for _ in range(n_layers)]
        )

    def create_position_embedding(self, d_model: int, max_len: int):
        position_embedding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        position_embedding[:, 0::2] = torch.sin(position * div_term)
        position_embedding[:, 1::2] = torch.cos(position * div_term)
        position_embedding = position_embedding.unsqueeze(0)
        return nn.Parameter(position_embedding, requires_grad=False)

    def forward(self, src: torch.Tensor):
        batch_size, seq_len = src.shape
        embeddings = []

        for i in range(self.n_layers):
            token_embedding = self.embedding_layers[i](src)
            position_embedding = self.position_embeddings[i][:, :seq_len, :]
            embedding = token_embedding + position_embedding
            embeddings.append(embedding)

        stacked_embeddings = torch.stack(embeddings, dim=2)
        matryoshka_embedding = torch.sum(stacked_embeddings, dim=2)

        return matryoshka_embedding


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def apply_rotary_pos_emb(q, k, cos, sin, offset=0):
    cos = cos[:, :, offset: q.shape[1] + offset, :]
    sin = sin[:, :, offset: q.shape[1] + offset, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


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
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
