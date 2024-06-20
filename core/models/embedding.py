import math

import torch
import torch.nn as nn


class MatryoshkaEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 5000, n_layers: int = 3):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.n_layers = n_layers

        self.embedding_layers = nn.ModuleList([nn.Embedding(vocab_size, d_model) for _ in range(n_layers)])
        self.position_embeddings = nn.ModuleList(
            [self.create_position_embedding(d_model, max_len) for _ in range(n_layers)])

    def create_position_embedding(self, d_model: int, max_len: int):
        position_embedding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
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
