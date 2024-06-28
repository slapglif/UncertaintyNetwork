from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class TimestepNorm(nn.Module):
    """
    A simplified Timestep Normalization layer.

    This implementation uses a running average of the input features to normalize
    them over time. It does not include group normalization or a custom CUDA kernel.

    Args:
        dim (int): The dimension of the input features.
        eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-5.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0))
        self.momentum = 0.99  # You can adjust this momentum parameter

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies Timestep Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            padding_mask (torch.Tensor, optional): Mask to indicate padded tokens.

        Returns:
            torch.Tensor: Normalized output tensor of shape (batch_size, seq_len, dim).
        """
        batch_size, seq_len, _ = x.shape

        # Calculate the masked mean and variance (if padding_mask is provided)
        if padding_mask is not None:
            masked_x = x * padding_mask.unsqueeze(-1)  # Apply padding mask
            valid_count = padding_mask.sum(dim=1, keepdim=True).float()
            mean = masked_x.sum(dim=1, keepdim=True) / (valid_count + self.eps)
            variance = ((masked_x - mean) ** 2).sum(dim=1, keepdim=True) / (valid_count + self.eps)
        else:
            mean = x.mean(dim=1, keepdim=True)
            variance = x.var(dim=1, keepdim=True)

        # Update running statistics
        self.count += batch_size
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.mean(dim=0)
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * variance.mean(dim=0)

        return (x - self.running_mean.to(x.device)) / torch.sqrt(
            self.running_var.to(x.device) + self.eps
        )
