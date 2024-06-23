# core/models/layers.py
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import PretrainedConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads

        self.W_q = nn.Linear(config.d_model, config.d_model)
        self.W_k = nn.Linear(config.d_model, config.d_model)
        self.W_v = nn.Linear(config.d_model, config.d_model)
        self.W_o = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)
        self.window_size = config.sliding_window_size

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        # Linear transformations
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Apply sliding window attention
        attn_scores = self._apply_sliding_window_attention(attn_scores)

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(attention_mask, batch_size, seq_len)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        # Compute attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute output
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)

        # Prepare outputs
        present_key_value = (k, v) if use_cache else None
        if output_attentions:
            return output, present_key_value, attn_probs
        else:
            return output, present_key_value, None

    def _apply_sliding_window_attention(self, attn_scores: torch.Tensor) -> torch.Tensor:
        batch_size, n_heads, seq_len, _ = attn_scores.shape

        # If sequence length is less than or equal to window size, no need to apply sliding window
        if seq_len <= self.window_size:
            return attn_scores

        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=attn_scores.device)).bool()

        # Reshape attention scores and causal mask for sliding window
        attn_scores = attn_scores.view(
            batch_size,
            n_heads,
            seq_len // self.window_size,
            self.window_size,
            seq_len // self.window_size,
            self.window_size,
        )
        causal_mask = causal_mask.view(
            seq_len // self.window_size,
            self.window_size,
            seq_len // self.window_size,
            self.window_size,
        )

        # Apply causal mask
        attn_scores = attn_scores.masked_fill(
            ~causal_mask.unsqueeze(0).unsqueeze(0),
            float("-inf"),
        )

        # Reshape back
        attn_scores = attn_scores.view(batch_size, n_heads, seq_len, seq_len)

        return attn_scores

    @staticmethod
    def _prepare_attention_mask(attention_mask: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Prepare the attention mask for multi-head attention.

        Args:
            attention_mask (torch.Tensor): The attention mask tensor.
            batch_size (int): The batch size.
            seq_len (int): The sequence length.

        Returns:
            torch.Tensor: The prepared attention mask.
        """
        if attention_mask.dim() == 2:
            # Resize the attention mask to have dimensions [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_len)
        elif attention_mask.dim() == 3:
            # Resize the attention mask to have dimensions [batch_size, 1, seq_len, seq_len]
            attention_mask = attention_mask.view(batch_size, 1, seq_len, seq_len)

        return attention_mask


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feedforward Network.

    This module applies a two-layer feedforward network to each position in the sequence.


    Example:
        >>> config = PretrainedConfig(d_model=512, d_ff=2048, ...)
        >>> ff_network = PositionwiseFeedForward(config)
        >>> x = torch.randn(1, 100, 512)  # Example input tensor
        >>> output = ff_network(x)
        >>> print(output.shape)
        torch.Size([1, 100, 512])
    """

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = (
            nn.GELU(approximate="tanh") if config.use_gelu_approximation else nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class GaussianProcessLayer(nn.Module):
    """
    Gaussian Process layer with a Radial Basis Function (RBF) kernel.

    This layer models uncertainty over a sequence by applying a Gaussian Process.
    It uses an RBF kernel to measure similarity between points in the sequence.

    Attributes:
        in_features (int): The input feature dimension.
        out_features (int): The output feature dimension.
        n_inducing (int): The number of inducing points for the Gaussian Process.
        embedding_dim (int): The embedding dimension (used for reshaping inducing points).

    Example:
        >>> gp_layer = GaussianProcessLayer(in_features=512, out_features=512, n_inducing=10, embedding_dim=512)
        >>> x = torch.randn(1, 100, 512) # Example input tensor (batch_size, seq_len, embedding_dim)
        >>> mean, variance = gp_layer(x, seq_len=100)
        >>> print(mean.shape, variance.shape) # Both outputs have shape (batch_size, seq_len, embedding_dim)
        torch.Size([1, 100, 512]) torch.Size([1, 100, 512])
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            n_inducing: int = 10,
            embedding_dim: int = 512,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_inducing = n_inducing
        self.embedding_dim = embedding_dim

        self.covar_module = nn.Linear(
            self.embedding_dim, n_inducing, bias=False
        )  # Projects inducing points to covariance matrix
        self.mean_module = nn.Linear(
            in_features, out_features
        )  # Calculates the mean based on the input

        self.noise = nn.Parameter(torch.tensor(0.1))  # Learnable noise parameter

    def forward(
            self, x: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Gaussian Process layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, in_features).
            seq_len (int): Length of the sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The mean and variance tensors.
        """
        batch_size, _, _ = x.shape
        # Generate inducing points linearly spaced over the sequence length
        inducing_points = torch.linspace(
            0, seq_len - 1, self.n_inducing, device=x.device
        )

        # Reshape inducing points to (batch_size, n_inducing, embedding_dim)
        inducing_points = (
            inducing_points.unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch_size, -1, self.embedding_dim)
        )

        # Calculate covariance matrix using the RBF kernel
        covar = self.covar_module(inducing_points)
        covar = F.softplus(covar)  # Apply Softplus for non-negative covariance

        # Calculate the mean
        mean = self.mean_module(x)

        # Calculate variance (summing over inducing point dimension)
        variance = covar.sum(dim=1, keepdim=True)
        variance = F.softplus(variance) + self.noise  # Apply Softplus and add noise

        return mean, variance


class CEMA(nn.Module):
    """
    Conditional Embedding with Memory Augmentation (CEMA) using Exponential Moving Average.

    This layer applies an exponential moving average (EMA) to the input sequence,
    effectively smoothing it and capturing long-range dependencies.

    Attributes:
        d_model (int): The input and output feature dimension.
        alpha (float): The decay factor for the EMA.

    Example:
        >>> cema_layer = CEMA(d_model=512, alpha=0.99)
        >>> x = torch.randn(1, 100, 512) # Example input (batch_size, seq_len, d_model)
        >>> output = cema_layer(x)
        >>> print(output.shape) # Output shape remains (batch_size, seq_len, d_model)
        torch.Size([1, 100, 512])
    """

    def __init__(self, d_model: int, alpha: float = 0.99):
        super().__init__()
        self.ema = None
        self.d_model = d_model
        self.alpha = alpha
        self.register_buffer(
            "ema", torch.zeros(d_model)
        )  # Buffer to store the EMA state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CEMA (EMA) to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The output tensor with EMA applied.
        """
        batch_size, seq_len, _ = x.shape
        output = []

        # Calculate EMA for each time step
        for i in range(seq_len):
            self.ema = self.alpha * self.ema + (1 - self.alpha) * x[:, i, :]
            output.append(self.ema)

        # Stack the EMA outputs to form the final output tensor
        return torch.stack(output, dim=1)


class MambaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand_factor = config.expand_factor
        self.d_inner = int(self.expand_factor * self.d_model)

        logger.debug(
            f"MambaLayer init: d_model={self.d_model}, d_state={self.d_state}, d_inner={self.d_inner}"
        )

        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)
        self.in_proj_bias = nn.Parameter(
            torch.randn(self.d_inner, self.d_state)
        )  # This is 'B' in the paper

        # Convolution layer
        self.conv = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.d_inner,
        )

        # Activation function
        self.activation = nn.SiLU()

        # Projections for Δ, B, and C
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2)

        # Learnable parameters
        self.dt = nn.Parameter(torch.randn(self.d_inner))
        self.A = nn.Parameter(torch.randn(self.d_state, self.d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model)

    def forward(self, x: torch.Tensor, **_) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        logger.debug(f"MambaLayer forward: input shape = {x.shape}")

        # Input projection and splitting
        x_and_res = self.in_proj(x)
        x, res = x_and_res.chunk(2, dim=-1)

        # Apply convolution
        x = x.transpose(1, 2)
        x = self.conv(x)[:, :, :seq_len]
        x = x.transpose(1, 2)

        # Apply activation
        x = self.activation(x)

        # Project x and split into B and C
        x_dbl = self.x_proj(x)
        B, C = x_dbl.chunk(2, dim=-1)

        # Compute Δ
        dt = torch.exp(self.dt.unsqueeze(0).unsqueeze(0))  # Shape: (1, 1, d_inner)
        logger.debug(f"MambaLayer: dt shape after exp = {dt.shape}")

        # Apply selective scan
        y = self._selective_scan(x, dt, self.A, B, C, self.D)

        # Add residual connection
        y = y + res

        # Final projection
        return self.out_proj(y)

    @classmethod
    def _selective_scan(
            cls,
            x: torch.Tensor,
            dt: torch.Tensor,
            A: torch.Tensor,
            B: torch.Tensor,
            C: torch.Tensor,
            D: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, d_inner = x.shape
        d_state = A.shape[0]

        # Ensure dt has the correct dimensions
        dt = dt[:, :, :d_state]

        logger.debug(
            f"_selective_scan: x shape = {x.shape}, dt shape = {dt.shape}, A shape = {A.shape}, B shape = {B.shape}, C shape = {C.shape}, D shape = {D.shape}")

        x = x * D.unsqueeze(0).unsqueeze(0)

        h = torch.zeros(batch_size, d_state, device=x.device)

        hs = []
        for i in range(seq_len):
            u = x[:, i, :]
            A_t = torch.exp(dt[:, :, i % dt.size(2)] * A)
            h = torch.tanh(A_t @ h.unsqueeze(-1)).squeeze(-1) + B[:, i, :] * h + C[:, i, :]
            hs.append(h)

        hs = torch.stack(hs, dim=1)
        return hs.repeat_interleave(d_inner // d_state, dim=-1)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.attention = MultiHeadAttention(config)
        self.mamba_layer = MambaLayer(config) if config.use_mamba else nn.Identity()
        self.feed_forward = PositionwiseFeedForward(config)
        self.layer_norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.layer_norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.layer_norm3 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor]]]:
        logger.debug(f"TransformerEncoderLayer input shape: {x.shape}")

        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension

        residual = x
        x = self.layer_norm1(x)
        attention_output, present_key_value, attention_probs = self.attention(
            x,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions
        )
        x = residual + self.dropout(attention_output)

        residual = x
        x = self.layer_norm2(x)
        mamba_outputs = self.mamba_layer(x)
        x = (
            mamba_outputs
            if isinstance(mamba_outputs, torch.Tensor)
            else mamba_outputs[0]
        )
        x = residual + self.dropout(x)

        residual = x
        x = self.layer_norm3(x)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)

        # Return present_key_value if use_cache is True
        if self.config.use_cache:
            return x, present_key_value
        else:
            return x
