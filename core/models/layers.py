# core/models/layers.py

from typing import Optional, Tuple
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import PretrainedConfig

from core.utils.utils import _check_nan_inf


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
        logger.debug(f"Input shape: {x.shape}")

        # Linear transformations
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Handle past key/value states
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
            seq_len = k.size(1)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        logger.debug(f"Query shape: {q.shape}, Key shape: {k.shape}, Value shape: {v.shape}")

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        logger.debug(f"Attention scores shape: {attn_scores.shape}")

        # Apply sliding window attention if window_size is set
        if self.window_size is not None:
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
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)

        logger.debug(f"Output shape: {output.shape}")

        # Prepare outputs
        present_key_value = (k, v) if use_cache else None
        if output_attentions:
            return output, present_key_value, attn_probs
        else:
            return output, present_key_value, None

    def _apply_sliding_window_attention(self, attn_scores: torch.Tensor) -> torch.Tensor:
        batch_size, n_heads, seq_len, _ = attn_scores.shape
        logger.debug(f"Applying sliding window attention. attn_scores shape: {attn_scores.shape}")

        if seq_len <= self.window_size:
            logger.warning(f"Sequence length {seq_len} is shorter than or equal to window size {self.window_size}. Applying causal mask only.")
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=attn_scores.device, dtype=torch.bool))
            return attn_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Calculate the number of windows and padding
        num_windows = (seq_len + self.window_size - 1) // self.window_size
        padding_len = num_windows * self.window_size - seq_len

        # Pad the attention scores if necessary
        if padding_len > 0:
            logger.debug(f"Padding attention scores with {padding_len} zeros")
            attn_scores = F.pad(attn_scores, (0, padding_len, 0, padding_len))

        # Reshape the attention scores to group by windows
        attn_scores = attn_scores.view(
            batch_size,
            n_heads,
            num_windows,
            self.window_size,
            num_windows,
            self.window_size
        )

        # Create a causal mask to prevent attending to future tokens within each window
        causal_mask = torch.tril(torch.ones(self.window_size, self.window_size, device=attn_scores.device, dtype=torch.bool))
        causal_mask = causal_mask.view(1, 1, 1, self.window_size, 1, self.window_size)

        # Apply the causal mask within each window
        attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))

        # Reshape the attention scores back to the original shape
        attn_scores = attn_scores.view(batch_size, n_heads, num_windows * self.window_size, num_windows * self.window_size)

        # Remove padding if it was added
        if padding_len > 0:
            attn_scores = attn_scores[:, :, :seq_len, :seq_len]

        logger.debug(f"Sliding window attention applied. Output shape: {attn_scores.shape}")
        return attn_scores

    @staticmethod
    def _prepare_attention_mask(attention_mask: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        logger.debug(f"Preparing attention mask. Input shape: {attention_mask.shape}, batch_size: {batch_size}, seq_len: {seq_len}")

        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        if attention_mask.dim() == 2:
            if attention_mask.shape[1] == 1:
                # Broadcast single token mask across sequence length
                attention_mask = attention_mask.expand(batch_size, seq_len)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        # Ensure the mask covers the entire sequence length
        if attention_mask.shape[-1] < seq_len:
            logger.warning(f"Attention mask is shorter than sequence length. Padding mask from {attention_mask.shape[-1]} to {seq_len}")
            padding = torch.ones(batch_size, 1, 1, seq_len - attention_mask.shape[-1], device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, padding], dim=-1)

        # Ensure the mask is properly broadcastable
        if attention_mask.shape[2] == 1:
            attention_mask = attention_mask.expand(-1, -1, seq_len, -1)

        logger.debug(f"Attention mask prepared. Output shape: {attention_mask.shape}")
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
        """
        Forward pass through the position-wise feedforward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
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
        self.register_buffer("ema", torch.zeros(d_model))

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
    """
    Mamba layer for sequence processing, inspired by the Mamba neural network architecture.

    This layer uses a combination of convolution, selective scan operation, and gating mechanisms
    to capture complex temporal dependencies in sequential data.

    Attributes:
        config (PretrainedConfig): Configuration object containing hyperparameters.
        d_model (int): Dimension of the input and output features.
        d_state (int): Dimension of the internal state.
        d_conv (int): Kernel size for the convolution operation.
        expand_factor (float): Expansion factor for the inner dimension.
        d_inner (int): Expanded inner dimension.
        in_proj (nn.Linear): Linear projection for the input.
        in_proj_bias (nn.Parameter): Bias term for the input projection.
        conv (nn.Conv1d): 1D convolution layer.
        activation (nn.Module): Activation function (SiLU).
        x_proj (nn.Linear): Linear projection for the intermediate output.
        dt (nn.Parameter): Learnable time decay parameter.
        A (nn.Parameter): Learnable parameter for state update.
        D (nn.Parameter): Learnable parameter for scaling.
        out_proj (nn.Linear): Linear projection for the output.
    """

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
        self.in_proj_bias = nn.Parameter(torch.randn(self.d_inner, self.d_state))

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
        """
        Forward pass through the Mamba layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
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
        dt = torch.exp(self.dt.unsqueeze(0).unsqueeze(0))
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
        """
        Performs the selective scan operation, a core component of the Mamba layer.

        Args:
            x (torch.Tensor): Intermediate tensor after convolution and activation.
            dt (torch.Tensor): Time decay parameter.
            A (torch.Tensor): Parameter for state update.
            B (torch.Tensor): Parameter for state gating.
            C (torch.Tensor): Parameter for state bias.
            D (torch.Tensor): Parameter for input scaling.

        Returns:
            torch.Tensor: Output tensor after the selective scan operation.
        """
        batch_size, seq_len, d_inner = x.shape
        d_state = A.shape[0]

        # Ensure dt has the correct dimensions:
        dt = dt[:, :, :d_state].repeat(1, 1, (seq_len // dt.size(2)) + 1)[:, :, :seq_len]

        logger.debug(
            f"_selective_scan: x shape = {x.shape}, dt shape = {dt.shape}, A shape = {A.shape}, B shape = {B.shape}, C shape = {C.shape}, D shape = {D.shape}"
        )

        # Scale the input
        x = x * D.unsqueeze(0).unsqueeze(0)

        # Initialize hidden state
        h = torch.zeros(batch_size, d_state, device=x.device)

        hs = []
        for i in range(seq_len):
            _ = x[:, i, :]
            # Calculate A_t using dt, ensuring correct indexing and dimension alignment
            A_t = torch.exp(dt[:, :, i] * A)  # Use i directly for indexing

            # Update hidden state h
            h = torch.tanh(A_t @ h.unsqueeze(-1)).squeeze(-1) + B[:, i, :] * h + C[:, i, :]

            # Check for NaN or Inf values in h
            _check_nan_inf(h, "hidden state h")

            hs.append(h)

        hs = torch.stack(hs, dim=1)

        # Repeat the hidden states to match the inner dimension
        return hs.repeat_interleave(d_inner // d_state, dim=-1)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer.

    This module combines multi-head attention, a Mamba layer (optional), and a position-wise
    feedforward network to form a single encoder layer in a Transformer architecture.

    Attributes:
        config (PretrainedConfig): Configuration object containing hyperparameters.
        attention (MultiHeadAttention): Multi-head attention module.
        mamba_layer (MambaLayer): Mamba layer (optional, used if config.use_mamba is True).
        feed_forward (PositionwiseFeedForward): Position-wise feedforward network.
        layer_norm1 (nn.LayerNorm): Layer normalization applied before attention.
        layer_norm2 (nn.LayerNorm): Layer normalization applied before Mamba layer.
        layer_norm3 (nn.LayerNorm): Layer normalization applied before feedforward network.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

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
        """
        Forward pass through the Transformer encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            attention_mask (Optional[torch.Tensor]): Attention mask to prevent attending to padded tokens.
            output_attentions (Optional[bool]): Whether to output attention probabilities.
            past_key_value (Optional[Tuple[torch.Tensor]]): Past key and value tensors for cached attention.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor]]]:
                - Output tensor of shape (batch_size, seq_len, d_model).
                - Tuple containing output tensor and present key/value tuple (if use_cache is True in config).
        """
        logger.debug(f"TransformerEncoderLayer input shape: {x.shape}")

        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        # Apply attention sub-layer
        residual = x
        x = self.layer_norm1(x)
        attention_output, present_key_value, attention_probs = self.attention(
            x,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        x = residual + self.dropout(attention_output)

        # Apply Mamba sub-layer (if enabled)
        residual = x
        x = self.layer_norm2(x)
        mamba_outputs = self.mamba_layer(x)
        x = (
            mamba_outputs
            if isinstance(mamba_outputs, torch.Tensor)
            else mamba_outputs[0]
        )
        x = residual + self.dropout(x)

        # Apply feedforward sub-layer
        residual = x
        x = self.layer_norm3(x)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)

        # Return output and present key/value tuple (if caching is enabled)
        return (x, present_key_value) if self.config.use_cache else x
