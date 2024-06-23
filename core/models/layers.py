# Importing the key components from your FasterKAN implementation

import math
from typing import Optional, Any
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.fasterkan.fasterkan_basis import ReflectionalSwitchFunction, SplineLinear
from core.models.uncertain_nn import UncertainTransformerConfig


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer with Sliding Window Attention (SWA) support.

    This module implements multi-head attention, optionally with Sliding Window Attention
    for efficient processing of long sequences.



    Example:
        >>> config = UncertainTransformerConfig(d_model=512, n_heads=8, max_position_embeddings=1024, ...)
        >>> attention_layer = MultiHeadAttention(config)
        >>> x = torch.randn(1, 100, 512) # Example input tensor
        >>> attention_mask = torch.ones(1, 100) # Example attention mask
        >>> outputs = attention_layer(x, attention_mask, use_cache=False)
        >>> print(outputs[0].shape)
        torch.Size([1, 100, 512])
    """

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
        self.window_size = (
            config.max_position_embeddings // 2
        )  # Example: Window size is half the max_position_embeddings

    # noinspection PyTypeChecker
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        _: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        # Project input into query, key, and value matrices
        q = (
            self.W_q(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        k = (
            self.W_k(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        v = (
            self.W_v(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        # Cache key and value tensors if use_cache is True
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply Sliding Window Attention (SWA) if sequence length exceeds window size
        if self.window_size < seq_len:
            attn_scores = self._apply_sliding_window_attention(attn_scores)

        # Apply attention mask (if provided)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf")
            )

        # Normalize scores with Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Weighted sum of values
        attn_output = torch.matmul(attn_probs, v)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # Final linear projection
        output = self.W_o(attn_output)

        past_key_value = (k, v) if use_cache else None
        if output_attentions:
            return output, past_key_value, attn_probs
        else:
            return output, past_key_value, None

    def _apply_sliding_window_attention(
        self, attn_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies sliding window attention to the attention scores.

        Args:
            attn_scores: Attention scores of shape (batch_size, n_heads, seq_len, seq_len).

        Returns:
            torch.Tensor: Attention scores with SWA applied.
        """
        batch_size, n_heads, seq_len, _ = attn_scores.shape

        # Create a causal mask to prevent attending to future tokens
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=attn_scores.device)
        ).bool()

        # Pad attention scores to handle cases where seq_len is not a multiple of window_size
        padding_len = (
            self.window_size - (seq_len % self.window_size)
        ) % self.window_size
        attn_scores = F.pad(
            attn_scores, (0, padding_len, 0, padding_len), value=float("-inf")
        )
        padded_len = seq_len + padding_len

        # Reshape attention scores for efficient sliding window calculations
        attn_scores = attn_scores.view(
            batch_size,
            n_heads,
            padded_len // self.window_size,
            self.window_size,
            padded_len // self.window_size,
            self.window_size,
        )

        # Apply causal mask within each window
        attn_scores = attn_scores.masked_fill(
            ~causal_mask.view(
                1,
                1,
                padded_len // self.window_size,
                self.window_size,
                padded_len // self.window_size,
                self.window_size,
            ),
            float("-inf"),
        )

        # Apply softmax within each window
        attn_scores = F.softmax(attn_scores, dim=-1)

        # Reshape back to the original shape
        attn_scores = attn_scores.view(batch_size, n_heads, padded_len, padded_len)

        # Remove padding
        attn_scores = attn_scores[:, :, :seq_len, :seq_len]

        return attn_scores


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feedforward Network.

    This module applies a two-layer feedforward network to each position in the sequence.


    Example:
        >>> config = UncertainTransformerConfig(d_model=512, d_ff=2048, ...)
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
    """
    Mamba layer for sequence modeling with selective state space modeling.

    This layer implements the Mamba architecture, which uses a selective
    state space model for efficient sequence processing. It includes
    mechanisms for adaptive time decay and efficient convolution operations.

    Attributes:
        config (UncertainTransformerConfig): Configuration object for the layer.
        in_proj (nn.Linear): Input projection layer.
        conv (nn.Conv1d): Depthwise convolution layer.
        x_proj (nn.Linear): Projection for input-dependent parameters.
        dt (nn.Parameter): Learnable time decay parameter.
        A (nn.Parameter): State transition matrix.
        D (nn.Parameter): Output projection matrix.
        out_proj (nn.Linear): Output projection layer.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand_factor = config.expand_factor
        self.d_inner = int(self.expand_factor * self.d_model)
        self.dt_rank = config.dt_rank or math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(self.d_model, self.d_inner)
        self.conv = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state)

        # Initialize A, D, and dt
        self.A = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.dt = nn.Parameter(torch.rand(self.dt_rank))
        self._initialize_dt()

        self.out_proj = nn.Linear(self.d_inner, self.d_model)

    def _initialize_dt(self):
        """Initialize the dt parameter within the specified range."""
        nn.init.uniform_(
            self.dt, a=math.log(self.config.dt_min), b=math.log(self.config.dt_max)
        )
        self.dt.data.exp_()

    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform the selective scan operation for the Mamba layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_inner).
            delta (torch.Tensor): Time decay tensor of shape (batch_size, seq_len, d_inner).
            A (torch.Tensor): State transition matrix of shape (d_inner, d_state).
            B (torch.Tensor): Input projection matrix of shape (batch_size, seq_len, d_state).
            C (torch.Tensor): Output projection matrix of shape (batch_size, seq_len, d_state).
            D (torch.Tensor): Direct feedthrough matrix of shape (d_inner,).
            z (Optional[torch.Tensor]): Optional initial state.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_inner).
        """
        batch_size, seq_len, _ = x.shape

        # Initialize state
        if z is None:
            z = torch.zeros(batch_size, self.d_state, device=x.device, dtype=x.dtype)

        output = []
        for t in range(seq_len):
            # Update state
            z = torch.einsum(
                "bd,bud->bu", z, torch.exp(-torch.clamp(delta[:, t] * A, min=-5, max=5))
            )
            z = z + torch.einsum("bud,bd->bu", B[:, t].unsqueeze(1), x[:, t])

            # Compute output
            y = torch.einsum("bu,bud->bd", z, C[:, t].unsqueeze(1))
            y = y + D * x[:, t]
            output.append(y)

        return torch.stack(output, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Mamba layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.in_proj(x)

        # Apply short convolution for local context
        x_conv = self.conv(x.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)

        # Project for delta, B, and C
        x_dbc = self.x_proj(x_conv)
        delta, B, C = torch.split(
            x_dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Compute delta
        delta = F.softplus(delta @ self.dt)

        # Perform selective scan
        y = self._selective_scan(x, delta, self.A, B, C, self.D)

        # Project output
        y = self.out_proj(y)

        return y


class SentenceEncoder(nn.Module):
    """
    Sentence Encoder implementing a Kolmogorov-Arnold Network (KAN) for transforming sentence representations.

    This module applies a series of RBF-based transformations followed by linear projections
    to create a semantically richer representation of the sentence embeddings.

    Attributes:
        input_dim (int): The input feature dimension (embedding dimension of the sentence).
        hidden_dim (int): The hidden dimension of the intermediate linear layers.
        output_dim (int): The output feature dimension.
        num_grids (int): The number of grids used in the RBF transformation.
        rbf (ReflectionalSwitchFunction): RBF transformation layer.
        spline_linear (SplineLinear): Linear projection after RBF.
        output_layer (nn.Linear): Final linear layer.

    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_grids: int = 8
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_grids = num_grids

        self.rbf = ReflectionalSwitchFunction(num_grids=num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SentenceEncoder (KAN).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_sentences, sentence_len, embedding_dim).

        Returns:
            torch.Tensor: Transformed sentence embedding of shape (batch_size, num_sentences, output_dim).
        """
        batch_size, num_sentences, sentence_len, _ = x.shape

        # Reshape for RBF transformation
        x = x.view(batch_size * num_sentences * sentence_len, -1)

        # Apply RBF transformation
        spline_basis = self.rbf(x).view(batch_size * num_sentences * sentence_len, -1)

        # Apply spline linear transformation
        x = self.spline_linear(spline_basis)

        # Reshape and pool across sentence length
        x = x.view(batch_size, num_sentences, sentence_len, -1).mean(dim=2)

        # Final linear projection
        x = self.output_layer(x)

        return x


class SentenceGP(nn.Module):
    """
    Sentence-level Gaussian Process layer with RBF kernel for uncertainty estimation.

    This module applies a Gaussian Process to sentence embeddings to model uncertainty
    at the sentence level using an RBF kernel.

    Attributes:
        input_dim (int): The input feature dimension.
        output_dim (int): The output feature dimension.
        n_inducing (int): The number of inducing points for the Gaussian Process.
        embedding_dim (int): The embedding dimension used for reshaping inducing points.
    """

    def __init__(
        self, input_dim: int, output_dim: int, n_inducing: int, embedding_dim: int
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_inducing = n_inducing
        self.embedding_dim = embedding_dim

        # Learnable inducing points
        self.inducing_points = nn.Parameter(torch.randn(n_inducing, input_dim))

        # Learnable length scale for RBF kernel
        self.log_lengthscale = nn.Parameter(torch.zeros(1))

        # Learnable noise parameter
        self.log_noise = nn.Parameter(torch.zeros(1))

        # Output projection
        self.output_proj = nn.Linear(n_inducing, output_dim)

    def rbf_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel between two sets of points.

        Args:
            x1 (torch.Tensor): First set of points of shape (..., input_dim).
            x2 (torch.Tensor): Second set of points of shape (..., input_dim).

        Returns:
            torch.Tensor: Kernel matrix of shape (..., x1.shape[:-1], x2.shape[:-1]).
        """
        dist = torch.cdist(x1, x2, p=2).pow(2)
        return torch.exp(-0.5 * dist / torch.exp(self.log_lengthscale).pow(2))

    def forward(
        self, x: torch.Tensor, num_sentences: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the SentenceGP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_sentences, input_dim).
            num_sentences (int): Number of sentences in the input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Mean tensor of shape (batch_size, num_sentences, output_dim).
                - Variance tensor of shape (batch_size, num_sentences, output_dim).
        """
        batch_size = x.shape[0]

        # Compute kernel matrices
        K_xx = self.rbf_kernel(x, x)
        K_xi = self.rbf_kernel(x, self.inducing_points)
        K_ii = self.rbf_kernel(self.inducing_points, self.inducing_points)

        # Compute predictive distribution
        K_ii_inv = torch.inverse(
            K_ii
            + torch.exp(self.log_noise) * torch.eye(self.n_inducing, device=x.device)
        )
        mean = K_xi @ K_ii_inv @ self.output_proj.weight.T
        var = K_xx - K_xi @ K_ii_inv @ K_xi.transpose(-1, -2)

        # Reshape outputs
        mean = mean.view(batch_size, num_sentences, self.output_dim)
        var = var.view(batch_size, num_sentences, num_sentences)
        var = (
            torch.diagonal(var, dim1=-2, dim2=-1)
            .unsqueeze(-1)
            .expand(-1, -1, self.output_dim)
        )

        return mean, F.softplus(var)  # Ensure positive variance


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with optional Mamba integration and CEMA.

    This layer implements a flexible transformer encoder that can use either
    standard self-attention or Mamba for sequence modeling, followed by a
    feedforward network and CEMA for enhanced temporal dependency modeling.

    Attributes:
        config (UncertainTransformerConfig): Configuration object for the layer.
        self_attn (nn.MultiheadAttention): Multi-head self-attention layer (if not using Mamba).
        mamba (MambaLayer): Mamba layer for sequence modeling (if using Mamba).
        ff_layer (nn.Sequential): Feedforward network.
        cema (CEMA): Conditional Embedding with Memory Augmentation layer.
        layer_norm1 (nn.LayerNorm): Layer normalization before self-attention or Mamba.
        layer_norm2 (nn.LayerNorm): Layer normalization before feedforward network.
        layer_norm3 (nn.LayerNorm): Layer normalization after CEMA.
        dropout (nn.Dropout): Dropout layer.
    """

    def __init__(self, config: UncertainTransformerConfig):
        super().__init__()
        self.config = config

        if not config.use_mamba:
            self.self_attn = nn.MultiheadAttention(
                config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
            )
        else:
            self.mamba = MambaLayer(config)

        self.ff_layer = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

        self.cema = CEMA(config.d_model)

        self.layer_norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.layer_norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.layer_norm3 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)

        if config.use_flash_attention and not config.use_mamba:
            try:
                from flash_attn import flash_attn_func

                self.flash_attn = flash_attn_func
            except ImportError:
                print(
                    "Flash Attention is not available. Falling back to standard attention."
                )
                self.flash_attn = None
        else:
            self.flash_attn = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[Tensor | Any]:
        """
        Forward pass of the TransformerEncoderLayer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, seq_len).
            position_ids (Optional[torch.Tensor]): Position IDs of shape (batch_size, seq_len).
            past_key_value (Optional[Tuple[torch.Tensor]]): Cached key and value tensor for incremental decoding.
            output_attentions (bool): Whether to output attention weights.
            use_cache (bool): Whether to use cached key and value tensors.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
                - Output tensor of shape (batch_size, seq_len, d_model)
                - Attention weights (if output_attentions is True)
                - Cached key and value tensors (if use_cache is True)
        """
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        if self.config.use_mamba:
            hidden_states = self.mamba(hidden_states)
            attn_weights = None  # Mamba doesn't produce attention weights

        elif self.flash_attn is not None and attention_mask is None:
            hidden_states = self.flash_attn(hidden_states, hidden_states, hidden_states)
        else:
            hidden_states, attn_weights = self.self_attn(
                hidden_states,
                hidden_states,
                hidden_states,
                attn_mask=attention_mask,
                need_weights=output_attentions,
            )
        hidden_states = residual + self.dropout(hidden_states)
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.ff_layer(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        # Apply CEMA
        residual = hidden_states
        hidden_states = self.layer_norm3(hidden_states)
        hidden_states = self.cema(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (past_key_value,)

        return outputs
