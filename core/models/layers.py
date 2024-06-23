import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor


class ReZero(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, sublayer):
        return x + self.alpha * sublayer(x)


class GELU_Approximation(nn.Module):
    def forward(self, x):
        return (
                0.5
                * x
                * (
                        1
                        + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
                )
        )


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ):
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        return torch.matmul(attn, v)


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

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.W_o(attn_output)

        if use_cache:
            past_key_value = (k, v)
        else:
            past_key_value = None

        if output_attentions:
            return output, past_key_value, attn_probs
        else:
            return output, past_key_value, None


class PositionwiseFeedForward(nn.Module):
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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.mamba = MambaLayer(config)
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.norm3 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        residual = x
        x = self.norm1(x)

        # Check for NaN/Inf before self-attention
        self._check_nan_inf(x, "input to self-attention")

        self_attn_outputs = self.self_attn(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attn_output = self_attn_outputs[0]
        outputs = self_attn_outputs[1:]

        # Check for NaN/Inf after self-attention
        self._check_nan_inf(attn_output, "output of self-attention")

        x = residual + self.dropout(attn_output)

        # Add Mamba layer
        residual = x
        x = self.norm2(x)

        # Check for NaN/Inf before Mamba layer
        self._check_nan_inf(x, "input to Mamba layer")

        x = self.mamba(x)

        # Check for NaN/Inf after Mamba layer
        self._check_nan_inf(x, "output of Mamba layer")

        x = residual + self.dropout(x)

        residual = x
        x = self.norm3(x)

        # Check for NaN/Inf before feed-forward
        self._check_nan_inf(x, "input to feed-forward")

        x = self.feed_forward(x)

        # Check for NaN/Inf after feed-forward
        self._check_nan_inf(x, "output of feed-forward")

        x = residual + self.dropout(x)

        if use_cache:
            outputs = (x,) + outputs
        else:
            outputs = (x,) + outputs[1:]

        return outputs

    def _check_nan_inf(self, tensor: Tensor, message: str):
        """
        Checks if the given tensor contains NaN or Inf values and logs a warning if found.

        Args:
            tensor (torch.Tensor): The tensor to check.
            message (str): A message to include in the warning log.
        """
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logger.warning(f"NaN or Inf detected in {message}.")


class GaussianProcessLayer(nn.Module):
    """
    Gaussian Process layer for uncertainty quantification.

    This layer applies Gaussian Process regression to the input features,
    providing both mean and variance estimates for uncertainty quantification.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        n_inducing (int): Number of inducing points for sparse GP approximation.
    """

    def __init__(self, in_features: int, out_features: int, n_inducing: int = 10):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_inducing = n_inducing

        # Initialize inducing points
        self.inducing_points = nn.Parameter(torch.randn(n_inducing, in_features))

        # Use separate linear layers for covariance and mean
        self.covar_module = nn.Linear(in_features, n_inducing)
        self.mean_module = nn.Linear(in_features, out_features)

        self.noise = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Gaussian Process regression to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and variance of the GP posterior.
        """
        # Ensure x has the correct shape
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Compute covariance with inducing points
        covar = self.covar_module(x)
        covar = F.softplus(covar)

        # Compute mean
        mean = self.mean_module(x)

        # Compute variance
        variance = covar.sum(dim=-1, keepdim=True)
        variance = F.softplus(variance) + self.noise

        return mean, variance


class NTKAttention(nn.Module):
    """
    Attention mechanism inspired by Neural Tangent Kernel theory.

    This attention mechanism approximates the behavior of an infinitely wide
    network in its NTK regime, providing more stable and generalizable attention.

    Attributes:
        d_model (int): The dimension of the model.
        n_heads (int): The number of attention heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute NTK-inspired attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model).
            mask (Optional[torch.Tensor]): Attention mask.

        Returns:
            torch.Tensor: Output tensor after applying NTK-inspired attention.
        """
        batch_size = q.size(0)

        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Compute output
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Apply output projection
        output = self.w_o(output)

        return output


class CEMA(nn.Module):
    """
    Conditional Embedding with Memory Augmentation (CEMA).

    This class implements the CEMA mechanism, which enhances the model's ability to
    capture long-range dependencies and handle uncertainty.

    Attributes:
        d (int): Input dimension.
        h (int): Hidden dimension.
        chunk_size (int): Size of chunks for processing.
        alpha (nn.Parameter): Learnable parameter for CEMA.
        delta (nn.Parameter): Learnable parameter for CEMA.
        omega (nn.Parameter): Learnable parameter for CEMA.
        beta (nn.Parameter): Learnable parameter for CEMA.
        eta (nn.Parameter): Learnable parameter for CEMA.
    """

    def __init__(self, d: int, h: int, chunk_size: int = 128):
        super().__init__()
        self.d = d
        self.h = h
        self.chunk_size = chunk_size
        self.alpha = nn.Parameter(torch.rand(d, h))
        self.delta = nn.Parameter(torch.rand(d, h))
        self.omega = nn.Parameter(torch.rand(h))
        self.beta = nn.Parameter(torch.randn(d, h))
        self.eta = nn.Parameter(torch.randn(d, h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CEMA to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d).

        Returns:
            torch.Tensor: Output tensor after applying CEMA.
        """
        # Handle 3D input
        if x.dim() == 3:
            batch_size, seq_len, d = x.shape
            x = x.view(batch_size * seq_len, d)
        else:
            batch_size, d = x.shape
            seq_len = 1

        assert d == self.d, f"Input dimension {d} must match CEMA's dimension {self.d}"

        # Compute theta
        theta = torch.outer(torch.arange(self.h, device=x.device), self.omega) * (
                2 * math.pi / self.h
        )
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Compute complex alpha and delta
        alpha_complex = self.alpha[:, None, :] * (
                cos_theta[None, :, :] + 1j * sin_theta[None, :, :]
        )
        delta_complex = self.delta[:, None, :] * (
                cos_theta[None, :, :] + 1j * sin_theta[None, :, :]
        )

        # Process in chunks
        output = []
        for i in range(0, batch_size * seq_len, self.chunk_size):
            chunk = x[i: i + self.chunk_size]

            # Compute u for the chunk
            u = torch.einsum("bd,dh->bdh", chunk, self.beta)

            # Apply CEMA for the chunk
            h = alpha_complex[None, :, :, :] * u[:, :, None, :]
            h += (
                         1 - alpha_complex[None, :, :, :] * delta_complex[None, :, :, :]
                 ) * torch.zeros(
                chunk.size(0), d, 1, self.h, dtype=torch.complex64, device=x.device
            )

            # Compute output for the chunk
            y = torch.einsum("bdhk,dh->bd", h.real, self.eta)
            output.append(y)

        output = torch.cat(output, dim=0)

        # Reshape output back to 3D if input was 3D
        if seq_len > 1:
            output = output.view(batch_size, seq_len, d)

        return output


class MambaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand_factor = config.mamba_expand_factor
        self.d_inner = int(self.expand_factor * self.d_model)
        self.dt_rank = config.mamba_dt_rank or math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(self.d_model, self.d_inner)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=self.d_conv, padding=self.d_conv - 1,
                              groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state)

        self.A = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, self.d_model)

        self.dt = nn.Parameter(torch.rand(self.dt_rank))
        self._initialize_dt(config.mamba_dt_min, config.mamba_dt_max)

    def _initialize_dt(self, dt_min, dt_max):
        nn.init.uniform_(self.dt, a=math.log(dt_min), b=math.log(dt_max))
        self.dt.data.exp_()

    def _selective_scan(self, x: torch.Tensor, delta: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Performs the selective scan operation for the Mamba layer with enhanced numerical stability.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_inner).
            delta (torch.Tensor): The delta tensor of shape (batch_size, seq_len, dt_rank).
            B (torch.Tensor): The B tensor of shape (batch_size, seq_len, d_state).
            C (torch.Tensor): The C tensor of shape (batch_size, seq_len, d_state).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, d_inner).
        """
        batch_size, seq_len, _ = x.shape
        H = torch.zeros(batch_size, seq_len, self.d_state, device=x.device, dtype=x.dtype)

        H[:, 0] = B[:, 0]

        for t in range(1, seq_len):
            delta_t = delta[:, t].unsqueeze(1).unsqueeze(2)
            A_t = self.A[:self.d_state].T.unsqueeze(0)

            # 1. Log-Domain Exponentiation and Scaling
            log_A_t = torch.log(torch.abs(A_t) + 1e-8)  # Logarithm of A_t with small epsilon
            log_delta_t = torch.log(torch.abs(delta_t) + 1e-8)  # Logarithm of delta_t
            log_exp_term = -log_delta_t - log_A_t  # Log of exp_term
            exp_term = torch.exp(log_exp_term - log_exp_term.max())  # Stable exponentiation

            # 2. Clamp exp_term to a safe range (adjust these values)
            exp_term_clip_min = 1e-4  # Adjust this value
            exp_term_clip_max = 1e4  # Adjust this value
            exp_term = torch.clamp(exp_term, min=exp_term_clip_min, max=exp_term_clip_max)

            B_t = B[:, t].unsqueeze(-1)
            H_prev = H[:, t - 1]

            # 3. Optional: Normalize H_prev before multiplication (experiment with this)
            # H_prev = F.normalize(H_prev, dim=-1)

            # Perform matrix multiplication
            H[:, t] = torch.matmul(exp_term, H_prev.unsqueeze(-1)).squeeze(-1) + B_t.squeeze(-1)

        C = C.view(batch_size, seq_len, self.d_state)

        # 4. Optional: Normalize H and C before multiplication (experiment with this)
        # H = F.normalize(H, dim=-1)
        # C = F.normalize(C, dim=-1)

        y = torch.einsum('bnd,bnd->bn', H, C).unsqueeze(-1) * x
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)

        # 5. Clamp the final output 'y' (adjust these values)
        y_clip_min = -1e3  # Adjust this value
        y_clip_max = 1e3  # Adjust this value
        y = torch.clamp(y, min=y_clip_min, max=y_clip_max)

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the Mamba layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, d_model).
        """
        B, L, _ = x.shape

        # 1. Apply Layer Normalization to the input
        x = nn.functional.layer_norm(x, x.shape[-1:])

        x = self.in_proj(x)
        x_conv = self.conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2)

        x_dbc = self.x_proj(x_conv)

        # 2. Apply Layer Normalization before softplus
        x_dbc = nn.functional.layer_norm(x_dbc, x_dbc.shape[-1:])

        # 3. Clamp dt before matrix multiplication
        self.dt.data = torch.clamp(self.dt.data, min=1e-6, max=1e6)

        delta = F.softplus(x_dbc[..., :self.dt_rank] @ self.dt)
        B = x_dbc[..., self.dt_rank:self.dt_rank + self.d_state]
        C = x_dbc[..., -self.d_state:]

        # 4. Apply Layer Normalization to B and C
        B = nn.functional.layer_norm(B, B.shape[-1:])
        C = nn.functional.layer_norm(C, C.shape[-1:])

        y = self._selective_scan(x, delta, B, C)
        output = self.out_proj(y)

        # 5. Apply Layer Normalization to the output
        output = nn.functional.layer_norm(output, output.shape[-1:])

        return output

    def _check_nan_inf(self, tensor: Tensor, message: str):
        """
        Checks if the given tensor contains NaN or Inf values and logs a warning if found.

        Args:
            tensor (torch.Tensor): The tensor to check.
            message (str): A message to include in the warning log.
        """
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logger.warning(f"NaN or Inf detected in {message}.")
