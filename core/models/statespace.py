# .\core\models\statespace.py
from typing import Optional, NamedTuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.types import Device


class MambaConfig:
    def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand_factor: float = 2.0,
            dt_rank: Optional[int] = None,
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dt_init: str = "random",
            dt_scale: float = 1.0,
            dt_init_floor: float = 1e-4,
            bias: bool = False,
            conv_bias: bool = True,
            pscan: bool = True,
            chunk_size: int = 64,
            n_heads: int = 8,
            headdim: int = None,  # Modify this line
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.d_inner = int(self.expand_factor * self.d_model)
        self.dt_rank = dt_rank if dt_rank is not None else self.d_inner
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.bias = bias
        self.conv_bias = conv_bias
        self.pscan = pscan
        self.chunk_size = chunk_size
        self.n_heads = n_heads
        # Calculate headdim to ensure compatibility
        self.headdim = self.d_inner // self.n_heads if headdim is None else headdim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


class InferenceCache(NamedTuple):
    conv_state: Tensor  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: Tensor  # (batch, nheads, headdim, d_state)

    @staticmethod
    def alloc(batch_size: int, args: MambaConfig, device: Device = None):
        return InferenceCache(
            torch.zeros(
                batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device
            ),
            torch.zeros(
                batch_size, args.n_heads, args.headdim, args.d_state, device=device
            ),
        )


class Mamba(nn.Module):
    """
    Mamba 2 Custom Implementation

    Intended Data Flow:
        Input: input_tensor (batch, seqlen, d_model)
        Linear Projection: projected_tensor (batch, seqlen, 2 * d_inner + 2 * d_state + n_heads)
        Split: z_tensor (batch, seqlen, d_inner), conv_input_tensor (batch, seqlen, d_inner + 2 * d_state), delta_timestep_tensor (batch, seqlen, n_heads)
        Convolution: conv_input_tensor (batch, seqlen, d_inner + 2 * d_state), conv_state (for caching)
        Split: x_tensor (batch, seqlen, d_inner), B_tensor (batch, seqlen, d_state), C_tensor (batch, seqlen, d_state)
        Permute delta_timestep: delta_timestep_tensor (batch, n_heads, seqlen)
        Padding: Pad x_tensor, B_tensor, C_tensor, delta_timestep_tensor, and decay_matrix along seqlen to be multiples of chunk_size.
        Rearrange x_tensor: x_tensor (batch, seqlen, n_heads, headdim)
        SSD: y_tensor (batch, seqlen, n_heads, headdim)
        Combine and Reshape: y_tensor (batch, seqlen, d_inner)
        Normalize and Project: output_tensor (batch, seqlen, d_model)
        Output: output_tensor, new_state
    """

    def __init__(self, config: "MambaConfig"):
        super().__init__()
        self.config = config

        # Adjust the input projection to match the expected input dimension
        self.in_proj = nn.Linear(
            config.d_model,
            2 * config.d_inner + 2 * config.d_state + config.n_heads,
            bias=False,
        )

        conv_dim = config.d_inner + 2 * config.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=config.d_conv,
            groups=conv_dim,
            padding=config.d_conv - 1,
        )

        self.delta_timestep_bias = nn.Parameter(torch.empty(config.n_heads))
        self.log_decay_matrix = nn.Parameter(torch.empty(config.n_heads))
        self.additive_skip_connection_scale = nn.Parameter(torch.empty(config.n_heads))
        self.norm = RMSNorm(config.d_inner)
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=False)

        self._initialize_weights()

    def forward(
            self,
            input_tensor: Tensor,
            state: Optional[torch.Tensor] | InferenceCache = None,
            use_cache: bool = False,
    ) -> tuple[Tensor, InferenceCache] | tuple[Any, InferenceCache | Any]:
        """
        Forward pass of the Mamba layer.

        Args:
            input_tensor (Tensor): Input tensor of shape (batch, seqlen, d_model).
            state (Optional[torch.Tensor] | InferenceCache): Optional state tensor or InferenceCache for caching.
            use_cache (bool): Whether to use caching during inference. Defaults to False.

        Returns:
            tuple[Tensor, InferenceCache] | tuple[Any, InferenceCache | Any]: Output tensor and InferenceCache if use_cache is True, else output tensor only.
        """
        if state:
            return self.step(input_tensor, state)

        # Compute negative exponential of log_decay_matrix parameter
        decay_matrix_neg_exp = -torch.exp(self.log_decay_matrix)  # (n_heads,)

        # Project input tensor to obtain projected_tensor
        projected_tensor = self.in_proj(input_tensor)  # (batch, seqlen, 2 * d_inner + 2 * d_state + n_heads)

        # Split projected_tensor into z_tensor, conv_input_tensor, and delta_timestep_tensor
        z_tensor, conv_input_tensor, delta_timestep_tensor = torch.split(
            projected_tensor,
            [
                self.config.d_inner,
                self.config.d_inner + 2 * self.config.d_state,
                self.config.n_heads,
            ],
            dim=-1,
        )

        # Apply softplus activation to delta_timestep_tensor with bias
        delta_timestep_tensor = F.softplus(delta_timestep_tensor + self.delta_timestep_bias)  # (batch, seqlen, n_heads)

        # Pad or truncate conv_input_tensor seqlen to match d_conv
        conv_state = F.pad(
            rearrange(conv_input_tensor, "b l d -> b d l"), (self.config.d_conv - input_tensor.shape[1], 0)
        )

        # Apply 1D convolution to conv_input_tensor and apply silu activation
        conv_input_tensor = silu(
            self.conv1d(conv_input_tensor.transpose(1, 2)).transpose(1, 2)[:, : input_tensor.shape[1], :]
        )  # (batch, seqlen, d_inner + 2 * d_state))

        # Split conv_input_tensor into x_tensor, B_tensor, and C_tensor
        x_tensor, B_tensor, C_tensor = torch.split(
            conv_input_tensor, [self.config.d_inner, self.config.d_state, self.config.d_state], dim=-1
        )

        # --- Permute delta_timestep_tensor dimensions ---
        batch_size, seq_len, _ = x_tensor.shape

        # Expand and permute decay_matrix_neg_exp to match the shape of x_tensor and delta_timestep_tensor
        decay_matrix_expanded = decay_matrix_neg_exp.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, seq_len).permute(
            0, 2, 1)  # (batch, seqlen, n_heads)

        # Permute delta_timestep_tensor and add a singleton dimension at the end
        delta_timestep_permuted = delta_timestep_tensor.permute(0, 1, 2).unsqueeze(-1)  # (batch, seqlen, n_heads, 1)
        # ---

        # --- Ensure d_inner = n_heads * headdim ---
        assert self.config.d_inner == self.config.n_heads * self.config.headdim, \
            f"Dimension mismatch: d_inner ({self.config.d_inner}) != n_heads ({self.config.n_heads}) * headdim ({self.config.headdim})"
        # ---

        # Rearrange x_tensor into (batch, seqlen, n_heads, headdim)
        x_rearranged = rearrange(x_tensor, "b l (h p) -> b l h p", p=self.config.headdim)

        # --- Multiply Before Padding ---
        x_delta_timestep_multiplied = x_rearranged * delta_timestep_permuted  # (batch, seqlen, n_heads, headdim)
        # ---

        # --- Consistent Padding Logic ---
        seqlen = x_rearranged.shape[1]
        padding_needed = self.config.chunk_size - (seqlen % self.config.chunk_size)
        if padding_needed != self.config.chunk_size:  # Pad if not already a multiple
            _ = (0, padding_needed)  # Padding for last dimension
            x_delta_timestep_multiplied = F.pad(x_delta_timestep_multiplied, (
                0, 0, 0, padding_needed, 0, 0))  # Pad x_delta_timestep_multiplied along seqlen
            decay_matrix_expanded = F.pad(decay_matrix_expanded,
                                          (0, padding_needed, 0, 0))  # Pad decay_matrix_expanded along seqlen
            B_tensor = F.pad(B_tensor, (0, 0, 0, padding_needed, 0, 0))  # Pad B_tensor along seqlen
            C_tensor = F.pad(C_tensor, (0, 0, 0, padding_needed, 0, 0))  # Pad C_tensor along seqlen
        # --- End Padding Logic ---

        # Apply Structured State Space (SSD) operation
        y_tensor, ssm_state = ssd(
            x_delta_timestep_multiplied,  # Pass the already multiplied x_tensor
            decay_matrix_expanded,  # Pass decay_matrix_expanded without modification
            rearrange(B_tensor, "b l n -> b l 1 n"),
            rearrange(C_tensor, "b l n -> b l 1 n"),
            self.config.chunk_size,
            device=self.config.device,
        )

        # Add skip connection and multiply with additive_skip_connection_scale parameter
        y_tensor = y_tensor + x_delta_timestep_multiplied * self.additive_skip_connection_scale.unsqueeze(-1)

        output_tensor = self.rearrange_and_project(y_tensor, "b l h p -> b l (h p)", z_tensor)
        # --- Remove Padding from Output ---
        if padding_needed != self.config.chunk_size:
            output_tensor = output_tensor[:, :seqlen, :]  # Slice to original sequence length
        # ---

        # Create the new state using conv_state and ssm_state
        new_state = InferenceCache(conv_state, ssm_state)

        return output_tensor, new_state if use_cache else output_tensor

    def step(
            self, input_tensor: Tensor, state: "InferenceCache"
    ) -> tuple[Tensor, "InferenceCache"]:
        """Take a single inference step for the current input and hidden state

        Unlike attention-based models, RNN-based models (eg Mamba) does not need
        to look back at all the past tokens to generate a new token. Instead, a
        hidden state (initialized to 0s initially) is updated for each input and
        passed to the next inference step. This means that the total inference
        time is linear with respect to the sequence length instead of quadratic
        in attention's case.

        Arguments
            input_tensor: (batch, 1, d_model)
            state: initial/running hidden state

        Return (output_tensor, state)
            output_tensor: (batch, 1, d_model)
            state: updated hidden state
        """
        assert input_tensor.shape[1] == 1, "Only one token can be decoded per inference step"

        projected_tensor = self.in_proj(input_tensor.squeeze(1))  # (batch, d_in_proj)
        z_tensor, conv_input_tensor, delta_timestep_tensor = torch.split(
            projected_tensor,
            [
                self.config.d_inner,
                self.config.d_inner + 2 * self.config.d_state,
                self.config.n_heads,
            ],
            dim=-1,
        )

        # Advance convolution input
        state.conv_state.copy_(torch.roll(state.conv_state, shifts=-1, dims=-1))
        state.conv_state[:, :, -1] = conv_input_tensor
        # Convolution step
        conv_input_tensor = torch.sum(
            state.conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        conv_input_tensor += self.conv1d.bias
        conv_input_tensor = silu(conv_input_tensor)

        x_tensor, B_tensor, C_tensor = torch.split(
            conv_input_tensor, [self.config.d_inner, self.config.d_state, self.config.d_state], dim=-1
        )
        decay_matrix = -torch.exp(self.log_decay_matrix)  # (nheads,)

        # SSM step
        delta_timestep_tensor = F.softplus(delta_timestep_tensor + self.delta_timestep_bias)  # (batch, nheads)
        delta_decay_matrix = torch.exp(delta_timestep_tensor * decay_matrix)  # (batch, nheads)
        x_tensor = rearrange(x_tensor, "b (h p) -> b h p", p=self.config.headdim)
        delta_B_x = torch.einsum("bh, bn, bhp -> bhpn", delta_timestep_tensor, B_tensor, x_tensor)
        state.ssm_state.copy_(state.ssm_state * rearrange(delta_decay_matrix, "b h -> b h 1 1") + delta_B_x)
        y_tensor = torch.einsum("bhpn, bn -> bhp", state.ssm_state, C_tensor)
        y_tensor = y_tensor + rearrange(self.additive_skip_connection_scale, "h -> h 1") * x_tensor
        output_tensor = self.rearrange_and_project(y_tensor, "b h p -> b (h p)", z_tensor)
        return output_tensor.unsqueeze(1), state

    def rearrange_and_project(self, y_tensor, rearrange_pattern, z_tensor):
        y_tensor = rearrange(y_tensor, rearrange_pattern)
        y_tensor = self.norm(y_tensor, z_tensor)
        return self.out_proj(y_tensor)

    def _initialize_weights(self):
        # Initialize log_decay_matrix
        decay_matrix = torch.arange(1, self.config.d_state + 1, dtype=torch.float32).repeat(
            self.config.d_inner, 1
        )
        decay_matrix = decay_matrix.to(
            self.log_decay_matrix.device)  # move decay_matrix to the same device as self.log_decay_matrix
        self.log_decay_matrix.data.copy_(
            torch.log(decay_matrix).flatten()[
            :self.config.n_heads
            ]
        )  # Flatten log(decay_matrix) and take the first n_heads elements

        self.log_decay_matrix._no_weight_decay = True

        # Initialize additive_skip_connection_scale
        self.additive_skip_connection_scale._no_weight_decay = True

        # Initialize delta_timestep_bias
        nn.init.uniform_(self.delta_timestep_bias, -1., 1.)


def segsum(x: Tensor, device: Device = None) -> Tensor:
    """Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.
    """
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x, A, B, C, chunk_size, initial_states=None, device: Device = None):
    """Structed State Space Duality (SSD) - the core of Mamba-2

    Arguments
        x: (batch, seqlen, n_heads, d_head)
        A: (batch, seqlen, n_heads)
        B: (batch, seqlen, n_heads, d_state)
        C: (batch, seqlen, n_heads, d_state)

    Return
        y: (batch, seqlen, n_heads, d_head)

    """
    # --- Removed Assertion ---
    # assert x.shape[1] % chunk_size == 0
    # ---

    # --- Calculate chunks ---
    seqlen = x.shape[1]
    chunks = (seqlen + chunk_size - 1) // chunk_size  # Calculate chunks to include remainder
    # ---

    # Rearrange into chunks
    # Step 1, 2 and 4 of SSD can be computed in parallel for each chunk across devices (sequence parallel)
    # This is not implemented and left as an exercise for the reader 😜
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A, device=device))
    L = rearrange(L, 'b h (c l1) (c l2) -> b h c l1 c l2', c=chunks, l1=chunk_size, l2=chunk_size)  # Reshape L

    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None):
        """Gated Root Mean Square Layer Normalization"""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x):
    """Applies the Sigmoid Linear Unit (SiLU), element-wise.

    Define this manually since torch's version doesn't seem to work on MPS.
    """
    return x * F.sigmoid(x)