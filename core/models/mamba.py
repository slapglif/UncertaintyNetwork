import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.utils import check_shapes, einsum_safe as einsum


class MambaConfig:
    def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand_factor: int = 2,
            dt_rank: Optional[int] = None,
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dt_init: str = "random",
            dt_scale: float = 1.0,
            dt_init_floor: float = 1e-4,
            bias: bool = False,
            conv_bias: bool = True,
            pscan: bool = True,
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


class Mamba(nn.Module):
    def __init__(self, config: 'MambaConfig'):
        super().__init__()
        self.config = config

        # Input projection
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        # Convolution layer
        self.conv = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            groups=config.d_inner,
            padding=config.d_conv - 1,
            bias=config.conv_bias,
        )

        # SSM parameters
        self.A_log = nn.Parameter(torch.randn(config.d_inner, config.d_state))
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.B = nn.Parameter(torch.randn(config.d_inner, config.d_state))
        self.C = nn.Parameter(torch.randn(config.d_inner, config.d_state))

        # Delta projection
        self.dt_proj = nn.Linear(config.d_inner, config.d_inner, bias=True)

        # Output projection
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize A_log
        A = torch.arange(1, self.config.d_state + 1, dtype=torch.float32).repeat(self.config.d_inner, 1)
        self.A_log.data.copy_(torch.log(A))
        self.A_log._no_weight_decay = True

        # Initialize D
        self.D._no_weight_decay = True

        # Initialize dt_proj
        dt_init_std = self.config.d_inner ** -0.5 * self.config.dt_scale
        if self.config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif self.config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Invalid dt_init method: {self.config.dt_init}")

        dt = torch.exp(
            torch.rand(self.config.d_inner) *
            (math.log(self.config.dt_max) - math.log(self.config.dt_min)) +
            math.log(self.config.dt_min)
        ).clamp(min=self.config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None, use_cache: bool = False) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0)

        batch_size, seq_len, _ = x.shape

        # Input projection and splitting
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Convolution
        x = x.transpose(1, 2)
        x = self.conv(x)[:, :, :seq_len]
        x = x.transpose(1, 2)

        # Compute Delta
        dt = self.dt_proj(x)
        delta = F.softplus(dt)

        # Selective scan
        A = -torch.exp(self.A_log.float())
        B = self.B.float()
        C = self.C.float()
        D = self.D.float()

        if self.config.pscan:
            y = self.parallel_scan(x, delta, A, B, C, D)
            new_state = None
        else:
            y, new_state = self.sequential_scan(x, delta, A, B, C, D, state)

        # Output projection
        y = y * F.silu(z)
        output = self.out_proj(y)

        # Restore original shape if necessary
        if len(original_shape) == 2:
            output = output.squeeze(0)

        return (output, new_state) if use_cache else (output, None)

    @classmethod
    def parallel_scan(cls, x: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                      D: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_inner = x.shape
        _, _, dt_rank = delta.shape
        d_state = A.shape[1]
        device, dtype = x.device, x.dtype

        check_shapes([A, B, C, D, x, delta],
                     [(d_inner, d_state), (d_inner, d_state), (d_inner, d_state), (d_inner,),
                      (batch_size, seq_len, d_inner), (batch_size, seq_len, dt_rank)],
                     ['A', 'B', 'C', 'D', 'x', 'delta'])

        padded_len = 2 ** math.ceil(math.log2(seq_len))
        x = F.pad(x, (0, 0, 0, padded_len - seq_len))
        delta = F.pad(delta, (0, 0, 0, padded_len - seq_len))

        h = torch.zeros(batch_size, padded_len, d_state, device=device, dtype=dtype)

        for i in range(int(math.log2(padded_len))):
            block_size = 2 ** i
            delta_exp = delta[:, ::2 * block_size].exp()

            # Perform the einsum operation without reshaping
            h_update = einsum('bnd,de->bne', delta_exp, A)
            h[:, ::2 * block_size] += h_update

            if i < int(math.log2(padded_len)) - 1:
                delta_odd = delta[:, block_size::2 * block_size].exp()
                h_odd = einsum('bnd,de->bne', delta_odd, B)
                h[:, block_size::2 * block_size] = h[:, ::2 * block_size] + h_odd

        for i in range(int(math.log2(padded_len)) - 1, -1, -1):
            block_size = 2 ** i
            delta_exp = delta[:, ::2 * block_size].exp()

            if i < int(math.log2(padded_len)) - 1:
                h_update = einsum('bnd,de->bne', delta_exp, A)
                h[:, block_size::2 * block_size] += einsum('bne,bne->bne', h_update, h[:, ::2 * block_size])

        return einsum('bne,de->bnd', h[:, :seq_len], C) + D * x[:, :seq_len]

    def sequential_scan(self, x, delta, A, B, C, D, state):
        batch_size, seq_len, d_inner = x.shape
        device, dtype = x.device, x.dtype

        if state is None:
            state = torch.zeros(batch_size, d_inner, self.config.d_state, device=device, dtype=dtype)

        outputs = []
        for t in range(seq_len):
            state = einsum('bd,de->be', delta[:, t].exp(), A) * state + einsum('bd,de->be', x[:, t], B)
            y = einsum('be,de->bd', state, C) + D * x[:, t]
            outputs.append(y)

        return torch.stack(outputs, dim=1), state

    def step(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, d_model = x.shape
        device, dtype = x.device, x.dtype

        # Input projection and splitting
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Convolution (assuming cached previous inputs)
        x = x.unsqueeze(-1)
        x = self.conv(x).squeeze(-1)

        # Compute Delta
        dt = self.dt_proj(x)
        delta = F.softplus(dt)

        # Initialize state if None
        if state is None:
            state = torch.zeros(batch_size, self.config.d_inner, self.config.d_state, device=device, dtype=dtype)

        # SSM parameters
        A = -torch.exp(self.A_log.float())
        B = self.B.float()
        C = self.C.float()
        D = self.D.float()

        # Update state
        state = einsum('bd,de->be', delta.exp(), A) * state + einsum('bd,de->be', x, B)

        # Compute output
        y = einsum('be,de->bd', state, C) + D * x

        # Output projection
        y = y * F.silu(z)
        output = self.out_proj(y)

        return output, state
