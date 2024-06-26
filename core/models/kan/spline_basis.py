# .\core\kan\fasterkan_basis.py

# .\core\kan\fasterkan_basis.py
import torch
import torch.nn as nn
from torch.autograd import Function


class RSWAFFunction(Function):
    @staticmethod
    def forward(ctx, input, grid, inv_denominator, train_grid, train_inv_denominator):
        diff = input[..., None] - grid
        diff_mul = diff.mul(inv_denominator)
        tanh_diff = torch.tanh(diff)
        tanh_diff_deriviative = (
            -tanh_diff.mul(tanh_diff) + 1
        )  # sech^2(x) = 1 - tanh^2(x)

        # Save tensors for backward pass
        ctx.save_for_backward(
            input, tanh_diff, tanh_diff_deriviative, diff, inv_denominator
        )
        ctx.train_grid = train_grid
        ctx.train_inv_denominator = train_inv_denominator

        return tanh_diff_deriviative

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input, tanh_diff, tanh_diff_deriviative, diff, inv_denominator = (
            ctx.saved_tensors
        )
        grad_grid = None
        grad_inv_denominator = None

        # Compute the backward pass for the input
        grad_input = -2 * tanh_diff * tanh_diff_deriviative * grad_output

        grad_input = grad_input.sum(dim=-1).mul(inv_denominator)

        # Compute the backward pass for grid
        if ctx.train_grid:
            grad_grid = -inv_denominator * grad_output.sum(dim=0).sum(
                dim=0
            )  # -(inv_denominator * grad_output * tanh_diff_deriviative).sum(dim=0) #-inv_denominator * grad_output.sum(dim=0).sum(dim=0)

        # Compute the backward pass for inv_denominator
        if ctx.train_inv_denominator:
            grad_inv_denominator = (
                grad_output * diff
            ).sum()  # (grad_output * diff * tanh_diff_deriviative).sum() #(grad_output* diff).sum()

        return (
            grad_input,
            grad_grid,
            grad_inv_denominator,
            None,
            None,
        )  # same number as tensors or parameters


class ReflectionalSwitchFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -1.2,
        grid_max: float = 0.2,
        num_grids: int = 8,
        exponent: int = 2,
        inv_denominator: float = 0.5,
        train_grid: bool = False,
        train_inv_denominator: bool = False,
        *args,
        **kwargs
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.train_grid = torch.tensor(train_grid, dtype=torch.bool)
        self.train_inv_denominator = torch.tensor(
            train_inv_denominator, dtype=torch.bool
        )
        self.grid = torch.nn.Parameter(grid, requires_grad=train_grid)
        # print(f"grid initial shape: {self.grid.shape }")
        self.inv_denominator = torch.nn.Parameter(
            torch.tensor(inv_denominator, dtype=torch.float32),
            requires_grad=train_inv_denominator,
        )  # Cache the inverse of the denominator

    def forward(self, x):
        return RSWAFFunction.apply(
            x,
            self.grid,
            self.inv_denominator,
            self.train_grid,
            self.train_inv_denominator,
        )


class SplineLinear(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, init_scale: float = 0.1, **kw
    ) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
