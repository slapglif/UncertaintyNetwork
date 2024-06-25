# .\core\kan\fasterkan_layers.py
from typing import List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from core.models.kan.spline_basis import ReflectionalSwitchFunction, SplineLinear


class SplineNetLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_grids: int = 8, *_, **kwargs):
        super().__init__()
        # LayerNorm now uses the correct normalized_shape
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = ReflectionalSwitchFunction(num_grids=num_grids, **kwargs)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(x)
        spline_basis = self.rbf(x)
        # Reshape to (batch_size, seq_len, input_dim * num_grids) instead of flattening
        spline_basis = spline_basis.view(x.size(0), x.size(1), -1)
        return self.spline_linear(spline_basis)


class SplineNet(nn.Module):
    """
    A network composed of multiple SplineNet layers.

    Args:
        layers_hidden (List[int]): A list of hidden layer dimensions.
        grid_min (float, optional): The minimum value of the grid for the reflectional switch function. Defaults to -1.2.
        grid_max (float, optional): The maximum value of the grid for the reflectional switch function. Defaults to 0.2.
        num_grids (int, optional): The number of grid points for the reflectional switch function. Defaults to 8.
        exponent (int, optional): The exponent for the reflectional switch function. Defaults to 2.
        inv_denominator (float, optional): The inverse of the denominator for the reflectional switch function. Defaults to 0.5.
        train_grid (bool, optional): Whether to train the grid points of the reflectional switch function. Defaults to False.
        train_inv_denominator (bool, optional): Whether to train the inverse of the denominator for the reflectional switch function. Defaults to False.
        base_activation (Callable, optional): The activation function to apply in the base update path. Defaults to None.
        spline_weight_init_scale (float, optional): The scaling factor for initializing the weights of the spline linear transformation. Defaults to 1.0.
    """

    def __init__(
            self,
            layers_hidden: List[int],
            grid_min: float = -1.2,
            grid_max: float = 0.2,
            num_grids: int = 8,
            exponent: int = 2,
            inv_denominator: float = 0.5,
            train_grid: bool = False,
            train_inv_denominator: bool = False,
            # use_base_update: bool = True,
            base_activation=None,
            spline_weight_init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SplineNetLayer(
                    in_dim,
                    out_dim,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    exponent=exponent,
                    inv_denominator=inv_denominator,
                    train_grid=train_grid,
                    train_inv_denominator=train_inv_denominator,
                    # use_base_update=use_base_update,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_weight_init_scale,
                )
                for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SplineNet network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualConnectionBlock(nn.Module):
    """
    A basic residual block with two convolutional layers and batch normalization.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        stride (int, optional): The stride of the convolutional layers. Defaults to 1.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualConnectionBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualConnectionBlock.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        identity = self.downsample(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)

        return out


class AttentionGateBlock(nn.Module):
    """
    Attention Gating block for channel attention.

    Args:
        channel (int): The number of input channels.
        reduction (int, optional): The reduction factor for the squeeze operation. Defaults to 16.
    """

    def __init__(self, channel, reduction=16):
        super(AttentionGateBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AttentionGateBlock.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the kernel.
        stride (int, optional): The stride of the convolution. Defaults to 1.
        padding (int, optional): The padding of the convolution. Defaults to 0.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DepthwiseSeparableConv.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SelfAttention(nn.Module):
    """
    Self-attention layer.

    Args:
        in_channels (int): The number of input channels.
    """

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SelfAttention layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        batch_size, C, width, height = x.size()
        proj_query = (
            self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        )
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out


class SplineNetConv(nn.Module):
    def __init__(
            self,
            layers_hidden: List[int],
            input_channels: int,
            hidden_dim: int,
            grid_min: float = -1.2,
            grid_max: float = 0.2,
            num_grids: int = 8,
            exponent: int = 2,
            inv_denominator: float = 0.5,
            train_grid: bool = False,
            train_inv_denominator: bool = False,
            spline_weight_init_scale: float = 1.0,
            uncertainty_output: bool = False,
    ):
        super(SplineNetConv, self).__init__()

        self.feature_extractor = Classification(input_channels, hidden_dim)
        self.uncertainty_output = uncertainty_output

        self.faster_kan_layers = nn.ModuleList([
            SplineNetLayer(
                in_dim,
                out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                exponent=exponent,
                inv_denominator=inv_denominator,
                train_grid=train_grid,
                train_inv_denominator=train_inv_denominator,
                spline_weight_init_scale=spline_weight_init_scale,
            )
            for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.feature_extractor(x)

        uncertainties = []
        for layer in self.faster_kan_layers:
            # Pass the correct input_dim to SplineNetLayer
            x = layer(x)
            if self.uncertainty_output:
                uncertainties.append(torch.var(x, dim=-1))

        if self.uncertainty_output:
            uncertainty = torch.stack(uncertainties).mean(dim=0)
            return x, uncertainty
        else:
            return x


class Classification(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int):
        super().__init__()
        self.initial_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # Reduce pooling size
            nn.Dropout(0.25),
            ResidualConnectionBlock(64, 128),
            AttentionGateBlock(128, reduction=16),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),  # Reduce pooling size
            nn.Dropout(0.25),
            DepthwiseSeparableConv(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            ResidualConnectionBlock(512, 512),
            AttentionGateBlock(512, reduction=16),
            nn.AdaptiveAvgPool2d(1),  # Use adaptive pooling instead
            nn.Dropout(0.25),
            SelfAttention(512),
        )
        self.fc = nn.Linear(512, hidden_dim).to("cuda")  # Move fc to GPU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check for NaN and Inf values
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error(f"NaN or Inf values found in the input tensor: {x.shape}")
            return x

        batch_size = x.size(0)
        x = self.initial_layers(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
