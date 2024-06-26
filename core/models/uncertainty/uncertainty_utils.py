# .\core\models\uncertainty\uncertainty_utils.py
import math
from typing import Optional, List
from typing import Tuple

import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from gpytorch.kernels import Kernel
from gpytorch.kernels import ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from loguru import logger
from torch import Tensor

from core.models.kan.spline_layers import SplineNetLayer


class UncertaintyModule(nn.Module):
    """
    Module for estimating uncertainty using Monte Carlo dropout and Gaussian Processes.

    Args:
        input_dim (int): The dimension of the input features.
        output_dim (int): The dimension of the output.
        n_gp_layers (int, optional): Number of Gaussian Process layers. Defaults to 2.
        n_inducing (int, optional): Number of inducing points for each GP layer. Defaults to 10.
        dropout_rate (float, optional): Dropout rate for MC dropout. Defaults to 0.1.
        mc_samples (int, optional): Number of MC samples to draw. Defaults to 5.
        temperature (float, optional): Temperature scaling factor for logits. Defaults to 1.0.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            n_gp_layers: int = 2,
            n_inducing: int = 10,
            dropout_rate: float = 0.1,
            mc_samples: int = 5,
            temperature: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_gp_layers = n_gp_layers
        self.n_inducing = n_inducing
        self.dropout_rate = dropout_rate
        self.mc_samples = mc_samples
        self.temperature = nn.Parameter(torch.tensor(temperature))

        logger.info(
            f"Initializing UncertaintyModule with {n_gp_layers} GP layers and {mc_samples} MC samples"
        )

        # Use a single GaussianProcessLayer for the single output
        self.gp_layer = GaussianProcessLayer(input_dim, 1, n_inducing)
        self.mc_dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(1, output_dim)  # Modify output_layer
        self.uncertainty_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the UncertaintyModule.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Scaled mean output of shape (batch_size, seq_len, output_dim)
                - Scaled uncertainty of shape (batch_size, seq_len, output_dim)
        """
        if x.dim() != 3 or x.size(2) != self.input_dim:
            logger.error(
                f"Invalid input shape. Expected (batch_size, seq_len, {self.input_dim}), got {x.shape}"
            )
            raise ValueError(
                f"Invalid input shape. Expected (batch_size, seq_len, {self.input_dim}), got {x.shape}"
            )

        batch_size, seq_len, _ = x.shape
        device = x.device

        total_uncertainty = torch.zeros(
            batch_size, seq_len, self.output_dim, device=device
        )
        outputs = []

        try:
            for _ in range(self.mc_samples):
                h = self.mc_dropout(x)

                # Normalize input to GP layer
                h_normalized = (h - h.mean(dim=1, keepdim=True)) / (
                        h.std(dim=1, keepdim=True) + 1e-5
                )

                # Apply the single GP layer to each timestep in the sequence
                mean, variance = self.gp_layer.predict_with_uncertainty(h_normalized)
                total_uncertainty += variance

                # Pass mean through output_layer
                outputs.append(self.output_layer(mean))

            mean_output = torch.stack(outputs).mean(
                dim=0
            )  # Shape: (batch_size, seq_len, output_dim)
            mean_uncertainty = (
                    total_uncertainty / self.mc_samples
            )  # Shape: (batch_size, seq_len, output_dim)

            scaled_mean = mean_output / self.temperature
            scaled_uncertainty = mean_uncertainty / (self.temperature ** 2)

            return scaled_mean, scaled_uncertainty
        except Exception as e:
            logger.error(f"Error in UncertaintyModule forward pass: {str(e)}")
            raise


class GaussianProcessLayer(ApproximateGP):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_inducing: int,
            kernel: Optional[Kernel] = None,
    ):
        # Initialize inducing points
        inducing_points = torch.randn(num_inducing, input_dim)
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = ConstantMean()
        self.covar_module = (
            kernel
            if kernel is not None
            else ScaleKernel(
                TSPKernel(
                    TSPEnergyFunction(input_dim, compression_dim=16), lengthscale=1.0
                )
            )
            # Default to TSPKernel
        )
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise_covar.raw_noise.data.fill_(
            -3
        )  # Initialize with higher noise
        self.output_dim = output_dim
        self.num_inducing = num_inducing
        self.input_dim = input_dim

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        logger.debug(f"GaussianProcessLayer input shape: {x.shape}")
        # Ensure input is 2D: (batch_size * seq_len, input_dim)
        if x.dim() == 3:
            x = x.view(-1, self.input_dim)

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        output = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        logger.debug(f"GaussianProcessLayer output shape: {output.shape}")
        return output

    def predict_with_uncertainty(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        self.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Reshape input if necessary
            original_shape = x.shape
            if x.dim() == 3:
                batch_size, seq_len, _ = original_shape
                x = x.view(batch_size * seq_len, self.input_dim)
            else:
                batch_size, seq_len = x.shape[0], 1

            # Make predictions
            predictions = self(x)
            mean = predictions.mean
            variance = predictions.variance

            # Reshape outputs
            mean = mean.view(batch_size, seq_len, self.output_dim)
            variance = variance.view(batch_size, seq_len, self.output_dim)

        return mean, variance


# todo: implement multi token inference
class MultiOutputUncertaintyModule(nn.Module):
    """
    Module for estimating uncertainty using Monte Carlo dropout and Gaussian Processes.

    Args:
        input_dim (int): The dimension of the input features.
        output_dim (int): The dimension of the output.
        n_gp_layers (int, optional): Number of Gaussian Process layers. Defaults to 2.
        n_inducing (int, optional): Number of inducing points for each GP layer. Defaults to 10.
        dropout_rate (float, optional): Dropout rate for MC dropout. Defaults to 0.1.
        mc_samples (int, optional): Number of MC samples to draw. Defaults to 5.
        temperature (float, optional): Temperature scaling factor for logits. Defaults to 1.0.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            n_gp_layers: int = 2,
            n_inducing: int = 10,
            dropout_rate: float = 0.1,
            mc_samples: int = 5,
            temperature: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_gp_layers = n_gp_layers
        self.n_inducing = n_inducing
        self.dropout_rate = dropout_rate
        self.mc_samples = mc_samples
        self.temperature = nn.Parameter(torch.tensor(temperature))

        logger.info(
            f"Initializing UncertaintyModule with {n_gp_layers} GP layers and {mc_samples} MC samples"
        )

        # Create a separate GaussianProcessLayer for each output dimension
        self.gp_layers = nn.ModuleList(
            [
                GaussianProcessLayer(
                    input_dim, 1, n_inducing
                )  # Output dimension is 1 for each layer
                for _ in range(output_dim)  # Create a layer for each output dimension
            ]
        )
        self.mc_dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(input_dim, output_dim)
        self.uncertainty_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the UncertaintyModule.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Scaled mean output of shape (batch_size, seq_len, output_dim)
                - Scaled uncertainty of shape (batch_size, seq_len, output_dim)
        """
        if x.dim() != 3 or x.size(2) != self.input_dim:
            logger.error(
                f"Invalid input shape. Expected (batch_size, seq_len, {self.input_dim}), got {x.shape}"
            )
            raise ValueError(
                f"Invalid input shape. Expected (batch_size, seq_len, {self.input_dim}), got {x.shape}"
            )

        batch_size, seq_len, _ = x.shape
        device = x.device

        total_uncertainty = torch.zeros(
            batch_size, seq_len, self.output_dim, device=device
        )
        outputs = []

        try:
            for _ in range(self.mc_samples):
                h = self.mc_dropout(x)

                # Apply GP layers to each dimension of the hidden state
                for i, gp_layer in enumerate(self.gp_layers):
                    h[:, :, i], variance = gp_layer(h[:, :, i])
                    total_uncertainty[:, :, i] += variance

                outputs.append(self.output_layer(h))

            mean_output = torch.stack(outputs).mean(dim=0)
            mean_uncertainty = total_uncertainty / self.mc_samples

            scaled_mean = mean_output / self.temperature
            scaled_uncertainty = mean_uncertainty / (self.temperature ** 2)

            return scaled_mean, scaled_uncertainty
        except Exception as e:
            logger.error(f"Error in UncertaintyModule forward pass: {str(e)}")
            raise


class RandomWalkKernel(Kernel):
    def __init__(
            self, input_dim: int, n_steps: int = 5, walk_type: str = "standard", **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.n_steps = n_steps
        self.walk_type = walk_type
        self.register_parameter(
            name="raw_lengthscale",
            parameter=torch.nn.Parameter(torch.zeros(1, input_dim)),
        )
        self.register_constraint("raw_lengthscale", gpytorch.constraints.Positive())

    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    def forward(
            self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params
    ) -> torch.Tensor:
        if x1.dim() == 3 and x2.dim() == 3:
            batch_size, seq_len, _ = x1.shape
            x1 = x1.view(batch_size * seq_len, -1)
            x2 = x2.view(batch_size * seq_len, -1)

        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
        dist = torch.sum(diff ** 2, dim=-1)

        modified_dist = self._random_walk_distances(dist)

        if diag:
            return torch.exp(
                -modified_dist
                / (2 * self.lengthscale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) ** 2)
            ).diagonal(dim1=-2, dim2=-1)
        else:
            return torch.exp(
                -modified_dist
                / (2 * self.lengthscale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) ** 2)
            )

    def _random_walk_distances(self, dist: torch.Tensor) -> torch.Tensor:
        walk_dist = dist.clone()
        for _ in range(self.n_steps):
            step_direction = (
                    torch.randint(0, 2, size=walk_dist.shape, device=walk_dist.device) * 2
                    - 1
            )
            if self.walk_type == "reflected":
                step_direction = step_direction.float() * 0.5
                walk_dist = self._reflect_distances(walk_dist, 10.0)
            elif self.walk_type == "biased":
                bias = 0.75
                step_direction = (
                                         torch.rand(walk_dist.shape, device=walk_dist.device) < bias
                                 ).float() * 2 - 1
            walk_dist = walk_dist + step_direction * dist
            walk_dist = walk_dist.clamp(min=0)
        return walk_dist

    @staticmethod
    def _reflect_distances(distances: torch.Tensor, clip_value: float) -> torch.Tensor:
        exceeding_mask = distances > clip_value
        distances[exceeding_mask] = 2 * clip_value - distances[exceeding_mask]
        return distances


class SplineCritic(nn.Module):
    def __init__(
            self, input_dim: int, hidden_dims: List[int] = [128, 64], num_grids: int = 8
    ):
        super().__init__()
        self.x_layers = nn.ModuleList()
        self.y_layers = nn.ModuleList()
        current_dim = input_dim  # Input dimension for each SplineNetLayer
        for hidden_dim in hidden_dims:
            self.x_layers.append(
                SplineNetLayer(current_dim, hidden_dim, num_grids=num_grids)
            )
            self.y_layers.append(
                SplineNetLayer(current_dim, hidden_dim, num_grids=num_grids)
            )
            current_dim = hidden_dim
        self.final_layer = nn.Linear(
            hidden_dims[-1] * 2, 1
        )  # Concatenate outputs of x and y layers

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Forward pass through the SplineCritic network.

        Args:
            x (Tensor): The first input tensor of shape (batch_size, seq_len_x, input_dim).
            y (Tensor): The second input tensor of shape (batch_size, seq_len_y, input_dim).

        Returns:
            Tensor: The output tensor of shape (batch_size, 1).
        """
        batch_size = x.size(0)  # Assuming x and y have the same batch size
        seq_len_x = x.size(1)
        seq_len_y = y.size(1)

        # Reshape x to (batch_size, seq_len_x, 1, input_dim)
        x = x.unsqueeze(2)
        # Reshape y to (batch_size, 1, seq_len_y, input_dim)
        y = y.unsqueeze(1)

        # Apply SplineNet layers to x
        for layer_x in self.x_layers:
            x = layer_x(x)  # x now has shape (batch_size, seq_len_x, 1, hidden_dim[-1])

        # Apply SplineNet layers to y (separate loop)
        for layer_y in self.y_layers:
            y = layer_y(y)  # y now has shape (batch_size, 1, seq_len_y, hidden_dim[-1])

        # Reshape x and y to (batch_size, seq_len_x, seq_len_y, hidden_dim[-1])
        x = x.view(batch_size, seq_len_x, 1, -1)
        y = y.view(batch_size, 1, seq_len_y, -1)

        # Expand x along the sequence dimension of y
        x_expanded = x.expand(-1, -1, seq_len_y, -1)
        # Expand y along the sequence dimension of x
        y_expanded = y.expand(-1, seq_len_x, -1, -1)

        # Concatenate transformed x and y representations
        combined = torch.cat(
            [x_expanded, y_expanded], dim=-1
        )  # (batch_size, seq_len_x, seq_len_y, 2 * hidden_dim[-1])

        # Reshape to (batch_size, seq_len_x * seq_len_y, 2 * hidden_dim[-1])
        combined = combined.view(batch_size, -1, 2 * self.x_layers[-1].output_dim)

        # Apply the final linear layer
        output = self.final_layer(combined)  # (batch_size, seq_len_x * seq_len_y, 1)

        # Average across the combined sequence dimension and keep the last dimension
        output = output.mean(dim=1, keepdim=True)  # (batch_size, 1, 1)

        # Squeeze the last dimension to get (batch_size, 1)
        output = output.squeeze(-1)

        return output


class TSPEnergyFunction(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            compression_dim: Optional[int] = None,
            lambda_ib: float = 0.01,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.compression_dim = compression_dim
        self.lambda_ib = lambda_ib

        if compression_dim is not None:
            self.compressor = nn.Linear(embedding_dim, compression_dim)
            self.mi_estimator = MutualInformationEstimator(
                compression_dim
            )  # Our custom MI estimator
        else:
            self.compressor = None
            self.mi_estimator = None

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        if not isinstance(x1, Tensor) or not isinstance(x2, Tensor):
            raise TypeError(
                f"Inputs must be tensors. Got types x1: {type(x1)}, x2: {type(x2)}"
            )

        if x1.dim() != 2 or x2.dim() != 2:
            raise ValueError(
                f"Inputs must be 2D tensors. Got shapes x1: {x1.shape}, x2: {x2.shape}"
            )

        if x1.size(1) != self.embedding_dim or x2.size(1) != self.embedding_dim:
            raise ValueError(
                f"Input embedding dimensions must match. Expected {self.embedding_dim}, "
                f"got x1: {x1.size(1)}, x2: {x2.size(1)}"
            )

        # Compute pairwise distances
        distances = 1 - torch.cosine_similarity(
            x1.unsqueeze(1), x2.unsqueeze(0), dim=-1
        )

        # Apply compression and mutual information regularization if enabled
        if self.compression_dim is not None:
            x1_compressed = self.compressor(x1)
            x2_compressed = self.compressor(x2)

            # Calculate the minimum sequence length
            min_seq_len = min(x1_compressed.size(1), x2_compressed.size(1))

            # Reshape the tensors to match the minimum length
            x1_compressed = x1_compressed[
                            :, :min_seq_len
                            ]  # Correct indexing for 2D tensors
            x2_compressed = x2_compressed[
                            :, :min_seq_len
                            ]  # Correct indexing for 2D tensors

            mi_estimate = self.mi_estimator(x1_compressed, x2_compressed)
            return distances + self.lambda_ib * mi_estimate

        return distances

    def extra_repr(self) -> str:
        return f"embedding_dim={self.embedding_dim}, compression_dim={self.compression_dim}, lambda_ib={self.lambda_ib}"


class TSPKernel(gpytorch.kernels.Kernel):
    def __init__(self, energy_function: TSPEnergyFunction, **kwargs):
        super().__init__(**kwargs)
        self.energy_function = energy_function

    def forward(
            self,
            x1: Tensor,
            x2: Tensor,
            diag: bool = False,
            last_dim_is_batch: bool = False,
            **params,
    ) -> Tensor:
        logger.debug(f"TSPKernel input shapes: x1 {x1.shape}, x2 {x2.shape}")
        if x1.dim() == 3:
            x1 = x1.reshape(-1, x1.size(-1))
        if x2.dim() == 3:
            x2 = x2.reshape(-1, x2.size(-1))

        energy = self.energy_function(x1, x2)
        if energy is None:
            raise ValueError(
                f"Energy function returned None for inputs: x1 shape {x1.shape}, x2 shape {x2.shape}"
            )
        if diag:
            return torch.exp(-energy.diag())
        output = torch.exp(-energy)
        logger.debug(f"TSPKernel output shape: {output.shape}")
        return output


class MutualInformationEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Estimates the mutual information between x and y using MINE.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size_x, input_dim) or (batch_size_x, seq_len_x, input_dim).
            y (torch.Tensor): Input tensor of shape (batch_size_y, input_dim) or (batch_size_y, seq_len_y, input_dim).

        Returns:
            torch.Tensor: Estimated mutual information scalar.
        """
        if x.dim() == 2:
            batch_size_x, input_dim = x.shape
            seq_len_x = 1
        elif x.dim() == 3:
            batch_size_x, seq_len_x, input_dim = x.shape
        else:
            raise ValueError(f"Input tensor x should be 2D or 3D, got {x.dim()}D")

        if y.dim() == 2:
            batch_size_y, input_dim_y = y.shape
            seq_len_y = 1
        elif y.dim() == 3:
            batch_size_y, seq_len_y, input_dim_y = y.shape
        else:
            raise ValueError(f"Input tensor y should be 2D or 3D, got {y.dim()}D")

        assert (
                input_dim == input_dim_y
        ), f"Input dimensions of x and y should match, got {input_dim} and {input_dim_y}"

        # Ensure x and y have the same sequence length
        min_seq_len = min(seq_len_x, seq_len_y)
        if seq_len_x != min_seq_len:
            x = x[:, :min_seq_len, :]
        if seq_len_y != min_seq_len:
            y = y[:, :min_seq_len, :]

        # Flatten x and y for critic input
        x_flat = x.view(-1, input_dim)
        y_flat = y.view(-1, input_dim)

        # Repeat or truncate x and y to match the larger size
        size_x = x_flat.size(0)
        size_y = y_flat.size(0)
        if size_x < size_y:
            x_flat = x_flat.repeat(math.ceil(size_y / size_x), 1)[:size_y]
        elif size_y < size_x:
            y_flat = y_flat.repeat(math.ceil(size_x / size_y), 1)[:size_x]

        # Positive samples: concatenate x and y
        xy = torch.cat((x_flat, y_flat), dim=-1)

        # Negative samples: shuffle y and concatenate with x
        y_shuffled = y_flat[torch.randperm(y_flat.size(0))]
        xy_shuffled = torch.cat((x_flat, y_shuffled), dim=-1)

        # Calculate critic scores
        t_xy = self.critic(xy)
        t_xy_shuffled = self.critic(xy_shuffled)

        # MINE objective
        mi_lower_bound = t_xy.mean() - (torch.exp(t_xy_shuffled).mean() + math.log(2))

        return mi_lower_bound


class MCDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.p, training=True)


class HeteroscedasticOutput(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mean_output = nn.Linear(input_dim, output_dim)
        self.var_output = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_output(x)
        log_var = self.var_output(x)
        return mean, torch.exp(log_var)


class ECELoss(nn.Module):
    def __init__(self, n_bins: int = 15):
        super().__init__()
        self.n_bins = n_bins

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, dim=1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin in torch.linspace(0, 1, self.n_bins + 1):
            in_bin = confidences.gt(bin) & confidences.le(bin + 1.0 / self.n_bins)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def epistemic_uncertainty(predictions: torch.Tensor) -> torch.Tensor:
    return torch.var(predictions, dim=0)


def aleatoric_uncertainty(variances: torch.Tensor) -> torch.Tensor:
    """
    Compute aleatoric uncertainty from model variances.

    Args:
        variances (torch.Tensor): Variance estimates from the model.

    Returns:
        torch.Tensor: Aleatoric uncertainty (always non-negative).
    """
    return F.softplus(torch.mean(variances, dim=0))


def total_uncertainty(epistemic: torch.Tensor, aleatoric: torch.Tensor) -> torch.Tensor:
    return epistemic + aleatoric


def entropy(probs: torch.Tensor) -> torch.Tensor:
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)


def mutual_information(
        mean_probs: torch.Tensor, all_probs: torch.Tensor
) -> torch.Tensor:
    entropy_mean = entropy(mean_probs)
    mean_entropy = torch.mean(entropy(all_probs), dim=0)
    return entropy_mean - mean_entropy


def calibration_plot(
        confidences: torch.Tensor, accuracies: torch.Tensor, n_bins: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    mean_accuracies = torch.zeros(n_bins)
    mean_confidences = torch.zeros(n_bins)

    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        if in_bin.sum() > 0:
            mean_accuracies[i] = accuracies[in_bin].mean()
            mean_confidences[i] = confidences[in_bin].mean()

    return mean_confidences, mean_accuracies


def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)


def expected_calibration_error(
        confidences: torch.Tensor, accuracies: torch.Tensor, n_bins: int = 10
) -> torch.Tensor:
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.tensor(0.0, device=confidences.device)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.mean(
        (probs - F.one_hot(labels, num_classes=probs.size(-1))).pow(2).sum(dim=-1)
    )


class EnsembleModel(nn.Module):
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = [model(x) for model in self.models]
        mean_output = torch.stack([out[0] for out in outputs]).mean(dim=0)
        variance = torch.stack([out[1] for out in outputs]).var(dim=0)
        return mean_output, variance


def out_of_distribution_detection(
        in_dist_uncertainties: torch.Tensor, out_dist_uncertainties: torch.Tensor
) -> Tuple[float, float]:
    """
    Perform out-of-distribution detection using uncertainty estimates.

    Args:
        in_dist_uncertainties (torch.Tensor): Uncertainties from in-distribution data
        out_dist_uncertainties (torch.Tensor): Uncertainties from out-of-distribution data

    Returns:
        Tuple[float, float]: AUROC and AUPRC scores
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    y_true = torch.cat(
        [
            torch.ones_like(in_dist_uncertainties),
            torch.zeros_like(out_dist_uncertainties),
        ]
    )
    y_score = torch.cat([in_dist_uncertainties, out_dist_uncertainties])

    auroc = roc_auc_score(y_true.cpu().numpy(), y_score.cpu().numpy())
    auprc = average_precision_score(y_true.cpu().numpy(), y_score.cpu().numpy())

    return auroc, auprc


def uncertainty_guided_sampling(
        logits: torch.Tensor,
        uncertainties: torch.Tensor,
        temperature: float = 1.0,
        alpha: float = 1.0,
) -> torch.Tensor:
    """
    Perform uncertainty-guided sampling for text generation.

    Args:
        logits (torch.Tensor): Logits from the model of shape (batch_size, vocab_size)
        uncertainties (torch.Tensor): Uncertainties associated with the logits of shape (batch_size, vocab_size)
        temperature (float): Temperature for softmax
        alpha (float): Weight for uncertainty influence

    Returns:
        torch.Tensor: Sampled token indices of shape (batch_size,)
    """
    if logits.shape != uncertainties.shape:
        raise ValueError(
            f"Shape mismatch: logits {logits.shape} != uncertainties {uncertainties.shape}"
        )

    scaled_logits = logits / temperature
    uncertainty_weight = F.softmax(-alpha * uncertainties, dim=-1)
    weighted_logits = scaled_logits * uncertainty_weight
    return torch.multinomial(F.softmax(weighted_logits, dim=-1), num_samples=1).squeeze(
        -1
    )


def active_learning_acquisition(
        uncertainties: torch.Tensor, n_samples: int
) -> torch.Tensor:
    """
    Select samples for active learning based on uncertainty.

    Args:
        uncertainties (torch.Tensor): Uncertainties for each sample
        n_samples (int): Number of samples to select

    Returns:
        torch.Tensor: Indices of selected samples
    """
    return torch.topk(uncertainties, k=n_samples, dim=0).indices


def uncertainty_weighted_loss(
        loss: torch.Tensor, uncertainty: torch.Tensor
) -> torch.Tensor:
    """
    Compute uncertainty-weighted loss.

    Args:
        loss (torch.Tensor): Standard loss values
        uncertainty (torch.Tensor): Uncertainty estimates

    Returns:
        torch.Tensor: Uncertainty-weighted loss
    """
    return (loss * torch.exp(-uncertainty) + uncertainty).mean()


def random_walk_distances(
        dist: torch.Tensor,
        n_steps: int,
        walk_type: str,
        step_scale: float,
        dist_clip: float,
) -> torch.Tensor:
    """
    Calculates modified distances using a random walk.

    Args:
        dist (torch.Tensor): Pairwise distance matrix.
        n_steps (int): Number of steps in the random walk.
        walk_type (str): Type of random walk ('standard', 'reflected', 'biased').
        step_scale (float): Scaling factor for the step size.
        dist_clip (float): Value at which to reflect distances (for 'reflected' walk).

    Returns:
        torch.Tensor: Distance matrix after the random walk.
    """
    walk_dist = dist.clone()
    for _ in range(n_steps):
        step_direction = (
                torch.randint(0, 2, size=(walk_dist.shape[:-1]), device=walk_dist.device)
                * 2
                - 1
        )

        if walk_type == "reflected":
            step_direction = step_direction.float() * step_scale
            walk_dist = _reflect_distances(walk_dist, dist_clip)
        elif walk_type == "biased":
            bias = 0.75
            step_direction = (
                                     torch.rand(walk_dist.shape[:-1], device=walk_dist.device) < bias
                             ).float() * 2 - 1

        walk_dist = walk_dist + step_direction.unsqueeze(-1).expand_as(dist) * dist
        walk_dist = walk_dist.clamp(min=0)
    return walk_dist


def _reflect_distances(distances: torch.Tensor, clip_value: float) -> torch.Tensor:
    """
    Reflects distances exceeding the clip value.

    Args:
        distances (torch.Tensor): The distance matrix.
        clip_value (float): The value at which to reflect the distances.

    Returns:
        torch.Tensor: The distance matrix with reflected values.
    """
    exceeding_mask = distances > clip_value
    distances[exceeding_mask] = 2 * clip_value - distances[exceeding_mask]
    return distances


class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(
            torch.Tensor(out_features, in_features).normal_(0, 0.1)
        )
        self.weight_rho = nn.Parameter(
            torch.Tensor(out_features, in_features).normal_(-3, 0.1)
        )
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(-3, 0.1))

        self.prior_std = prior_std
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mu, -bound, bound)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))

        weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)

        out = F.linear(x, weight, bias)

        kl_weight = self._kl_divergence(self.weight_mu, weight_std, self.prior_std)
        kl_bias = self._kl_divergence(self.bias_mu, bias_std, self.prior_std)

        return out, kl_weight + kl_bias

    @staticmethod
    def _kl_divergence(
            mu: torch.Tensor, std: torch.Tensor, prior_std: float
    ) -> torch.Tensor:
        kl = 0.5 * (
                2 * torch.log(prior_std / std)
                - 1
                + (std / prior_std).pow(2)
                + (mu / prior_std).pow(2)
        )
        return kl.sum()
