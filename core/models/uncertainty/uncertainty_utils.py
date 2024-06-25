# .\core\models\uncertainty\uncertainty_utils.py
import math
from typing import List
from typing import Optional, Tuple

import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from gpytorch.kernels import Kernel, RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from loguru import logger
from torch import Tensor
from torch_mist.critic import Critic
from torch_mist.estimators import MINE

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

        logger.info(f"Initializing UncertaintyModule with {n_gp_layers} GP layers and {mc_samples} MC samples")

        self.gp_layers = nn.ModuleList([
            GaussianProcessLayer(input_dim, input_dim, n_inducing)
            for _ in range(n_gp_layers)
        ])
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
            logger.error(f"Invalid input shape. Expected (batch_size, seq_len, {self.input_dim}), got {x.shape}")
            raise ValueError(f"Invalid input shape. Expected (batch_size, seq_len, {self.input_dim}), got {x.shape}")

        batch_size, seq_len, _ = x.shape
        device = x.device

        total_uncertainty = torch.zeros(batch_size, seq_len, self.output_dim, device=device)
        outputs = []

        try:
            for _ in range(self.mc_samples):
                h = self.mc_dropout(x)

                for gp_layer in self.gp_layers:
                    h, variance = gp_layer(h)
                    total_uncertainty += variance

                outputs.append(self.output_layer(h))

            mean_output = torch.stack(outputs).mean(dim=0)
            mean_uncertainty = total_uncertainty / self.mc_samples

            scaled_mean = mean_output / self.temperature
            scaled_uncertainty = mean_uncertainty / (self.temperature ** 2)

            return scaled_mean, scaled_uncertainty
        except Exception as e:
            logger.error(f"Error in UncertaintyModule forward pass: {str(e)}")
            raise


class RandomWalkSimilarity(nn.Module):
    """
    Calculates a similarity matrix based on a random walk.
    """

    def __init__(self, n_steps: int = 5, walk_type: str = 'standard'):
        super().__init__()
        self.n_steps = n_steps
        self.walk_type = walk_type

        if self.walk_type not in ['standard', 'reflected', 'biased']:
            raise ValueError(f"Invalid walk_type: {self.walk_type}. Choose from 'standard', 'reflected', 'biased'.")

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Computes the similarity matrix.

        Args:
            dist (torch.Tensor): Pairwise distance matrix (batch_size, seq_len1, seq_len2).

        Returns:
            torch.Tensor: Similarity matrix (batch_size, seq_len1, seq_len2).
        """
        walk_dist = dist.clone()  # Create a copy for the walk
        for _ in range(self.n_steps):
            # Generate random steps (in the same direction for all dimensions)
            step_direction = torch.randint(0, 2, size=(walk_dist.shape[:-1]), device=walk_dist.device) * 2 - 1

            # Apply different walk types
            if self.walk_type == 'reflected':
                step_direction = step_direction.float() * 0.5  # Smaller steps for reflection
            elif self.walk_type == 'biased':
                bias = 0.75  # Example bias value (adjust as needed)
                step_direction = (torch.rand(walk_dist.shape[:-1], device=walk_dist.device) < bias).float() * 2 - 1

            # Update walk_dist based on the step direction
            walk_dist = walk_dist + step_direction.unsqueeze(-1).expand_as(dist) * dist  # Add step to the distance

        # Calculate similarity (you can customize this)
        similarity = torch.exp(-walk_dist)
        return similarity


def random_walk_distances(dist: torch.Tensor, n_steps: int, walk_type: str, step_scale: float,
                          dist_clip: float) -> torch.Tensor:
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
        step_direction = torch.randint(0, 2, size=(walk_dist.shape[:-1]), device=walk_dist.device) * 2 - 1

        if walk_type == 'reflected':
            step_direction = step_direction.float() * step_scale
            walk_dist = _reflect_distances(walk_dist, dist_clip)
        elif walk_type == 'biased':
            bias = 0.75
            step_direction = (torch.rand(walk_dist.shape[:-1], device=walk_dist.device) < bias).float() * 2 - 1

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


class RandomWalkKernel(gpytorch.kernels.Kernel):
    """
    A stateless kernel that incorporates random walk similarity into an RBF kernel.
    """

    def __init__(self, input_dim: int, n_steps: int = 5, walk_type: str = 'standard',
                 step_scale: float = 0.2, dist_clip: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.n_steps = n_steps
        self.walk_type = walk_type
        self.step_scale = step_scale
        self.dist_clip = dist_clip
        self.rbf_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim, **kwargs)

    def forward(self, x1, x2, diag=False, **params):
        # Calculate the squared Euclidean distance
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
        dist = diff.pow(2).sum(dim=-1)

        # Get the modified distances from the random walk
        modified_dist = random_walk_distances(
            dist, self.n_steps, self.walk_type, self.step_scale, self.dist_clip
        )

        # Apply RBF kernel to the modified distances
        return self.rbf_kernel(modified_dist, modified_dist, diag=diag, **params)


class GaussianProcessLayer(ApproximateGP):
    """
    A Gaussian Process layer using an approximate GP model.

    This layer uses variational inference for scalable Gaussian Process modeling.

    Attributes:
        mean_module (gpytorch.means.Mean): The mean module for the GP.
        covar_module (gpytorch.kernels.Kernel): The covariance module (kernel) for the GP.
        variational_strategy (gpytorch.variational.VariationalStrategy): The variational strategy.

    Args:
        input_dim (int): The input dimension.
        output_dim (int): The output dimension.
        num_inducing (int): The number of inducing points.
        kernel (Optional[Kernel]): The kernel to use. If None, RBFKernel is used.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_inducing: int = 10,
            kernel: Optional[Kernel] = None
    ):
        # Initialize inducing points
        inducing_points = torch.randn(num_inducing, input_dim)

        # Set up variational distribution and strategy
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=num_inducing)
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        # Set up mean and covariance modules
        self.mean_module = ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel if kernel is not None else RBFKernel(ard_num_dims=input_dim)
        )

        # Set output dimension
        self.output_dim = output_dim

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Forward pass of the GaussianProcessLayer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            gpytorch.distributions.MultivariateNormal: A multivariate normal distribution.
        """
        # Reshape input for compatibility with GP
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size * seq_len, -1)

        # Compute mean and covariance
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        # Return multivariate normal distribution
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SplineCriticLayer(SplineNetLayer):
    """
    Spline-based critic layer for use in MINE estimator.
    Inherits from SplineNetLayer but with output tailored for critic function.
    """

    def __init__(self, input_dim: int, output_dim: int, num_grids: int = 8, *_, **kwargs):
        super().__init__(input_dim, output_dim, num_grids=num_grids, *_, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SplineCriticLayer. Flattens the output for compatibility with KANCritic.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: The flattened output tensor of shape (batch_size, seq_len * output_dim).
        """
        x = super().forward(x)
        return x.view(x.size(0), -1)  # Flatten the output, keeping batch dimension


class KANCritic(Critic):
    """
    KAN-based critic network for MINE estimator, inheriting from torch_mist's Critic.
    """

    def __init__(self, input_dim, hidden_dims: List[int], grid_min=-1.2, grid_max=0.2, num_grids=8, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList(
            [
                SplineCriticLayer(  # Use SplineCriticLayer here
                    in_dim,
                    out_dim,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    **kwargs
                )
                for in_dim, out_dim in zip([input_dim] + hidden_dims[:-1], hidden_dims)
            ]
        )
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the critic network, taking both x and y as inputs (as expected by torch_mist).

        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: The critic's output (single value).
        """
        # Combine x and y (you can customize this based on your needs)
        combined_input = torch.cat((x, y), dim=-1)
        for layer in self.layers:
            combined_input = layer(combined_input)
        return self.output_layer(combined_input).squeeze(-1)


class TSPEnergyFunction(nn.Module):
    """
    Computes the energy between two sequences of word embeddings based on the Traveling
    Salesman Problem (TSP) with dual Information Bottleneck (IB) regularization.

    Attributes:
        embedding_dim (int): The dimension of the word embeddings.
        compression_dim (Optional[int]): The dimension of the compressed representation for dual IB.
        lambda_ib (float): Regularization strength for the dual IB term.
        compressor (nn.Linear): Linear layer for compressing input if compression_dim is not None.
        critic (Critic): The critic network for MINE estimator.
        mine_estimator (MINE): The MINE estimator for mutual information estimation.

    Args:
        embedding_dim (int): The dimension of the word embeddings.
        compression_dim (Optional[int]): The dimension of the compressed representation for dual IB.
        lambda_ib (float): Regularization strength for the dual IB term.
    """

    def __init__(self, embedding_dim: int, compression_dim: Optional[int] = None, lambda_ib: float = 0.01):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.compression_dim = compression_dim
        self.lambda_ib = lambda_ib

        if compression_dim is not None:
            self.compressor = nn.Linear(embedding_dim, compression_dim)
            self.critic = Critic(input_dim=compression_dim * 2, hidden_dims=[128, 64])
            self.mine_estimator = MINE(critic=self.critic)
        else:
            self.compressor = None
            self.critic = None
            self.mine_estimator = None

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Computes the TSP-inspired energy between two sequences of word embeddings.

        Args:
            x1 (torch.Tensor): First sequence of embeddings (batch_size, seq_len1, embedding_dim).
            x2 (torch.Tensor): Second sequence of embeddings (batch_size, seq_len2, embedding_dim).

        Returns:
            torch.Tensor: Energy matrix of shape (batch_size, seq_len1, seq_len2).
        """
        # Compute pairwise distances (cosine similarity)
        distances = 1 - torch.nn.functional.cosine_similarity(x1.unsqueeze(2), x2.unsqueeze(1), dim=-1)

        # Apply positional weighting (NTK-inspired)
        seq_len1, seq_len2 = distances.shape[-2:]
        pos_matrix1 = torch.arange(seq_len1, device=distances.device).unsqueeze(1)
        pos_matrix2 = torch.arange(seq_len2, device=distances.device).unsqueeze(0)
        pos_diff = (pos_matrix1 - pos_matrix2).abs()
        decay_factor = 0.8  # Control decay rate (hyperparameter)
        positional_weights = torch.exp(-decay_factor * pos_diff)
        weighted_distances = distances * positional_weights

        # Apply Dual IB Compression and Regularization (if enabled)
        if self.compression_dim is not None:
            x1_compressed = self.compressor(x1)
            x2_compressed = self.compressor(x2)

            # Pad sequences to match lengths
            max_seq_len = max(x1_compressed.shape[1], x2_compressed.shape[1])
            x1_compressed = self._pad_sequence(x1_compressed, max_seq_len)
            x2_compressed = self._pad_sequence(x2_compressed, max_seq_len)

            ib_regularizer = self.lambda_ib * self.mine_estimator(x1_compressed, x2_compressed)
            weighted_distances = weighted_distances + ib_regularizer.unsqueeze(-1).unsqueeze(-1)

        return weighted_distances

    @staticmethod
    def _pad_sequence(x: torch.Tensor, max_seq_len: int) -> torch.Tensor:
        """
        Pads the input tensor to the specified maximum sequence length.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            max_seq_len (int): The desired sequence length after padding.

        Returns:
            torch.Tensor: Padded tensor of shape (batch_size, max_seq_len, dim).
        """
        batch_size, seq_len, dim = x.shape
        if seq_len < max_seq_len:
            padding = torch.zeros(batch_size, max_seq_len - seq_len, dim, device=x.device)
            return torch.cat((x, padding), dim=1)
        return x


class TSPKernel(gpytorch.kernels.RBFKernel):  # Inherit from a kernel with lengthscale
    """
    A kernel based on the TSP energy function for Gaussian Processes.

    Args:
        energy_function (TSPEnergyFunction): The TSP-inspired energy function.
        lengthscale (float, optional): Initial lengthscale value for the kernel. Defaults to 1.0.
    """

    def __init__(self, energy_function: TSPEnergyFunction, **kwargs):
        super().__init__(**kwargs)  # Pass any additional arguments to the base kernel
        self.energy_function = energy_function
        self.lengthscale = self.lengthscale  # Use the lengthscale from the base kernel

    def forward(self, x1, x2, **params):
        energy = self.energy_function(x1, x2)
        # Use the base kernel's functionality to apply the lengthscale
        return super().forward(x1, x2, **params) * torch.exp(-energy)


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


def mutual_information(mean_probs: torch.Tensor, all_probs: torch.Tensor) -> torch.Tensor:
    entropy_mean = entropy(mean_probs)
    mean_entropy = torch.mean(entropy(all_probs), dim=0)
    return entropy_mean - mean_entropy


def calibration_plot(confidences: torch.Tensor, accuracies: torch.Tensor, n_bins: int = 10) -> Tuple[
    torch.Tensor, torch.Tensor]:
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


def expected_calibration_error(confidences: torch.Tensor, accuracies: torch.Tensor, n_bins: int = 10) -> torch.Tensor:
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
    return torch.mean((probs - F.one_hot(labels, num_classes=probs.size(-1))).pow(2).sum(dim=-1))


class EnsembleModel(nn.Module):
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = [model(x) for model in self.models]
        mean_output = torch.stack([out[0] for out in outputs]).mean(dim=0)
        variance = torch.stack([out[1] for out in outputs]).var(dim=0)
        return mean_output, variance


def out_of_distribution_detection(in_dist_uncertainties: torch.Tensor,
                                  out_dist_uncertainties: torch.Tensor) -> Tuple[float, float]:
    """
    Perform out-of-distribution detection using uncertainty estimates.

    Args:
        in_dist_uncertainties (torch.Tensor): Uncertainties from in-distribution data
        out_dist_uncertainties (torch.Tensor): Uncertainties from out-of-distribution data

    Returns:
        Tuple[float, float]: AUROC and AUPRC scores
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    y_true = torch.cat([torch.ones_like(in_dist_uncertainties),
                        torch.zeros_like(out_dist_uncertainties)])
    y_score = torch.cat([in_dist_uncertainties, out_dist_uncertainties])

    auroc = roc_auc_score(y_true.cpu().numpy(), y_score.cpu().numpy())
    auprc = average_precision_score(y_true.cpu().numpy(), y_score.cpu().numpy())

    return auroc, auprc


def uncertainty_guided_sampling(
        logits: torch.Tensor,
        uncertainties: torch.Tensor,
        temperature: float = 1.0,
        alpha: float = 1.0
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
        raise ValueError(f"Shape mismatch: logits {logits.shape} != uncertainties {uncertainties.shape}")

    scaled_logits = logits / temperature
    uncertainty_weight = F.softmax(-alpha * uncertainties, dim=-1)
    weighted_logits = scaled_logits * uncertainty_weight
    return torch.multinomial(F.softmax(weighted_logits, dim=-1), num_samples=1).squeeze(-1)


def active_learning_acquisition(uncertainties: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Select samples for active learning based on uncertainty.

    Args:
        uncertainties (torch.Tensor): Uncertainties for each sample
        n_samples (int): Number of samples to select

    Returns:
        torch.Tensor: Indices of selected samples
    """
    return torch.topk(uncertainties, k=n_samples, dim=0).indices


def uncertainty_weighted_loss(loss: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
    """
    Compute uncertainty-weighted loss.

    Args:
        loss (torch.Tensor): Standard loss values
        uncertainty (torch.Tensor): Uncertainty estimates

    Returns:
        torch.Tensor: Uncertainty-weighted loss
    """
    return (loss * torch.exp(-uncertainty) + uncertainty).mean()


class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-3, 0.1))
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
    def _kl_divergence(mu: torch.Tensor, std: torch.Tensor, prior_std: float) -> torch.Tensor:
        kl = 0.5 * (2 * torch.log(prior_std / std) - 1 + (std / prior_std).pow(2) + (mu / prior_std).pow(2))
        return kl.sum()
