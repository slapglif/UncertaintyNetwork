# .\core\models\uncertainty\uncertainty_utils.py

import math
from typing import List, Tuple

import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch_mist.estimators import MINE


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


class GaussianProcessLayer(nn.Module):
    """
    A single Gaussian Process layer using a quantum-inspired kernel combined with a Matern kernel.

    Args:
        input_dim (int): The dimension of the input features.
        output_dim (int): The dimension of the output.
        num_inducing (int, optional): Number of inducing points. Defaults to 10.
    """

    def __init__(self, input_dim: int, output_dim: int, num_inducing: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_inducing = num_inducing

        self.inducing_points = nn.Parameter(torch.randn(num_inducing, input_dim))

        # Robust Kernel Combination with Individual Lengthscales (managed by gpytorch)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.AdditiveKernel(
                QuantumWalkKernel(input_dim, n_steps=3),  # No lengthscale here
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
            )
        )

        self.mean_module = gpytorch.means.ZeroMean()

        # Initialize variational distribution without a fixed batch shape
        self.variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=self.num_inducing
        )

        self.variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            self.inducing_points,
            self.variational_distribution,
            learn_inducing_locations=True
        )

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GaussianProcessLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Mean output of shape (batch_size, seq_len, output_dim).
                - Variance of shape (batch_size, seq_len, output_dim).
        """
        variational_dist_f = self.variational_strategy(x)
        variational_dist_y = self.likelihood(variational_dist_f)

        mean = variational_dist_y.mean
        variance = variational_dist_y.variance

        return mean, variance


class QuantumWalkKernel(gpytorch.kernels.Kernel):
    """
    A stateless quantum walk-inspired kernel for Gaussian Processes.

    This kernel is not based on true quantum computation but uses concepts from quantum walks
    to define the similarity between data points. It can capture complex non-linear relationships
    in the data.

    Args:
        input_dim (int): The dimension of the input features.
        coin_param (float, optional): Parameter controlling the 'coin flip' probability in the walk.
            Defaults to 0.5.
        n_steps (int, optional): Number of steps in the quantum walk. Defaults to 5.
    """

    def __init__(self, input_dim: int, coin_param: float = 0.5, n_steps: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.coin_param = coin_param
        self.n_steps = n_steps

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **params) -> torch.Tensor:
        """
        Computes the covariance matrix between two sets of input points.

        Args:
            x1 (torch.Tensor): First set of input points (batch_size, seq_len, input_dim).
            x2 (torch.Tensor): Second set of input points (batch_size, seq_len, input_dim).
            **params: Additional parameters for the kernel (including lengthscale from gpytorch).

        Returns:
            torch.Tensor: Covariance matrix of shape (batch_size, seq_len, seq_len).
        """
        # Handle the last_dim_is_batch argument correctly
        if params.get('last_dim_is_batch', False):
            x1 = x1.transpose(-1, -2)
            x2 = x2.transpose(-1, -2)

        # Calculate pairwise squared distances directly, avoiding cdist
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
        dist = diff.pow(2).sum(dim=-1)

        # Simulate quantum walk without in-place modification
        walk_dist = dist.clone()  # Create a copy for the walk
        for _ in range(self.n_steps):
            coin_flip = (torch.rand(walk_dist.shape, device=walk_dist.device) < self.coin_param).float()

            # Correct the distance update using broadcasting:
            walk_dist = walk_dist * coin_flip + walk_dist.transpose(-1, -2) * (1 - coin_flip)

        # Ensure walk_dist has the same shape as dist
        walk_dist = walk_dist.expand_as(dist)

        # Return the kernel value (the lengthscale will be applied by gpytorch)
        if params.get('last_dim_is_batch', False):
            return torch.exp(-walk_dist).transpose(-1, -2)
        else:
            return torch.exp(-walk_dist)


class TSPEnergyFunction(nn.Module):
    """
    Computes the energy between two sequences of word embeddings based on the Traveling
    Salesman Problem (TSP) with dual Information Bottleneck (IB) regularization and
    Neural Tangent Kernel (NTK) inspiration.

    Args:
        embedding_dim (int): The dimension of the word embeddings.
        compression_dim (int, optional): The dimension of the compressed representation for dual IB.
            If None, dual IB is not applied. Defaults to None.
        lambda_ib (float, optional): Regularization strength for the dual IB term. Defaults to 0.01.
    """

    def __init__(self, embedding_dim: int, compression_dim: int = None, lambda_ib: float = 0.01):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.compression_dim = compression_dim
        self.lambda_ib = lambda_ib

        if compression_dim is not None:
            self.compressor = nn.Linear(embedding_dim, compression_dim)
            self.mine_estimator = MINE()  # Initialize the MINE estimator

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Computes the TSP-inspired energy between two sequences of word embeddings.

        Args:
            x1 (torch.Tensor): First sequence of embeddings (batch_size, seq_len1, embedding_dim).
            x2 (torch.Tensor): Second sequence of embeddings (batch_size, seq_len2, embedding_dim).

        Returns:
            torch.Tensor: Energy matrix of shape (batch_size, seq_len1, seq_len2).
        """
        # 1. Compute Pairwise Distances (Cosine)
        distances = 1 - torch.nn.functional.cosine_similarity(x1.unsqueeze(2), x2.unsqueeze(1), dim=-1)

        # 2. Apply Positional Weighting (NTK-inspired)
        seq_len1, seq_len2 = distances.shape[-2:]
        pos_matrix1 = torch.arange(seq_len1, device=distances.device).unsqueeze(1)
        pos_matrix2 = torch.arange(seq_len2, device=distances.device).unsqueeze(0)
        pos_diff = (pos_matrix1 - pos_matrix2).abs()
        decay_factor = 0.8  # Control decay rate (hyperparameter)
        positional_weights = torch.exp(-decay_factor * pos_diff)
        weighted_distances = distances * positional_weights

        # 3. Apply Dual IB Compression and Regularization (if enabled)
        if self.compression_dim is not None:
            x1_compressed = self.compressor(x1)
            x2_compressed = self.compressor(x2)
            ib_regularizer = self.lambda_ib * self.mine_estimator(x1_compressed, x2_compressed)
        else:
            ib_regularizer = 0.0

        return weighted_distances.sum(dim=-1) + ib_regularizer


class TSPKernel(gpytorch.kernels.Kernel):
    """
    A kernel based on the TSP energy function for Gaussian Processes.

    Args:
        energy_function (TSPEnergyFunction): The TSP-inspired energy function.
        lengthscale (float, optional): Initial lengthscale value for the kernel. Defaults to 1.0.
    """

    def __init__(self, energy_function: TSPEnergyFunction, lengthscale: float = 1.0):
        super().__init__()
        self.energy_function = energy_function
        self.lengthscale = nn.Parameter(torch.tensor(lengthscale))

    def forward(self, x1, x2, **params):
        energy = self.energy_function(x1, x2)
        return torch.exp(-energy / self.lengthscale ** 2)


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
