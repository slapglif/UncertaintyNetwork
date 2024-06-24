# core/utils/uncertainty.py

import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class UncertaintyModule(nn.Module):
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

        try:
            self.gp_layers = nn.ModuleList([
                GaussianProcessLayer(input_dim, input_dim, n_inducing)
                for _ in range(n_gp_layers)
            ])
            self.mc_dropout = MCDropout(p=dropout_rate)
            self.output_layer = HeteroscedasticOutput(input_dim, output_dim)
        except Exception as e:
            logger.error(f"Error initializing UncertaintyModule components: {str(e)}")
            raise

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3 or x.size(2) != self.input_dim:
            logger.error(f"Invalid input shape. Expected (batch_size, seq_len, {self.input_dim}), got {x.shape}")
            raise ValueError(f"Invalid input shape. Expected (batch_size, seq_len, {self.input_dim}), got {x.shape}")

        batch_size, seq_len, _ = x.shape
        device = x.device

        total_uncertainty = torch.zeros(batch_size, seq_len, self.input_dim, device=device)

        outputs = []
        try:
            for _ in range(self.mc_samples):
                h = self.mc_dropout(x)

                gp_uncertainty = torch.zeros_like(h)
                for gp_layer in self.gp_layers:
                    h, uncertainty = gp_layer(h)
                    gp_uncertainty += uncertainty

                mean, het_uncertainty = self.output_layer(h)

                outputs.append(mean)
                total_uncertainty += gp_uncertainty + het_uncertainty

            mean_output = torch.stack(outputs).mean(dim=0)
            mean_uncertainty = total_uncertainty / self.mc_samples

            scaled_mean = mean_output / self.temperature
            scaled_uncertainty = mean_uncertainty / (self.temperature ** 2)

            return scaled_mean, scaled_uncertainty
        except Exception as e:
            logger.error(f"Error in UncertaintyModule forward pass: {str(e)}")
            raise


class GaussianProcessLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_inducing: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_inducing = num_inducing

        self.inducing_points = nn.Parameter(torch.randn(num_inducing, input_dim))
        self.covar_module = nn.Linear(input_dim, num_inducing, bias=False)
        self.mean_module = nn.Linear(input_dim, output_dim)

        self.log_lengthscale = nn.Parameter(torch.zeros(1))
        self.log_variance = nn.Parameter(torch.zeros(1))

    def kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(x1, x2, p=2).pow(2)
        return self.log_variance.exp() * torch.exp(-0.5 * dist / self.log_lengthscale.exp().pow(2))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        covar = F.relu(self.covar_module(x))
        mean = self.mean_module(x)

        kernel = self.kernel(x, self.inducing_points)
        weight = torch.einsum('bni,bno->bio', kernel, covar)
        variance = torch.sum(weight * kernel, dim=1)

        return mean, variance.unsqueeze(-1).expand_as(mean)


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
    return torch.mean(variances, dim=0)


def total_uncertainty(epistemic: torch.Tensor, aleatoric: torch.Tensor) -> torch.Tensor:
    return epistemic + aleatoric


# Continuing from where we left off in uncertainty.py

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


def uncertainty_guided_sampling(logits: torch.Tensor, uncertainties: torch.Tensor,
                                temperature: float = 1.0, alpha: float = 1.0) -> torch.Tensor:
    """
    Perform uncertainty-guided sampling for text generation.

    Args:
        logits (torch.Tensor): Logits from the model
        uncertainties (torch.Tensor): Uncertainties associated with the logits
        temperature (float): Temperature for softmax
        alpha (float): Weight for uncertainty influence

    Returns:
        torch.Tensor: Sampled token indices
    """
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

# Add any additional utility functions or classes as needed for uncertainty analysis
