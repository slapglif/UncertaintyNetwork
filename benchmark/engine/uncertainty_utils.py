# uncertainty_utils.py
# Generated stub for advanced benchmarking tool


from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, average_precision_score


def monte_carlo_dropout(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, num_samples: int) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    model.train()  # Enable dropout
    outputs = [model(input_ids, attention_mask=attention_mask) for _ in range(num_samples)]
    mean = torch.stack([o.logits for o in outputs]).mean(0)
    variance = torch.stack([o.logits for o in outputs]).var(0)
    return mean, variance


def calibrate_model(model: nn.Module, val_dataloader: torch.utils.data.DataLoader) -> nn.Module:
    temperatures = []
    for batch in val_dataloader:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        temperature = calibrate_temperature(logits, labels)
        temperatures.append(temperature)

    final_temperature = np.mean(temperatures)
    model.temperature = nn.Parameter(torch.tensor([final_temperature]))
    return model


def calibrate_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    def temperature_scale(t):
        return nn.CrossEntropyLoss()(logits / t, labels)

    return minimize_scalar(temperature_scale, bounds=(0.1, 10.0), method='brent').x


def expected_calibration_error(confidences: torch.Tensor, accuracies: torch.Tensor, num_bins: int = 10) -> float:
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.zeros(1, device=confidences.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


def uncertainty_decomposition(total_uncertainty: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Assuming total_uncertainty is the variance of the predictions
    # Aleatoric uncertainty is estimated as the mean of the variances
    aleatoric_uncertainty = total_uncertainty.mean(dim=0)

    # Epistemic uncertainty is estimated as the variance of the means
    epistemic_uncertainty = total_uncertainty.var(dim=0)

    return aleatoric_uncertainty, epistemic_uncertainty


def out_of_distribution_detection(in_dist_uncertainties: torch.Tensor, out_dist_uncertainties: torch.Tensor) -> Tuple[
    float, float]:
    y_true = torch.cat([torch.ones_like(in_dist_uncertainties), torch.zeros_like(out_dist_uncertainties)])
    y_score = torch.cat([in_dist_uncertainties, out_dist_uncertainties])

    auroc = roc_auc_score(y_true.cpu().numpy(), y_score.cpu().numpy())
    auprc = average_precision_score(y_true.cpu().numpy(), y_score.cpu().numpy())

    return auroc, auprc


def compute_uncertainty_metrics(model: nn.Module, test_dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
    model.eval()
    uncertainties = []
    accuracies = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            mean, variance = monte_carlo_dropout(model, input_ids, attention_mask, num_samples=10)

            predictions = mean.argmax(dim=-1)
            uncertainties.extend(variance.mean(dim=-1).cpu().numpy())
            accuracies.extend((predictions == labels).float().cpu().numpy())

    uncertainties = torch.tensor(uncertainties)
    accuracies = torch.tensor(accuracies)

    ece = expected_calibration_error(uncertainties, accuracies)
    aleatoric, epistemic = uncertainty_decomposition(uncertainties)

    return {
        "expected_calibration_error": ece,
        "mean_aleatoric_uncertainty": aleatoric.mean().item(),
        "mean_epistemic_uncertainty": epistemic.mean().item(),
        "total_uncertainty": uncertainties.mean().item()
    }


def entropy(probs: torch.Tensor) -> torch.Tensor:
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)


def mutual_information(mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    expected_entropy = entropy(torch.softmax(mean, dim=-1))
    expected_p = torch.softmax(mean / (1 + torch.exp(variance)), dim=-1)
    entropy_expected_p = entropy(expected_p)
    return entropy_expected_p - expected_entropy


def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)


def bald_score(mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    return mutual_information(mean, variance)


