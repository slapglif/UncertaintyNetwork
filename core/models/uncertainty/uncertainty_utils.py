# .\core\models\uncertainty\uncertainty_utils.py
from typing import Tuple

import torch


def total_uncertainty(epistemic: torch.Tensor, aleatoric: torch.Tensor) -> torch.Tensor:
    return epistemic + aleatoric


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


def entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.

    Returns:
        torch.Tensor: Entropy of the distribution.
    """
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)


def uncertainty_decomposition(_total_uncertainty: torch.Tensor, aleatoric_uncertainty: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
    Decompose total uncertainty into aleatoric and epistemic components.

    Args:
        _total_uncertainty (torch.Tensor): Total uncertainty.
        aleatoric_uncertainty (torch.Tensor): Aleatoric uncertainty.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Aleatoric and epistemic uncertainties.
    """
    epistemic_uncertainty = _total_uncertainty - aleatoric_uncertainty
    return aleatoric_uncertainty, epistemic_uncertainty
