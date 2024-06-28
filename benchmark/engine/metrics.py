# metrics.py

from typing import List, Dict

import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# metrics.py (continued)

def expected_calibration_error(confidences: List[float], accuracies: List[float], num_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.array(confidences)
    accuracies = np.array(accuracies)

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def calculate_brier_score(probabilities: List[float], outcomes: List[int]) -> float:
    return np.mean((np.array(probabilities) - np.array(outcomes)) ** 2)


def calculate_auc_roc(probabilities: List[float], labels: List[int]) -> float:
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels, probabilities)


def calculate_log_loss(probabilities: List[float], labels: List[int]) -> float:
    from sklearn.metrics import log_loss
    return log_loss(labels, probabilities)


def calculate_mutual_information(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    mutual_info = torch.mean(entropy)
    return mutual_info.item()


def calculate_predictive_entropy(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    return entropy.mean().item()


def calculate_diversity_metrics(generated_texts: List[str]) -> Dict[str, float]:
    from nltk import ngrams

    def distinct_n_grams(text, n):
        n_grams = ngrams(text.split(), n)
        return len(set(n_grams)) / len(list(n_grams))

    diversity_1 = np.mean([distinct_n_grams(text, 1) for text in generated_texts])
    diversity_2 = np.mean([distinct_n_grams(text, 2) for text in generated_texts])

    return {
        "diversity_1": diversity_1,
        "diversity_2": diversity_2
    }




def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from the loss."""
    return torch.exp(torch.tensor(loss)).item()


def calculate_accuracy(predictions: List[int], labels: List[int]) -> float:
    """Calculate accuracy given predictions and labels."""
    return accuracy_score(labels, predictions)


def calculate_precision_recall_f1(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score."""
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def calculate_bleu_score(references: List[List[str]], hypotheses: List[str]) -> float:
    """Calculate BLEU score for generated text."""
    return corpus_bleu([[ref] for ref in references], hypotheses)


def calculate_rouge_scores(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores for generated text."""
    rouge = Rouge()
    scores = rouge.get_scores(hypotheses, references, avg=True)
    return {
        "rouge-1": scores["rouge-1"]["f"],
        "rouge-2": scores["rouge-2"]["f"],
        "rouge-l": scores["rouge-l"]["f"]
    }


def calculate_uncertainty_metrics(uncertainties: List[float], accuracies: List[float]) -> Dict[str, float]:
    """Calculate uncertainty metrics including ECE and Brier score."""
    ece = expected_calibration_error(uncertainties, accuracies)
    brier_score = calculate_brier_score(uncertainties, accuracies)
    return {
        "expected_calibration_error": ece,
        "brier_score": brier_score
    }
