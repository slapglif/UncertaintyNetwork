# visualization_utils.py

import os
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from loguru import logger
from pandas.plotting._matplotlib import HistPlot


def visualize_results(results: Dict[str, Any]):
    for task, task_results in results.items():
        if task == "language_modeling":
            plot_perplexity(task_results["perplexity"])
        elif task == "text_generation":
            plot_generation_metrics(task_results)
        elif task == "uncertainty_estimation":
            plot_uncertainty_metrics(task_results)


def plot_perplexity(perplexity: float):
    plt.figure(figsize=(10, 6))
    plt.bar(["Perplexity"], [perplexity])
    plt.title("Language Modeling Perplexity")
    plt.ylabel("Perplexity")
    plt.show()


def plot_generation_metrics(results: Dict[str, float]):
    metrics = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(12, 6))
    plt.bar(metrics, values)
    plt.title("Text Generation Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.show()


def plot_uncertainty_metrics(results: Dict[str, float]):
    plt.figure(figsize=(12, 6))
    plt.bar(results.keys(), results.values())
    plt.title("Uncertainty Estimation Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_uncertainty_decomposition(uncertainties: List[float]):
    plt.figure(figsize=(10, 6))
    HistPlot(uncertainties, kde=True)
    plt.title("Uncertainty Distribution")
    plt.xlabel("Uncertainty")
    plt.ylabel("Frequency")
    plt.show()


def plot_attention_weights(attention_weights: torch.Tensor, tokenizer):
    # Assuming attention_weights shape: [batch_size, num_heads, seq_len, seq_len]
    # We'll visualize the average attention across all heads for the first example in the batch
    avg_attention = attention_weights[0].mean(dim=0).cpu().numpy()

    plt.figure(figsize=(12, 8))
    sns.heatmap(avg_attention, cmap='viridis')
    plt.title("Average Attention Weights")
    plt.xlabel("Token Position (Target)")
    plt.ylabel("Token Position (Source)")
    plt.show()


def plot_uncertainty_vs_error(uncertainties: List[float], errors: List[float]):
    plt.figure(figsize=(10, 6))
    plt.scatter(uncertainties, errors, alpha=0.5)
    plt.title("Uncertainty vs. Error")
    plt.xlabel("Uncertainty")
    plt.ylabel("Error")
    plt.show()


def plot_learning_curves(train_losses: List[float], val_losses: List[float]):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Learning Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def visualize_embeddings(embeddings: torch.Tensor, labels: List[str]):
    from sklearn.manifold import TSNE

    # Reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings.cpu().numpy())

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=range(len(labels)), cmap='viridis')
    plt.colorbar(scatter)
    plt.title("Token Embeddings Visualization")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()


def save_visualizations(results: Dict[str, Any], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for task, task_results in results.items():
        if task == "language_modeling":
            plt.figure()
            plot_perplexity(task_results["perplexity"])
            plt.savefig(os.path.join(output_dir, "perplexity.png"))
            plt.close()

        elif task == "text_generation":
            plt.figure()
            plot_generation_metrics(task_results)
            plt.savefig(os.path.join(output_dir, "generation_metrics.png"))
            plt.close()

        elif task == "uncertainty_estimation":
            plt.figure()
            plot_uncertainty_metrics(task_results)
            plt.savefig(os.path.join(output_dir, "uncertainty_metrics.png"))
            plt.close()

    logger.info(f"Visualizations saved to {output_dir}")
