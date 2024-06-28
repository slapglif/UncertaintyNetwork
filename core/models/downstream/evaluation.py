# core/tests/evaluation.py
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from transformers import PreTrainedModel, StoppingCriteria

from core.models.uncertainty.uncertainty import (
    UncertainTransformerLMHeadModel,
    UncertainTransformerConfig,
)
from core.utils.metrics import (
    calculate_bleu_score,
    calculate_perplexity,
    calculate_rouge_scores,
)
from core.utils.tokenizer import Tokenizer
from core.utils.utils import generate_text


class MaxLengthCriteria(StoppingCriteria):
    """
    Custom stopping criteria based on maximum length.

    This class implements a stopping criteria that halts generation
    when the generated sequence reaches a specified maximum length.

    Args:
        max_length (int): The maximum length of the generated sequence.
    """

    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        """
        Check if the stopping condition is met.

        Args:
            input_ids (torch.LongTensor): The generated token IDs.
            scores (torch.FloatTensor): The scores for each token.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True if the stopping condition is met, False otherwise.
        """
        return input_ids.shape[-1] >= self.max_length


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: Tokenizer,
    prompts: List[str],
    checkpoint_path: Optional[str] = None,
    max_length: int = 100,
    num_return_sequences: int = 1,
    device: torch.device = torch.device("cuda" if torch.cpu.is_available() else "cpu"),
) -> None:
    """
    Evaluate the model's performance on a set of prompts.

    Args:
        model (PreTrainedModel): The pre-trained language model.
        tokenizer (Tokenizer): The tokenizer for encoding and decoding text.
        prompts (List[str]): A list of prompts to generate text from.
        checkpoint_path (Optional[str]): The path to the model checkpoint (default: None).
        max_length (int): The maximum length of the generated text (default: 100).
        num_return_sequences (int): The number of sequences to generate for each prompt (default: 1).
        device (torch.device): The device to run the model on (default: CUDA if available, else CPU).
    """
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))
        logger.info(f"Loaded model checkpoint from: {checkpoint_path}")

    model.to(device)
    model.eval()

    perplexities = []
    bleu_scores = []
    rouge_scores = []

    for prompt in tqdm(prompts, desc="Evaluating"):
        generated_texts = generate_text(
            model,
            tokenizer,
            prompt,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            device=device,
        )

        # Check if text was generated:
        if len(generated_texts) == 0:
            logger.warning(f"No text generated for prompt: {prompt}")
            continue

        for generated_text in generated_texts:
            perplexity = calculate_perplexity(model, tokenizer, generated_text, device)
            bleu_score = calculate_bleu_score([[prompt]], [generated_text])
            rouge_score = calculate_rouge_scores([prompt], [generated_text])

            perplexities.append(perplexity)
            bleu_scores.append(bleu_score)
            rouge_scores.append(rouge_score)

    avg_perplexity = np.mean(perplexities) if perplexities else float("nan")
    avg_bleu_score = np.mean(bleu_scores) if bleu_scores else float("nan")
    avg_rouge_scores = (
        {
            key: np.mean([score[key] for score in rouge_scores])
            for key in rouge_scores[0]
        }
        if rouge_scores
        else {}
    )

    logger.info(f"Average Perplexity: {avg_perplexity:.2f}")
    logger.info(f"Average BLEU Score: {avg_bleu_score:.2f}")
    logger.info("Average ROUGE Scores:")
    for key, value in avg_rouge_scores.items():
        logger.info(f"  {key}: {value:.2f}")

    # Visualize evaluation metrics
    visualize_metrics(perplexities, bleu_scores, rouge_scores)


def visualize_metrics(
    perplexities: List[float], bleu_scores: List[float], rouge_scores: List[dict]
) -> None:
    """
    Visualize the evaluation metrics.

    Args:
        perplexities (List[float]): A list of perplexity scores.
        bleu_scores (List[float]): A list of BLEU scores.
        rouge_scores (List[dict]): A list of ROUGE score dictionaries.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Perplexity histogram
    axs[0, 0].hist(perplexities, bins=20, edgecolor="black")
    axs[0, 0].set_title("Perplexity Distribution")
    axs[0, 0].set_xlabel("Perplexity")
    axs[0, 0].set_ylabel("Frequency")

    # BLEU score histogram
    axs[0, 1].hist(bleu_scores, bins=20, edgecolor="black")
    axs[0, 1].set_title("BLEU Score Distribution")
    axs[0, 1].set_xlabel("BLEU Score")
    axs[0, 1].set_ylabel("Frequency")

    # ROUGE-1 score histogram
    rouge_1_scores = [score["rouge-1"] for score in rouge_scores]
    axs[1, 0].hist(rouge_1_scores, bins=20, edgecolor="black")
    axs[1, 0].set_title("ROUGE-1 Score Distribution")
    axs[1, 0].set_xlabel("ROUGE-1 Score")
    axs[1, 0].set_ylabel("Frequency")

    # ROUGE-L score histogram
    rouge_l_scores = [score["rouge-l"] for score in rouge_scores]
    axs[1, 1].hist(rouge_l_scores, bins=20, edgecolor="black")
    axs[1, 1].set_title("ROUGE-L Score Distribution")
    axs[1, 1].set_xlabel("ROUGE-L Score")
    axs[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def main():
    # Set up logging
    logger.add(
        "evaluation.log",
        format="{time} {level} {message}",
        level="INFO",
        rotation="10 MB",
    )

    # Load the tokenizer
    tokenizer = Tokenizer()

    # Define the model configuration
    config = UncertainTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_layers=6,
        dropout=0.1,
        max_position_embeddings=1024,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Load the model
    model = UncertainTransformerLMHeadModel(config)
    checkpoint_path = "path/to/checkpoint.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        logger.info(f"Loaded model checkpoint from: {checkpoint_path}")
    else:
        logger.warning(
            "Checkpoint path does not exist. Using a randomly initialized model."
        )

    # Set the device
    device = torch.device("cuda" if torch.cpu.is_available() else "cpu")
    model.to(device)

    # Define the evaluation prompts
    prompts = [
        "Once upon a time,",
        "The quick brown fox",
        "In a world where",
        "The future of artificial intelligence",
        "Climate change is a pressing issue",
    ]

    # Run the evaluation
    evaluate_model(
        model,
        tokenizer,
        prompts,
        max_length=100,
        num_return_sequences=1,
        device=device,
    )


if __name__ == "__main__":
    main()
