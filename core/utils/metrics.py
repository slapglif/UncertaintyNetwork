# core/utils/metrics.py
import math
from typing import List, Union

import nltk
import torch
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from transformers import PreTrainedModel

from core.utils.tokenizer import Tokenizer

nltk.download("punkt")


def calculate_perplexity(
    model: PreTrainedModel,
    tokenizer: Tokenizer,
    text: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> float:
    """
    Calculate the perplexity of the given text using the specified model and tokenizer.

    Args:
        model (PreTrainedModel): The pre-trained language model.
        tokenizer (Tokenizer): The tokenizer for encoding the text.
        text (str): The input text to calculate perplexity for.
        device (torch.device): The device to run the model on (default: CUDA if available, else CPU).

    Returns:
        float: The perplexity of the text.
    """
    model.to(device)
    model.eval()

    tokens = tokenizer.encode(text)
    input_ids = tokens.unsqueeze(0).to(device)
    num_tokens = input_ids.shape[1]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return torch.exp(loss / num_tokens).item()


def calculate_bleu_score(
    references: List[List[str]],
    hypotheses: List[str],
    smoothing_function: SmoothingFunction = SmoothingFunction().method1,
    weights: Union[List[float], tuple] = (0.25, 0.25, 0.25, 0.25),
    lowercase: bool = False,
) -> float:
    """
    Calculate the BLEU score between the reference texts and the generated hypotheses.

    Args:
        references (List[List[str]]): A list of reference texts, where each reference is a list of tokens.
        hypotheses (List[str]): A list of generated hypotheses, where each hypothesis is a string.
        smoothing_function (SmoothingFunction): The smoothing function to use for BLEU score calculation.
        weights (Union[List[float], tuple]): The weights for different n-gram scores (default: equal weights).
        lowercase (bool): Whether to lowercase the tokens before calculating the BLEU score.

    Returns:
        float: The corpus-level BLEU score.
    """
    references = [
        [ref.lower() if lowercase else ref for ref in refs] for refs in references
    ]
    hypotheses = [hyp.lower() if lowercase else hyp for hyp in hypotheses]

    return corpus_bleu(
        [[ref] for ref in references],
        hypotheses,
        smoothing_function=smoothing_function,
        weights=weights,
    )


def calculate_rouge_scores(
    references: List[str],
    hypotheses: List[str],
    lowercase: bool = False,
) -> dict:
    """
    Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) between the reference texts and the generated hypotheses.

    Args:
        references (List[str]): A list of reference texts, where each reference is a string.
        hypotheses (List[str]): A list of generated hypotheses, where each hypothesis is a string.
        lowercase (bool): Whether to lowercase the tokens before calculating the ROUGE scores.

    Returns:
        dict: A dictionary containing the average ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    """
    from rouge import Rouge

    rouge = Rouge()

    if lowercase:
        references = [ref.lower() for ref in references]
        hypotheses = [hyp.lower() for hyp in hypotheses]

    scores = rouge.get_scores(hypotheses, references, avg=True)
    return {
        "rouge-1": scores["rouge-1"]["f"],
        "rouge-2": scores["rouge-2"]["f"],
        "rouge-l": scores["rouge-l"]["f"],
    }
