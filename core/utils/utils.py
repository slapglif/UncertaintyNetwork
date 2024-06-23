# core/utils/utils.py
import math
from typing import List, Union

import torch
from loguru import logger
from torch import Tensor
from transformers import PreTrainedModel, GPT2Tokenizer


def _check_nan_inf(tensor: Tensor, message: str):
    """
    Checks if the given tensor contains NaN or Inf values and logs a warning if found.

    Args:
        tensor (torch.Tensor): The tensor to check.
        message (str): A message to include in the warning log.
    """
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        logger.warning(f"NaN or Inf detected in {message}.")


def softplus(x: torch.Tensor) -> torch.Tensor:
    """
    Implements the softplus activation function.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after applying softplus.
    """
    return torch.log(1 + torch.exp(x))


def generate_text(
    model: PreTrainedModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_length: int = 1024,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    num_return_sequences: int = 1,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> List[str]:
    """
    Generates text using the provided model and tokenizer.

    Args:
        model (UncertainTransformerLMHeadModel): The model to use for text generation.
        tokenizer (GPT2Tokenizer): The tokenizer to use for encoding and decoding text.
        prompt (str): The initial text to start generation from.
        max_length (int, optional): The maximum length of the generated text. Defaults to 1024.
        temperature (float, optional): The temperature for sampling. Defaults to 0.7.
        top_k (int, optional): The number of top-k tokens to consider during sampling. Defaults to 50.
        top_p (float, optional): The cumulative probability threshold for nucleus sampling. Defaults to 0.95.
        repetition_penalty (float, optional): The penalty for repeating tokens. Defaults to 1.2.
        num_return_sequences (int, optional): The number of sequences to generate. Defaults to 1.
        device (torch.device, optional): The device to run the model on. Defaults to CUDA if available, else CPU.

    Returns:
        List[str]: The list of generated text sequences.
    """
    model.to(device)
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    generated_texts = []
    try:
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                do_sample=True,
            )
            generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output] # Generate all sequences in one call
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        logger.error(f"Input shape: {input_ids.shape}")
        logger.error(f"Input device: {input_ids.device}")
        logger.error(f"Model device: {next(model.parameters()).device}")
        raise

    return generated_texts


def calculate_perplexity(
        model: PreTrainedModel,
        tokenizer: GPT2Tokenizer,
        text: Union[str, List[str]],
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Union[float, List[float]]:
    """
    Calculates the perplexity of the given text using the provided model and tokenizer.

    Perplexity is a measure of how well a language model predicts a sample of text.
    Lower perplexity indicates better predictive performance.

    Args:
        model (UncertainTransformerLMHeadModel): The model to use for perplexity calculation.
        tokenizer (GPT2Tokenizer): The tokenizer to use for encoding and decoding text.
        text (Union[str, List[str]]): The text to calculate the perplexity for. Can be a single string or a list of strings.
        device (torch.device, optional): The device to run the model on. Defaults to CUDA if available, else CPU.

    Returns:
        Union[float, List[float]]: The perplexity of the text. If the input `text` is a string, returns a float.
                                   If `text` is a list of strings, returns a list of floats representing the perplexity
                                   of each string in the list.
    """
    model.to(device)
    model.eval()

    if isinstance(text, list):
        return [
            calculate_perplexity(model, tokenizer, t, device) for t in text if t.strip()
        ]

    if not text.strip():
        raise ValueError("Input text is empty.")

    input_ids = (
        torch.tensor(tokenizer.encode(text, add_special_tokens=True))
        .unsqueeze(0)
        .to(device)
    )

    if input_ids.numel() == 0:
        raise ValueError("No valid tokens in the input text.")

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError(f"Invalid loss value: {loss.item()}. Check the model outputs.")

    return math.exp(min(loss.item(), 100))
