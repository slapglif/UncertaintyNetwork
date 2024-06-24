# core/utils/utils.py

from typing import List

import torch
from loguru import logger
from torch import Tensor
from transformers import PreTrainedModel, GenerationConfig

from core.utils.tokenizer import Tokenizer


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


# noinspection PyTypeChecker
@torch.inference_mode()
def generate_text(
        model: PreTrainedModel,
        tokenizer: "Tokenizer",
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        num_return_sequences: int = 1,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> List[str]:
    """
    Generates text using the specified model and tokenizer with given parameters.

    Args:
        model (PreTrainedModel): The pre-trained language model.
        tokenizer (Tokenizer): The tokenizer for encoding and decoding text.
        prompt (str): The input prompt to generate text from.
        max_length (int, optional): The maximum length of the generated text (default: 100).
        temperature (float, optional): The temperature parameter for sampling (default: 0.7).
        top_k (int, optional): The number of top-k tokens to consider for sampling (default: 50).
        top_p (float, optional): The probability threshold for nucleus sampling (default: 0.95).
        repetition_penalty (float, optional): The repetition penalty (default: 1.2).
        num_return_sequences (int, optional): The number of sequences to generate for each prompt (default: 1).
        device (torch.device, optional): The device to run the model on (default: CUDA if available, else CPU).

    Returns:
        List[str]: A list of generated texts.
    """
    model.to(device)
    model.eval()

    try:
        logger.info(f"Encoding prompt: {prompt}")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        logger.debug(f"Input IDs shape: {input_ids.shape}, device: {input_ids.device}")

        attention_mask = torch.ones_like(input_ids)

        generation_config = GenerationConfig(
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )

        logger.info(f"Starting text generation with generation config: {generation_config}")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )

        logger.info("Text generation completed")

        return tokenizer.batch_decode(outputs, skip_special_tokens=True)
    except Exception as e:
        logger.exception(f"Error during text generation: {str(e)}")
        return []


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
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # Convert to tensor
    num_tokens = input_ids.shape[1]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return torch.exp(loss / num_tokens).item()
