# core/utils/utils.py
import math
from typing import List, Optional
from typing import Union

import torch
from loguru import logger
from torch import Tensor, nn
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
        pad_token_id: Optional[int] = None,
) -> List[str]:
    """
    Generates text using the provided model and tokenizer.

    Args:
        model (PreTrainedModel): The model to use for text generation.
        tokenizer (GPT2Tokenizer): The tokenizer to use for encoding and decoding text.
        prompt (str): The initial text to start generation from.
        max_length (int): The maximum length of the generated text. Defaults to 1024.
        temperature (float): The temperature for sampling. Defaults to 0.7.
        top_k (int): The number of top-k tokens to consider during sampling. Defaults to 50.
        top_p (float): The cumulative probability threshold for nucleus sampling. Defaults to 0.95.
        repetition_penalty (float): The penalty for repeating tokens. Defaults to 1.2.
        num_return_sequences (int): The number of sequences to generate. Defaults to 1.
        device (torch.device): The device to run the model on. Defaults to CUDA if available, else CPU.
        pad_token_id (Optional[int]): The ID of the padding token. If None, uses the model's pad_token_id.

    Returns:
        List[str]: The list of generated text sequences.
    """
    model.to(device)
    model.eval()

    # Encode the prompt
    try:
        encoded_prompt = tokenizer.encode(
            prompt, add_special_tokens=True, return_tensors="pt"
        )
        encoded_prompt = encoded_prompt.to(device)
    except Exception as e:
        logger.error(f"Error encoding prompt: {str(e)}")
        return []

    # Set pad_token_id if not provided
    if pad_token_id is None:
        pad_token_id = (
                getattr(model.config, "pad_token_id", None) or tokenizer.eos_token_id
        )

    # Log input shapes and devices for debugging
    logger.info(f"Encoded prompt shape: {encoded_prompt.shape}")
    logger.info(f"Encoded prompt device: {encoded_prompt.device}")
    logger.info(f"Model device: {next(model.parameters()).device}")

    try:
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=encoded_prompt.repeat(num_return_sequences, 1),
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=pad_token_id,
                num_return_sequences=num_return_sequences,
                do_sample=True,
            )

        # Decode the generated sequences
        generated_texts = []
        for generated_sequence in output_sequences:
            generated_sequence = generated_sequence.tolist()
            text = tokenizer.decode(
                generated_sequence, clean_up_tokenization_spaces=True
            )
            text = text[
                   len(
                       tokenizer.decode(
                           encoded_prompt[0], clean_up_tokenization_spaces=True
                       )
                   ):
                   ]
            generated_texts.append(text.strip())

        return generated_texts

    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error(f"CUDA out of memory error: {str(e)}")
            torch.cuda.empty_cache()
        else:
            logger.error(f"Runtime error during text generation: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        logger.exception("Full traceback:")
        return []


def calculate_perplexity(
        model: PreTrainedModel,
        tokenizer: GPT2Tokenizer,
        text: Union[str, List[str]],
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Union[float, List[float]]:
    """
    Calculates the perplexity of the given text using the provided model and tokenizer,
    specifically handling text generated using sampling.

    Args:
        model (PreTrainedModel): The model to use for perplexity calculation.
        tokenizer (GPT2Tokenizer): The tokenizer to use for encoding and decoding text.
        text (Union[str, List[str]]): The text to calculate the perplexity for.
            Can be a single string or a list of strings.
        device (torch.device, optional): The device to run the model on.
            Defaults to CUDA if available, else CPU.

    Returns:
        Union[float, List[float]]: The perplexity of the text. If the input `text` is a string,
            returns a float. If `text` is a list of strings, returns a list of floats representing the
            perplexity of each string in the list.
    """
    model.to(device)
    model.eval()

    if isinstance(text, list):
        return [
            calculate_perplexity(model, tokenizer, t, device) for t in text if t.strip()
        ]

    if not text.strip():
        raise ValueError("Input text is empty.")

    tokens = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)

    total_loss = 0.0
    num_tokens = len(tokens) - 1  # Exclude the first token (start token)
    loss_fct = nn.CrossEntropyLoss()

    for i in range(1, len(tokens)):
        # Take the sequence up to the current token
        current_input_ids = input_ids[:, :i]
        with torch.no_grad():
            outputs = model(current_input_ids)
            logits = outputs.logits

        # Get the logits for the next token prediction
        next_token_logits = logits[:, -1, :]
        target = input_ids[:, i]

        # Calculate the loss for this prediction
        loss = loss_fct(next_token_logits, target)
        total_loss += loss.item()

    # Calculate average loss and perplexity
    avg_loss = total_loss / num_tokens
    return math.exp(min(avg_loss, 100))
