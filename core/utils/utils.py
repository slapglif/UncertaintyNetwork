# core/utils/utils.py

from typing import Tuple, List, Union

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


def check_shapes(tensors, expected_shapes, names):
    for tensor, expected_shape, name in zip(tensors, expected_shapes, names):
        assert tensor.shape == expected_shape, f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}"

import torch
from loguru import logger
from typing import Tuple, List, Union, Any
import inspect

def einsum_safe(equation: str, *operands: torch.Tensor,
                 check_shapes: bool = True,
                 reshape_operands: bool = True,
                 verbose: bool = False) -> torch.Tensor:
    """
    A safe wrapper for torch.einsum that automatically checks for shape compatibility and attempts to
    reshape operands to resolve mismatches, even those that occur before the einsum operation.

    Args:
        equation (str): The einsum equation string.
        *operands (torch.Tensor): The input tensors.
        check_shapes (bool): Whether to perform shape compatibility checks. Default: True.
        reshape_operands (bool): Whether to attempt to reshape operands to resolve shape mismatches.
            Default: True.
        verbose (bool): Whether to print debug information. Default: False.

    Returns:
        torch.Tensor: The result of the einsum operation.

    Raises:
        ValueError: If shape compatibility cannot be resolved after reshaping.

    Example:
        >>> x = torch.randn(2, 3, 4)
        >>> y = torch.randn(4, 5)
        >>> # This would normally raise an error due to shape mismatch
        >>> # result = torch.einsum('ijk,kl->ijl', x, y)
        >>> # Using the safe wrapper:
        >>> result = einsum_safe('ijk,kl->ijl', x, y)  # Shape mismatch is resolved automatically
        >>> result.shape
        torch.Size([2, 3, 5])
    """

    if check_shapes:
        # Check if shapes are compatible for the given einsum equation
        try:
            torch.einsum(equation, *operands)
        except RuntimeError as e:
            if verbose:
                logger.warning(f"Shape mismatch detected in einsum: {e}")
            if not reshape_operands:
                raise

            # Attempt to reshape operands to resolve the mismatch
            operands = _reshape_operands_for_einsum(equation, *operands, verbose=verbose)
            if operands is None:
                raise ValueError("Shape mismatch cannot be resolved after reshaping.") from e
            if verbose:
                logger.info("Reshaped operands successfully.")
            return torch.einsum(equation, *operands)
    # Perform einsum if shapes are compatible or reshaping was successful
    return torch.einsum(equation, *operands)


def _reshape_operands_for_einsum(equation: str, *operands: torch.Tensor, verbose: bool = False) -> Union[Tuple[torch.Tensor], None]:
    """
    Attempts to reshape operands to resolve shape mismatches in einsum.

    Args:
        equation (str): The einsum equation string.
        *operands (torch.Tensor): The input tensors.
        verbose (bool): Whether to print debug information. Default: False.

    Returns:
        Tuple[torch.Tensor]: The reshaped operands if successful, otherwise None.
    """

    # Extract operand dimensions from the einsum equation
    operand_dims = [list(dim) for dim in equation.split(',')]
    if verbose:
        logger.debug(f"Operand dimensions: {operand_dims}")

    # Iterate through each operand and attempt to reshape it
    for i, operand in enumerate(operands):
        operand_shape = list(operand.shape)
        if verbose:
            logger.debug(f"Operand {i} shape: {operand_shape}")

        # Find the dimension in the operand that needs to be adjusted
        target_dim = operand_dims[i]
        target_dim_size = len(target_dim)
        if verbose:
            logger.debug(f"Target dimension: {target_dim}, size: {target_dim_size}")

        # Reshape the operand if necessary
        if len(operand_shape) != target_dim_size:
            if verbose:
                logger.debug(f"Reshaping operand {i} to match target dimension.")
            try:
                operand = operand.reshape(
                    *[operand_shape[j] for j in range(target_dim_size)]
                )
                operands = list(operands)  # Convert tuple to list for modification
                operands[i] = operand
                operands = tuple(operands)  # Convert back to tuple
                if verbose:
                    logger.debug(f"Reshaped operand {i} shape: {operand.shape}")
            except RuntimeError as e:
                if verbose:
                    logger.warning(f"Reshaping failed for operand {i}: {e}")
                return None

    # Check if any operand is a function call
    for i, operand in enumerate(operands):
        if callable(operand):
            if verbose:
                logger.debug(f"Operand {i} is a function call. Attempting to resolve shape mismatch.")
            try:
                # Get the function's signature
                signature = inspect.signature(operand)
                # Get the arguments to the function call
                args = inspect.getcallargs(operand, *operands[i + 1:])
                # Call the function with the resolved arguments
                result = operand(**args)
                # Replace the function call with the result
                operands = list(operands)
                operands[i] = result
                operands = tuple(operands)
                if verbose:
                    logger.debug(f"Function call resolved. Operand {i} shape: {result.shape}")
            except Exception as e:
                if verbose:
                    logger.warning(f"Function call resolution failed for operand {i}: {e}")
                return None

    return operands
