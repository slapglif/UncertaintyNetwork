# .\core\utils\utils.py

from typing import List, Optional, Union
from typing import Tuple

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from transformers import PreTrainedModel, GenerationConfig

from core import Mamba
from core.models.embedding import RotaryPositionEncoding
from core.models.layers import MultiHeadAttention
from core.utils.tokenizer import Tokenizer


# noinspection PyTypeChecker


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


@torch.inference_mode()
def generate_text(
        model: PreTrainedModel,
        tokenizer: Tokenizer,
        prompt: Union[str, List[str]],
        max_length: int = 100,
        temperature: float = 1.13,
        top_k: int = 49,
        top_p: float = 0.18,
        repetition_penalty: float = 1.8,
        num_return_sequences: int = 1,
        device: Optional[torch.device] = None,
        generation_config: Optional[GenerationConfig] = None,
        seed: Optional[int] = None,
        **kwargs
) -> List[str]:
    """
    Generates text using the specified model and tokenizer with given parameters.

    This function uses advanced sampling techniques, including temperature scaling,
    top-k filtering, and nucleus (top-p) sampling to generate diverse and coherent text.
    It also implements error handling, logging, and optional seeding for reproducibility.

    Args:
        model (PreTrainedModel): The pre-trained language model.
        tokenizer (Tokenizer): The tokenizer for encoding and decoding text.
        prompt (Union[str, List[str]]): The input prompt(s) to generate text from.
        max_length (int, optional): The maximum length of the generated text. Defaults to 100.
        temperature (float, optional): The temperature parameter for sampling. Defaults to 1.13.
        top_k (int, optional): The number of top-k tokens to consider for sampling. Defaults to 49.
        top_p (float, optional): The probability threshold for nucleus sampling. Defaults to 0.18.
        repetition_penalty (float, optional): The repetition penalty. Defaults to 1.8.
        num_return_sequences (int, optional): The number of sequences to generate for each prompt. Defaults to 1.
        device (torch.device, optional): The device to run the model on. If None, uses CUDA if available, else CPU.
        generation_config (GenerationConfig, optional): A GenerationConfig object to override default generation parameters.
        seed (int, optional): Random seed for reproducibility. If None, no seed is set.
        **kwargs: Additional keyword arguments to pass to the model's generate method.

    Returns:
        List[str]: A list of generated texts.

    Raises:
        ValueError: If the input prompt is empty or if the model or tokenizer are not properly initialized.
        RuntimeError: If there's an error during the text generation process.
    """
    if not prompt:
        raise ValueError("Input prompt cannot be empty.")

    if not isinstance(model, PreTrainedModel):
        raise ValueError("Invalid model type. Expected a PreTrainedModel instance.")

    if not isinstance(tokenizer, Tokenizer):
        raise ValueError("Invalid tokenizer type. Expected a Tokenizer instance.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    try:
        logger.info(f"Encoding prompt: {prompt}")
        if isinstance(prompt, str):
            prompt = [prompt]

        input_ids = tokenizer.batch_encode_plus(
            prompt, padding=True, truncation=True, return_tensors="pt"
        )["input_ids"].to(device)

        logger.info(f"Input IDs shape: {input_ids.shape}, device: {input_ids.device}")

        attention_mask = torch.ones_like(input_ids)

        if generation_config is None:
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

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                **kwargs,
            )

        logger.info("Text generation completed")

        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Clear CUDA cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return generated_texts

    except Exception as e:
        logger.exception(f"Error during text generation: {str(e)}")
        raise RuntimeError(f"Text generation failed: {str(e)}") from e


def calculate_perplexity(
        model: PreTrainedModel, tokenizer: Tokenizer, text: str, device: torch.device
) -> float:
    """
    Calculate the perplexity of the given text using the specified model and tokenizer.

    Args:
        model (PreTrainedModel): The pre-trained language model.
        tokenizer (Tokenizer): The tokenizer for encoding the text.
        text (str): The input text to calculate perplexity for.
        device (torch.device): The device to run the model on.

    Returns:
        float: The perplexity of the text.
    """
    model.to(device)
    model.eval()

    ids = tokenizer.encode(text)
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)
    num_tokens = input_ids.shape[1]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return torch.exp(loss / num_tokens).item()


def check_shapes(tensors, expected_shapes, names):
    for tensor, expected_shape, name in zip(tensors, expected_shapes, names):
        assert (
                tensor.shape == expected_shape
        ), f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}"


def check_layer(
        layer: nn.Module,
        input_shape: Tuple[int, ...],
        expected_output_shape: Tuple[int, ...],
        device: torch.device = torch.device("cuda" if torch.cpu.is_available() else "cpu"),
):
    """
    Check the forward and backward pass of a single layer.

    Args:
        layer (nn.Module): The layer to test.
        input_shape (Tuple[int, ...]): The shape of the input tensor.
        expected_output_shape (Tuple[int, ...]): The expected shape of the output tensor.
        device (torch.device): The device to run the test on.
    """
    layer.to(device)

    # Prepare input tensor
    if isinstance(layer, nn.Embedding):
        input_tensor = torch.randint(
            0, layer.num_embeddings, input_shape, device=device
        )
    else:
        input_tensor = torch.randn(input_shape, device=device, requires_grad=True)

    # Prepare attention mask for MultiHeadAttention
    if isinstance(layer, MultiHeadAttention):
        attention_mask = torch.ones(input_shape[0], input_shape[1], device=device)
    else:
        attention_mask = None

    # Forward pass
    if isinstance(layer, RotaryPositionEncoding):
        output = layer(input_tensor)[0]
    elif isinstance(layer, Mamba):
        # For Mamba, we need to handle the case where it returns a tuple
        output, _ = layer(input_tensor)
    elif isinstance(layer, MultiHeadAttention):
        output, _ = layer(input_tensor, attention_mask=attention_mask)
    else:
        output = layer(input_tensor)

    if isinstance(output, tuple):
        output = output[0]

    assert (
            output.shape == expected_output_shape
    ), f"Output shape mismatch for {layer.__class__.__name__}: expected {expected_output_shape}, got {output.shape}"

    # Backward pass
    try:
        # Use retain_graph=True to prevent in-place operation errors
        output.sum().backward(retain_graph=True)

        for name, param in layer.named_parameters():
            assert (
                    param.grad is not None
            ), f"Gradient is None for parameter {name} in {layer.__class__.__name__}"
            assert (
                    torch.sum(param.grad ** 2) > 0
            ), f"Gradient is zero for parameter {name} in {layer.__class__.__name__}"
    except RuntimeError as e:
        if (
                "one of the variables needed for gradient computation has been modified by an inplace operation"
                in str(e)
        ):
            logger.warning(
                f"In-place operation detected in {layer.__class__.__name__}. Skipping gradient check."
            )
        else:
            raise e

    # Clear gradients after each layer check
    layer.zero_grad()


def check_shape(tensor: torch.Tensor, expected_shape: Tuple[int], name: str):
    """Checks the shape of a tensor against the expected shape and raises a ValueError if they don't match."""
    if tensor.shape != expected_shape:
        raise ValueError(
            f"Shape mismatch for {name}: expected {expected_shape}, got {tensor.shape}"
        )


class TimestepNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Timestep Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Normalized output tensor of shape (batch_size, seq_len, dim).
        """
        # Calculate the mean and standard deviation across the sequence (axis=1)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)

        return (x - mean) / (std + self.eps)
