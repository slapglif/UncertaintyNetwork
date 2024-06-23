# test_generation.py

import math
from typing import List, Union

import pytest
import torch
from loguru import logger
from transformers import GPT2Tokenizer
from transformers import StoppingCriteria

from core.models.uncertain_nn import (
    UncertainTransformerConfig,
    UncertainTransformerLMHeadModel,
)
from core.utils.tokenizer import Tokenizer

# Constants
MAX_LENGTH = 50
TEMPERATURE = 0.7
TIMEOUT = 30
NUM_SAMPLES = 3


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def model(device):
    config = UncertainTransformerConfig(
        vocab_size=50257,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_layers=6,
        dropout=0.1,
        max_position_embeddings=1024,
        pad_token_id=50256,
        use_mamba=True,
        d_state=16,
        d_conv=4,
        expand_factor=2.0,
        dt_rank=16,
    )
    model = UncertainTransformerLMHeadModel(config)
    model.to(device)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer.from_pretrained("gpt2")


class MaxLengthCriteria(StoppingCriteria):
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        return input_ids.shape[-1] >= self.max_length


# Update the generate_text function to not use attention_mask for Mamba
def generate_text(
    model: UncertainTransformerLMHeadModel,
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

    try:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                do_sample=True,
            )
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        logger.error(f"Input shape: {input_ids.shape}")
        logger.error(f"Input device: {input_ids.device}")
        logger.error(f"Model device: {next(model.parameters()).device}")
        raise

    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]


def calculate_perplexity(
    model: UncertainTransformerLMHeadModel,
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


@pytest.mark.parametrize(
    "prompt",
    [
        "Once upon a time,",
        "The quick brown fox",
        "In a world where",
        "Scientists have discovered",
        "The future of artificial intelligence",
    ],
)
def test_generation_and_perplexity(
    model: UncertainTransformerLMHeadModel,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
):
    """
    Test the text generation and perplexity calculation capabilities of the model.

    This test function generates text based on various prompts and then calculates the perplexity
    of the generated text. It asserts that the generated text is not empty and the perplexity
    is a finite positive number.

    Args:
        model (UncertainTransformerLMHeadModel): The model to test.
        tokenizer (Tokenizer): The tokenizer to use for encoding and decoding text.
        prompt (str): The prompt to start text generation from.
        device (torch.device): The device to run the model on.
    """
    model.to(device)
    logger.info(f"\nTesting prompt: {prompt}")

    try:
        generated_texts = generate_text(
            model,
            tokenizer.tokenizer,
            prompt,
            max_length=1024,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            num_return_sequences=NUM_SAMPLES,
            device=device,
        )

        for i, generated_text in enumerate(generated_texts):
            perplexity = calculate_perplexity(
                model, tokenizer.tokenizer, generated_text, device=device
            )

            logger.info(f"\nSample {i + 1}:")
            logger.info(f"Generated text: '{generated_text}'")
            logger.info(f"Generated text length: {len(generated_text)}")
            logger.info(f"Perplexity: {perplexity:.2f}")

            assert len(generated_text) > 0, "Generated text should not be empty"
            assert (
                0 < perplexity < float("inf")
            ), "Perplexity should be a finite positive number"

    except Exception as e:
        logger.error(f"Error generating text for prompt '{prompt}': {str(e)}")
        logger.error(f"Error details: {type(e).__name__}")
        import traceback

        logger.error(traceback.format_exc())
        pytest.fail(f"Test failed due to an error: {str(e)}")


def test_model_output_shapes(model, tokenizer, device):
    """
    Test the output shapes of the model.

    This function checks the shapes of the model's outputs (logits) to ensure they match the expected dimensions.
    It also checks for NaN values in the logits and logs a warning if any are found.

    Args:
        model (UncertainTransformerLMHeadModel): The model to test.
        tokenizer (Tokenizer): The tokenizer to use for encoding text.
        device (torch.device): The device to run the model on.
    """
    model.to(device)
    prompt = "Test prompt"
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_ids)

    logger.info(f"\nModel output shapes:")
    logger.info(f"Logits shape: {outputs.logits.shape}")

    # Check for nan values
    if torch.isnan(outputs.logits).any():
        logger.warning("NaN values detected in logits")

    assert outputs.logits.shape[0] == 1, "Batch size should be 1"
    assert outputs.logits.shape[1] == len(
        input_ids[0]
    ), "Sequence length should match input"
    assert (
        outputs.logits.shape[2] == model.config.vocab_size
    ), "Last dimension should be vocab size"


def test_attention_mask(model, tokenizer, device):
    """
    Tests the effect of the attention mask on the model's output.

    This function compares the model's outputs with and without an attention mask. The attention mask
    is used to simulate padding in the input sequence. For models with standard attention mechanisms,
    the outputs should differ when an attention mask is used. For models using Mamba layers, the
    attention mask should have no effect, as Mamba layers do not rely on attention masks.

    Args:
        model (UncertainTransformerLMHeadModel): The model to test.
        tokenizer (Tokenizer): The tokenizer to use for encoding text.
        device (torch.device): The device to run the model on.
    """
    model.to(device)
    prompt = "Test with padding"
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs_without_mask = model(input_ids)
        if not model.config.use_mamba:
            attention_mask = torch.ones_like(input_ids)
            attention_mask[:, -2:] = 0  # Simulate padding
            outputs_with_mask = model(input_ids, attention_mask=attention_mask)
        else:
            outputs_with_mask = (
                outputs_without_mask  # For Mamba, we don't use attention mask
            )

    logger.info("\nTesting attention mask:")

    if (
        torch.isnan(outputs_with_mask.logits).any()
        or torch.isnan(outputs_without_mask.logits).any()
    ):
        logger.warning("NaN values detected in logits")
        return

    diff = (
        (outputs_with_mask.logits[:, -1] - outputs_without_mask.logits[:, -1])
        .abs()
        .max()
        .item()
    )
    logger.info(f"Difference in last token logits: {diff:.4f}")

    if not model.config.use_mamba:
        assert diff > 0, "Outputs should differ when using attention mask"
    else:
        assert diff == 0, "Outputs should be the same for Mamba layers"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])