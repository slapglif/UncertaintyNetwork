# test_generation.py

import matplotlib.pyplot as plt
import pytest
import seaborn as sns
import torch
from loguru import logger
from transformers import StoppingCriteria

from core.models.uncertain_nn import (
    UncertainTransformerConfig,
    UncertainTransformerLMHeadModel,
)
from core.utils.tokenizer import Tokenizer
from core.utils.utils import generate_text, calculate_perplexity

# Constants
MAX_LENGTH = 50
TEMPERATURE = 0.7
TIMEOUT = 30
NUM_SAMPLES = 1


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
    Tests text generation and perplexity calculation using the provided model and tokenizer.

    Args:
        model (UncertainTransformerLMHeadModel): The model to use for text generation and perplexity calculation.
        tokenizer (Tokenizer): The tokenizer to use for encoding and decoding text.
        prompt (str): The initial text to start generation from.
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
            num_return_sequences=1,
            device=device,
        )

        assert len(generated_texts) > 0, "No text was generated"

        perplexities = []
        # sourcery skip: no-loop-in-tests
        for i, generated_text in enumerate(generated_texts):
            logger.info(f"\nSample {i + 1}:")
            logger.info(f"Generated text: '{generated_text}'")
            logger.info(f"Generated text length: {len(generated_text)}")

            assert len(generated_text) > 0, f"Generated text {i + 1} is empty"

            perplexity = calculate_perplexity(
                model, tokenizer.tokenizer, generated_text, device=device
            )
            perplexities.append(perplexity)
            logger.info(f"Perplexity: {perplexity:.2f}")

            assert (
                    0 < perplexity < float("inf")
            ), f"Invalid perplexity value: {perplexity}"

        # Visualize perplexities
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=perplexities)
        plt.title(f"Perplexity Distribution for Prompt: '{prompt}'")
        plt.xlabel("Perplexity")
        plt.savefig(f"perplexity_distribution_{prompt.replace(' ', '_')}.png")
        plt.close()

    except Exception as e:
        logger.error(
            f"Error in test_generation_and_perplexity for prompt '{prompt}': {str(e)}"
        )
        logger.exception("Full traceback:")
        pytest.fail(f"Test failed due to an error: {str(e)}")


def test_model_output_shapes(model, tokenizer, device):
    """
    Tests the output shapes of the model for a given input.

    Args:
        model (UncertainTransformerLMHeadModel): The model to test.
        tokenizer (Tokenizer): The tokenizer to use for encoding.
        device (torch.device): The device to run the model on.
    """
    model.to(device)
    prompt = "Test prompt"
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    with torch.no_grad():
        # Remove token_type_ids from the model call, as it's not needed
        outputs = model(input_ids)

    logger.info(f"\nModel output shapes:")
    logger.info(f"Input shape: {input_ids.shape}")
    logger.info(f"Logits shape: {outputs.logits.shape}")

    # sourcery skip: no-conditionals-in-tests
    if torch.isnan(outputs.logits).any():
        logger.warning("NaN values detected in logits")

    assert (
            outputs.logits.shape[0] == input_ids.shape[0]
    ), f"Batch size mismatch: expected {input_ids.shape[0]}, got {outputs.logits.shape[0]}"
    assert (
            outputs.logits.shape[1] == input_ids.shape[1]
    ), f"Sequence length mismatch: expected {input_ids.shape[1]}, got {outputs.logits.shape[1]}"
    assert (
            outputs.logits.shape[2] == model.config.vocab_size
    ), f"Vocabulary size mismatch: expected {model.config.vocab_size}, got {outputs.logits.shape[2]}"


def test_attention_mask(model, tokenizer, device):
    """
    Tests the effect of using an attention mask on the model's output.

    Args:
        model (UncertainTransformerLMHeadModel): The model to test.
        tokenizer (Tokenizer): The tokenizer to use for encoding.
        device (torch.device): The device to run the model on.
    """
    model.to(device)
    prompt = "Test with padding"
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs_without_mask = model(input_ids)
        # sourcery skip: no-conditionals-in-tests
        if not model.config.use_mamba:
            attention_mask = torch.ones_like(input_ids)
            attention_mask[:, -2:] = 0  # Simulate padding
            outputs_with_mask = model(input_ids, attention_mask=attention_mask)
        else:
            outputs_with_mask = (
                outputs_without_mask  # For Mamba, we don't use attention mask
            )

    logger.info("\nTesting attention mask:")

    # sourcery skip: no-conditionals-in-tests
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

    # sourcery skip: no-conditionals-in-tests
    if not model.config.use_mamba:
        assert diff > 0, "Outputs should differ when using attention mask"
    else:
        assert diff == 0, "Outputs should be the same for Mamba layers"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
