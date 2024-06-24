# core/tests/test_generation.py

import pytest
import torch
from loguru import logger
from transformers import StoppingCriteria

from core.models.uncertainty.uncertain_nn import UncertainTransformerConfig, UncertainTransformerLMHeadModel
from core.utils.tokenizer import Tokenizer
from core.models.uncertainty.uncertainty import uncertainty_guided_sampling
from core.utils.utils import calculate_perplexity

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
        sliding_window_size=512,
    )
    model = UncertainTransformerLMHeadModel(config)
    model.to(device)
    return model


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer.from_pretrained("gpt2")


class MaxLengthCriteria(StoppingCriteria):
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
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
def test_generation_and_perplexity_with_uncertainty(
        model: UncertainTransformerLMHeadModel,
        tokenizer: Tokenizer,
        prompt: str,
        device: torch.device,
):
    model.to(device)
    logger.info(f"\nTesting prompt: {prompt}")

    try:
        torch.cuda.empty_cache()
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        generated_outputs = model.generate(
            input_ids,
            max_length=MAX_LENGTH,
            num_return_sequences=NUM_SAMPLES,
            do_sample=True,
            temperature=TEMPERATURE,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_sequences = generated_outputs.sequences
        generated_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)

        assert len(generated_texts) > 0, "No text was generated"

        for i, generated_text in enumerate(generated_texts):
            logger.info(f"\nSample {i + 1}:")
            logger.info(f"Generated text: '{generated_text}'")
            logger.info(f"Generated text length: {len(generated_text)}")

            assert len(generated_text) > 0, f"Generated text {i + 1} is empty"

            # Calculate perplexity
            perplexity = calculate_perplexity(model, tokenizer, generated_text, device=device)
            logger.info(f"Perplexity: {perplexity:.2f}")

            assert 0 < perplexity < float("inf"), f"Invalid perplexity value: {perplexity}"

        # Test uncertainty-guided sampling
        logits = generated_outputs.scores[-1]
        uncertainties = \
            model.transformer.uncertainty_module(
                model.transformer.final_layer_norm(generated_outputs.hidden_states[-1]))[
                -1]
        sampled_tokens = uncertainty_guided_sampling(logits, uncertainties)
        assert sampled_tokens.shape == (NUM_SAMPLES,)

    except Exception as e:
        logger.error(f"Error in test_generation_and_perplexity_with_uncertainty for prompt '{prompt}': {str(e)}")
        logger.exception("Full traceback:")
        raise


def test_model_output_shapes_with_uncertainty(model, tokenizer, device):
    model.to(device)
    prompt = "Test prompt"
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        logger.debug(f"Input IDs shape: {input_ids.shape}, device: {input_ids.device}")

        with torch.no_grad():
            outputs, uncertainty = model(input_ids)

        logger.debug(f"\nModel output shapes:")
        logger.debug(f"Input shape: {input_ids.shape}")
        logger.debug(f"Logits shape: {outputs.logits.shape}")
        logger.debug(f"Uncertainty shape: {uncertainty.shape}")

        assert outputs.logits.shape[0] == input_ids.shape[
            0], f"Batch size mismatch: expected {input_ids.shape[0]}, got {outputs.logits.shape[0]}"
        assert outputs.logits.shape[1] == input_ids.shape[
            1], f"Sequence length mismatch: expected {input_ids.shape[1]}, got {outputs.logits.shape[1]}"
        assert outputs.logits.shape[
                   2] == model.config.vocab_size, f"Vocabulary size mismatch: expected {model.config.vocab_size}, got {outputs.logits.shape[2]}"
        assert uncertainty.shape == outputs.logits.shape, f"Uncertainty shape mismatch: expected {outputs.logits.shape}, got {uncertainty.shape}"

    except Exception as e:
        logger.error(f"Error in test_model_output_shapes_with_uncertainty: {str(e)}")
        logger.exception("Full traceback:")
        raise

# Add more tests as needed
