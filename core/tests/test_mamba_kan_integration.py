import os
from typing import Tuple, Dict, Any

import pytest
import torch
from loguru import logger
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from core.models.uncertainty.uncertain_nn import UncertainTransformerConfig, UncertainTransformerLMHeadModel
from core.models.uncertainty.uncertainty_utils import (
    epistemic_uncertainty,
    aleatoric_uncertainty,
    total_uncertainty,
    uncertainty_guided_sampling
)
from core.utils.tokenizer import Tokenizer

# Constants
BATCH_SIZE = 2
SEQ_LEN = 10
VOCAB_SIZE = 50257  # GPT-2's vocabulary size
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 6
D_FF = 2048
D_STATE = 16
D_CONV = 4
EXPAND_FACTOR = 2.0
DT_RANK = 16
MAX_LENGTH = 20
NUM_RETURN_SEQUENCES = 1
NUM_MC_SAMPLES = 5

# Set CUDA_LAUNCH_BLOCKING for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Device to use for testing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def config() -> UncertainTransformerConfig:
    return UncertainTransformerConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        dropout=0.1,
        max_position_embeddings=1024,
        pad_token_id=50256,
        use_mamba=True,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand_factor=EXPAND_FACTOR,
        dt_rank=DT_RANK,
        sliding_window_size=512,
    )


@pytest.fixture(scope="module")
def model(config: UncertainTransformerConfig) -> UncertainTransformerLMHeadModel:
    model = UncertainTransformerLMHeadModel(config)
    model.to(DEVICE)
    return model


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained("gpt2")


def log_model_info(model: UncertainTransformerLMHeadModel) -> None:
    """Log information about the model's parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")


def log_tensor_info(name: str, tensor: torch.Tensor) -> None:
    """Log information about a tensor."""
    logger.info(f"{name} shape: {tensor.shape}, device: {tensor.device}")
    logger.info(f"{name} dtype: {tensor.dtype}")
    logger.info(f"{name} min: {tensor.min().item()}, max: {tensor.max().item()}")
    if torch.is_floating_point(tensor):
        logger.info(f"{name} mean: {tensor.mean().item()}")
    else:
        logger.info(f"{name} mean: {tensor.float().mean().item()}")


def check_output_validity(outputs: Dict[str, Any], batch_size: int, seq_len: int, vocab_size: int) -> None:
    """Check the validity of the model's output."""
    assert hasattr(outputs, 'logits'), "Outputs should have 'logits' attribute"
    assert isinstance(outputs.logits, Tensor), f"Expected logits to be a Tensor, got {type(outputs.logits)}"
    assert outputs.logits.shape == (batch_size, seq_len, vocab_size), \
        f"Expected logits shape {(batch_size, seq_len, vocab_size)}, got {outputs.logits.shape}"
    log_tensor_info("Logits", outputs.logits)

    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        assert isinstance(outputs.hidden_states, Tensor), "hidden_states should be a Tensor"
        assert outputs.hidden_states.shape == (batch_size, seq_len, outputs.logits.shape[-1]), \
            f"Expected hidden states shape {(batch_size, seq_len, outputs.logits.shape[-1])}, got {outputs.hidden_states.shape}"
        log_tensor_info("Hidden states", outputs.hidden_states)
    else:
        logger.warning("Hidden states not available in the output.")


def extract_uncertainty(model: UncertainTransformerLMHeadModel, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
    """Extract mean and uncertainty from the model's hidden states."""
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(1)
    mean, uncertainty = model.uncertainty_module(hidden_states)
    return mean.squeeze(1), uncertainty.squeeze(1)


@pytest.mark.parametrize("batch_size, seq_len", [(2, 10), (4, 20)])
def test_model_forward_with_uncertainty(
        model: UncertainTransformerLMHeadModel,
        config: UncertainTransformerConfig,
        batch_size: int,
        seq_len: int
) -> None:
    """
    Test the forward pass of the UncertainTransformerLMHeadModel.

    Args:
        model (UncertainTransformerLMHeadModel): The model to test.
        config (UncertainTransformerConfig): The model configuration.
        batch_size (int): The batch size to use for testing.
        seq_len (int): The sequence length to use for testing.
    """
    try:
        log_model_info(model)

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=DEVICE)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=DEVICE)

        log_tensor_info("Input IDs", input_ids)
        log_tensor_info("Attention mask", attention_mask)

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        check_output_validity(outputs, batch_size, seq_len, config.vocab_size)

        # Additional checks
        assert outputs.loss is None, "Loss should be None when no labels are provided"
        assert outputs.past_key_values is not None, "past_key_values should not be None"
        assert outputs.attentions is not None, "attentions should not be None"

        logger.info(f"Model forward pass completed successfully with batch_size={batch_size}, seq_len={seq_len}")

    except Exception as e:
        logger.error(f"Error in test_model_forward_with_uncertainty: {str(e)}")
        logger.exception("Full traceback:")
        raise


def test_model_generation_with_uncertainty(
        model: UncertainTransformerLMHeadModel,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = MAX_LENGTH,
        num_return_sequences: int = NUM_RETURN_SEQUENCES
) -> None:
    try:
        model.eval()

        input_text = "Hello, world!"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)
        input_ids = torch.clamp(input_ids, 0, model.config.vocab_size - 1)

        log_tensor_info("Input IDs", input_ids)

        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                output_scores=True,
                return_dict_in_generate=True,
            )

        assert isinstance(output_sequences.sequences, torch.Tensor), "Expected sequences to be a torch.Tensor"
        assert output_sequences.sequences.shape[0] == num_return_sequences, \
            f"Expected {num_return_sequences} sequences, got {output_sequences.sequences.shape[0]}"
        assert all(seq_len <= max_length for seq_len in output_sequences.sequences.shape[1:]), \
            f"Some sequences exceed max_length {max_length}"

        log_tensor_info("Generated sequences", output_sequences.sequences)

        if output_sequences.scores:
            all_scores = torch.stack(output_sequences.scores, dim=1)
            log_tensor_info("All scores", all_scores)

        sample_text = tokenizer.decode(output_sequences.sequences[0], skip_special_tokens=True)
        logger.info(f"Sample generated text: {sample_text}")

        logger.info(f"Text generation successful with max_length={max_length}, "
                    f"num_return_sequences={num_return_sequences}")

    except Exception as e:
        logger.error(f"Error in test_model_generation_with_uncertainty: {str(e)}")
        logger.exception("Full traceback:")
        raise


@pytest.mark.parametrize("batch_size, seq_len, temperature, alpha", [
    (2, 10, 0.7, 1.0),
    (4, 20, 0.5, 2.0)
])
def test_uncertainty_guided_sampling(
        model: UncertainTransformerLMHeadModel,
        config: UncertainTransformerConfig,
        batch_size: int,
        seq_len: int,
        temperature: float,
        alpha: float
) -> None:
    try:
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=DEVICE)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=DEVICE)

        log_tensor_info("Input IDs", input_ids)
        log_tensor_info("Attention mask", attention_mask)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits[:, -1, :]  # Take the last token's logits
        log_tensor_info("Logits", logits)

        if outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[:, -1, :]  # Take the last hidden state
            _, uncertainties = extract_uncertainty(model, hidden_states)
            log_tensor_info("Uncertainties", uncertainties)

            sampled_tokens = uncertainty_guided_sampling(logits, uncertainties, temperature=temperature, alpha=alpha)
            log_tensor_info("Sampled tokens", sampled_tokens)

            assert isinstance(sampled_tokens, torch.Tensor), "Expected sampled_tokens to be a torch.Tensor"
            assert sampled_tokens.shape == (batch_size,), f"Expected shape ({batch_size},), got {sampled_tokens.shape}"
            assert torch.all(sampled_tokens >= 0), "Sampled tokens contain negative values"
            assert torch.all(sampled_tokens < config.vocab_size), "Sampled tokens exceed vocabulary size"

        logger.info(f"Uncertainty-guided sampling successful with batch_size={batch_size}, "
                    f"seq_len={seq_len}, temperature={temperature}, alpha={alpha}")

    except Exception as e:
        logger.error(f"Error in test_uncertainty_guided_sampling: {str(e)}")
        logger.exception("Full traceback:")
        raise


def test_model_uncertainty_decomposition(
        model: UncertainTransformerLMHeadModel,
        config: UncertainTransformerConfig
) -> None:
    try:
        model.eval()

        batch_size = BATCH_SIZE
        seq_len = SEQ_LEN
        num_mc_samples = NUM_MC_SAMPLES

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=DEVICE)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=DEVICE)

        log_tensor_info("Input IDs", input_ids)
        log_tensor_info("Attention mask", attention_mask)

        mc_logits = []
        mc_uncertainties = []
        with torch.no_grad():
            for _ in range(num_mc_samples):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                mc_logits.append(outputs.logits)
                if outputs.hidden_states is not None:
                    _, uncertainties = extract_uncertainty(model, outputs.hidden_states[-1])
                    mc_uncertainties.append(uncertainties)

        mc_logits = torch.stack(mc_logits)
        log_tensor_info("MC logits", mc_logits)

        if mc_uncertainties:
            mc_uncertainties = torch.stack(mc_uncertainties)
            log_tensor_info("MC uncertainties", mc_uncertainties)

            epistemic_unc = epistemic_uncertainty(mc_logits)
            aleatoric_unc = aleatoric_uncertainty(mc_uncertainties)
            total_unc = total_uncertainty(epistemic_unc, aleatoric_unc)

            log_tensor_info("Epistemic uncertainty", epistemic_unc)
            log_tensor_info("Aleatoric uncertainty", aleatoric_unc)
            log_tensor_info("Total uncertainty", total_unc)

            assert torch.all(epistemic_unc >= 0), "Epistemic uncertainty contains negative values"
            assert torch.all(aleatoric_unc >= 0), "Aleatoric uncertainty contains negative values"
            assert torch.all(total_unc >= 0), "Total uncertainty contains negative values"
            assert torch.all(total_unc >= epistemic_unc), "Total uncertainty is less than epistemic uncertainty"
            assert torch.all(total_unc >= aleatoric_unc), "Total uncertainty is less than aleatoric uncertainty"

        logger.info("Model uncertainty decomposition successful")

    except Exception as e:
        logger.error(f"Error in test_model_uncertainty_decomposition: {str(e)}")
        logger.exception("Full traceback:")
        raise


def run_all_tests(model: UncertainTransformerLMHeadModel, config: UncertainTransformerConfig,
                  tokenizer: PreTrainedTokenizerBase) -> None:
    logger.info("Starting all tests...")

    try:
        test_model_forward_with_uncertainty(model, config, BATCH_SIZE, SEQ_LEN)
        test_model_generation_with_uncertainty(model, tokenizer, MAX_LENGTH, NUM_RETURN_SEQUENCES)
        test_uncertainty_guided_sampling(model, config, BATCH_SIZE, SEQ_LEN, temperature=0.7, alpha=1.0)
        test_model_uncertainty_decomposition(model, config)

        logger.info("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Error occurred while running tests: {str(e)}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    logger.info(f"Running tests on device: {DEVICE}")
    config = config()
    model = model(config)
    tokenizer = tokenizer()
    run_all_tests(model, config, tokenizer)
