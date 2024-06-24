# core/tests/test_mamba_kan_integration.py

import pytest
import torch

from core.models.uncertain_nn import UncertainTransformerConfig, UncertainTransformerLMHeadModel
from core.utils.tokenizer import Tokenizer
from core.utils.uncertainty import total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty

# Constants
BATCH_SIZE = 2
SEQ_LEN = 10
VOCAB_SIZE = 1000
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
D_FF = 512
D_STATE = 16
D_CONV = 4
EXPAND_FACTOR = 2.0
DT_RANK = 16

# Device to use for testing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def config():
    return UncertainTransformerConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        dropout=0.1,
        max_position_embeddings=SEQ_LEN,
        pad_token_id=0,
        use_mamba=True,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand_factor=EXPAND_FACTOR,
        dt_rank=DT_RANK,
        sliding_window_size=SEQ_LEN,
    )


@pytest.fixture
def model(config):
    model = UncertainTransformerLMHeadModel(config)
    return model.to(DEVICE)


@pytest.fixture
def tokenizer():
    return Tokenizer.from_pretrained("gpt2")


def test_model_forward_with_uncertainty(model: UncertainTransformerLMHeadModel, config: UncertainTransformerConfig):
    """
    Tests the model's forward pass with uncertainty estimation.
    """
    input_ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long, device=DEVICE)

    outputs, uncertainty = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs.logits.shape == (BATCH_SIZE, SEQ_LEN, config.vocab_size)
    assert uncertainty.shape == (BATCH_SIZE, SEQ_LEN, config.vocab_size)
    assert outputs.past_key_values is not None
    assert len(outputs.past_key_values) == config.n_layers


def test_model_generation_with_uncertainty(model, tokenizer):
    """
    Tests the model's text generation with uncertainty.
    """
    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=DEVICE)

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=20,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        output_scores=True,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_attentions=True,
    )

    assert output_sequences.sequences.shape[1] <= 20
    assert output_sequences.sequences.shape[0] == 1
    assert output_sequences.scores is not None
    assert output_sequences.hidden_states is not None
    assert output_sequences.attentions is not None


def test_model_uncertainty_decomposition(model: UncertainTransformerLMHeadModel, config: UncertainTransformerConfig):
    """
    Tests the decomposition of the model's uncertainty into epistemic and aleatoric components.
    """
    input_ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long, device=DEVICE)

    outputs, uncertainty = model(input_ids=input_ids, attention_mask=attention_mask)

    # Perform multiple forward passes for Monte Carlo estimation
    mc_outputs = [model(input_ids=input_ids, attention_mask=attention_mask) for _ in range(5)]
    mc_logits = torch.stack([out[0].logits for out in mc_outputs])

    epistemic_unc = epistemic_uncertainty(mc_logits)
    aleatoric_unc = aleatoric_uncertainty(torch.stack([out[1] for out in mc_outputs]))
    total_unc = total_uncertainty(epistemic_unc, aleatoric_unc)

    assert epistemic_unc.shape == (BATCH_SIZE, SEQ_LEN, config.vocab_size)
    assert aleatoric_unc.shape == (BATCH_SIZE, SEQ_LEN, config.vocab_size)
    assert total_unc.shape == (BATCH_SIZE, SEQ_LEN, config.vocab_size)


def test_uncertainty_guided_sampling(model: UncertainTransformerLMHeadModel, config: UncertainTransformerConfig):
    """
    Tests the uncertainty-guided sampling function for text generation.
    """
    from core.utils.uncertainty import uncertainty_guided_sampling

    input_ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long, device=DEVICE)

    outputs, uncertainty = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, -1, :]  # Get logits for the last token
    uncertainties = uncertainty[:, -1, :]  # Get uncertainties for the last token

    sampled_tokens = uncertainty_guided_sampling(logits, uncertainties)

    assert sampled_tokens.shape == (BATCH_SIZE,)
    assert torch.all(sampled_tokens < config.vocab_size)