# core/tests/test_e2e.py

import pytest
import torch
from transformers import GPT2Tokenizer

from core.models.layers import (
    MambaLayer,
    TransformerEncoderLayer,
)
from core.models.uncertainty.uncertain_nn import UncertainTransformerConfig, UncertainTransformerLMHeadModel
from core.models.uncertainty.uncertainty import UncertaintyModule, uncertainty_guided_sampling

# Constants
BATCH_SIZE = 2
SEQ_LEN = 10
EMBED_DIM = 512
N_HEADS = 8
D_FF = 2048
N_INDUCING = 10
D_STATE = 16
D_CONV = 4
EXPAND_FACTOR = 2.0
DT_RANK = 16

# Device to use for testing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer for text encoding and decoding
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


@pytest.fixture
def config():
    return UncertainTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=EMBED_DIM,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=2,  # Reduced for faster testing
        dropout=0.1,
        max_position_embeddings=SEQ_LEN,
        pad_token_id=tokenizer.pad_token_id,
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
    model.to(DEVICE)
    return model


def test_transformer_encoder_layer_with_uncertainty(config):
    encoder_layer = TransformerEncoderLayer(config).to(DEVICE)
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, device=DEVICE)
    attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long, device=DEVICE)
    attention_mask[:, 5:] = 0

    output = encoder_layer(input_tensor, attention_mask=attention_mask)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, EMBED_DIM)


def test_mamba_layer(config):
    mamba_layer = MambaLayer(config).to(DEVICE)
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, device=DEVICE)
    output = mamba_layer(input_tensor)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, EMBED_DIM)


def test_uncertainty_module(config):
    uncertainty_module = UncertaintyModule(
        input_dim=EMBED_DIM,
        output_dim=config.vocab_size,
        n_gp_layers=2,
        n_inducing=N_INDUCING,
        dropout_rate=0.1
    ).to(DEVICE)
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, device=DEVICE)
    mean, uncertainty = uncertainty_module(input_tensor)
    assert mean.shape == (BATCH_SIZE, SEQ_LEN, config.vocab_size)
    assert uncertainty.shape == (BATCH_SIZE, SEQ_LEN, config.vocab_size)


def test_model_generation_with_uncertainty(model, config):
    model.eval()
    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=DEVICE)

    generated_outputs = model.generate(
        input_ids,
        max_length=20,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
    )

    assert generated_outputs.sequences.shape[1] <= 20
    assert generated_outputs.sequences.shape[0] == 1
    assert generated_outputs.scores is not None

    # Test uncertainty-guided sampling
    logits = generated_outputs.scores[-1]
    uncertainties = \
        model.transformer.uncertainty_module(model.transformer.final_layer_norm(generated_outputs.hidden_states[-1]))[
            -1]
    sampled_tokens = uncertainty_guided_sampling(logits, uncertainties)
    assert sampled_tokens.shape == (1,)


def test_model_forward_with_uncertainty(model, config):
    input_ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long, device=DEVICE)

    outputs, uncertainty = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs.logits.shape == (BATCH_SIZE, SEQ_LEN, config.vocab_size)
    assert uncertainty.shape == (BATCH_SIZE, SEQ_LEN, config.vocab_size)

# Add more tests as needed
