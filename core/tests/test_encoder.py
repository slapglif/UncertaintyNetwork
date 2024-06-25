# core/tests/test_encoder.py

import pytest
import torch
from core.models.embedding import RotaryPositionEncoding, SentenceEncoder, SentenceGP
from core.models.uncertainty.uncertain_nn import UncertainTransformerConfig, UncertainTransformerLMHeadModel
from core.utils.tokenizer import Tokenizer
from core.models.uncertainty.uncertainty_utils import UncertaintyModule

# Constants
BATCH_SIZE = 2
SEQ_LEN = 10
NUM_SENTENCES = 3
EMBED_DIM = 512
N_INDUCING = 10
LEARNING_RATE = 1e-5
NUM_EPOCHS = 5
D_STATE = 16
D_CONV = 4
EXPAND_FACTOR = 2.0
DT_RANK = 16
N_HEADS = 8
D_FF = 2048

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tokenizer():
    return Tokenizer.from_pretrained("gpt2")


@pytest.fixture
def config():
    return UncertainTransformerConfig(
        vocab_size=50257,
        d_model=EMBED_DIM,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=2,
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


def test_rotary_position_encoding_shape():
    rotary_pe = RotaryPositionEncoding(dim=EMBED_DIM, n_heads=N_HEADS).to(DEVICE)
    input_tensor = torch.randn(1, SEQ_LEN, EMBED_DIM, device=DEVICE)
    cos, sin = rotary_pe(input_tensor)
    assert cos.shape == (1, N_HEADS, SEQ_LEN, EMBED_DIM // N_HEADS)
    assert sin.shape == (1, N_HEADS, SEQ_LEN, EMBED_DIM // N_HEADS)


def test_sentence_encoder_shape(tokenizer, config):
    sentence_encoder = SentenceEncoder(
        vocab_size=config.vocab_size,
        hidden_dim=EMBED_DIM,
        output_dim=EMBED_DIM,
        kan_config={
            'layers_hidden': [1024, 2048],
            'grid_min': -1.2,
            'grid_max': 0.2,
            'num_grids': 8,
            'exponent': 2,
            'inv_denominator': 0.5,
            'train_grid': False,
            'train_inv_denominator': False,
            'spline_weight_init_scale': 1.0,
        }
    ).to(DEVICE)

    input_tensor = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    output = sentence_encoder(input_tensor)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, EMBED_DIM)


def test_sentence_gp_output_type(config):
    sentence_gp = SentenceGP(
        input_dim=EMBED_DIM,
        output_dim=EMBED_DIM,
        n_inducing=N_INDUCING,
        embedding_dim=EMBED_DIM,
    ).to(DEVICE)
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, device=DEVICE)
    mean, variance = sentence_gp(input_tensor)
    assert isinstance(mean, torch.Tensor)
    assert isinstance(variance, torch.Tensor)


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


def test_model_with_uncertainty(config):
    model = UncertainTransformerLMHeadModel(config).to(DEVICE)
    input_ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long, device=DEVICE)

    outputs, uncertainty = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs.logits.shape == (BATCH_SIZE, SEQ_LEN, config.vocab_size)
    assert uncertainty.shape == (BATCH_SIZE, SEQ_LEN, config.vocab_size)

