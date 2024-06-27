import os

import pytest
import torch

from core.models.uncertainty.uncertainty import UncertainTransformerConfig, UncertainTransformerLMHeadModel
from core.utils.tokenizer import Tokenizer

# Constants
BATCH_SIZE = 2
SEQ_LEN = 10
VOCAB_SIZE = 1000  # Reduced for testing
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 2  # Reduced for faster testing
D_FF = 2048
D_STATE = 16
D_CONV = 4
EXPAND_FACTOR = 2.0
DT_RANK = 16
MAX_LENGTH = 20
NUM_RETURN_SEQUENCES = 1
NUM_MC_SAMPLES = 3  # Reduced for faster testing

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
        max_position_embeddings=512,
        pad_token_id=0,
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


def test_model_generation_with_uncertainty(
        model: UncertainTransformerLMHeadModel,
        tokenizer: Tokenizer,
        max_length: int = MAX_LENGTH,
        num_return_sequences: int = NUM_RETURN_SEQUENCES
) -> None:
    try:
        model.eval()

        input_text = "Hello, world!"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)

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
    finally:
        pass
