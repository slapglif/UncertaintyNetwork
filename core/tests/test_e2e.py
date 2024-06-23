import pytest
import torch
from transformers import GPT2Tokenizer

from core.models.embedding import RotaryPositionEncoding, SentenceEncoder, SentenceGP
from core.models.layers import (
    MultiHeadAttention,
    PositionwiseFeedForward,
    GaussianProcessLayer,
    CEMA,
    MambaLayer,
    TransformerEncoderLayer,
)
from core.models.uncertain_nn import UncertainTransformerConfig, UncertainTransformerLMHeadModel
from core.utils.utils import generate_text, calculate_perplexity

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


def test_transformer_encoder_layer(config):
    """Tests the TransformerEncoderLayer with attention mask and past key values."""
    encoder_layer = TransformerEncoderLayer(config).to(DEVICE)

    # Input tensors
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, device=DEVICE)
    attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long, device=DEVICE)
    attention_mask[:, 5:] = 0

    # Test with attention mask
    output = encoder_layer(input_tensor, attention_mask=attention_mask)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, EMBED_DIM)

    # Test with past key values
    past_key_value = (
        torch.randn(BATCH_SIZE, N_HEADS, 5, EMBED_DIM // N_HEADS, device=DEVICE),
        torch.randn(BATCH_SIZE, N_HEADS, 5, EMBED_DIM // N_HEADS, device=DEVICE),
    )
    output, present_key_value = encoder_layer(
        input_tensor, past_key_value=past_key_value, use_cache=True
    )
    assert output.shape == (BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    assert len(present_key_value) == 2
    assert present_key_value[0].shape == (
        BATCH_SIZE,
        N_HEADS,
        SEQ_LEN + 5,
        EMBED_DIM // N_HEADS,
    )
    assert present_key_value[1].shape == (
        BATCH_SIZE,
        N_HEADS,
        SEQ_LEN + 5,
        EMBED_DIM // N_HEADS,
    )


def test_mamba_layer(config):
    """Tests the MambaLayer with various input shapes."""
    mamba_layer = MambaLayer(config).to(DEVICE)

    # Test with standard input shape
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, device=DEVICE)
    output = mamba_layer(input_tensor)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, EMBED_DIM)

    # Test with single-sequence input (batch size of 1)
    input_tensor = torch.randn(1, SEQ_LEN, EMBED_DIM, device=DEVICE)
    output = mamba_layer(input_tensor)
    assert output.shape == (1, SEQ_LEN, EMBED_DIM)

    # Test with different sequence length
    input_tensor = torch.randn(BATCH_SIZE, 15, EMBED_DIM, device=DEVICE)
    output = mamba_layer(input_tensor)
    assert output.shape == (BATCH_SIZE, 15, EMBED_DIM)


def test_multihead_attention(config):
    """Tests the MultiHeadAttention with various input shapes."""
    attention = MultiHeadAttention(config).to(DEVICE)

    # Test with standard input shape
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, device=DEVICE)
    output, _, _ = attention(input_tensor)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, EMBED_DIM)

    # Test with single-sequence input (batch size of 1)
    input_tensor = torch.randn(1, SEQ_LEN, EMBED_DIM, device=DEVICE)
    output, _, _ = attention(input_tensor)
    assert output.shape == (1, SEQ_LEN, EMBED_DIM)

    # Test with different sequence length
    input_tensor = torch.randn(BATCH_SIZE, 15, EMBED_DIM, device=DEVICE)
    output, _, _ = attention(input_tensor)
    assert output.shape == (BATCH_SIZE, 15, EMBED_DIM)

    # Test with attention mask
    attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long, device=DEVICE)
    attention_mask[:, 5:] = 0  # Mask last 5 tokens
    output, _, _ = attention(input_tensor, attention_mask=attention_mask)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, EMBED_DIM)


def test_positionwise_feedforward(config):
    """Tests the PositionwiseFeedForward with various input shapes."""
    feed_forward = PositionwiseFeedForward(config).to(DEVICE)

    # Test with standard input shape
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, device=DEVICE)
    output = feed_forward(input_tensor)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, EMBED_DIM)

    # Test with single-sequence input (batch size of 1)
    input_tensor = torch.randn(1, SEQ_LEN, EMBED_DIM, device=DEVICE)
    output = feed_forward(input_tensor)
    assert output.shape == (1, SEQ_LEN, EMBED_DIM)

    # Test with different sequence length
    input_tensor = torch.randn(BATCH_SIZE, 15, EMBED_DIM, device=DEVICE)
    output = feed_forward(input_tensor)
    assert output.shape == (BATCH_SIZE, 15, EMBED_DIM)


def test_gaussian_process_layer(config):
    """Tests the GaussianProcessLayer with various input shapes."""
    gp_layer = GaussianProcessLayer(
        in_features=EMBED_DIM,
        out_features=EMBED_DIM,
        n_inducing=N_INDUCING,
        embedding_dim=EMBED_DIM,
    ).to(DEVICE)

    # Test with standard input shape
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, device=DEVICE)
    mean, variance = gp_layer(input_tensor, seq_len=SEQ_LEN)
    assert mean.shape == (BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    assert variance.shape == (BATCH_SIZE, SEQ_LEN, EMBED_DIM)

    # Test with single-sequence input (batch size of 1)
    input_tensor = torch.randn(1, SEQ_LEN, EMBED_DIM, device=DEVICE)
    mean, variance = gp_layer(input_tensor, seq_len=SEQ_LEN)
    assert mean.shape == (1, SEQ_LEN, EMBED_DIM)
    assert variance.shape == (1, SEQ_LEN, EMBED_DIM)

    # Test with different sequence length
    input_tensor = torch.randn(BATCH_SIZE, 15, EMBED_DIM, device=DEVICE)
    mean, variance = gp_layer(input_tensor, seq_len=15)
    assert mean.shape == (BATCH_SIZE, 15, EMBED_DIM)
    assert variance.shape == (BATCH_SIZE, 15, EMBED_DIM)


def test_cema(config):
    """Tests the CEMA layer with various input shapes."""
    cema_layer = CEMA(d_model=EMBED_DIM, alpha=0.99).to(DEVICE)

    # Test with standard input shape
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, device=DEVICE)
    output = cema_layer(input_tensor)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, EMBED_DIM)

    # Test with single-sequence input (batch size of 1)
    input_tensor = torch.randn(1, SEQ_LEN, EMBED_DIM, device=DEVICE)
    output = cema_layer(input_tensor)
    assert output.shape == (1, SEQ_LEN, EMBED_DIM)

    # Test with different sequence length
    input_tensor = torch.randn(BATCH_SIZE, 15, EMBED_DIM, device=DEVICE)
    output = cema_layer(input_tensor)
    assert output.shape == (BATCH_SIZE, 15, EMBED_DIM)


def test_rotary_position_encoding(config):
    """Tests the RotaryPositionEncoding layer."""
    rotary_pe = RotaryPositionEncoding(dim=EMBED_DIM).to(DEVICE)

    # Test with standard input shape
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, device=DEVICE)
    cos, sin = rotary_pe(input_tensor)
    assert cos.shape == (BATCH_SIZE, 1, SEQ_LEN, EMBED_DIM)
    assert sin.shape == (BATCH_SIZE, 1, SEQ_LEN, EMBED_DIM)

    # Test with different sequence length
    input_tensor = torch.randn(BATCH_SIZE, 15, EMBED_DIM, device=DEVICE)
    cos, sin = rotary_pe(input_tensor, seq_len=15)
    assert cos.shape == (BATCH_SIZE, 1, 15, EMBED_DIM)
    assert sin.shape == (BATCH_SIZE, 1, 15, EMBED_DIM)


def test_sentence_encoder(config):
    """Tests the SentenceEncoder layer."""
    sentence_encoder = SentenceEncoder(
        input_dim=EMBED_DIM, hidden_dim=EMBED_DIM * 2, output_dim=EMBED_DIM
    ).to(DEVICE)

    # Test with standard input shape
    input_tensor = torch.randn(
        BATCH_SIZE, 3, SEQ_LEN, EMBED_DIM, device=DEVICE
    )  # Batch, Sentences, Seq_len, Embed_dim
    output = sentence_encoder(input_tensor)
    assert output.shape == (BATCH_SIZE, 3, EMBED_DIM)

    # Test with different number of sentences
    input_tensor = torch.randn(BATCH_SIZE, 5, SEQ_LEN, EMBED_DIM, device=DEVICE)
    output = sentence_encoder(input_tensor)
    assert output.shape == (BATCH_SIZE, 5, EMBED_DIM)


def test_sentence_gp(config):
    """Tests the SentenceGP layer."""
    sentence_gp = SentenceGP(
        input_dim=EMBED_DIM,
        output_dim=EMBED_DIM,
        n_inducing=N_INDUCING,
        embedding_dim=EMBED_DIM,
    ).to(DEVICE)

    # Test with standard input shape
    input_tensor = torch.randn(BATCH_SIZE, 3, EMBED_DIM, device=DEVICE)
    mean, variance = sentence_gp(input_tensor, num_sentences=3)
    assert mean.shape == (BATCH_SIZE, 3, EMBED_DIM)
    assert variance.shape == (BATCH_SIZE, 3, EMBED_DIM)

    # Test with different number of sentences
    input_tensor = torch.randn(BATCH_SIZE, 5, EMBED_DIM, device=DEVICE)
    mean, variance = sentence_gp(input_tensor, num_sentences=5)
    assert mean.shape == (BATCH_SIZE, 5, EMBED_DIM)
    assert variance.shape == (BATCH_SIZE, 5, EMBED_DIM)


def test_model_generation(model, config):
    """Tests the UncertainTransformerLMHeadModel with text generation."""
    model.to(DEVICE)
    model.eval()

    # Test generation with a simple prompt
    prompt = "The quick brown fox"
    generated_texts = generate_text(
        model,
        tokenizer,
        prompt,
        max_length=20,  # Reduced max length for testing
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        num_return_sequences=1,
        device=DEVICE,
    )
    assert len(generated_texts) == 1
    assert generated_texts[0].startswith(prompt)


def test_model_perplexity(model, config):
    """Tests the UncertainTransformerLMHeadModel with perplexity calculation."""
    model.to(DEVICE)
    model.eval()

    # Test perplexity calculation on a simple sentence
    sentence = "The quick brown fox jumps over the lazy dog."
    perplexity = calculate_perplexity(model, tokenizer, sentence, device=DEVICE)
    assert perplexity > 0


# Integration Test for UncertainTransformerLMHeadModel:
def test_uncertain_transformer_end_to_end(config, model):
    """Tests the UncertainTransformerLMHeadModel for forward pass and loss calculation."""
    model = UncertainTransformerLMHeadModel(config).to(DEVICE)
    model.eval()

    input_ids = torch.randint(
        0, tokenizer.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE
    )
    labels = input_ids.clone()

    # Test forward pass
    outputs = model(input_ids=input_ids, labels=labels)
    assert outputs.logits.shape == (BATCH_SIZE, SEQ_LEN, tokenizer.vocab_size)
    assert outputs.loss is not None

    # Test loss calculation
    loss = outputs.loss
    assert loss > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
