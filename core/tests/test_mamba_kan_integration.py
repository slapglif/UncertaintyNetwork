# .\core\tests\test_mamba_kan_integration.py
import pytest
import torch

from core.models.uncertain_nn import UncertainTransformerConfig, UncertainTransformerLMHeadModel
from core.utils.tokenizer import Tokenizer

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
    """
    Configures the UncertainTransformerConfig for the test.

    Returns:
        UncertainTransformerConfig: The configuration for the model.
    """
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
    try:
        model.transformer.embedding.to(DEVICE)
        print("Embedding moved to device successfully")
        model.transformer.rotary_pos_emb.to(DEVICE)
        print("Rotary position embedding moved to device successfully")
        model.transformer.sentence_encoder.to(DEVICE)
        print("Sentence encoder moved to device successfully")
        model.transformer.sentence_gp.to(DEVICE)
        print("Sentence GP moved to device successfully")
        model.transformer.gp_projection.to(DEVICE)
        print("GP projection moved to device successfully")
        model.transformer.cema.to(DEVICE)
        print("CEMA moved to device successfully")
        for i, layer in enumerate(model.transformer.layers):
            layer.to(DEVICE)
            print(f"Layer {i} moved to device successfully")
        model.transformer.final_layer_norm.to(DEVICE)
        print("Final layer norm moved to device successfully")
        model.lm_head.to(DEVICE)
        print("LM head moved to device successfully")
    except RuntimeError as e:
        print(f"Error occurred: {str(e)}")
        raise
    return model


@pytest.fixture
def tokenizer():
    """
    Creates the Tokenizer for the test.

    Returns:
        Tokenizer: The initialized tokenizer.
    """
    return Tokenizer.from_pretrained("gpt2")


def test_model_forward_with_uncertainty(model: UncertainTransformerLMHeadModel, config: UncertainTransformerConfig):
    """
    Tests the forward pass of the model with uncertainty estimation.

    Args:
        model (UncertainTransformerLMHeadModel): The model to test.
        config (UncertainTransformerConfig): The configuration for the model.
    """
    input_ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long, device=DEVICE)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs.logits.shape == (BATCH_SIZE, SEQ_LEN, config.vocab_size)
    # We removed the uncertainty head and the corresponding calculation
    # assert outputs.uncertainties is not None
    # assert outputs.uncertainties.shape == (BATCH_SIZE, SEQ_LEN)
    assert outputs.past_key_values is not None
    assert len(outputs.past_key_values) == config.n_layers


def test_model_generation_with_uncertainty(model, tokenizer):
    """
    Tests the text generation capabilities of the model with uncertainty estimation.

    Args:
        model (UncertainTransformerLMHeadModel): The model to test.
        tokenizer (Tokenizer): The tokenizer for encoding and decoding.
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


def test_model_loss_with_uncertainty(model, config):
    """
    Tests the loss calculation of the model with uncertainty estimation.

    Args:
        model (UncertainTransformerLMHeadModel): The model to test.
        config (UncertainTransformerConfig): The configuration for the model.
    """
    input_ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long, device=DEVICE)
    labels = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert outputs.loss is not None
    assert outputs.logits.shape == (BATCH_SIZE, SEQ_LEN, config.vocab_size)
    # We removed the uncertainty head and the corresponding calculation
    # assert outputs.uncertainties.shape == (BATCH_SIZE, SEQ_LEN)
    assert outputs.past_key_values is None


def test_reorder_cache(model, config):
    """
    Tests the _reorder_cache method for reordering past key values during beam search.

    Args:
        model (UncertainTransformerLMHeadModel): The model to test.
        config (UncertainTransformerConfig): The configuration for the model.
    """
    past_key_values = tuple(
        (
            tuple(torch.randn(2, BATCH_SIZE, config.d_model, device=DEVICE) for _ in range(2)),
            tuple(torch.randn(2, BATCH_SIZE, config.n_heads, SEQ_LEN, SEQ_LEN, device=DEVICE) for _ in range(2))
        )
        for _ in range(config.n_layers)
    )
    beam_idx = torch.tensor([0, 1], device=DEVICE)

    reordered_past = model._reorder_cache(past_key_values, beam_idx)

    assert len(reordered_past) == config.n_layers
    for layer_past, reordered_layer_past in zip(past_key_values, reordered_past):
        mamba_state, transformer_past = layer_past
        reordered_mamba_state, reordered_transformer_past = reordered_layer_past
        for past_state, reordered_past_state in zip(mamba_state, reordered_mamba_state):
            assert torch.allclose(past_state.index_select(1, beam_idx), reordered_past_state)
        for past_state, reordered_past_state in zip(transformer_past, reordered_transformer_past):
            assert torch.allclose(past_state.index_select(1, beam_idx), reordered_past_state)


def test_model_forward(model, config):
    """
    Tests the forward pass of the model without labels.

    Args:
        model (UncertainTransformerLMHeadModel): The model to test.
        config (UncertainTransformerConfig): The configuration for the model.
    """
    input_ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long, device=DEVICE)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs.logits.shape == (BATCH_SIZE, SEQ_LEN, config.vocab_size)
    assert outputs.past_key_values is not None
    assert len(outputs.past_key_values) == config.n_layers


def test_model_generation(model, tokenizer):
    """
    Tests the text generation capabilities of the model.

    Args:
        model (UncertainTransformerLMHeadModel): The model to test.
        tokenizer (Tokenizer): The tokenizer for encoding and decoding.
    """
    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=DEVICE)

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=20,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
    )

    assert output_sequences.shape[1] <= 20  # Check max length
    assert output_sequences.shape[0] == 1  # Check num return sequences


def test_attention_mask(model, config):
    """
    Tests the effect of using an attention mask on the model's output.

    Args:
        model (UncertainTransformerLMHeadModel): The model to test.
        config (UncertainTransformerConfig): The configuration for the model.
    """
    input_ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long, device=DEVICE)
    attention_mask[:, SEQ_LEN // 2:] = 0  # Mask out second half of the sequence

    outputs_with_mask = model(input_ids=input_ids, attention_mask=attention_mask)
    outputs_without_mask = model(input_ids=input_ids)

    # Check that the outputs are different when using a mask
    assert not torch.allclose(outputs_with_mask.logits, outputs_without_mask.logits)


def test_gradient_checkpointing(model, config):
    """
    Tests the gradient checkpointing functionality of the model.

    Args:
        model (UncertainTransformerLMHeadModel): The model to test.
        config (UncertainTransformerConfig): The configuration for the model.
    """
    model.gradient_checkpointing_enable()

    input_ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    labels = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)

    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels)

    # Backward pass
    outputs.loss.backward()

    # Check that gradients are computed
    for param in model.parameters():
        assert param.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])