from typing import Tuple

import pytest
import torch
import torch.nn as nn
from loguru import logger

from core import Mamba, MambaConfig, check_layer
from core.models.layers import TransformerEncoderLayer, KANFeedForward
from core.models.uncertainty.uncertainty import (
    UncertainTransformerConfig,
    UncertainTransformerLMHeadModel,
)


def test_uncertain_transformer():
    """
    Test the UncertainTransformerLMHeadModel and its components.

    This function creates an instance of the model, tests individual layers,
    and checks the model's learnability.
    """
    # Enable anomaly detection for more informative error messages
    torch.autograd.set_detect_anomaly(True)

    # Use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cpu.is_available() else "cpu")

    # Define model configuration
    config = UncertainTransformerConfig(
        vocab_size=256128,
        d_model=128,
        n_heads=4,
        d_ff=256,
        n_layers=2,
        dropout=0.1,
        max_position_embeddings=512,
        use_mamba=True,
        d_state=8,
        d_conv=2,
        expand_factor=1.5,
        dt_rank=8,
    )

    # Initialize the model and move it to the appropriate device
    model = UncertainTransformerLMHeadModel(config).to(device)

    # Test individual layers
    check_layer(model.transformer.embedding, (2, 10), (2, 10, config.d_model))
    check_layer(
        model.transformer.rotary_pos_emb,
        (2, 10, config.d_model),
        (2, config.n_heads, 10, config.d_model // config.n_heads),
    )

    # sourcery skip: no-loop-in-tests
    for layer in model.transformer.layers:
        mamba, transformer, kan_ff, projection = layer
        check_layer(mamba, (2, 10, config.d_model), (2, 10, config.d_model))
        check_layer(transformer, (2, 10, config.d_model), (2, 10, config.d_model))
        check_layer(kan_ff, (2, 10, config.d_model), (2, 10, config.d_model))
        check_layer(projection, (2, 10, config.d_model), (2, 10, config.d_model))

    check_layer(
        model.transformer.final_layer_norm,
        (2, 10, config.d_model),
        (2, 10, config.d_model),
    )
    check_layer(model.lm_head, (2, 10, config.d_model), (2, 10, config.vocab_size))

    # Test full model learnability
    check_model_learnability(model, (2, 10), device)
    logger.info("Model passed all tests")


def check_model_learnability(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    num_steps: int = 5,
):
    """
    Check if the model is learnable by performing a few optimization steps.

    Args:
        model (nn.Module): The model to test.
        input_shape (Tuple[int, ...]): The shape of the input tensor.
        device (torch.device): The device to run the test on.
        num_steps (int, optional): The number of optimization steps to perform. Defaults to 5.
    """
    _criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    initial_loss = None
    for step in range(num_steps):
        input_ids = torch.randint(
            0, model.config.vocab_size, input_shape, device=device
        )
        attention_mask = torch.ones_like(input_ids, device=device)
        labels = torch.randint(0, model.config.vocab_size, input_shape, device=device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        logger.info(f"Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}")

        if step == 0:
            initial_loss = loss.item()
        elif step == num_steps - 1:
            assert (
                loss.item() < initial_loss
            ), f"Loss did not decrease: initial {initial_loss:.4f}, final {loss.item():.4f}"


def test_mamba():
    config = MambaConfig(
        d_model=128,
        d_state=8,
        d_conv=2,
        expand_factor=1.5,
        dt_rank=8,
    )
    mamba = Mamba(config)
    check_layer(mamba.to("cuda"), (2, 10, config.d_model), (2, 10, config.d_model))


def test_transformer_encoder_layer():
    config = UncertainTransformerConfig(
        d_model=128,
        n_heads=4,
        d_ff=256,
        dropout=0.1,
    )
    transformer = TransformerEncoderLayer(config).to("cpu")
    check_layer(transformer, (2, 10, config.d_model), (2, 10, config.d_model))


def test_kan_feed_forward():
    config = UncertainTransformerConfig(
        d_model=128,
        d_ff=256,
        dropout=0.1,
    )
    kan_ff = KANFeedForward(config).to("cpu")
    check_layer(kan_ff, (2, 10, config.d_model), (2, 10, config.d_model))


if __name__ == "__main__":
    pytest.main([__file__])
