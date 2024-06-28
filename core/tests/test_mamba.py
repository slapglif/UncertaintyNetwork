import os
import unittest

import torch
from torch.cpu.amp import autocast, GradScaler

from core.models.statespace import Mamba, MambaConfig, InferenceCache
from core.models.uncertainty.uncertainty import UncertainTransformerConfig, UncertainTransformerLMHeadModel
from core.utils.tokenizer import Tokenizer
from core.utils.utils import check_layer

# Set CUDA_LAUNCH_BLOCKING for debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Device to use for testing
DEVICE = torch.device("cuda" if torch.cpu.is_available() else "cpu")


class TestMambaIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer()
        cls.config = UncertainTransformerConfig(
            vocab_size=cls.tokenizer.vocab_size,
            d_model=64,  # Reduced for testing
            n_heads=2,  # Reduced for testing
            d_ff=128,  # Reduced for testing
            n_layers=1,  # Reduced for testing
            dropout=0.1,
            max_position_embeddings=512,
            pad_token_id=0,
            use_mamba=True,
            d_state=8,
            d_conv=2,
            expand_factor=1.5,
            dt_rank=8,
            n_inducing=5,
            uncertainty_weight=0.1,
        )
        cls.model = UncertainTransformerLMHeadModel(cls.config).to(DEVICE)
        cls.mamba_config = MambaConfig(
            d_model=cls.config.d_model,
            d_state=cls.config.d_state,
            d_conv=cls.config.d_conv,
            expand_factor=cls.config.expand_factor,
            dt_rank=cls.config.dt_rank,
        )
        cls.mamba_layer = Mamba(cls.mamba_config).to(DEVICE)
        cls.batch_size = 4
        cls.seq_len = 20

    def test_mamba_output_shape(self):
        """Verifies that the Mamba layer outputs tensors of the expected shape."""
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.config.d_model).to(DEVICE)
        output_tensor, _ = self.mamba_layer(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.seq_len, self.config.d_model))

    def test_mamba_inference_cache_shape(self):
        """Ensures the InferenceCache produced by Mamba has the correct shape."""
        input_tensor = torch.randn(self.batch_size, 1, self.config.d_model).to(DEVICE)
        state = InferenceCache.alloc(self.batch_size, self.mamba_config, device=DEVICE)
        _, new_state = self.mamba_layer(input_tensor, state=state, use_cache=True)
        self.assertEqual(new_state.conv_state.shape, (
        self.batch_size, self.mamba_config.d_inner + 2 * self.mamba_config.d_state, self.mamba_config.d_conv))
        self.assertEqual(new_state.ssm_state.shape, (
        self.batch_size, self.mamba_config.n_heads, self.mamba_config.headdim, self.mamba_config.d_state))

    def test_mamba_integration_forward_pass(self):
        """Tests the forward pass of the entire model with the Mamba layer."""
        self.initialize_model("logits")

    def test_mamba_gradient_flow(self):
        """Checks if gradients flow correctly through the Mamba layer during training."""
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len)).to(DEVICE)
        attention_mask = torch.ones_like(input_ids).to(DEVICE)
        labels = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len)).to(DEVICE)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        scaler = GradScaler()

        self.model.train()

        with autocast():
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        for name, param in self.mamba_layer.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient is None for parameter {name} in Mamba layer")
            self.assertGreater(torch.sum(param.grad ** 2), 0, f"Gradient is zero for parameter {name} in Mamba layer")

        scaler.step(optimizer)
        scaler.update()

    def test_mamba_integration_with_uncertainty_module(self):
        """Ensures the Mamba layer interacts correctly with the UncertaintyModule."""
        self.initialize_model("uncertainties")


    def initialize_model(self, arg0):
        input_ids = torch.randint(
            0, self.config.vocab_size, (self.batch_size, self.seq_len)
        ).to(DEVICE)
        attention_mask = torch.ones_like(input_ids).to(DEVICE)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        self.assertEqual(
            outputs[arg0].shape,
            (self.batch_size, self.seq_len, self.config.vocab_size),
        )

    def test_mamba_layer_checkpointing(self):
        """Tests if gradient checkpointing works correctly with the Mamba layer."""
        self.config.use_gradient_checkpointing = True
        checkpointed_model = UncertainTransformerLMHeadModel(self.config).to(DEVICE)
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len)).to(DEVICE)
        attention_mask = torch.ones_like(input_ids).to(DEVICE)
        labels = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len)).to(DEVICE)

        optimizer = torch.optim.Adam(checkpointed_model.parameters(), lr=1e-4)
        checkpointed_model.train()

        outputs = checkpointed_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        # Add assertions to check for successful gradient checkpointing
        self.assertTrue(any(
            name.startswith("transformer.layers.0.0") and p.grad is not None
            for name, p in checkpointed_model.named_parameters()
        ), "No gradients found for Mamba layer parameters, checkpointing might have failed.")

    def test_mamba_padding(self):
        # sourcery skip: no-loop-in-tests
        """Validates the padding logic within the Mamba layer for different sequence lengths."""

        for seq_len in [15, 32, 65]:  # Test with lengths requiring padding and not
            input_tensor = torch.randn(self.batch_size, seq_len, self.config.d_model).to(DEVICE)
            output_tensor, _ = self.mamba_layer(input_tensor)
            self.assertEqual(output_tensor.shape, (self.batch_size, seq_len, self.config.d_model))

    def test_mamba_numerical_stability(self):
        # sourcery skip: no-loop-in-tests
        """Checks for NaN or Inf values in the Mamba layer's output for various inputs."""

        for _ in range(10):  # Run multiple tests with random inputs
            input_tensor = torch.randn(self.batch_size, self.seq_len, self.config.d_model).to(DEVICE)
            output_tensor, _ = self.mamba_layer(input_tensor)
            self.assertFalse(torch.isnan(output_tensor).any(), "NaN values detected in Mamba output")
            self.assertFalse(torch.isinf(output_tensor).any(), "Inf values detected in Mamba output")

    def test_mamba_step_function(self):
        """Specifically tests the step function used during inference."""
        input_tensor = torch.randn(self.batch_size, 1, self.config.d_model).to(DEVICE)
        state = InferenceCache.alloc(self.batch_size, self.mamba_config, device=DEVICE)
        output_tensor, new_state = self.mamba_layer(input_tensor, state=state, use_cache=True)
        self.assertEqual(output_tensor.shape, (self.batch_size, 1, self.config.d_model))
        self.assertIsInstance(new_state, InferenceCache)

    def test_mamba_with_different_configs(self):
        # sourcery skip: no-loop-in-tests
        """
        Tests the Mamba layer with different configurations to ensure compatibility
        and robustness to varying hyperparameters.
        """
        for d_state in [4, 16, 32]:
            for d_conv in [2, 4, 6]:
                for expand_factor in [1.5, 2.0, 2.5]:
                    config = MambaConfig(
                        d_model=self.config.d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand_factor=expand_factor,
                        dt_rank=8,
                    )
                    mamba = Mamba(config).to(DEVICE)
                    input_tensor = torch.randn(self.batch_size, self.seq_len, self.config.d_model).to(DEVICE)
                    check_layer(
                        mamba,
                        (self.batch_size, self.seq_len, self.config.d_model),
                        (self.batch_size, self.seq_len, self.config.d_model),
                        device=DEVICE,
                    )


if __name__ == "__main__":
    unittest.main()
