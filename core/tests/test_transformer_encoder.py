import unittest

import torch

from core.models.layers import TransformerEncoderLayer
from core.models.uncertainty.uncertainty import UncertainTransformerConfig


class TestTransformerEncoderLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda"
        cls.config = UncertainTransformerConfig(
            d_model=64,
            n_heads=2,
            d_ff=128,
            dropout=0.1,
        )
        cls.layer = TransformerEncoderLayer(cls.config).to(cls.device)
        cls.batch_size = 4
        cls.seq_len = 20

    def test_output_shape(self):
        """Verifies the output shape of the TransformerEncoderLayer."""
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.config.d_model).to(self.device)
        output_tensor = self.layer(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.seq_len, self.config.d_model))

    def test_attention_mask(self):
        """Tests the application of the attention mask."""
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.config.d_model).to(self.device)
        attention_mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool).to(self.device)
        attention_mask[:, self.seq_len // 2:] = 0
        output_tensor_with_mask = self.layer(input_tensor, attention_mask=attention_mask)
        output_tensor_without_mask = self.layer(input_tensor)
        self.assertFalse(torch.allclose(output_tensor_with_mask, output_tensor_without_mask))

    def test_gradient_flow(self):
        """Ensures gradients can propagate through the layer."""
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.config.d_model, requires_grad=True).to(
            self.device)
        output_tensor = self.layer(input_tensor)
        loss = output_tensor.sum()
        loss.backward()
        # sourcery skip: no-loop-in-tests
        for name, param in self.layer.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient is None for parameter {name}")
