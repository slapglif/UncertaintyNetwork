import unittest

import torch

from core.models.layers import KANFeedForward
from core.models.uncertainty.uncertainty import UncertainTransformerConfig


class TestKANFeedForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda"
        cls.config = UncertainTransformerConfig(
            d_model=64,
            d_ff=128,
            dropout=0.1,
        )
        cls.layer = KANFeedForward(cls.config).to(cls.device)
        cls.batch_size = 4
        cls.seq_len = 20

    def test_output_shape(self):
        """Checks the output shape of the KANFeedForward layer."""
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.config.d_model).to(self.device)
        output_tensor = self.layer(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.seq_len, self.config.d_model))

    def test_gradient_flow(self):
        """Ensures gradients propagate correctly through the layer."""
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.config.d_model, requires_grad=True).to(
            self.device)
        output_tensor = self.layer(input_tensor)
        loss = output_tensor.sum()
        loss.backward()
        # sourcery skip: no-loop-in-tests
        for name, param in self.layer.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient is None for parameter {name}")
