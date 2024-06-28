import unittest

import torch

from core.models.embedding import RotaryPositionEncoding


class TestRotaryPositionEncoding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = 'cuda'
        cls.d_model = 64
        cls.n_heads = 2
        cls.max_position_embeddings = 512
        cls.rotary_pe = RotaryPositionEncoding(cls.d_model, cls.n_heads, cls.max_position_embeddings).to(cls.device)
        cls.batch_size = 4
        cls.seq_len = 20

    def test_output_shape(self):
        """Verifies the output shape of the RotaryPositionEncoding."""
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.d_model).to(self.device)
        cos, sin = self.rotary_pe(input_tensor)
        self.assertEqual(cos.shape, (1, self.seq_len, 1, self.d_model // self.n_heads))
        self.assertEqual(sin.shape, (1, self.seq_len, 1, self.d_model // self.n_heads))

    def test_output_range(self):
        """Checks that the output values of cos and sin are within the expected range."""
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.d_model).to(self.device)
        cos, sin = self.rotary_pe(input_tensor)
        self.assertTrue(torch.all(cos >= -1.0) and torch.all(cos <= 1.0))
        self.assertTrue(torch.all(sin >= -1.0) and torch.all(sin <= 1.0))

    def test_different_sequence_lengths(self):
        """Ensures correct handling of different sequence lengths."""
        # sourcery skip: no-loop-in-tests
        for seq_len in [10, 30, 50]:
            input_tensor = torch.randn(self.batch_size, seq_len, self.d_model).to(self.device)
            cos, sin = self.rotary_pe(input_tensor)
            self.assertEqual(cos.shape, (1, seq_len, 1, self.d_model // self.n_heads))
            self.assertEqual(sin.shape, (1, seq_len, 1, self.d_model // self.n_heads))

    def test_gradient_flow(self):
        """Verifies that gradients flow correctly through the module."""
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True).to(
            self.device) # Enable gradient calculation
        cos, sin = self.rotary_pe(input_tensor)
        loss = (cos.sum() + sin.sum())
        loss.backward()
        self.assertIsNotNone(input_tensor.grad, "Gradient is None for input tensor")