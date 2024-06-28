import unittest

import torch

from core.models.layers import CEMA


class TestCEMA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda"
        cls.d_model = 64
        cls.cema = CEMA(cls.d_model).to(cls.device)
        cls.batch_size = 4
        cls.seq_len = 20

    def test_output_shape(self):
        """Checks the output shape of the CEMA layer."""
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.d_model).to(self.device)
        output_tensor = self.cema(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_cumulative_effect(self):
        """Verifies that the output reflects the cumulative average."""
        input_tensor = torch.ones(self.batch_size, self.seq_len, self.d_model).to(self.device)
        output_tensor = self.cema(input_tensor)
        # As input is all ones, the cumulative average should be increasing
        self.assertTrue(torch.all(output_tensor[:, 1:] > output_tensor[:, :-1]))

    def test_gradient_flow(self):
        """Ensures gradients can flow through the CEMA layer."""
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True).to(self.device)
        output_tensor = self.cema(input_tensor)
        loss = output_tensor.sum()
        loss.backward()
        self.assertIsNotNone(input_tensor.grad, "Gradient is None for input tensor")

    def test_different_alpha_values(self):
        """Tests the CEMA layer with different smoothing factors (alpha)."""
        # sourcery skip: no-loop-in-tests
        for alpha in [0.1, 0.5, 0.9, 0.99]:
            cema = CEMA(self.d_model, alpha=alpha).to(self.device)
            input_tensor = torch.randn(self.batch_size, self.seq_len, self.d_model).to(self.device)
            output_tensor = cema(input_tensor)
            self.assertEqual(output_tensor.shape, (self.batch_size, self.seq_len, self.d_model))
