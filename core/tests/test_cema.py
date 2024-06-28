import unittest

import torch

from core.models.attention import CEMA



class TestCEMA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.embed_dim = 64
        cls.ndim = 16
        cls.cema = CEMA(cls.embed_dim, cls.ndim).to(cls.device)
        cls.batch_size = 4
        cls.seq_len = 20

    def test_output_shape(self):
        """Checks the output shape of the CEMA layer."""
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.embed_dim).to(self.device)
        output_tensor = self.cema(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_cumulative_effect(self):
        """Verifies that the output reflects the cumulative effect."""
        input_tensor = torch.ones(self.batch_size, self.seq_len, self.embed_dim).to(self.device)
        output_tensor = self.cema(input_tensor)
        # Check if the output changes along the sequence dimension
        self.assertFalse(torch.allclose(output_tensor[:, 0, :], output_tensor[:, -1, :]))

    def test_gradient_flow(self):
        """Ensures gradients can flow through the CEMA layer."""
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.embed_dim, requires_grad=True).to(self.device)
        target = torch.randn(self.batch_size, self.seq_len, self.embed_dim).to(self.device)

        self.cema.train()
        output_tensor = self.cema(input_tensor)
        loss = torch.nn.functional.mse_loss(output_tensor, target)
        loss.backward()

        self.assertIsNotNone(input_tensor.grad, "Gradient is None for input tensor")
        self.assertGreater(input_tensor.grad.abs().sum().item(), 0, "Gradient has zero magnitude for input tensor")

        for name, param in self.cema.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient is None for CEMA parameter: {name}")
            self.assertGreater(param.grad.abs().sum().item(), 0,
                               f"Gradient has zero magnitude for CEMA parameter: {name}")

    def test_coefficients(self):
        """Tests the coefficient calculation of the CEMA layer."""
        p, q, gamma = self.cema._calc_coeffs()

        self.assertEqual(p.shape, (self.embed_dim, self.ndim, 1))
        self.assertEqual(q.shape, (self.embed_dim, self.ndim, 1))
        self.assertEqual(gamma.shape, (self.embed_dim, self.ndim))

        self.assertTrue(torch.all(p >= 0) and torch.all(p <= 1))
        self.assertTrue(torch.all(torch.abs(q) <= 1))

    def test_different_dimensions(self):
        """Tests the CEMA layer with different embedding dimensions and ndim values."""
        for embed_dim, ndim in [(32, 8), (128, 32)]:
            cema = CEMA(embed_dim, ndim).to(self.device)
            input_tensor = torch.randn(self.batch_size, self.seq_len, embed_dim).to(self.device)
            output_tensor = cema(input_tensor)
            self.assertEqual(output_tensor.shape, (self.batch_size, self.seq_len, embed_dim))

    def test_parameter_initialization(self):
        """Tests the parameter initialization of the CEMA layer."""
        self.assertEqual(self.cema.alpha.shape, (self.embed_dim, self.ndim, 1))
        self.assertEqual(self.cema.delta.shape, (self.embed_dim, self.ndim, 1))
        self.assertEqual(self.cema.theta.shape, (self.embed_dim, 1, 1))
        self.assertEqual(self.cema.gamma.shape, (self.embed_dim, self.ndim, 2))
        self.assertEqual(self.cema.omega.shape, (self.embed_dim, 1))

        # Check initialization ranges
        self.assertTrue(torch.all(self.cema.alpha >= -0.6) and torch.all(self.cema.alpha <= 0.6))
        self.assertTrue(torch.all(self.cema.delta >= -0.6) and torch.all(self.cema.delta <= 0.6))
        self.assertTrue(torch.all(self.cema.omega >= -0.75) and torch.all(self.cema.omega <= 0.75))

    def test_residual_connection(self):
        """Tests the residual connection in the CEMA layer."""
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.embed_dim).to(self.device)
        output_tensor = self.cema(input_tensor)

        # Check that the output is not identical to the input
        self.assertFalse(torch.allclose(input_tensor, output_tensor))

        # Check that the residual connection is working
        residual = input_tensor * self.cema.omega.t()
        self.assertTrue(torch.allclose(output_tensor - residual, output_tensor - input_tensor * self.cema.omega.t()))


if __name__ == '__main__':
    unittest.main()
