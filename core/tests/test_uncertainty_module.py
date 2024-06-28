import unittest

import torch

from core.models.uncertainty.uncertainty_layers import UncertaintyModule
from core.models.uncertainty.uncertainty import UncertainTransformerConfig


class TestUncertaintyModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda"
        cls.config = UncertainTransformerConfig(
            vocab_size=1000,
            d_model=64,
            n_inducing=5,
            uncertainty_weight=0.1,
        )
        cls.uncertainty_module = UncertaintyModule(
            input_dim=cls.config.d_model,
            output_dim=cls.config.vocab_size,
            n_gp_layers=1,
            n_inducing=cls.config.n_inducing,
            dropout_rate=0.1,
            mc_samples=3,
        ).to(cls.device)
        cls.batch_size = 4
        cls.seq_len = 20

    def test_output_shape(self):
        """Verifies the output shape of the UncertaintyModule."""
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.config.d_model).to(self.device)
        mean, uncertainty = self.uncertainty_module(input_tensor)
        self.assertEqual(mean.shape, (self.batch_size, self.seq_len, self.config.vocab_size))
        self.assertEqual(uncertainty.shape, (self.batch_size, self.seq_len, self.config.vocab_size))

    def test_mc_dropout_effect(self):
        """Tests if MC dropout is applied and affects the outputs."""
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.config.d_model).to(self.device)
        mean1, uncertainty1 = self.uncertainty_module(input_tensor)
        mean2, uncertainty2 = self.uncertainty_module(input_tensor)
        self.assertFalse(torch.allclose(mean1, mean2))
        self.assertFalse(torch.allclose(uncertainty1, uncertainty2))

    def test_gradient_flow(self):
        """Ensures gradients propagate correctly through the module."""
        # --- Create linearly related test data ---
        num_samples = 100
        input_dim = self.config.d_model
        x = torch.linspace(-5, 5, num_samples).unsqueeze(-1)
        input_tensor = x.repeat(self.batch_size, self.seq_len, 1)
        input_tensor = input_tensor + torch.randn_like(input_tensor) * 0.5  # Add some noise
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        # ------------------------------------------

        mean, uncertainty = self.uncertainty_module(input_tensor)
        loss = mean.sum() + uncertainty.sum()

        noise_param = self.uncertainty_module.gp_layers[0].noise
        noise_grad = torch.autograd.grad(loss, noise_param, retain_graph=True)[0]

        self.assertIsNotNone(noise_grad, "Gradient is None for noise parameter")
        self.assertGreater(noise_grad.abs().sum().item(), 0, "Gradient has zero magnitude for noise parameter")

        loss.backward()
        for name, param in self.uncertainty_module.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient is None for parameter {name}")
