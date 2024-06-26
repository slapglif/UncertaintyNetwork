# test_uncertainty_utils.py
import unittest

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from loguru import logger

from core.models.uncertainty.uncertainty_utils import (
    TSPEnergyFunction,
    GaussianProcessLayer,
    TSPKernel,
    RandomWalkKernel,
)


class TestUncertaintyUtils(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len1 = 6
        self.seq_len2 = 8
        self.input_dim = 5
        self.output_dim = 5
        self.num_inducing = 8
        self.batch_size = 3
        self.seq_len = 6

    def test_gaussian_process_layer_forward(self) -> None:
        """
        Test the forward pass of the GaussianProcessLayer.

        This test checks the shape of the output distribution in both train and eval modes,
        as well as the shape of the predicted mean and variance.
        """
        logger.info("Starting test_gaussian_process_layer_forward")

        # Initialize the GaussianProcessLayer
        gp_layer: GaussianProcessLayer = GaussianProcessLayer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_inducing=self.num_inducing
        ).to(self.device)

        # Generate random input tensor
        x: torch.Tensor = torch.randn(
            self.batch_size, self.seq_len, self.input_dim, device=self.device
        )

        # Test in train mode
        gp_layer.train()
        x_flat: torch.Tensor = x.view(-1, self.input_dim)
        output_dist: MultivariateNormal = gp_layer(x_flat)

        # Check output distribution shapes
        self.assertEqual(
            output_dist.loc.shape,
            torch.Size([self.batch_size * self.seq_len, self.output_dim]),
            "Mismatch in output distribution location shape"
        )
        self.assertEqual(
            output_dist.covariance_matrix.shape,
            torch.Size([self.batch_size * self.seq_len, self.batch_size * self.seq_len]),
            "Mismatch in output distribution covariance matrix shape"
        )

        # Test in eval mode
        gp_layer.eval()
        pred_mean, pred_var = gp_layer.predict_with_uncertainty(x)

        # Check predicted mean and variance shapes
        self.assertEqual(
            pred_mean.shape,
            torch.Size([self.batch_size, self.seq_len, self.output_dim]),
            "Mismatch in predicted mean shape"
        )
        self.assertEqual(
            pred_var.shape,
            torch.Size([self.batch_size, self.seq_len, self.output_dim]),
            "Mismatch in predicted variance shape"
        )

        logger.info("Completed test_gaussian_process_layer_forward")

    def test_gaussian_process_layer_training(self):
        gp_layer = GaussianProcessLayer(
            self.input_dim, self.output_dim, self.num_inducing
        ).to(self.device)
        gp_layer.train()
        optimizer = torch.optim.Adam(gp_layer.parameters(), lr=0.01)

        x = torch.randn(
            self.batch_size, self.seq_len, self.input_dim, device=self.device
        )
        y = torch.randn(
            self.batch_size * self.seq_len, self.output_dim, device=self.device
        )  # Flatten y

        mll = gpytorch.mlls.VariationalELBO(
            gp_layer.likelihood, gp_layer, num_data=x.size(0) * x.size(1)
        )

        for _ in range(5):  # Train for a few iterations
            optimizer.zero_grad()
            x_flat = x.view(self.batch_size * self.seq_len, self.input_dim)
            output_dist = gp_layer(x_flat)  # GP layer expects flattened input
            loss = -mll(output_dist, y)
            loss.backward()
            optimizer.step()

        self.assertFalse(
            torch.isnan(gp_layer.variational_strategy.inducing_points).any(),
            "NaN values detected in inducing points",
        )
        self.assertFalse(
            torch.isinf(gp_layer.variational_strategy.inducing_points).any(),
            "Inf values detected in inducing points",
        )

        with torch.no_grad():
            pred_mean, pred_var = gp_layer.predict_with_uncertainty(x_flat)

        self.assertEqual(
            pred_mean.shape,
            torch.Size([self.batch_size * self.seq_len, self.output_dim]),
        )
        self.assertEqual(
            pred_var.shape,
            torch.Size([self.batch_size * self.seq_len, self.output_dim]),
        )

    def test_random_walk_kernel_positive_definite(self):
        kernel = RandomWalkKernel(self.input_dim, walk_type="standard").to(self.device)
        x = torch.randn(self.batch_size * 10, self.input_dim, device=self.device)
        covar_matrix = (
            kernel(x, x).evaluate().cpu()
        )  # Evaluate the kernel to get a dense matrix
        eigenvalues = torch.linalg.eigvalsh(covar_matrix)
        self.assertTrue(torch.all(eigenvalues >= 0))

    def test_tsp_energy_function_shape(self):
        energy_function = TSPEnergyFunction(self.input_dim, compression_dim=None).to(
            self.device
        )
        x1 = torch.randn(
            self.batch_size, self.seq_len, self.input_dim, device=self.device
        )
        x2 = torch.randn(
            self.batch_size, self.seq_len, self.input_dim, device=self.device
        )
        energy_matrix = energy_function(
            x1.view(self.batch_size, -1), x2.view(self.batch_size, -1)
        )  # Flatten input tensors
        self.assertEqual(
            energy_matrix.shape,
            torch.Size([self.batch_size, self.seq_len, self.seq_len]),
        )

    def test_gaussian_process_layer_with_tsp_kernel(self):
        energy_function = TSPEnergyFunction(self.input_dim, compression_dim=16).to(
            self.device
        )
        kernel = ScaleKernel(
            TSPKernel(energy_function, lengthscale=1.0).to(self.device)
        )
        gp_layer = GaussianProcessLayer(
            self.input_dim, self.output_dim, self.num_inducing, kernel=kernel
        ).to(self.device)

        x = torch.randn(
            self.batch_size, self.seq_len1, self.input_dim, device=self.device
        )
        output_distribution = gp_layer(x.view(-1, self.input_dim))
        mean = output_distribution.mean
        variance = output_distribution.variance

        expected_shape = (self.batch_size * self.seq_len1,)
        self.assertEqual(mean.shape, expected_shape)
        self.assertEqual(variance.shape, expected_shape)

        self.assertFalse(
            torch.isnan(mean).any(),
            "NaN values detected in GP layer mean output (with TSP kernel)",
        )
        self.assertFalse(
            torch.isinf(mean).any(),
            "Inf values detected in GP layer mean output (with TSP kernel)",
        )
        self.assertFalse(
            torch.isnan(variance).any(),
            "NaN values detected in GP layer variance output (with TSP kernel)",
        )
        self.assertFalse(
            torch.isinf(variance).any(),
            "Inf values detected in GP layer variance output (with TSP kernel)",
        )


if __name__ == "__main__":
    unittest.main()
