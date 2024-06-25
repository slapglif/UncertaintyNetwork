import unittest

import gpytorch
import torch

from core.models.uncertainty.uncertainty_utils import TSPEnergyFunction, GaussianProcessLayer, \
    TSPKernel, RandomWalkKernel, random_walk_distances


class TestUncertaintyUtils(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = 5  # Adjusted input dimension to match kernel setup
        self.output_dim = 5  # Output dimension should match input dimension for this test
        self.num_inducing = 8
        self.batch_size = 3
        self.seq_len1 = 6
        self.seq_len2 = 8

    def test_random_walk_kernel_shape(self):
        kernel = RandomWalkKernel(self.input_dim, n_steps=4, walk_type='standard').to(self.device)
        x1 = torch.randn(self.batch_size, self.seq_len1, self.input_dim, device=self.device)
        x2 = torch.randn(self.batch_size, self.seq_len2, self.input_dim, device=self.device)
        covar_matrix = kernel(x1, x2)
        self.assertEqual(covar_matrix.shape, torch.Size([self.batch_size, self.seq_len1, self.seq_len2]))

    def test_random_walk_kernel_positive_definite(self):
        kernel = RandomWalkKernel(self.input_dim, walk_type='standard').to(self.device)
        x = torch.randn(self.batch_size * 10, self.input_dim, device=self.device)
        covar_matrix = kernel(x, x).cpu()
        eigenvalues = torch.linalg.eigvalsh(covar_matrix)
        self.assertTrue(torch.all(eigenvalues >= 0))

    def test_random_walk_kernel_with_scale_kernel(self):
        base_kernel = RandomWalkKernel(self.input_dim, n_steps=3, walk_type='standard').to(self.device)
        scale_kernel = gpytorch.kernels.ScaleKernel(base_kernel).to(self.device)
        x1 = torch.randn(self.batch_size, self.seq_len1, self.input_dim, device=self.device)
        x2 = torch.randn(self.batch_size, self.seq_len2, self.input_dim, device=self.device)
        covar_matrix = scale_kernel(x1, x2)
        self.assertEqual(covar_matrix.shape, torch.Size([self.batch_size, self.seq_len1, self.seq_len2]))

    def test_tsp_energy_function_shape(self):
        energy_function = TSPEnergyFunction(self.input_dim, compression_dim=None).to(self.device)
        x1 = torch.randn(self.batch_size, self.seq_len1, self.input_dim, device=self.device)
        x2 = torch.randn(self.batch_size, self.seq_len2, self.input_dim, device=self.device)
        energy_matrix = energy_function(x1, x2)
        # Expect the shape to include the maximum sequence length after padding
        max_seq_len = max(self.seq_len1, self.seq_len2)
        self.assertEqual(energy_matrix.shape, torch.Size([self.batch_size, max_seq_len, max_seq_len]))

    def test_tsp_energy_function_with_compression(self):
        energy_function = TSPEnergyFunction(self.input_dim, compression_dim=16).to(self.device)
        x1 = torch.randn(self.batch_size, self.seq_len1, self.input_dim, device=self.device)
        x2 = torch.randn(self.batch_size, self.seq_len2, self.input_dim, device=self.device)
        energy_matrix = energy_function(x1, x2)
        self.assertEqual(energy_matrix.shape, torch.Size([self.batch_size, self.seq_len1, self.seq_len2]))

    def test_tsp_energy_function_gradients(self):
        energy_function = TSPEnergyFunction(self.input_dim, compression_dim=16).to(self.device)
        x1 = torch.randn(self.batch_size, self.seq_len1, self.input_dim, device=self.device, requires_grad=True)
        x2 = torch.randn(self.batch_size, self.seq_len2, self.input_dim, device=self.device, requires_grad=True)
        energy_matrix = energy_function(x1, x2)
        scalar = energy_matrix.sum()
        scalar.backward()
        self.assertIsNotNone(x1.grad)
        self.assertIsNotNone(x2.grad)

    def test_gaussian_process_layer_forward(self):
        gp_layer = GaussianProcessLayer(self.input_dim, self.output_dim, self.num_inducing).to(self.device)
        x = torch.randn(self.batch_size, self.seq_len1, self.input_dim, device=self.device)
        output_distribution = gp_layer(x)
        mean = output_distribution.mean
        variance = output_distribution.variance
        # Expect the shapes to reflect the reshaped input
        self.assertEqual(mean.shape, torch.Size([self.batch_size * self.seq_len1, self.output_dim]))
        self.assertEqual(variance.shape, torch.Size([self.batch_size * self.seq_len1, self.output_dim]))
        self.assertFalse(torch.isnan(mean).any(), "NaN values detected in GP layer mean output")
        self.assertFalse(torch.isinf(mean).any(), "Inf values detected in GP layer mean output")
        self.assertFalse(torch.isnan(variance).any(), "NaN values detected in GP layer variance output")
        self.assertFalse(torch.isinf(variance).any(), "Inf values detected in GP layer variance output")

    def test_gaussian_process_layer_training(self):
        gp_layer = GaussianProcessLayer(self.input_dim, self.output_dim, self.num_inducing).to(self.device)
        gp_layer.train()
        optimizer = torch.optim.Adam(gp_layer.parameters(), lr=0.01)
        x = torch.randn(self.batch_size, self.seq_len1, self.input_dim, device=self.device)
        y = torch.randn(self.batch_size, self.seq_len1, self.output_dim, device=self.device)
        # Reshape y to match the output of gp_layer(x)
        y = y.view(self.batch_size * self.seq_len1, self.output_dim)
        mll = gpytorch.mlls.VariationalELBO(gp_layer.likelihood, gp_layer, num_data=x.size(0) * x.size(1))

        for _ in range(5):  # Train for a few iterations
            optimizer.zero_grad()
            output = gp_layer(x)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()

        self.assertFalse(torch.isnan(gp_layer.inducing_points.grad).any(),
                         "NaN values detected in inducing points gradients")
        self.assertFalse(torch.isinf(gp_layer.inducing_points.grad).any(),
                         "Inf values detected in inducing points gradients")

    def test_gaussian_process_layer_with_tsp_kernel(self):
        energy_function = TSPEnergyFunction(self.input_dim, compression_dim=16).to(self.device)
        kernel = TSPKernel(energy_function, lengthscale=1.0).to(self.device)  # Explicitly set lengthscale
        gp_layer = GaussianProcessLayer(self.input_dim, self.output_dim, self.num_inducing, kernel=kernel).to(
            self.device)

        x = torch.randn(self.batch_size, self.seq_len1, self.input_dim, device=self.device)
        output_distribution = gp_layer(x)
        mean = output_distribution.mean
        variance = output_distribution.variance

        self.assertEqual(mean.shape, torch.Size([self.batch_size, self.seq_len1, self.output_dim]))
        self.assertEqual(variance.shape, torch.Size([self.batch_size, self.seq_len1, self.output_dim]))
        self.assertFalse(torch.isnan(mean).any(), "NaN values detected in GP layer mean output (with TSP kernel)")
        self.assertFalse(torch.isinf(mean).any(), "Inf values detected in GP layer mean output (with TSP kernel)")
        self.assertFalse(torch.isnan(variance).any(),
                         "NaN values detected in GP layer variance output (with TSP kernel)")
        self.assertFalse(torch.isinf(variance).any(),
                         "Inf values detected in GP layer variance output (with TSP kernel)")


if __name__ == '__main__':
    unittest.main()