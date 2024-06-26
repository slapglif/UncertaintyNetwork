import math
from typing import Optional, Tuple, List

import gpytorch
import torch
import torch.nn.functional as F
from gpytorch.constraints import Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.priors import GammaPrior
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from loguru import logger
from torch import Tensor
from torch import nn

from core.models.kan.spline_layers import SplineNetLayer


class UncertaintyModule(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 n_gp_layers: int = 1,
                 n_inducing: int = 5,
                 dropout_rate: float = 0.1,
                 mc_samples: int = 3
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mc_samples = mc_samples

        self.gp_layers = nn.ModuleList([
            GaussianProcessLayer(input_dim, output_dim, n_inducing)
            for _ in range(n_gp_layers)
        ])
        self.mc_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # sourcery skip: low-code-quality
        """
        Forward pass of the uncertainty module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim),
                              (batch_size * seq_len, input_dim), or (input_dim,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the mean output and the uncertainty.

        Raises:
            ValueError: If the input tensor has an incorrect shape.
        """
        original_shape = x.shape
        logger.debug(f"Original input shape: {original_shape}")

        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected 1D, 2D or 3D input, but got shape {x.shape}")

        batch_size, seq_len, input_dim = x.shape
        logger.debug(f"Reshaped input: batch_size={batch_size}, seq_len={seq_len}, input_dim={input_dim}")

        if input_dim != self.input_dim:
            raise ValueError(f"Expected input dimension of {self.input_dim}, but got {input_dim}")

        x = x.view(-1, input_dim)
        logger.debug(f"Flattened input shape: {x.shape}")

        total_mean = None
        total_variance = None

        for i in range(self.mc_samples):
            x_dropout = self.mc_dropout(x)

            for j, gp_layer in enumerate(self.gp_layers):
                try:
                    gp_output = gp_layer(x_dropout)
                    logger.debug(f"MC sample {i}, GP layer {j} output type: {type(gp_output)}")

                    if isinstance(gp_output, MultivariateNormal):
                        mean = gp_output.mean
                        variance = gp_output.variance
                    elif isinstance(gp_output, torch.Tensor):
                        mean = gp_output
                        variance = torch.zeros_like(mean)
                    else:
                        raise TypeError(f"Unexpected output type from GP layer: {type(gp_output)}")

                    logger.debug(f"Mean shape: {mean.shape}, Variance shape: {variance.shape}")

                    if total_mean is None:
                        total_mean = mean
                        total_variance = variance
                    else:
                        if total_mean.shape != mean.shape:
                            raise ValueError(f"Shape mismatch: total_mean {total_mean.shape}, mean {mean.shape}")
                        total_mean += mean
                        total_variance += variance
                except Exception as e:
                    logger.error(f"Error in MC sample {i}, GP layer {j}: {str(e)}")
                    raise

        mean_output = total_mean / (len(self.gp_layers) * self.mc_samples)
        uncertainty = total_variance / (len(self.gp_layers) * self.mc_samples)

        logger.debug(f"Mean output shape before reshape: {mean_output.shape}")
        logger.debug(f"Uncertainty shape before reshape: {uncertainty.shape}")

        mean_output = mean_output.view(batch_size, seq_len, -1)
        uncertainty = uncertainty.view(batch_size, seq_len, -1)

        logger.debug(f"Final output shapes: mean={mean_output.shape}, uncertainty={uncertainty.shape}")
        return mean_output, uncertainty


class GaussianProcessLayer(ApproximateGP):
    """
    Gaussian Process Layer with data normalization and improved kernel initialization.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_inducing: int,
            kernel: Optional[Kernel] = None,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Initialize the GaussianProcessLayer.

        Args:
            input_dim (int): Dimension of the input.
            output_dim (int): Dimension of the output.
            num_inducing (int): Number of inducing points.
            kernel (Optional[gpytorch.kernels.Kernel]): Kernel to use. If None, a default RBF kernel is used.
            device (torch.device): Device to use for computation.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_inducing = num_inducing
        self.device = device

        # Initialize inducing points
        inducing_points = torch.randn(output_dim, num_inducing, input_dim, device=self.device)

        # Set up variational distribution and strategy
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([output_dim])
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        # Set up mean and covariance modules
        self.mean_module = ConstantMean(batch_shape=torch.Size([output_dim])).to(self.device)
        self.covar_module = (
            kernel
            if kernel is not None
            else ScaleKernel(
                RandomWalkKernel(input_dim, lengthscale_prior=GammaPrior(3.0, 6.0)).to(self.device),
                batch_shape=torch.Size([output_dim]),
            ).to(self.device)
        )

        # Set up likelihood
        self.likelihood = GaussianLikelihood(batch_shape=torch.Size([output_dim])).to(self.device)
        self.likelihood.noise_covar.raw_noise.data.fill_(-3)  # Initialize with low noise

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        Forward pass of the Gaussian Process Layer with data normalization and eigenvalue thresholding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size * seq_len, input_dim).

        Returns:
            gpytorch.distributions.MultivariateNormal: The output distribution.
        """

        # Normalize the input data
        x = (x - x.mean(dim=0)) / x.std(dim=0)

        # Apply eigenvalue thresholding to the inducing points covariance matrix
        with gpytorch.settings.prior_mode(True):
            induc_induc_covar = self.covar_module(self.variational_strategy.inducing_points)
            # Call _eigenvalue_thresholding on the base_kernel of the ScaleKernel
            induc_induc_covar = self.covar_module.base_kernel._eigenvalue_thresholding(induc_induc_covar)

        # The rest of the forward pass remains the same
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# todo: implement multi token inference
class MultiOutputUncertaintyModule(nn.Module):
    """
    Module for estimating uncertainty using Monte Carlo dropout and Gaussian Processes.

    Args:
        input_dim (int): The dimension of the input features.
        output_dim (int): The dimension of the output.
        n_gp_layers (int, optional): Number of Gaussian Process layers. Defaults to 2.
        n_inducing (int, optional): Number of inducing points for each GP layer. Defaults to 10.
        dropout_rate (float, optional): Dropout rate for MC dropout. Defaults to 0.1.
        mc_samples (int, optional): Number of MC samples to draw. Defaults to 5.
        temperature (float, optional): Temperature scaling factor for logits. Defaults to 1.0.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            n_gp_layers: int = 2,
            n_inducing: int = 10,
            dropout_rate: float = 0.1,
            mc_samples: int = 5,
            temperature: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_gp_layers = n_gp_layers
        self.n_inducing = n_inducing
        self.dropout_rate = dropout_rate
        self.mc_samples = mc_samples
        self.temperature = nn.Parameter(torch.tensor(temperature))

        logger.info(
            f"Initializing UncertaintyModule with {n_gp_layers} GP layers and {mc_samples} MC samples"
        )

        # Create a separate GaussianProcessLayer for each output dimension
        self.gp_layers = nn.ModuleList(
            [
                GaussianProcessLayer(
                    input_dim, 1, n_inducing
                )  # Output dimension is 1 for each layer
                for _ in range(output_dim)  # Create a layer for each output dimension
            ]
        )
        self.mc_dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(input_dim, output_dim).to(self.device).to(self.device)
        self.uncertainty_layer = nn.Linear(input_dim, output_dim).to(self.device).to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the UncertaintyModule.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Scaled mean output of shape (batch_size, seq_len, output_dim)
                - Scaled uncertainty of shape (batch_size, seq_len, output_dim)
        """
        if x.dim() != 3 or x.size(2) != self.input_dim:
            logger.error(
                f"Invalid input shape. Expected (batch_size, seq_len, {self.input_dim}), got {x.shape}"
            )
            raise ValueError(
                f"Invalid input shape. Expected (batch_size, seq_len, {self.input_dim}), got {x.shape}"
            )

        batch_size, seq_len, _ = x.shape
        device = x.device

        _total_uncertainty = torch.zeros(
            batch_size, seq_len, self.output_dim, device=device
        )
        outputs = []

        try:
            for _ in range(self.mc_samples):
                h = self.mc_dropout(x)

                # Apply GP layers to each dimension of the hidden state
                for i, gp_layer in enumerate(self.gp_layers):
                    h[:, :, i], variance = gp_layer(h[:, :, i])
                    _total_uncertainty[:, :, i] += variance

                outputs.append(self.output_layer(h))

            mean_output = torch.stack(outputs).mean(dim=0)
            mean_uncertainty = _total_uncertainty / self.mc_samples

            scaled_mean = mean_output / self.temperature
            scaled_uncertainty = mean_uncertainty / (self.temperature ** 2)

            return scaled_mean, scaled_uncertainty
        except Exception as e:
            logger.error(f"Error in UncertaintyModule forward pass: {str(e)}")
            raise


class RandomWalkKernel(Kernel):
    """
    Custom kernel implementing a random walk covariance function with numerical stability checks.

    This kernel can be used to model sequential data where the covariance
    between points depends on their distance in the sequence.

    Attributes:
        input_dim (int): Dimension of the input.
        n_steps (int): Number of steps in the random walk.
        walk_type (str): Type of random walk. Options: "standard", "reflected".
        eigenvalue_threshold (float): Minimum allowed eigenvalue for the covariance matrix.
    """

    def __init__(
            self,
            input_dim: int,
            n_steps: int = 5,
            walk_type: str = "standard",
            lengthscale_prior: Optional[torch.distributions.Distribution] = None,
            eigenvalue_threshold: float = 1e-6,
            active_dims: Optional[Tuple[int, ...]] = None,
            **kwargs,
    ):
        """
        Initialize the RandomWalkKernel.

        Args:
            input_dim (int): Dimension of the input.
            n_steps (int): Number of steps in the random walk.
            walk_type (str): Type of random walk. Options: "standard", "reflected".
            lengthscale_prior (Optional[torch.distributions.Distribution]): Prior distribution for the lengthscale parameter.
            eigenvalue_threshold (float): Minimum allowed eigenvalue for the covariance matrix.
            active_dims (Optional[Tuple[int, ...]]): See gpytorch.kernels.Kernel
            **kwargs: Additional keyword arguments for the base Kernel class.
        """
        super().__init__(active_dims=active_dims, **kwargs)
        self.input_dim = input_dim
        self.n_steps = n_steps
        self.walk_type = walk_type
        self.eigenvalue_threshold = eigenvalue_threshold

        # Use a lengthscale parameter with a prior
        self.register_parameter(
            name="raw_lengthscale", parameter=torch.nn.Parameter(torch.ones(1, input_dim))
        )
        self.register_constraint("raw_lengthscale", Positive())

        if lengthscale_prior is not None:
            self.register_prior(
                "lengthscale_prior",
                lengthscale_prior,
                lambda module: module.lengthscale,  # Corrected lambda function
                lambda module, value: module._set_lengthscale(value),
            )

    @property
    def lengthscale(self) -> Tensor:
        """
        Get the lengthscale parameter after applying the constraint.

        Returns:
            torch.Tensor: The lengthscale parameter.
        """
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value: Tensor):
        """
        Set the lengthscale parameter.

        Args:
            value (torch.Tensor): The new lengthscale value.
        """
        self._set_lengthscale(value)

    def _set_lengthscale(self, value: Tensor):
        """
        Set the lengthscale parameter after applying the constraint.

        Args:
            value (torch.Tensor): The new lengthscale value.
        """
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(
            raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value)
        )

    def forward(
            self,
            x1: Tensor,
            x2: Tensor,
            diag: bool = False,
            last_dim_is_batch: bool = False,
            **params,
    ) -> Tensor:
        """
        Compute the kernel matrix between inputs x1 and x2 with numerical stability checks.

        Args:
            x1 (torch.Tensor): First input tensor.
            x2 (torch.Tensor): Second input tensor.
            diag (bool): If True, return only the diagonal of the kernel matrix.
            last_dim_is_batch (bool): If True, the last dimension of the input is treated as the batch dimension.
            **params: Additional parameters.

        Returns:
            torch.Tensor: The computed kernel matrix.
        """
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)

        # Use squared Euclidean distance instead of Manhattan distance
        diff = self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params)

        if self.walk_type == "reflected":
            diff = self._reflect_distances(diff, 10.0)

        # Check for NaN or Inf in the distances
        if torch.isnan(diff).any() or torch.isinf(diff).any():
            logger.warning("NaN or Inf detected in computed distances. Replacing with small values.")
            diff = torch.where(torch.isnan(diff) | torch.isinf(diff), torch.tensor(1e-6), diff)

        return diff

    def _eigenvalue_thresholding(self, covar_matrix: Tensor) -> Tensor:
        """
        Performs eigenvalue thresholding on the covariance matrix.

        Args:
            covar_matrix (torch.Tensor): The covariance matrix.

        Returns:
            torch.Tensor: The covariance matrix with eigenvalues clipped to the threshold.
        """
        eigenvalues, eigenvectors = torch.linalg.eigh(covar_matrix)
        eigenvalues = torch.clamp(eigenvalues, min=self.eigenvalue_threshold)
        return torch.matmul(torch.matmul(eigenvectors, torch.diag(eigenvalues)), eigenvectors.t())

    @classmethod
    def _reflect_distances(cls, distances: Tensor, clip_value: float) -> Tensor:
        """
        Reflect distances exceeding the clip value.

        Args:
            distances (torch.Tensor): The distance matrix.
            clip_value (float): The value at which to reflect the distances.

        Returns:
            torch.Tensor: The distance matrix with reflected values.
        """
        exceeding_mask = distances > clip_value
        distances[exceeding_mask] = 2 * clip_value - distances[exceeding_mask]
        return distances


class SplineCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, num_grids: int = 8):
        if hidden_dims is None:
            hidden_dims = [128, 64]
        super().__init__()
        self.x_layers = nn.ModuleList()
        self.y_layers = nn.ModuleList()
        current_dim = input_dim  # Input dimension for each SplineNetLayer
        for hidden_dim in hidden_dims:
            self.x_layers.append(
                SplineNetLayer(current_dim, hidden_dim, num_grids=num_grids)
            )
            self.y_layers.append(
                SplineNetLayer(current_dim, hidden_dim, num_grids=num_grids)
            )
            current_dim = hidden_dim
        self.final_layer = nn.Linear(
            hidden_dims[-1] * 2, 1
        ).to(self.device)  # Concatenate outputs of x and y layers

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Forward pass through the SplineCritic network.

        Args:
            x (Tensor): The first input tensor of shape (batch_size, seq_len_x, input_dim).
            y (Tensor): The second input tensor of shape (batch_size, seq_len_y, input_dim).

        Returns:
            Tensor: The output tensor of shape (batch_size, 1).
        """
        batch_size = x.size(0)  # Assuming x and y have the same batch size
        seq_len_x = x.size(1)
        seq_len_y = y.size(1)

        # Reshape x to (batch_size, seq_len_x, 1, input_dim)
        x = x.unsqueeze(2)
        # Reshape y to (batch_size, 1, seq_len_y, input_dim)
        y = y.unsqueeze(1)

        # Apply SplineNet layers to x
        for layer_x in self.x_layers:
            x = layer_x(x)  # x now has shape (batch_size, seq_len_x, 1, hidden_dim[-1])

        # Apply SplineNet layers to y (separate loop)
        for layer_y in self.y_layers:
            y = layer_y(y)  # y now has shape (batch_size, 1, seq_len_y, hidden_dim[-1])

        # Reshape x and y to (batch_size, seq_len_x, seq_len_y, hidden_dim[-1])
        x = x.view(batch_size, seq_len_x, 1, -1)
        y = y.view(batch_size, 1, seq_len_y, -1)

        # Expand x along the sequence dimension of y
        x_expanded = x.expand(-1, -1, seq_len_y, -1)
        # Expand y along the sequence dimension of x
        y_expanded = y.expand(-1, seq_len_x, -1, -1)

        # Concatenate transformed x and y representations
        combined = torch.cat(
            [x_expanded, y_expanded], dim=-1
        )  # (batch_size, seq_len_x, seq_len_y, 2 * hidden_dim[-1])

        # Reshape to (batch_size, seq_len_x * seq_len_y, 2 * hidden_dim[-1])
        combined = combined.view(batch_size, -1, 2 * self.x_layers[-1].output_dim)

        # Apply the final linear layer
        output = self.final_layer(combined)  # (batch_size, seq_len_x * seq_len_y, 1)

        # Average across the combined sequence dimension and keep the last dimension
        output = output.mean(dim=1, keepdim=True)  # (batch_size, 1, 1)

        # Squeeze the last dimension to get (batch_size, 1)
        output = output.squeeze(-1)

        return output


class TSPEnergyFunction(nn.Module):
    """
    Energy function for the Traveling Salesman Problem (TSP) using neural networks.

    This class implements an energy function that can be used to estimate the "cost"
    or "energy" between pairs of embeddings, which is useful for solving TSP-like problems.

    Attributes:
        embedding_dim (int): Dimension of the input embeddings.
        compression_dim (Optional[int]): Dimension for compressed representations.
        lambda_ib (float): Weight for the information bottleneck term.
        device (torch.device): Device to use for computation.
        compressor (Optional[nn.Linear]): Linear layer for compressing input if compression_dim is specified.
        mi_estimator (Optional[MutualInformationEstimator]): Estimator for mutual information if compression is used.
    """

    def __init__(
            self,
            embedding_dim: int,
            compression_dim: Optional[int] = None,
            lambda_ib: float = 0.01,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        """
        Initialize the TSPEnergyFunction.

        Args:
            embedding_dim (int): Dimension of the input embeddings.
            compression_dim (Optional[int]): Dimension for compressed representations. If None, no compression is applied.
            lambda_ib (float): Weight for the information bottleneck term.
            device (torch.device): Device to use for computation.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.compression_dim = compression_dim
        self.lambda_ib = lambda_ib
        self.device = device

        if compression_dim is not None:
            self.compressor = nn.Linear(embedding_dim, compression_dim).to(self.device)
            self.mi_estimator = MutualInformationEstimator(compression_dim)
        else:
            self.compressor = None
            self.mi_estimator = None

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Computes the energy between two sets of embeddings, potentially with an added mutual information term.

        Args:
            x1 (torch.Tensor): First set of embeddings, of shape (batch_size, seq_len1, embedding_dim).
            x2 (torch.Tensor): Second set of embeddings, of shape (batch_size, seq_len2, embedding_dim).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, seq_len1, seq_len2) representing the pairwise energies.

        Raises:
            ValueError: If the input tensors have incorrect shapes or types.
        """
        if not isinstance(x1, torch.Tensor) or not isinstance(x2, torch.Tensor):
            raise ValueError(f"Inputs must be tensors. Got types x1: {type(x1)}, x2: {type(x2)}")

        if x1.dim() != 3 or x2.dim() != 3:
            raise ValueError(f"Inputs must be 3D tensors. Got shapes x1: {x1.shape}, x2: {x2.shape}")

        if x1.size(2) != self.embedding_dim or x2.size(2) != self.embedding_dim:
            raise ValueError(f"Input embedding dimension mismatch. Expected {self.embedding_dim}, "
                             f"got x1: {x1.size(2)}, x2: {x2.size(2)}")

        # Compute pairwise distances
        x1_norm = x1 / x1.norm(dim=-1, keepdim=True)
        x2_norm = x2 / x2.norm(dim=-1, keepdim=True)
        distances = 1 - torch.bmm(x1_norm, x2_norm.transpose(-2, -1))

        # Add mutual information term if compression dimension is specified
        if self.compression_dim is not None:
            x1_compressed = self.compressor(x1)
            x2_compressed = self.compressor(x2)
            mi_estimate = self.mi_estimator(x1_compressed, x2_compressed)
            return distances + self.lambda_ib * mi_estimate.unsqueeze(1).unsqueeze(2)

        return distances


class TSPKernel(gpytorch.kernels.Kernel):
    def __init__(self, energy_function: TSPEnergyFunction, **kwargs):
        super().__init__(**kwargs)
        self.energy_function = energy_function

    def forward(
            self,
            x1: Tensor,
            x2: Tensor,
            diag: bool = False,
            last_dim_is_batch: bool = False,
            **params,
    ) -> Tensor:
        logger.debug(f"TSPKernel input shapes: x1 {x1.shape}, x2 {x2.shape}")

        # Reshape inputs to 3D tensors if necessary
        if x1.dim() == 2:
            x1 = x1.unsqueeze(1)
        if x2.dim() == 2:
            x2 = x2.unsqueeze(1)

        energy = self.energy_function(x1, x2)
        if energy is None:
            raise ValueError(
                f"Energy function returned None for inputs: x1 shape {x1.shape}, x2 shape {x2.shape}"
            )
        if diag:
            return torch.exp(-energy.diag())
        output = torch.exp(-energy)
        logger.debug(f"TSPKernel output shape: {output.shape}")
        return output


class MutualInformationEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, device='cuda'):
        self.device = device
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim).to(self.device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim).to(self.device),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1).to(self.device),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Estimates the mutual information between x and y using MINE.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size_x, input_dim) or (batch_size_x, seq_len_x, input_dim).
            y (torch.Tensor): Input tensor of shape (batch_size_y, input_dim) or (batch_size_y, seq_len_y, input_dim).

        Returns:
            torch.Tensor: Estimated mutual information scalar.
        """
        if x.dim() == 2:
            batch_size_x, input_dim = x.shape
            seq_len_x = 1
        elif x.dim() == 3:
            batch_size_x, seq_len_x, input_dim = x.shape
        else:
            raise ValueError(f"Input tensor x should be 2D or 3D, got {x.dim()}D")

        if y.dim() == 2:
            batch_size_y, input_dim_y = y.shape
            seq_len_y = 1
        elif y.dim() == 3:
            batch_size_y, seq_len_y, input_dim_y = y.shape
        else:
            raise ValueError(f"Input tensor y should be 2D or 3D, got {y.dim()}D")

        assert (
                input_dim == input_dim_y
        ), f"Input dimensions of x and y should match, got {input_dim} and {input_dim_y}"

        # Ensure x and y have the same sequence length
        min_seq_len = min(seq_len_x, seq_len_y)
        if seq_len_x != min_seq_len:
            x = x[:, :min_seq_len, :]
        if seq_len_y != min_seq_len:
            y = y[:, :min_seq_len, :]

        # Flatten x and y for critic input
        x_flat = x.view(-1, input_dim)
        y_flat = y.view(-1, input_dim)

        # Repeat or truncate x and y to match the larger size
        size_x = x_flat.size(0)
        size_y = y_flat.size(0)
        if size_x < size_y:
            x_flat = x_flat.repeat(math.ceil(size_y / size_x), 1)[:size_y]
        elif size_y < size_x:
            y_flat = y_flat.repeat(math.ceil(size_x / size_y), 1)[:size_x]

        # Positive samples: concatenate x and y
        xy = torch.cat((x_flat, y_flat), dim=-1)

        # Negative samples: shuffle y and concatenate with x
        y_shuffled = y_flat[torch.randperm(y_flat.size(0))]
        xy_shuffled = torch.cat((x_flat, y_shuffled), dim=-1)

        # Calculate critic scores
        t_xy = self.critic(xy)
        t_xy_shuffled = self.critic(xy_shuffled)

        return t_xy.mean() - (torch.exp(t_xy_shuffled).mean() + math.log(2))


class MCDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.p, training=True)


class HeteroscedasticOutput(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mean_output = nn.Linear(input_dim, output_dim).to(self.device)
        self.var_output = nn.Linear(input_dim, output_dim).to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_output(x)
        log_var = self.var_output(x)
        return mean, torch.exp(log_var)


class ECELoss(nn.Module):
    def __init__(self, n_bins: int = 15):
        super().__init__()
        self.n_bins = n_bins

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, dim=1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin in torch.linspace(0, 1, self.n_bins + 1):
            in_bin = confidences.gt(bin) & confidences.le(bin + 1.0 / self.n_bins)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class UncertaintyAwareLoss(nn.Module):
    def __init__(self, base_loss: str = 'cross_entropy', uncertainty_weight: float = 0.1):
        super().__init__()
        self.base_loss = base_loss
        self.uncertainty_weight = uncertainty_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, uncertainties: torch.Tensor) -> torch.Tensor:
        if self.base_loss == 'cross_entropy':
            base_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
        else:
            raise ValueError(f"Unsupported base loss: {self.base_loss}")

        # Reshape uncertainties to match base_loss shape
        uncertainties = uncertainties.view(-1)

        return torch.mean(
            base_loss * torch.exp(-uncertainties)
            + self.uncertainty_weight * uncertainties
        )