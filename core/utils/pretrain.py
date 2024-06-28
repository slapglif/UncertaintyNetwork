import os
from typing import Tuple, List, Optional

import torch
import gpytorch
import numpy as np
from gpytorch.constraints import Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, ScaleKernel, RBFKernel, MaternKernel, LinearKernel, PeriodicKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.priors import GammaPrior
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from loguru import logger
from torch import Tensor
from torch import nn
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

from core.models.uncertainty.uncertainty import (
    UncertainTransformerConfig,  # Import your uncertainty network configuration
    UncertainTransformerLMHeadModel,  # Import your uncertainty network
)
from core.models.uncertainty.uncertainty_layers import (  # Import your custom kernels
    TSPKernel,
    TSPEnergyFunction,
)

# Set CUDA_LAUNCH_BLOCKING for debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Device to use for testing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Synthetic Text Data Generation ---
def generate_synthetic_text_data(num_samples: int, max_length: int, model_name="gpt2-small"):
    """Generates synthetic text data using a pretrained language model.

    Args:
        num_samples (int): The number of text samples to generate.
        max_length (int): The maximum length of each text sample.
        model_name (str): The name of the pretrained language model to use (e.g., "gpt2-small").

    Returns:
        List[str]: A list of generated text samples.
    """
    logger.info(f"✨ Generating synthetic text data using {model_name}...")

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    texts = []
    with tqdm(total=num_samples, desc="Generating Text", leave=False) as pbar:
        for _ in range(num_samples):
            input_ids = tokenizer.encode("The", add_special_tokens=True, return_tensors="pt").to(
                DEVICE)  # Start with a prompt
            outputs = model.generate(input_ids, max_length=max_length, do_sample=True, top_k=50, top_p=0.95)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            texts.append(text)
            pbar.update(1)

    logger.info(f"🎉 Synthetic text data generation complete! 🎉")
    return texts


# --- Preprocessing ---
def preprocess_text(text: str) -> str:
    """Cleans up generated text by removing redundant words and punctuation.

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The cleaned-up text.
    """
    # --- Add your desired preprocessing logic here ---
    text = text.replace("The The", "The")  # Remove duplicate "The"
    text = text.replace("The the", "The")  # Remove "The the"
    text = text.replace("  ", " ")  # Remove extra spaces
    text = text.strip()  # Remove leading and trailing spaces
    return text


# --- Tokenization and Embedding ---
def preprocess_text_data(texts: List[str], tokenizer: GPT2Tokenizer, embedding_dim: int) -> Tuple[Tensor, Tensor]:
    """Tokenizes and embeds text data for use with Gaussian Process.

    Args:
        texts (List[str]): A list of text samples.
        tokenizer (GPT2Tokenizer): The tokenizer to use for encoding.
        embedding_dim (int): The desired dimension for the embeddings.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the input tensors (tokenized and embedded) and target tensors.
    """
    logger.info("✨ Processing text data... ✨")

    inputs = []
    targets = []
    with tqdm(total=len(texts), desc="Processing Text", leave=False) as pbar:
        for text in texts:
            # Clean up the text
            text = preprocess_text(text)

            # Encode the text into token IDs
            input_ids = tokenizer.encode(text, add_special_tokens=True)

            # Create an embedding using a random tensor (replace with your desired embedding method)
            embedding = torch.randn(embedding_dim)

            # Store the token IDs and embedding
            inputs.append(input_ids)
            targets.append(embedding)
            pbar.update(1)

    logger.info(f"🎉 Text data processing complete! 🎉")

    # Convert to tensors for use with Gaussian Process
    inputs = torch.tensor(inputs, dtype=torch.long).to(DEVICE)  # Shape: (num_samples, seq_length)
    targets = torch.stack(targets).to(DEVICE)  # Shape: (num_samples, embedding_dim)

    return inputs, targets


# --- Gaussian Process Layer ---
class GaussianProcessLayer(ApproximateGP):
    """Gaussian Process Layer with data normalization and eigenvalue thresholding."""

    def __init__(
            self,
            input_dim: int,  # This should now be the embedding dimension
            output_dim: int,
            num_inducing: int,
            kernel: Optional[Kernel] = None,
            device: torch.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
    ):
        """
        Initialize the GaussianProcessLayer.

        Args:
            input_dim (int): Dimension of the input (embedding dimension).
            output_dim (int): Dimension of the output.
            num_inducing (int): Number of inducing points.
            kernel (Optional[Kernel]): Kernel to use. If None, uses RandomWalkKernel with cosine similarity.
            device (torch.device): Device to use for computation.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_inducing = num_inducing
        self.device = device

        inducing_points = torch.randn(num_inducing, input_dim, device=self.device)
        logger.info(f"Inducing points shape: {inducing_points.shape}")

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=torch.Size([output_dim])
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = ConstantMean(batch_shape=torch.Size([output_dim])).to(
            self.device
        )
        self.noise = nn.Parameter(torch.tensor([-5.0], device=self.device))

        # --- Modification:  Set up all kernels and apply scaling ---
        self.covar_module = ScaleKernel(kernel).to(self.device)
        # --------------------------------------------------------

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        Forward pass of the Gaussian Process Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size * seq_len, embedding_dim).

        Returns:
            gpytorch.distributions.MultivariateNormal: The output distribution.
        """
        logger.info(f"Input tensor shape: {x.shape}")
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D")

        batch_size_seq_len, _ = x.shape

        # Normalize the input data
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
        logger.info(f"Normalized input tensor shape: {x.shape}")

        # Apply eigenvalue thresholding to the inducing points covariance matrix
        with gpytorch.settings.prior_mode(True):
            induc_induc_covar = self.covar_module(
                self.variational_strategy.inducing_points
            )
            logger.info(f"Inducing points covariance matrix shape: {induc_induc_covar.shape}")

        # Compute the mean and covariance
        mean_x = self.mean_module(x)
        logger.info(f"Mean tensor shape: {mean_x.shape}")

        # Apply noise after scaling
        covar_x = self.covar_module(x)
        noise_variance = self.noise
        expanded_noise_variance = noise_variance.unsqueeze(0).unsqueeze(-1).expand(-1, covar_x.size(-2),
                                                                                   covar_x.size(-1))
        covar_x = covar_x + expanded_noise_variance

        logger.info(f"Covariance matrix shape: {covar_x.shape}")

        return MultivariateNormal(mean_x, covar_x)


# --- Uncertainty Module ---
class UncertaintyModule(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, n_gp_layers: int = 1, n_inducing: int = 5,
                 dropout_rate: float = 0.1, mc_samples: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mc_samples = mc_samples

        self.gp_layers = nn.ModuleList([
            GaussianProcessLayer(input_dim, output_dim, n_inducing) for _ in range(n_gp_layers)
        ])
        self.mc_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0)

        batch_size, seq_len, _ = x.shape
        x = x.view(-1, self.input_dim)

        total_mean = 0
        total_variance = 0

        for _ in range(self.mc_samples):
            x_dropout = self.mc_dropout(x)
            for gp_layer in self.gp_layers:
                # --- Modification: Get the full distribution and then compute mean and variance ---
                gp_output = gp_layer(x_dropout)
                mean = gp_output.mean
                variance = gp_output.variance
                # ---------------------------------------------------------------------------------
                total_mean += mean
                total_variance += variance + mean.pow(2)

        mean_output = total_mean / (len(self.gp_layers) * self.mc_samples)
        uncertainty = (total_variance / (len(self.gp_layers) * self.mc_samples)) - mean_output.pow(2)

        mean_output = mean_output.view(batch_size, seq_len, -1)
        uncertainty = uncertainty.view(batch_size, seq_len, -1)

        if original_shape == mean_output.shape[1:]:
            mean_output = mean_output.squeeze(0)
            uncertainty = uncertainty.squeeze(0)

        return mean_output, uncertainty


# --- Main Script ---
if __name__ == "__main__":
    logger.info("🚀 Starting Gaussian Process Kernel Benchmarking! 🚀")

    # --- Configure Hyperparameters ---
    embedding_dim = 64
    num_inducing = 5
    num_samples = 500
    learning_rate = 1e-3
    num_epochs = 10
    # --------------------------------------

    # --- Define Kernels ---
    rbf_kernel = RBFKernel(ard_num_dims=embedding_dim).to(DEVICE)
    matern_kernel = MaternKernel(nu=1.5, ard_num_dims=embedding_dim).to(DEVICE)
    linear_kernel = LinearKernel(ard_num_dims=embedding_dim).to(DEVICE)
    periodic_kernel = PeriodicKernel(ard_num_dims=embedding_dim).to(DEVICE)
    tsp_kernel = TSPKernel(energy_function=TSPEnergyFunction(embedding_dim)).to(DEVICE)
    # -------------------------

    # --- Generate Synthetic Text Data ---
    texts = generate_synthetic_text_data(num_samples, max_length=100)
    # --------------------------------------

    # --- Tokenize and Embed Data ---
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-small")
    inputs, targets = preprocess_text_data(texts, tokenizer, embedding_dim)
    # ------------------------------------

    logger.info("✨ Kernels: ✨")
    for kernel in [rbf_kernel, matern_kernel, linear_kernel, periodic_kernel, tsp_kernel]:
        logger.info(f"   - {type(kernel).__name__}")

    # --- Benchmark Kernels ---
    logger.info("🚀 Benchmarking Kernels: 🚀")
    for kernel in [rbf_kernel, matern_kernel, linear_kernel, periodic_kernel, tsp_kernel]:
        logger.info(f"✨ Testing kernel: {type(kernel).__name__} ✨")

        # ---  Set up Uncertainty Network  ---
        config = UncertainTransformerConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=embedding_dim,  # Match embedding dimension
            n_heads=2,
            d_ff=2 * embedding_dim,
            n_layers=1,
            dropout=0.1,
            max_position_embeddings=1024,
            pad_token_id=tokenizer.pad_token_id,
            use_mamba=True,
            d_state=8,
            d_conv=2,
            expand_factor=1.5,
            dt_rank=8,
            n_inducing=5,
            uncertainty_weight=0.1,
        )
        uncertainty_network = UncertainTransformerLMHeadModel(config).to(DEVICE)  # Load your model
        uncertainty_network.eval()

        # --- Create the GaussianProcessLayer ---
        gp_layer = GaussianProcessLayer(
            input_dim=embedding_dim,  # Use embedding dimension as input dimension
            output_dim=output_dim,
            num_inducing=num_inducing,
            kernel=kernel,
        ).to(DEVICE)

        # ---  Set up Pipeline  ---
        generator = pipeline("text-generation", model=uncertainty_network, device=DEVICE)  # Load your model
        prompt_template = "The {} was so {} that {}."  # Example prompt template
        # -------------------------

        # ---  Train the model  ---
        gp_layer.train()
        optimizer = torch.optim.Adam(gp_layer.parameters(), lr=learning_rate)
        for _ in range(num_epochs):
            optimizer.zero_grad()
            # ---  Generate Text using Pipeline and Model  ---
            generated_texts = generator(
                prompt_template.format("cat", "fluffy", "everyone loved it"),
                max_length=100,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )
            text = generated_texts[0]["generated_text"]

            # Preprocess and encode the text
            text = preprocess_text(text)
            input_ids = tokenizer.encode(text, add_special_tokens=True)

            # ---  Convert to Tensor ---
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
            embedding = torch.randn(embedding_dim)
            target_tensor = torch.tensor([embedding], dtype=torch.float).to(DEVICE)
            # --------------------------

            output = gp_layer(target_tensor)  # Use target embeddings as input to GP
            loss = -gp_layer.likelihood(output, input_tensor).log_prob().mean()
            loss.backward()
            optimizer.step()

        # ---  Evaluate the model  ---
        gp_layer.eval()
        with torch.no_grad():
            predictions = gp_layer(targets)
            mse = torch.mean((predictions.mean - inputs) ** 2)
            logger.info(f"  Mean Squared Error: {mse.item()}")
            # Add other metrics you want to benchmark here (e.g., accuracy, perplexity)

    logger.info("🎉 Kernel benchmarking complete! 🎉")