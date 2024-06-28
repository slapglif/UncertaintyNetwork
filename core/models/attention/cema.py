import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger




class CEMA(nn.Module):
    """
    Implements the Cumulative Effect Modulation Attention (CEMA) mechanism.

    CEMA applies a cumulative effect to the input sequence, modulated by a
    learnable positional encoding. This effect is designed to enhance the
    representation of sequential information by emphasizing the influence of
    past elements on subsequent ones.

    Args:
        embed_dim (int): The dimensionality of the input embeddings.
        ndim (int, optional): The dimensionality of the internal positional encoding.
            Defaults to 16.
        eps (float, optional): A small constant for numerical stability. Defaults to 1e-5.

    Attributes:
        embed_dim (int): The dimensionality of the input embeddings.
        ndim (int): The dimensionality of the internal positional encoding.
        eps (float): A small constant for numerical stability.
        omega (nn.Parameter): Learnable parameter for scaling input activations.
        alpha (nn.Parameter): Learnable parameter for sinusoidal component of positional encoding.
        beta (nn.Parameter): Learnable parameter for cosinusoidal component of positional encoding.
        gamma (nn.Parameter): Learnable parameter for scaling the CEMA effect.
        t (torch.Tensor): Buffer for precomputed time values used in positional encoding.
    """

    def __init__(self, embed_dim: int, ndim: int = 16, eps: float = 1e-5) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ndim = ndim
        self.eps = eps

        # Learnable parameters
        self.omega = nn.Parameter(torch.randn(embed_dim))
        self.alpha = nn.Parameter(torch.randn(ndim, embed_dim))
        self.beta = nn.Parameter(torch.randn(ndim, embed_dim))
        self.gamma = nn.Parameter(torch.ones(embed_dim))

        # Initialize t for positional encoding calculation
        self.register_buffer('t', torch.linspace(0, 2 * torch.pi, self.ndim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the CEMA modulation to the input tensor.

        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
        torch.Tensor: Output tensor with CEMA modulation applied, of the same
            shape as the input tensor.
        """
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, f"Input embed_dim ({embed_dim}) does not match self.embed_dim ({self.embed_dim})"

        # Compute positional encoding
        pos_enc = self._compute_positional_encoding(seq_len)
        logger.debug(f"Positional encoding shape: {pos_enc.shape}, min: {pos_enc.min()}, max: {pos_enc.max()}")

        # Compute CEMA effect
        omega_activated = F.softplus(self.omega)
        x_scaled = F.softplus(x * omega_activated.view(1, 1, -1))  # Ensure non-negative values
        cema_effect = torch.cumsum(x_scaled, dim=1) * pos_enc.unsqueeze(0)

        logger.debug(f"CEMA effect shape: {cema_effect.shape}, min: {cema_effect.min()}, max: {cema_effect.max()}")

        # Combine CEMA effect with input using learnable scaling
        gamma_activated = F.softplus(self.gamma)
        output = x + gamma_activated.view(1, 1, -1) * cema_effect

        logger.debug(f"Output shape: {output.shape}, min: {output.min()}, max: {output.max()}")

        return output

    def _compute_positional_encoding(self, seq_len: int) -> torch.Tensor:
        """
        Computes the positional encoding for the given sequence length.

        The positional encoding is calculated using a combination of sinusoidal
        functions with learnable parameters, ensuring each position in the
        sequence is mapped to a unique representation.

        Args:
            seq_len (int): The length of the sequence for which to compute the
                positional encoding.

        Returns:
            torch.Tensor: A tensor of shape (seq_len, embed_dim) representing
                the positional encoding for each position in the sequence.
        """
        t = self.t.view(1, -1, 1)
        alpha = self.alpha.unsqueeze(0)
        beta = self.beta.unsqueeze(0)

        pos = torch.arange(seq_len, dtype=torch.float32, device=self.t.device).view(-1, 1, 1)

        angles = pos * t
        pos_enc = torch.sum(torch.tanh(alpha * torch.sin(angles) + beta * torch.cos(angles)), dim=1)
        return F.softplus(pos_enc)  # Ensure non-negative values

