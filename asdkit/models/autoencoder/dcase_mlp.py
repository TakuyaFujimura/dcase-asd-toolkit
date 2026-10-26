import torch
from torch import nn


class DCASEMLPEncoder(nn.Module):
    """Encoder for DCASE ASD AutoEncoder"""

    def __init__(
        self,
        input_size: int,
        h_dim: int,
        z_dim: int,
    ) -> None:
        super().__init__()
        if type(input_size) is not int:
            raise ValueError("input_size must be int")
        self.input_size = input_size
        output_size = z_dim

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, h_dim),
            nn.BatchNorm1d(h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(h_dim, output_size),
            nn.BatchNorm1d(output_size, momentum=0.01, eps=1e-03),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature [B, input_size]
        Returns:
            z: Latent feature [B, z_dim]
        """
        return self.layers(x)


class DCASEMLPDecoder(nn.Module):
    """Decoder for DCASE ASD AutoEncoder"""

    def __init__(
        self,
        output_size: int,
        h_dim: int,
        z_dim: int,
    ) -> None:
        super().__init__()
        if type(output_size) is not int:
            raise ValueError("output_size must be int")
        self.output_size = output_size

        self.layers = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.BatchNorm1d(h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(h_dim, self.output_size),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent feature [B, z_dim]
        Returns:
            x_est: Reconstructed feature [B, output_size]
        """
        return self.layers(z)
