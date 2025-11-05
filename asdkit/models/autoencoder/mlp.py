from torch import nn


class AE_MLP(nn.Module):
    """AutoEncoder for ASD"""

    def __init__(self, input_dim: int, h_dim: int, z_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.z_dim),
            nn.BatchNorm1d(self.z_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.input_dim),
        )

    def forward(self, x, condition=None):
        """
        Args:
            x: Audio feature [B, D]
            condition: Condition feature [B, D]
        Returns:
            x_est: Reconstructed audio feature [B, D]
            z: Latent feature [B, D]
        """
        z = self.encoder(x)
        x_est = self.decoder(z)
        return x_est, z
