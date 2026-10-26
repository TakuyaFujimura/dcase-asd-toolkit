# Copyright 2025 Takuya Fujimura, Kevin Wilkinghoff

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseClassificationHead


class AdaProj(BaseClassificationHead):
    def __init__(
        self,
        embedding_dim: int,
        n_classes: int,
        subspace_dim: int = 32,
        trainable: bool = False,
        dynamic: bool = True,
        reduction: str = "mean",
        eps: float = 1e-7,
    ) -> None:
        """
        Args:
            embedding_dim: Number of input embedding features.
            n_classes: Number of target classes.
            subspace_dim: Number of basis vectors per class.
            trainable: Whether class weights receive gradients.
            dynamic: Update the logit scale during training.
            reduction: Loss reduction: ``mean``, ``sum``, or ``none``.
            eps: Numerical stability constant.
        """
        super().__init__()
        self.n_classes = n_classes
        self.subspace_dim = subspace_dim
        self.s_init = np.sqrt(2) * np.log(n_classes - 1)
        self.eps = eps

        # Weight initialization
        self.W = nn.Parameter(
            torch.Tensor(n_classes, subspace_dim, embedding_dim),
            requires_grad=trainable,
        )
        nn.init.xavier_uniform_(self.W.data)

        # Scale factor
        self.register_buffer("s", torch.tensor(self.s_init))
        self.reduction = reduction
        self.dynamic = dynamic

    def _projection_cosines(self, x: torch.Tensor) -> torch.Tensor:
        """Return class projection cosines before applying the adaptive scale."""
        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.W, p=2.0, dim=2)  # num_classes x subspace_dim x emb_dim
        logits = torch.tensordot(x, W, dims=[[1], [2]])  # type: ignore
        # batchsize x num_classes x subspace_dim
        x_proj = (torch.unsqueeze(logits, dim=3) * torch.unsqueeze(W, dim=0)).sum(dim=2)
        # batchsize x num_classes x emb_dim
        x_proj = F.normalize(x_proj, p=2.0, dim=2)
        logits = (torch.unsqueeze(x, dim=1) * x_proj).sum(dim=2)
        # batchsize x num_classes
        return logits

    def calc_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embeddings shaped ``[batch, embedding_dim]``.

        Returns:
            Unnormalized class logits shaped ``[batch, n_classes]``
        """
        logits = self._projection_cosines(x)
        logits *= self.s
        return logits

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embeddings shaped ``[batch, embedding_dim]``.
            y: One-hot or probability targets shaped ``[batch, n_classes]``.

        Returns:
            Scalar loss for ``mean``/``sum`` or ``[batch]`` for ``none``.
        """
        y_org = y.clone()

        logits = self._projection_cosines(x)
        theta = torch.acos(torch.clamp(logits, -1.0 + self.eps, 1.0 - self.eps))

        if self.training and self.dynamic:
            with torch.no_grad():
                max_s_logits = torch.max(self.s * logits)
                B_avg = torch.where(
                    y < 1,
                    torch.exp(self.s * logits - max_s_logits),
                    torch.exp(torch.zeros_like(self.s * logits) - max_s_logits),
                )  # re-scaling trick
                B_avg = torch.mean(torch.sum(B_avg, dim=1))
                theta_class = torch.sum(y * theta, dim=1)
                theta_med = torch.median(theta_class)
                new_s = max_s_logits + torch.log(B_avg)  # re-scaling trick
                new_s /= (
                    torch.cos(torch.minimum(torch.tensor(np.pi / 4), theta_med))
                    + self.eps
                )
                self.s.copy_(new_s)

        logits *= self.s
        loss = F.cross_entropy(logits, y_org, reduction=self.reduction)
        return loss
