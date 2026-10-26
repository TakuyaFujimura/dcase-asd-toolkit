# Copyright 2024 Takuya Fujimura

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseClassificationHead


class ArcFace(BaseClassificationHead):
    def __init__(
        self,
        embedding_dim: int,
        n_classes: int,
        m: float = 0.5,
        s: float = 64,
        trainable: bool = False,
        reduction: str = "mean",
        eps: float = 1e-7,
    ) -> None:
        """
        Args:
            embedding_dim: Number of input embedding features.
            n_classes: Number of target classes.
            m: Angular margin added to cosine angles.
            s: Logit scale.
            trainable: Whether class weights receive gradients.
            reduction: Loss reduction: ``mean``, ``sum``, or ``none``.
            eps: Numerical stability constant.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.W = nn.Parameter(
            torch.Tensor(embedding_dim, n_classes), requires_grad=trainable
        )
        nn.init.xavier_uniform_(self.W.data)
        self.eps = eps
        self.loss_fn = nn.CrossEntropyLoss(reduction=reduction)

    def calc_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embeddings shaped ``[batch, embedding_dim]``.

        Returns:
            Class scores shaped ``[batch, n_classes]``.
        """
        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.W, p=2, dim=0)
        logits = torch.mm(x, W)
        theta = torch.acos(torch.clamp(logits, -1.0 + self.eps, 1.0 - self.eps))
        logits = self.s * torch.cos(theta + self.m)
        return logits

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embeddings shaped ``[batch, embedding_dim]``.
            y: One-hot or probability targets shaped ``[batch, n_classes]``.

        Returns:
            Scalar loss for ``mean``/``sum`` or ``[batch]`` for ``none``.
        """
        logits = self.calc_logits(x)
        loss = self.loss_fn(logits, y)
        return loss
