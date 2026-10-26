# Copyright 2024 Takuya Fujimura

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseClassificationHead


class SubClusterAdaCos(BaseClassificationHead):
    def __init__(
        self,
        embedding_dim: int,
        n_classes: int,
        n_subclusters: int = 1,
        trainable: bool = False,
        dynamic: bool = True,
        reduction: str = "mean",
        eps: float = 1e-7,
    ) -> None:
        """
        Args:
            embedding_dim: Number of input embedding features.
            n_classes: Number of target classes.
            n_subclusters: Number of learned centers per class.
            trainable: Whether class weights receive gradients.
            dynamic: Update the logit scale during training.
            reduction: Loss reduction: ``mean``, ``sum``, or ``none``.
            eps: Numerical stability constant.
        """
        super().__init__()
        self.n_classes = n_classes
        self.n_subclusters = n_subclusters
        self.s_init = np.sqrt(2) * np.log(n_classes * n_subclusters - 1)
        self.eps = eps

        # Weight initialization
        self.W = nn.Parameter(
            torch.Tensor(embedding_dim, n_classes * n_subclusters),
            requires_grad=trainable,
        )
        nn.init.xavier_uniform_(self.W.data)

        # Scale factor
        self.register_buffer("s", torch.tensor(self.s_init))
        self.reduction = reduction
        self.dynamic = dynamic

    def calc_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embeddings shaped ``[batch, embedding_dim]``.

        Returns:
            Unnormalized class logits shaped ``[batch, n_classes]``
        """
        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.W, p=2, dim=0)
        logits = torch.mm(x, W)
        logits *= self.s
        logits = logits.view(-1, self.n_classes, self.n_subclusters)  # (B, C, S)
        class_logits = torch.logsumexp(logits, dim=2)  # (B, C)
        return class_logits

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embeddings shaped ``[batch, embedding_dim]``.
            y: One-hot or probability targets shaped ``[batch, n_classes]``.

        Returns:
            Scalar loss for ``mean``/``sum`` or ``[batch]`` for ``none``.
        """
        y_org = y.clone()
        y = y.repeat_interleave(self.n_subclusters, dim=1)

        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.W, p=2, dim=0)
        logits = torch.mm(x, W)
        theta = torch.acos(torch.clamp(logits, -1.0 + self.eps, 1.0 - self.eps))

        if self.training and self.dynamic:
            with torch.no_grad():
                max_s_logits = torch.max(self.s * logits)
                B_avg = torch.exp(self.s * logits - max_s_logits)  # re-scaling trick
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
        logits = logits.view(-1, self.n_classes, self.n_subclusters)
        class_logits = torch.logsumexp(logits, dim=2)
        loss = F.cross_entropy(class_logits, y_org, reduction=self.reduction)
        return loss
