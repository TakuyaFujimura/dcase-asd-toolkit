# Copyright 2024 Takuya Fujimura

import torch
import torch.nn.functional as F
from torch import nn


class ArcFace(nn.Module):
    def __init__(
        self,
        embed_size: int,
        n_classes: int,
        m: float = 0.5,
        s: float = 64,
        trainable: bool = False,
        reduction: str = "mean",
        eps: float = 1e-7,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.W = nn.Parameter(
            torch.Tensor(embed_size, n_classes), requires_grad=trainable
        )
        nn.init.xavier_uniform_(self.W.data)
        self.eps = eps
        self.loss_fn = nn.CrossEntropyLoss(reduction=reduction)

    def calc_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.W, p=2, dim=0)
        logits = torch.mm(x, W)
        theta = torch.acos(torch.clamp(logits, -1.0 + self.eps, 1.0 - self.eps))
        logits = self.s * torch.cos(theta + self.m)
        return logits

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): embedding
            y (torch.Tensor): onehot label

        Returns:
            torch.Tensor: CrossEntropyLoss
        """
        logits = self.calc_logits(x)
        loss = self.loss_fn(logits, y)
        return loss
