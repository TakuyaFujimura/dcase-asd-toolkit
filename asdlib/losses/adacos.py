# Copyright 2024 Takuya Fujimura

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class AdaCos(nn.Module):
    def __init__(
        self,
        embed_size: int,
        n_classes: int,
        trainable: bool = False,
        dynamic: bool = True,
        reduction: str = "mean",
        eps: float = 1e-7,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.n_classes = n_classes
        self.s = np.sqrt(2) * np.log(n_classes - 1)
        self.W = nn.Parameter(
            torch.Tensor(embed_size, n_classes), requires_grad=trainable
        )
        nn.init.xavier_uniform_(self.W.data)
        self.eps = eps
        self.loss_fn = nn.CrossEntropyLoss(reduction=reduction)
        self.dynamic = dynamic

    def calc_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.W, p=2, dim=0)
        logits = torch.mm(x, W)
        logits *= self.s
        return logits

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.W, p=2, dim=0)
        logits = torch.mm(x, W)
        if self.training and self.dynamic:
            theta = torch.acos(torch.clamp(logits, -1.0 + self.eps, 1.0 - self.eps))
            with torch.no_grad():
                B_avg = torch.where(
                    y < 1, torch.exp(self.s * logits), torch.zeros_like(logits)
                )
                B_avg = torch.sum(B_avg) / x.size(0)
                theta_med = torch.median(theta[y == 1])
                self.s = torch.log(B_avg) / torch.cos(
                    torch.min(np.pi / 4 * torch.ones_like(theta_med), theta_med)
                )
        logits *= self.s
        loss = self.loss_fn(logits, y)
        return loss
