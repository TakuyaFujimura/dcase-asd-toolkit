# Copyright 2024 Takuya Fujimura

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SCAdaCos(nn.Module):
    def __init__(
        self,
        embed_size: int,
        n_classes: int,
        n_subclusters: int = 1,
        trainable: bool = False,
        dynamic: bool = True,
        reduction: str = "mean",
        eps: float = 1e-7,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_subclusters = n_subclusters
        self.s_init = np.sqrt(2) * np.log(n_classes * n_subclusters - 1)
        self.eps = eps

        # Weight initialization
        self.W = nn.Parameter(
            torch.Tensor(embed_size, n_classes * n_subclusters), requires_grad=trainable
        )
        nn.init.xavier_uniform_(self.W.data)

        # Scale factor
        self.s = nn.Parameter(torch.tensor(self.s_init), requires_grad=False)
        self.reduction = reduction
        self.dynamic = dynamic

    def calc_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.W, p=2, dim=0)
        logits = torch.mm(x, W)
        logits *= self.s
        prob = F.softmax(logits, dim=1)
        prob = prob.view(-1, self.n_classes, self.n_subclusters)
        # (B, C, n_subclusters)
        prob = torch.sum(prob, dim=2)  # (B, C)
        logits = torch.log(prob)
        # softmax(logits) = softmax(log(softmax(logits)))
        #                 = softmax(log(prob))
        return logits

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_org = y.clone()
        y = y.repeat(1, self.n_subclusters)

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
                self.s.data = max_s_logits + torch.log(B_avg)  # re-scaling trick
                self.s.data /= (
                    torch.cos(torch.minimum(torch.tensor(np.pi / 4), theta_med))
                    + self.eps
                )

        logits *= self.s
        prob = F.softmax(logits, dim=1)
        prob = prob.view(-1, self.n_classes, self.n_subclusters)
        # (B, C, n_subclusters)
        prob = torch.sum(prob, dim=2)  # (B, C)
        loss = torch.sum(-torch.log(prob) * y_org, dim=-1)  # (B)
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            pass
        else:
            raise NotImplementedError()
        return loss
