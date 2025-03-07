# Copyright 2025 Takuya Fujimura, Kevin Wilkinghoff

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class AdaProj(nn.Module):
    def __init__(
        self,
        embed_size: int,
        n_classes: int,
        subspace_dim: int = 32,
        trainable: bool = False,
        dynamic: bool = True,
        reduction: str = "mean",
        eps: float = 1e-7,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.subspace_dim = subspace_dim
        self.s_init = np.sqrt(2) * np.log(n_classes * n_subclusters - 1)
        self.eps = eps

        # Weight initialization
        self.W = nn.Parameter(
            torch.Tensor(n_classes, subspace_dim, embed_size), requires_grad=trainable
        )
        nn.init.xavier_uniform_(self.W.data)

        # Scale factor
        self.s = nn.Parameter(torch.tensor(self.s_init), requires_grad=False)
        self.reduction = reduction
        self.dynamic = dynamic

    def calc_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.W, p=2.0, dim=2)  # num_classes x subspace_dim x emb_dim
        logits = torch.tensordot(x, W, dims=[[1],[2]])  # batchsize x num_classes x subspace_dim
        x_proj = (torch.unsqueeze(logits, dim=3)*torch.unsqueeze(W, dim=0)).sum(dim=2)  # batchsize x num_classes x emb_dim
        x_proj = F.normalize(x_proj, p=2.0, dim=2)
        logits = (torch.unsqueeze(x, dim=1)*x_proj).sum(dim=2)  # batchsize x num_classes
        logits *= self.s
        prob = F.softmax(logits, dim=1)
        logits = torch.log(prob)
        return logits

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_org = y.clone()

        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.W, p=2.0, dim=2)  # num_classes x subspace_dim x emb_dim
        logits = torch.tensordot(x, W, dims=[[1],[2]])  # batchsize x num_classes x subspace_dim
        x_proj = (torch.unsqueeze(logits, dim=3)*torch.unsqueeze(W, dim=0)).sum(dim=2)  # batchsize x num_classes x emb_dim
        x_proj = F.normalize(x_proj, p=2.0, dim=2)
        logits = (torch.unsqueeze(x, dim=1)*x_proj).sum(dim=2)  # batchsize x num_classes
        theta = torch.acos(torch.clamp(logits, -1.0 + self.eps, 1.0 - self.eps))

        if self.training and self.dynamic:
            with torch.no_grad():
                max_s_logits = torch.max(self.s * logits)
                B_avg = torch.where(y<1, torch.exp(self.s*logits-max_logits), torch.exp(torch.zeros_like(self.s*logits)-max_logits))  # re-scaling trick
                B_avg = torch.mean(torch.sum(B_avg, dim=1))
                theta_class = torch.sum(y * theta, dim=1)
                theta_med = torch.median(theta_class)
                self.s.data = max_s_logits + torch.log(B_avg)  # re-scaling trick
                self.s.data /= (
                    torch.cos(min(torch.tensor(np.pi / 4), theta_med)) + self.eps
                )

        logits *= self.s
        prob = F.softmax(logits, dim=1)
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
