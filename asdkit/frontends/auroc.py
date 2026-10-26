from typing import Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


class AUROC:
    def __init__(self):
        self.y_score = []
        self.y_target = []

    def update(self, score: torch.Tensor, target: torch.Tensor) -> None:
        self.y_score.append(score)
        self.y_target.append(target)

    def compute(self) -> Optional[float]:
        if len(self.y_score) == 0 or len(self.y_target) == 0:
            return None

        y_score = torch.cat(self.y_score).detach().cpu().numpy()
        y_target = torch.cat(self.y_target).detach().cpu().numpy()
        # y_target is normal / anomaly labels

        if np.unique(y_target).size < 2:
            return None
        else:
            return roc_auc_score(y_true=y_target, y_score=y_score)  # type: ignore

    def reset(self) -> None:
        self.y_score = []
        self.y_target = []


# We don't use torchmetrics.AUROC because it applies sigmoid to anomaly scores.
# For large anomaly scores, sigmoid saturates at 1.0, making it unsuitable for AUC calculation.
