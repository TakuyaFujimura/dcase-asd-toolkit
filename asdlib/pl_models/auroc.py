import torch
from sklearn.metrics import roc_auc_score


class AUROC:
    def __init__(self):
        self.y_score = []
        self.y_target = []

    def update(self, score: torch.Tensor, target: torch.Tensor) -> None:
        self.y_score.append(score)
        self.y_target.append(target)

    def compute(self) -> float:
        y_score = torch.cat(self.y_score).cpu().numpy()
        y_target = torch.cat(self.y_target).cpu().numpy()
        return roc_auc_score(y_true=y_target, y_score=y_score)  # type: ignore

    def reset(self) -> None:
        self.y_score = []
        self.y_target = []


# We don't use torchmetrics.AUROC because it applies sigmoid to anomaly scores.
# If anomaly scores are large, sigmoid will return 1.0, which is unsuitable for AUC calculation.
