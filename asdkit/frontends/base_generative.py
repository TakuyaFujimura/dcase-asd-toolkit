"""Scalar generative objectives and inference scores from shared model results."""

from abc import abstractmethod
from typing import Any, Dict

import torch

from .base_plmodel import BasePLAUCFrontend


class BaseGenerativePLFrontend(BasePLAUCFrontend):
    """Share training and inference over concrete sample-wise losses and scores."""

    @abstractmethod
    def compute_outputs(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute sample-wise objectives, scores and embeddings.

        Returns:
            loss_main: Required training objective, shape (B,).
            loss_<name>: Optional additional objectives, shape (B,).
            AS-<name>: Concrete anomaly scores for evaluation, shape (B,).
            Other keys: Optional outputs such as embed, shape (B, ...).

        All entries share the input sample axis. Do not average over the batch
        here. The concrete model defines scores independently of
        losses; identical values may reference the same tensor.
        """
        raise NotImplementedError

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Return sample-wise losses, scores and embeddings."""
        return self.compute_outputs(batch)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self.compute_outputs(batch)
        loss_dict = {
            key.removeprefix("loss_"): value.mean()
            for key, value in outputs.items()
            if key.startswith("loss_")
        }
        self._log_step_losses(
            loss_dict=loss_dict, batch_size=len(outputs["loss_main"]), stage="train"
        )
        return loss_dict["main"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        loss_outputs = self.compute_outputs(batch)

        loss_dict = {
            key.removeprefix("loss_"): value.mean()
            for key, value in loss_outputs.items()
            if key.startswith("loss_")
        }
        self._log_step_losses(
            loss_dict=loss_dict,
            batch_size=len(loss_outputs["loss_main"]),
            stage="valid",
        )
        anomaly_score_dict = {
            key: value for key, value in loss_outputs.items() if key.startswith("AS-")
        }
        self.validation_step_auc(anomaly_score_dict=anomaly_score_dict, batch=batch)
