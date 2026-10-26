"""Classification training and label-available GT anomaly scoring."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, final

from torch import Tensor, nn

from asdkit.utils.common import instantiate

from .base_plmodel import BasePLFrontend


def classification_losses(
    embedding: Tensor,
    heads: nn.ModuleDict,
    batch: Dict[str, Tensor],
    weights: Dict[str, float],
) -> Dict[str, Tensor]:
    """Return scalar label losses and their weighted total; require all targets."""
    losses: Dict[str, Tensor] = {"main": embedding.new_zeros(())}
    for label, weight in weights.items():
        loss = heads[label](embedding, batch[f"onehot_{label}"])
        if loss.ndim != 0:
            raise ValueError("Scalar training losses require reduction='mean' or 'sum'")
        losses[f"label_{label}"] = loss
        losses["main"] += loss * weight
    return losses


def classification_probability_scores(
    logits: Tensor, target: Tensor, name: str
) -> Tensor:
    """Return the named score (B,) for p = sum(target * softmax(logits))."""
    log_complement_mass = (logits + (1 - target).log()).logsumexp(dim=-1)
    if name == "one_minus_prob":
        return (log_complement_mass - logits.logsumexp(dim=-1)).exp()
    if name == "neg_log_odds":
        log_target_mass = (logits + target.log()).logsumexp(dim=-1)
        return log_complement_mass - log_target_mass
    raise ValueError(f"Unknown anomaly score type: {name}")


def classification_anomaly_scores(
    embedding: Tensor,
    heads: nn.ModuleDict,
    batch: Dict[str, Tensor],
    score_types: Sequence[str],
) -> Dict[str, Tensor]:
    """Return sample-wise GT scores for available labels without updating heads."""
    if not score_types:
        return {}
    scores: Dict[str, Tensor] = {}
    for label, head in heads.items():
        target_key = f"onehot_{label}"
        if target_key not in batch:
            continue
        # Evaluate each head once, even when exporting both score types.
        logits = head.calc_logits(embedding)
        for name in score_types:
            scores[f"AS-label_{label}-{name}-gt"] = classification_probability_scores(
                logits=logits, target=batch[target_key], name=name
            )
    return scores


class BaseDisPLFrontend(BasePLFrontend):
    def set_anomaly_score_type(
        self, anomaly_score_type: Optional[List[str]] = None
    ) -> None:
        """Configure score types; None selects both and an empty list disables scores."""
        if anomaly_score_type is None:
            anomaly_score_type = ["one_minus_prob", "neg_log_odds"]
        if isinstance(anomaly_score_type, str) or any(
            name not in ("one_minus_prob", "neg_log_odds")
            for name in anomaly_score_type
        ):
            raise ValueError(
                "anomaly_score_type must be a list containing only "
                "one_minus_prob and neg_log_odds"
            )
        self.anomaly_score_type = list(anomaly_score_type)

    def set_head_dict(self, label_loss_weight_dict: Dict[str, float]) -> None:
        """
        Args:
            label_loss_weight_dict: Labels requiring classification heads.
        """
        self.head_dict = nn.ModuleDict({})
        for label_name in label_loss_weight_dict:
            self.head_dict[label_name] = instantiate(
                cfg=self.loss_cfg,
                n_classes=self.num_class_dict[label_name],
                embedding_dim=self.embedding_dim,
            )

    @final
    def set_augmentations(self, augmentation_cfg_list: List[Dict[str, Any]]) -> None:
        """
        Args:
            augmentation_cfg_list: Batch augmentation configurations, in order.
        """
        self.augmentations = nn.ModuleList(
            [instantiate(cfg) for cfg in augmentation_cfg_list]
        )

    @abstractmethod
    def compute_losses(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Return scalar objectives, including their weighted total under main."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Return inference outputs defined by the concrete frontend."""
        raise NotImplementedError

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        for augmentation in self.augmentations:
            batch = augmentation(batch)
        loss_dict = self.compute_losses(batch)
        self._log_step_losses(
            loss_dict=loss_dict, batch_size=len(batch["wave"]), stage="train"
        )
        return loss_dict["main"]

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        loss_dict = self.compute_losses(batch)
        self._log_step_losses(
            loss_dict=loss_dict, batch_size=len(batch["wave"]), stage="valid"
        )
