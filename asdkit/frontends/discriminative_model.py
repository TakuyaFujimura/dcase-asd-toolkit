from pathlib import Path
from typing import Any, Dict, List, Optional, final

from torch import Tensor

from asdkit.utils.common import instantiate

from .base_discriminative import (
    BaseDisPLFrontend,
    classification_anomaly_scores,
    classification_losses,
)


class BasicDisPLFrontend(BaseDisPLFrontend):
    """Train label-specific classification heads on waveform embeddings."""

    def __init__(
        self,
        model_cfg: Dict[str, Any],
        optim_cfg: Dict[str, Any],
        lrscheduler_cfg: Optional[Dict[str, Any]] = None,
        label_dict_path: Optional[Dict[str, Path]] = None,
        save_only_trainable: bool = False,
        optimizer_log_options: Optional[Dict[str, bool]] = None,
    ) -> None:
        super().__init__(
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            lrscheduler_cfg=lrscheduler_cfg,
            label_dict_path=label_dict_path,
            save_only_trainable=save_only_trainable,
            optimizer_log_options=optimizer_log_options,
        )
        self.construct_model(**model_cfg)
        if self.save_only_trainable:
            self.save_state_names = self.collect_trainable_state_names()
            self.save_state_names.update(self.head_dict.state_dict(prefix="head_dict."))

    def compute_outputs(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            batch: Input with wave shaped (B, T).

        Returns:
            Dictionary containing embed shaped (B, D).
        """
        return {"embed": self.extractor(batch["wave"])}

    def get_metadata(self) -> Dict[str, Any]:
        return {"embeddings": {"embed": {"representation": "vector"}}}

    @final
    def construct_model(
        self,
        extractor_cfg: Dict[str, Any],
        loss_cfg: Dict[str, Any],
        label_loss_weight_dict: Dict[str, float],
        augmentation_cfg_list: Optional[List[Dict[str, Any]]] = None,
        anomaly_score_type: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            extractor_cfg: Extractor returning (B, D) embeddings and embedding_dim.
            loss_cfg: Classification loss configuration shared by all heads.
            label_loss_weight_dict: Label names mapped to objective weights.
            augmentation_cfg_list: Optional batch augmentation configurations.
            anomaly_score_type: Inference score names. Defaults to both
                one_minus_prob and neg_log_odds. Does not change loss.
        """
        if loss_cfg.get("reduction") == "none":
            raise ValueError("Scalar training losses require reduction='mean' or 'sum'")
        self.set_anomaly_score_type(anomaly_score_type)
        self.extractor = instantiate(extractor_cfg)
        self.loss_cfg = loss_cfg
        self.embedding_dim = self.extractor.embedding_dim
        self.label_loss_weight_dict = label_loss_weight_dict
        self.set_head_dict(self.label_loss_weight_dict)
        self.set_augmentations(augmentation_cfg_list or [])

    def compute_losses(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Compute the weighted classification objective using concrete embeddings."""
        outputs = self.compute_outputs(batch)
        return classification_losses(
            embedding=outputs["embed"],
            heads=self.head_dict,
            batch=batch,
            weights=self.label_loss_weight_dict,
        )

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Return embeddings and available GT scores from one model evaluation."""
        outputs = self.compute_outputs(batch)
        scores = classification_anomaly_scores(
            embedding=outputs["embed"],
            heads=self.head_dict,
            batch=batch,
            score_types=self.anomaly_score_type,
        )
        return {**outputs, **scores}
