from pathlib import Path
from typing import Any, Dict, List, Optional, final

import torch
from torch import Tensor

from asdkit.models.pretrained_models.heads import BaseHead
from asdkit.models.pretrained_models.wrapper import PretrainedWrapper
from asdkit.utils.common import instantiate

from .base_discriminative import (
    BaseDisPLFrontend,
    classification_anomaly_scores,
    classification_losses,
)


class PretrainedDisPLFrontend(BaseDisPLFrontend):
    """Train a discriminative head using pretrained features."""

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
            self.save_state_names.update(
                self.embedding_head.state_dict(prefix="embedding_head.")
            )

    @final
    def construct_model(
        self,
        extractor_cfg: Dict[str, Any],
        embedding_head_cfg: Dict[str, Any],
        loss_cfg: Dict[str, Any],
        label_loss_weight_dict: Dict[str, float],
        head_input_mode: Optional[str] = None,
        augmentation_cfg_list: Optional[List[Dict[str, Any]]] = None,
        anomaly_score_type: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            extractor_cfg: PretrainedWrapper configuration.
            embedding_head_cfg: BaseHead mapping pretrained features to (B, D) vectors.
            loss_cfg: Classification loss configuration shared by all label heads.
            label_loss_weight_dict: Label names mapped to objective weights.
            head_input_mode: Optional feature layout passed to transform_features.
            augmentation_cfg_list: Optional batch augmentation configurations.
            anomaly_score_type: Inference score names; defaults to both
                one_minus_prob and neg_log_odds. An empty list disables scores.
        """
        if loss_cfg.get("reduction") == "none":
            raise ValueError("Scalar training losses require reduction='mean' or 'sum'")
        self.set_anomaly_score_type(anomaly_score_type)
        self.extractor = instantiate(extractor_cfg)
        if not isinstance(self.extractor, PretrainedWrapper):
            raise TypeError("extractor_cfg must instantiate PretrainedWrapper")
        self.embedding_representation = self.extractor.embedding_representation
        self.transform_features = self.extractor.transform_features
        self.head_input_mode = head_input_mode
        self.embedding_head: BaseHead = instantiate(
            cfg=embedding_head_cfg,
            input_dim=embedding_head_cfg.get("input_dim", self.extractor.embedding_dim),
        )
        self.embedding_dim = self.embedding_head.embedding_dim
        self.loss_cfg = loss_cfg
        self.label_loss_weight_dict = label_loss_weight_dict
        self.set_head_dict(label_loss_weight_dict=self.label_loss_weight_dict)
        self.set_augmentations(augmentation_cfg_list or [])

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "embeddings": {
                "embed": {"representation": self.embedding_representation},
                "embed_dis": {"representation": "vector"},
            }
        }

    def compute_outputs(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            batch: Input with wave shaped (B, T).

        Returns:
            embed in the extractor layout and embed_dis shaped (B, D).
        """
        embed = self.extractor(batch["wave"])
        head_input = self.transform_features(
            features=embed, output_mode=self.head_input_mode
        )
        embed_dis = self.embedding_head(head_input)
        assert embed_dis.ndim == 2
        return {"embed": embed, "embed_dis": embed_dis}

    def compute_losses(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Compute the weighted classification objective using concrete embeddings."""
        outputs = self.compute_outputs(batch)
        return classification_losses(
            embedding=outputs["embed_dis"],
            heads=self.head_dict,
            batch=batch,
            weights=self.label_loss_weight_dict,
        )

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Return embeddings and available GT scores from one model evaluation."""
        outputs = self.compute_outputs(batch)
        scores = classification_anomaly_scores(
            embedding=outputs["embed_dis"],
            heads=self.head_dict,
            batch=batch,
            score_types=self.anomaly_score_type,
        )
        return {**outputs, **scores}
