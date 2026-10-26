from pathlib import Path
from typing import Any, Dict, List, Optional, final

import torch
from torch import Tensor

from asdkit.augmentations.featex import FeatEx
from asdkit.utils.common.hydra import instantiate

from .base_discriminative import BaseDisPLFrontend


class FeatExPLFrontend(BaseDisPLFrontend):
    """
    K. Wilkinghoff, "Self-supervised learning for anomalous sound detection," Proc. ICASSP, 2024.
    """

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
                self.featex_head_dict.state_dict(prefix="featex_head_dict.")
            )

    def compute_outputs(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            batch: Input with wave shaped (B, T).

        Returns:
            Dictionary containing embed shaped (B, D).
        """
        return {"embed": self.extractor(batch["wave"])}

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Return embeddings for inference."""
        return self.compute_outputs(batch)

    def get_metadata(self) -> Dict[str, Any]:
        return {"embeddings": {"embed": {"representation": "vector"}}}

    def set_head_dict(self, label_loss_weight_dict: Dict[str, float]) -> None:
        """
        Args:
            label_loss_weight_dict: Labels requiring classification heads.
        """
        if self.embedding_dim % self.subspace_embedding_dim != 0:
            raise ValueError(
                f"embedding_dim {self.embedding_dim} should be divisible by subspace_embedding_dim {self.subspace_embedding_dim}"
            )
        embed_num = self.embedding_dim // self.subspace_embedding_dim
        self.head_dict = torch.nn.ModuleDict({})
        self.featex_head_dict = torch.nn.ModuleDict({})
        for label_name in label_loss_weight_dict:
            self.head_dict[label_name] = instantiate(
                cfg=self.loss_cfg,
                n_classes=self.num_class_dict[label_name],
                embedding_dim=self.embedding_dim,
            )
            self.featex_head_dict[label_name] = instantiate(
                cfg=self.loss_cfg,
                n_classes=self.num_class_dict[label_name] * (embed_num + 1),
                embedding_dim=self.embedding_dim,
                trainable=True,
            )

    @final
    def construct_model(
        self,
        extractor_cfg: Dict[str, Any],
        loss_cfg: Dict[str, Any],
        label_loss_weight_dict: Dict[str, float],
        augmentation_cfg_list: Optional[List[Dict[str, Any]]] = None,
        subspace_embedding_dim: Optional[int] = None,
        featex_loss_weight: float = 1.0,
        featex_prob: float = 0.5,
    ) -> None:
        """
        Args:
            extractor_cfg: Waveform extractor configuration.
            loss_cfg: Classification loss configuration shared by all heads.
            label_loss_weight_dict: Label names mapped to objective weights.
            augmentation_cfg_list: Optional batch augmentation configurations.
            subspace_embedding_dim: Number of features per exchanged subspace.
            featex_loss_weight: Weight for the auxiliary exchange objective.
            featex_prob: Probability of exchanging each feature subspace.
        """
        if loss_cfg.get("reduction") == "none":
            raise ValueError("Scalar training losses require reduction='mean' or 'sum'")
        if subspace_embedding_dim is None or subspace_embedding_dim <= 0:
            raise ValueError("subspace_embedding_dim must be a positive integer")
        self.subspace_embedding_dim = subspace_embedding_dim
        self.featex_loss_weight = featex_loss_weight
        self.featex = FeatEx(
            prob=featex_prob, subspace_embedding_dim=subspace_embedding_dim
        )
        self.extractor = instantiate(extractor_cfg)
        self.loss_cfg = loss_cfg
        self.embedding_dim = self.extractor.embedding_dim
        self.label_loss_weight_dict = label_loss_weight_dict
        # Auxiliary objectives are used for training; export embeddings only.
        self.set_anomaly_score_type([])
        self.set_head_dict(label_loss_weight_dict)
        self.set_augmentations(augmentation_cfg_list or [])

    def compute_losses(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            batch: Waveforms under wave and one-hot targets (B, C) under
                onehot_<label>. All configured labels are required.

        Returns:
            Scalar original, exchange, and combined losses plus main total.
        """
        embed = self.compute_outputs(batch)["embed"]
        assert embed.shape[1] == self.embedding_dim
        embed_ex, batch_ex = self.featex(embed=embed, batch=batch)
        loss_dict: Dict[str, Tensor] = {"main": embed.new_zeros(())}

        for label_name, weight in self.label_loss_weight_dict.items():
            loss_dict[f"{label_name}_main"] = self.head_dict[label_name](
                embed, batch[f"onehot_{label_name}"]
            )
            loss_dict[f"{label_name}_featex"] = self.featex_head_dict[label_name](
                embed_ex, batch_ex[f"onehot_{label_name}"]
            )
            loss_dict[f"label_{label_name}"] = (
                loss_dict[f"{label_name}_main"]
                + self.featex_loss_weight * loss_dict[f"{label_name}_featex"]
            )
            loss_dict["main"] += loss_dict[f"label_{label_name}"] * weight

        return loss_dict
