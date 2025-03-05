# Copyright 2024 Takuya Fujimura

import logging
from typing import Any, Dict

from asdlib.utils.config_class import DMConfig, MainTrainConfig

from .base import BaseExtractDMConfigMaker

logger = logging.getLogger(__name__)


class BasicExtractDMConfigMaker(BaseExtractDMConfigMaker):
    def __init__(
        self,
        past_cfg: MainTrainConfig,  # given by extract.py
        machine: str,  # given by extract.py
        dataloader_cfg: Dict[str, Any] = {},
        collator_cfg: Dict[str, Any] = {},
        dataset_cfg: Dict[str, Any] = {},
    ) -> None:
        self.past_cfg = past_cfg
        self.machine = machine

        if self.past_cfg.datamodule.valid is not None:
            past_dmcfg = self.past_cfg.datamodule.valid
        else:
            past_dmcfg = self.past_cfg.datamodule.train

        self.dataloader_cfg = {
            "batch_size": 64,
            "num_workers": 0,
            **dataloader_cfg,
            "shuffle": False,
            "pin_memory": False,
        }
        self.collator_cfg = {
            "tgt_class": "asdlib.datasets.BasicCollator",
            "sec": past_dmcfg.collator["sec"],
            "sr": past_dmcfg.collator["sr"],
            **collator_cfg,
            "need_feat": False,
            "shuffle": False,
        }
        self.dataset_cfg = {
            "tgt_class": "asdlib.datasets.BasicDataset",
            **dataset_cfg,
        }

    def get_config(self, split: str) -> DMConfig:

        path_selector_list = [
            f"{self.past_cfg.data_dir}/formatted/{self.past_cfg.dcase}/raw/{self.machine}/{split}/*.wav"
        ]

        dmcfg_dict: Dict[str, Any] = {
            "dataloader": self.dataloader_cfg,
            "dataset": {**self.dataset_cfg, "path_selector_list": path_selector_list},
            "collator": self.collator_cfg,
            "batch_sampler": None,
        }

        return DMConfig(**dmcfg_dict)
