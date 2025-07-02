# Copyright 2024 Takuya Fujimura

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from asdkit.datasets import PLDataModule
from asdkit.frontends.base import BaseFrontend
from asdkit.utils.asdkit_utils.restore import get_past_cfg, restore_plfrontend
from asdkit.utils.common import instantiate_tgt, re_match_any
from asdkit.utils.config_class import DMConfig, MainExtractConfig, MainTrainConfig

logger = logging.getLogger(__name__)


def extraction_setup_restore(
    cfg: MainExtractConfig,
) -> Tuple[BaseFrontend, Dict[str, DataLoader]]:
    if cfg.restore_model_ver is None:
        raise ValueError(
            "restore_model_ver must be specified when restore_or_scratch is restore"
        )
    if cfg.restore_ckpt_ver is None:
        raise ValueError(
            "restore_ckpt_ver must be specified when restore_or_scratch is restore"
        )
    past_cfg = get_past_cfg(cfg)

    # Frontend
    frontend = restore_plfrontend(cfg=cfg, past_cfg=past_cfg)

    # DataLoader
    check_cfg_with_past_cfg(cfg=cfg, past_cfg=past_cfg)
    dataloader_dict = {
        "train": PLDataModule.get_loader(dm_config=cfg.datamodule.train),
        "test": PLDataModule.get_loader(dm_config=cfg.datamodule.test),
    }
    return frontend, dataloader_dict


def extraction_setup_scratch(
    cfg: MainExtractConfig,
) -> Tuple[BaseFrontend, Dict[str, DataLoader]]:
    if cfg.scratch_frontend is None:
        raise ValueError(
            "scratch_frontend must be specified when restore_or_scratch is scratch"
        )

    # Frontend
    frontend = instantiate_tgt(cfg.scratch_frontend)

    # DataLoader
    dataloader_dict = {
        "train": PLDataModule.get_loader(dm_config=cfg.datamodule.train),
        "test": PLDataModule.get_loader(dm_config=cfg.datamodule.test),
    }
    return frontend, dataloader_dict


def loader2dict(
    dataloader: DataLoader,
    frontend: BaseFrontend,
    device: str,
    extract_items: List[str],
) -> Dict[str, np.ndarray]:

    extract_dict = defaultdict(list)

    for batch in tqdm.tqdm(dataloader):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with torch.no_grad():
            output_dict = frontend.extract(batch)
            output_dict.update(batch)
            for key, value in output_dict.items():
                if not re_match_any(patterns=extract_items, string=key):
                    continue
                elif isinstance(value, torch.Tensor):
                    extract_dict[key].append(value.detach().cpu().numpy())
                else:
                    extract_dict[key].append(np.asarray(value))

    extract_np_dict = {
        key: np.concatenate(list_of_array)
        for key, list_of_array in extract_dict.items()
    }
    return extract_np_dict


def check_collator_args(current_collator: dict, past_collator: dict) -> None:
    collator_args = ["sec", "sr", "pad_mode"]
    for key in collator_args:
        if current_collator.get(key) != past_collator.get(key):
            logger.warning(
                f"cfg.datamodule.collator.{key} ({current_collator.get(key)}) is different from past_cfg.datamodule.collator.{key} ({past_collator.get(key)})."
            )


def check_cfg_with_past_cfg(cfg: MainExtractConfig, past_cfg: MainTrainConfig) -> None:
    # Check data_dir and dcase
    if past_cfg.data_dir != cfg.data_dir:
        logger.warning(
            f"cfg.data_dir ({cfg.data_dir}) is different from past_cfg.data_dir ({past_cfg.data_dir})."
        )
    if past_cfg.dcase != cfg.dcase:
        logger.warning(
            f"cfg.dcase ({cfg.dcase}) is different from past_cfg.dcase ({past_cfg.dcase})."
        )

    # Check datamodule config
    dm_config_dict = {"train": cfg.datamodule.train, "test": cfg.datamodule.test}
    for split, dm_config in dm_config_dict.items():
        # Dataloader
        if dm_config.dataloader.get("shuffle") is not False:
            logger.warning(f"datamodule.{split}.dataloader.shuffle is not False.")

        # Collator
        expected_collator = "asdkit.datasets.DCASEWaveCollator"
        if dm_config.collator["tgt_class"] != expected_collator:
            logger.warning(
                f"cfg.datamodule.{split}.collator['tgt_class'] is not '{expected_collator}'. Skipping checks"
            )
        else:
            is_checked = False
            if past_cfg.datamodule.train.collator["tgt_class"] == expected_collator:
                is_checked = True
                check_collator_args(
                    current_collator=dm_config.collator,
                    past_collator=past_cfg.datamodule.train.collator,
                )
            if (
                past_cfg.datamodule.valid is not None
                and past_cfg.datamodule.valid.collator["tgt_class"] == expected_collator
            ):
                is_checked = True
                check_collator_args(
                    current_collator=dm_config.collator,
                    past_collator=past_cfg.datamodule.valid.collator,
                )
            if not is_checked:
                logger.warning(
                    f"cfg.datamodule.{split}.collator is not checked because past_cfg.datamodule.train/valid.collator.tgt_class is not '{expected_collator}'."
                )

        if dm_config.collator.get("shuffle") is not False:
            logger.warning(f"cfg.datamodule.{split}.collator.shuffle is not False.")

        # Dataset
        expected_dataset = "asdkit.datasets.WaveDataset"
        if dm_config.dataset["tgt_class"] != expected_dataset:
            logger.warning(
                f"cfg.datamodule.{split}.dataset['tgt_class'] is not '{expected_dataset}'."
            )
