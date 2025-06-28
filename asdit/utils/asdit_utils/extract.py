# Copyright 2024 Takuya Fujimura

import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from asdit.datasets import PLDataModule
from asdit.frontends.base import BaseFrontend
from asdit.utils.asdit_utils.restore import get_past_cfg, restore_plfrontend
from asdit.utils.common import instantiate_tgt, re_match_any
from asdit.utils.config_class import DMConfig, MainExtractConfig, MainTrainConfig

logger = logging.getLogger(__name__)

# TODO: remove
# def sorted_cols(df: pd.DataFrame) -> pd.DataFrame:
#     col_list = df.columns.tolist()
#     embed_cols = [col for col in col_list if col.startswith("e_")]
#     logits_cols = [col for col in col_list if col.startswith("l_")]
#     col_list = list(set(col_list) - set(embed_cols) - set(logits_cols) - set(INFOLIST))
#     col_list = INFOLIST + sorted(col_list) + logits_cols + embed_cols
#     return df[col_list]


def setup_frontend(cfg: MainExtractConfig) -> BaseFrontend:
    if cfg.restore_or_scratch == "restore":
        if cfg.restore_model_ver is None:
            raise ValueError(
                "restore_model_ver must be specified when restore_or_scratch is restore"
            )
        if cfg.restore_ckpt_ver is None:
            raise ValueError(
                "restore_ckpt_ver must be specified when restore_or_scratch is restore"
            )
        past_cfg = get_past_cfg(cfg=cfg)
        frontend = restore_plfrontend(cfg=cfg, past_cfg=past_cfg)
        check_cfg_with_past_cfg(cfg=cfg, past_cfg=past_cfg)

    elif cfg.restore_or_scratch == "scratch":
        if cfg.scratch_frontend is None:
            raise ValueError(
                "scratch_frontend must be specified when restore_or_scratch is scratch"
            )
        frontend = instantiate_tgt(cfg.scratch_frontend)
    else:
        raise ValueError(f"Unexpected restore_or_scratch: {cfg.restore_or_scratch}")

    return frontend


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


def get_extract_dataloader(cfg: MainExtractConfig, split: str) -> DataLoader:
    path_selector_list = [
        f"{cfg.data_dir}/formatted/{cfg.dcase}/raw/{cfg.machine}/{split}/*.wav"
    ]
    dataset_cfg = {
        **cfg.datamodule.dataset,
        "path_selector_list": path_selector_list,
    }

    datamoduleconfig = DMConfig(
        dataloader=cfg.datamodule.dataloader,
        dataset=dataset_cfg,
        collator=cfg.datamodule.collator,
        batch_sampler=None,
    )
    dataloader = PLDataModule.get_loader(
        datamoduleconfig=datamoduleconfig, label_dict_path=cfg.label_dict_path
    )
    return dataloader  # type: ignore


def check_cfg_with_past_cfg(cfg: MainExtractConfig, past_cfg: MainTrainConfig) -> None:
    if past_cfg.data_dir != cfg.data_dir:
        logger.warning(
            f"data_dir in cfg ({cfg.data_dir}) is different from past_cfg ({past_cfg.data_dir})."
        )
    if past_cfg.dcase != cfg.dcase:
        logger.warning(
            f"dcase in cfg ({cfg.dcase}) is different from past_cfg ({past_cfg.dcase})."
        )

    # Check args in collator
    collator_args = ["sec", "sr", "pad_mode"]
    is_checked = False
    if (past_cfg.datamodule.valid is not None) and (
        past_cfg.datamodule.valid.collator["tgt_class"]
        == "asdit.datasets.DCASEWaveCollator"
    ):
        for key in collator_args:
            if cfg.datamodule.collator.get(
                key
            ) != past_cfg.datamodule.valid.collator.get(key):
                logger.warning(
                    f"{key} in cfg.datamodule.collator ({cfg.datamodule.collator.get(key)}) is different from past_cfg.valid.collator ({past_cfg.datamodule.valid.collator.get(key)})."
                )
        is_checked = True
    if (
        past_cfg.datamodule.train.collator["tgt_class"]
        == "asdit.datasets.DCASEWaveCollator"
    ):
        for key in collator_args:
            if cfg.datamodule.collator.get(
                key
            ) != past_cfg.datamodule.train.collator.get(key):
                logger.warning(
                    f"{key} in cfg.datamodule.collator ({cfg.datamodule.collator.get(key)}) is different from past_cfg.train.collator ({past_cfg.datamodule.train.collator.get(key)})."
                )
        is_checked = True
    if not is_checked:
        logger.warning(
            "collator in past_cfg is not DCASEWaveCollator, so cfg.datamodule.collator is not checked."
        )

    # Check cfg values
    value = cfg.datamodule.dataloader.get("shuffle")
    if value is not False:
        logger.warning(
            f"Expected 'shuffle' in datamodule.dataloader to be False but got {value}"
        )

    value = cfg.datamodule.collator.get("tgt_class")
    if value != "asdit.datasets.DCASEWaveCollator":
        logger.warning(
            f"Expected 'tgt_class' in datamodule.collator to be 'asdit.datasets.DCASEWaveCollator' but got {value}"
        )

    value = cfg.datamodule.collator.get("shuffle")
    if value is not False:
        logger.warning(
            f"Expected 'shuffle' in datamodule.collator to be False but got {value}"
        )

    value = cfg.datamodule.dataset.get("tgt_class")
    if value != "asdit.datasets.WaveDataset":
        logger.warning(
            f"Expected 'tgt_class' in datamodule.dataset to be 'asdit.datasets.WaveDataset' but got {value}"
        )
