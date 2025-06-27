# Copyright 2024 Takuya Fujimura

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader

from asdit.datasets import PLDataModule
from asdit.frontends import BaseFrontend
from asdit.utils.common import item_match
from asdit.utils.config_class import DMConfig, MainExtractConfig, MainTrainConfig
from asdit.utils.dcase_utils import INFOLIST

logger = logging.getLogger(__name__)


def sorted_cols(df: pd.DataFrame) -> pd.DataFrame:
    col_list = df.columns.tolist()
    embed_cols = [col for col in col_list if col.startswith("e_")]
    logits_cols = [col for col in col_list if col.startswith("l_")]
    col_list = list(set(col_list) - set(embed_cols) - set(logits_cols) - set(INFOLIST))
    col_list = INFOLIST + sorted(col_list) + logits_cols + embed_cols
    return df[col_list]


def make_df(info__dict_of_list: Dict[str, list]) -> pd.DataFrame:
    df_list = []
    for key, values_list in info__dict_of_list.items():
        values_np = np.concatenate(values_list, axis=0)
        if key.startswith("embed-"):
            key_tmp = key[len("embed-") :]
            embed_cols = [f"e_{key_tmp}_{i}" for i in range(values_np.shape[-1])]
            df = pd.DataFrame(columns=embed_cols, data=values_np)
        elif key.startswith("logits-"):
            key_tmp = key[len("logits-") :]
            logits_cols = [f"l_{key_tmp}_{i}" for i in range(values_np.shape[-1])]
            df = pd.DataFrame(columns=logits_cols, data=values_np)
        elif values_np.ndim == 2 and values_np.shape[1] == 1:
            df = pd.DataFrame(columns=[key], data=values_np[:, 0])
        elif values_np.ndim == 1:
            df = pd.DataFrame(columns=[key], data=values_np)
        else:
            raise NotImplementedError(f"Unexpected shape: {values_np.shape}")
        df_list.append(df)
    df = pd.concat(df_list, axis=1)
    df = sorted_cols(df)
    return df


def loader2df(
    dataloader: Optional[DataLoader],
    frontend: BaseFrontend,
    device: str,
    extract_items: List[str],
) -> pd.DataFrame:

    if dataloader is None:
        logger.info("No dataloader is given")
        return pd.DataFrame()

    logger.info("Start extract_loader")
    extract__dict_of_list = defaultdict(list)
    extract_items = list(set(INFOLIST + extract_items))

    for batch in tqdm.tqdm(dataloader):

        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            pl_output = frontend.extract(batch)
            for key1 in ["embed", "logits", "AS"]:
                for key2, value in getattr(pl_output, key1).items():
                    key = f"{key1}-{key2}"
                    if item_match(item=key, patterns=extract_items):
                        extract__dict_of_list[key].append(value.cpu().numpy())

            for key, value in batch.items():
                if not item_match(item=key, patterns=extract_items):
                    continue

                if isinstance(value, torch.Tensor):
                    extract__dict_of_list[key].append(batch[key].cpu().numpy())
                elif isinstance(value, list):
                    extract__dict_of_list[key].append(value)
                else:
                    raise NotImplementedError(f"Unexpected type: {type(value)}")

    df = make_df(extract__dict_of_list)
    return df


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

    # check cfg values
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
    assert dataloader is not None
    return dataloader
