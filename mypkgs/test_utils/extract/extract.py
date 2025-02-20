# Copyright 2024 Takuya Fujimura

import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import tqdm
from lightning import LightningModule
from torch.utils.data import DataLoader

INFOLIST = ["path", "section", "is_normal", "is_target"]

logger = logging.getLogger(__name__)


def sorted_cols(df: pd.DataFrame, embed_cols: List[str]) -> pd.DataFrame:
    col_list = df.columns.tolist()
    col_list = list(set(col_list) - set(embed_cols))
    for key in INFOLIST:
        col_list.remove(key)
    col_list = INFOLIST + sorted(col_list) + embed_cols
    return df[col_list]


def make_df(info__dict_of_list: Dict[str, list]) -> pd.DataFrame:
    df_list = []
    embed_cols = []
    for key, values_list in info__dict_of_list.items():
        values_np = np.concatenate(values_list, axis=0)
        if key == "embed":
            embed_cols = [f"e{i}" for i in range(values_np.shape[-1])]
            df = pd.DataFrame(columns=embed_cols, data=values_np)
        elif values_np.ndim == 2 and values_np.shape[1] == 1:
            df = pd.DataFrame(columns=[key], data=values_np[:, 0])
        elif values_np.ndim == 1:
            df = pd.DataFrame(columns=[key], data=values_np)
        elif key.startswith("logits_"):
            raise NotImplementedError("I have not implemented storing logits.")
        else:
            raise NotImplementedError(f"Unexpected shape: {values_np.shape}")
        df_list.append(df)
    df = pd.concat(df_list, axis=1)
    df = sorted_cols(df, embed_cols)
    return df


def get_info_key_list(plmodel: LightningModule) -> List[str]:
    info_key_list: List[str] = INFOLIST
    # for key in plmodel.num_class_dict:
    #     info_key_list.append(key)
    return list(set(info_key_list))


def get_output_key_list(plmodel: LightningModule) -> List[str]:
    output_key_list: List[str] = ["embed"]
    # for key in plmodel.num_class_dict:
    #     output_key_list.append(f"logits_{key}")
    return output_key_list


def extract(dataloader: DataLoader, plmodel: LightningModule, device: str):
    logger.info("Start extract_loader")
    plmodel.eval()
    extract__dict_of_list = defaultdict(list)
    info_key_list = get_info_key_list(plmodel)
    output_key_list = get_output_key_list(plmodel)

    for batch in tqdm.tqdm(dataloader):
        with torch.no_grad():
            wave = batch["wave"].to(device)
            output_dict = plmodel(wave)
            for key in output_key_list:
                extract__dict_of_list[key].append(output_dict[key].cpu().numpy())

            for key in info_key_list:
                value = batch[key]
                if isinstance(value, torch.Tensor):
                    extract__dict_of_list[key].append(batch[key].cpu().numpy())
                elif isinstance(value, list):
                    extract__dict_of_list[key].append(value)
                else:
                    raise NotImplementedError(f"Unexpected type: {type(value)}")
    df = make_df(extract__dict_of_list)
    return df
