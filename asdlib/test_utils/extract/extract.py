# Copyright 2024 Takuya Fujimura

import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader

from ...pl_models import BasePLModel
from ...utils.config_class.output_config import PLOutput

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
        if key.startswith("embed-"):
            key_tmp = key.replace("embed-", "")
            embed_cols = [f"e_{key_tmp}_{i}" for i in range(values_np.shape[-1])]
            df = pd.DataFrame(columns=embed_cols, data=values_np)
        elif key.startswith("logits-"):
            key_tmp = key.replace("logits-", "")
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
    df = sorted_cols(df, embed_cols)
    return df


def get_info_key_list(plmodel: BasePLModel) -> List[str]:
    info_key_list: List[str] = INFOLIST
    # for key in plmodel.num_class_dict:
    #     info_key_list.append(key)
    return list(set(info_key_list))


def extract(dataloader: DataLoader, plmodel: BasePLModel, device: str):
    logger.info("Start extract_loader")
    plmodel.eval()
    extract__dict_of_list = defaultdict(list)
    info_key_list = get_info_key_list(plmodel)
    output_key_list = ["embed", "AS"]  # Currently, I exclude "logits" from output

    for batch in tqdm.tqdm(dataloader):

        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            pl_output: PLOutput = plmodel(batch)
            for key1 in output_key_list:
                for key2, value in getattr(pl_output, key1).items():
                    extract__dict_of_list[f"{key1}-{key2}"].append(value.cpu().numpy())
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
