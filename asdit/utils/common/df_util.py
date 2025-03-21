from typing import List

import numpy as np
import pandas as pd

from .match import item_match


def get_embed_from_df(df: pd.DataFrame, embed_key: str = "e_main") -> np.ndarray:
    e_cols = []
    for col in df.keys():
        if col.startswith(f"{embed_key}_") and int(col[len(embed_key) + 1 :]) >= 0:
            e_cols.append(col)
    return df[e_cols].values


def pickup_cols(df: pd.DataFrame, extract_items: List[str]) -> pd.DataFrame:
    used_cols = []
    for col in df.columns:
        if item_match(item=col, patterns=extract_items):
            used_cols.append(col)
    return df[used_cols].copy()
