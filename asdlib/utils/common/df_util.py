import numpy as np
import pandas as pd


def get_embed_from_df(df: pd.DataFrame, embed_key: str = "e_main") -> np.ndarray:
    e_cols = []
    for col in df.keys():
        if col.startswith(f"{embed_key}_") and int(col[len(embed_key) + 1 :]) >= 0:
            e_cols.append(col)
    return df[e_cols].values
