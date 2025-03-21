import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import umap

from asdit.utils.common import get_embed_from_df
from asdit.utils.common.df_util import pickup_cols
from asdit.utils.dcase_utils import INFOLIST

logger = logging.getLogger(__name__)


def get_df(output_dir: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    train_df = pd.read_csv(output_dir / "train_extract.csv")
    test_df = pd.read_csv(output_dir / "test_extract.csv")
    df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    is_test = np.array([0] * len(train_df) + [1] * len(test_df))
    return df, is_test


def trans_umap(
    output_dir: Path,
    metric: str,
    embed_key: str,
    save_path: Path,
    extract_items: List[str],
) -> None:
    df, is_test = get_df(output_dir=output_dir)
    embed = get_embed_from_df(df=df, embed_key=embed_key)
    umap_model = umap.UMAP(random_state=0, metric=metric)
    umap_embed = umap_model.fit_transform(embed)  # (N, 2)
    extract_items = list(set(INFOLIST + extract_items))
    umap_df = pickup_cols(df=df, extract_items=extract_items)
    additional_df = pd.DataFrame(
        {
            "is_test": is_test,  # type: ignore
            "u0": umap_embed[:, 0],  # type: ignore
            "u1": umap_embed[:, 1],  # type: ignore
        }
    )
    umap_df = pd.concat([umap_df, additional_df], axis=1)
    umap_df.to_csv(save_path, index=False)
    logger.info(f"Saved: {save_path}")
