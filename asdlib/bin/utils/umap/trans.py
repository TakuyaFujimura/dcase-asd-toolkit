import logging
from pathlib import Path

import pandas as pd
import umap

from asdlib.utils.common import get_embed_from_df

logger = logging.getLogger(__name__)


def get_df(output_dir: Path) -> pd.DataFrame:
    train_df = pd.read_csv(output_dir / "train_extract.csv")
    test_df = pd.read_csv(output_dir / "test_extract.csv")
    df = pd.concat([train_df, test_df], axis=0)
    df["is_test"] = [0] * len(train_df) + [1] * len(test_df)
    return df


def trans_umap(output_dir: Path, metric: str, embed_key: str, save_path: Path) -> None:
    logger.info(f"Start UMAP transformation: {output_dir}")

    df = get_df(output_dir=output_dir)
    embed = get_embed_from_df(df=df, embed_key=embed_key)
    umap_model = umap.UMAP(random_state=0, metric=metric)
    umap_embed = umap_model.fit_transform(embed)  # (N, 2)
    umap_df = pd.DataFrame(
        {
            "path": df.path.values,
            "section": df.section.values,
            "is_normal": df.is_normal.values,
            "is_target": df.is_target.values,
            "is_test": df.is_test.values,
            "u0": umap_embed[:, 0],  # type: ignore
            "u1": umap_embed[:, 1],  # type: ignore
        }
    )
    umap_df.to_csv(save_path, index=False)
    logger.info(f"Saved: {save_path}")
