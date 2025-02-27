from datetime import datetime
from pathlib import Path

import pandas as pd
import umap

from ...utils.common import get_embed_from_df


def get_df(machine_dir: Path) -> pd.DataFrame:
    train_df = pd.read_csv(machine_dir / "train_extraction.csv")
    test_df = pd.read_csv(machine_dir / "test_extraction.csv")
    df = pd.concat([train_df, test_df], axis=0)
    df["is_test"] = [0] * len(train_df) + [1] * len(test_df)
    return df


def get_umap_df(
    machine_dir: Path, metric: str, overwrite: bool, embed_key: str
) -> pd.DataFrame:
    umap_path = machine_dir / "umap.csv"
    if umap_path.exists() and not overwrite:
        return pd.read_csv(umap_path)

    df = get_df(machine_dir=machine_dir)
    embed = get_embed_from_df(df=df, embed_key=embed_key)
    umap_model = umap.UMAP(random_state=0, metric=metric)
    umap_embed = umap_model.fit_transform(embed)  # (N, 2)
    umap_df = pd.DataFrame(
        {
            "path": df.path.values,
            "is_normal": df.is_normal.values,
            "is_target": df.is_target.values,
            "is_test": df.is_test.values,
            "u0": umap_embed[:, 0],  # type: ignore
            "u1": umap_embed[:, 1],  # type: ignore
        }
    )
    umap_df.to_csv(umap_path, index=False)

    with open(machine_dir / "umap_info.txt", "a") as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write(f"metric: {metric}\n")
        f.write(f"embed_key: {embed_key}\n")

    return umap_df
