import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from asdkit.utils.common.match import re_match_any


def load_score_df(
    output_dir: Path,
    split: str,
    extract_dict: Dict[str, Any],
    extract_items: List[str],
    overwrite_keep_existing: bool = False,
) -> pd.DataFrame:
    """Load existing scores or initialize score rows from extraction results.

    Reusing an existing CSV requires paths in both inputs with identical values,
    length, and order. Missing or mismatched paths raise ValueError.
    """
    score_path = output_dir / f"{split}_score.csv"
    if overwrite_keep_existing:
        if score_path.exists():
            score_df = pd.read_csv(score_path)
            if "path" not in extract_dict or "path" not in score_df:
                raise ValueError(
                    f"Missing 'path' in extract_dict or score_df. "
                    f"score_path: {score_path}, split: {split}"
                )
            if not np.array_equal(score_df["path"].to_numpy(), extract_dict["path"]):
                raise ValueError(
                    f"Existing score path does not match extract path. "
                    f"score_path: {score_path}, split: {split}"
                )
            return score_df
        logging.warning(f"overwrite_keep_existing is True, but {score_path} not found.")

    # Initialize score_df with extract_dict
    return pd.DataFrame(
        {
            key: value
            for key, value in extract_dict.items()
            if re_match_any(patterns=extract_items, string=key)
        }
    )


def make_backend_name(value: Any) -> str:
    """Render target configs and ordinary containers with explicit nesting."""
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        if "_target_" in value:
            name = value["_target_"].split(".")[-1]
            arguments: List[str] = []
            for key in sorted(value):
                if key == "_target_":
                    continue
                label = "backend" if key == "backend_cfg" else key
                arguments.append(f"{label}={make_backend_name(value[key])}")
            return f"{name}({','.join(arguments)})"
        dict_items = [f"{key}={make_backend_name(value[key])}" for key in sorted(value)]
        return "{" + ",".join(dict_items) + "}"
    if isinstance(value, list):
        return "[" + ",".join(make_backend_name(item) for item in value) + "]"
    if isinstance(value, tuple):
        tuple_items = ",".join(make_backend_name(item) for item in value)
        suffix = "," if len(value) == 1 else ""
        return f"({tuple_items}{suffix})"
    if isinstance(value, str):
        if (
            not value
            or value in ("None", "True", "False")
            or any(
                character.isspace() or character in ",=(){}[]\"'\\"
                for character in value
            )
        ):
            return json.dumps(value, ensure_ascii=False)
        return value
    return str(value)


def sort_score_columns(score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort columns of the DataFrame based on a specific order.
    """
    col_order = ["path", "section", "is_normal", "is_target"]
    sorted_cols = [col for col in col_order if col in score_df.columns]
    sorted_cols += sorted(list(set(score_df.columns) - set(sorted_cols)))
    return score_df[sorted_cols].copy()


def save_scores(score_df: pd.DataFrame, score_path: Path) -> None:
    """Write one score CSV with stable column ordering."""
    # Save
    sort_score_columns(score_df).to_csv(score_path, index=False)
    logging.info(f"Saved at {score_path}")
