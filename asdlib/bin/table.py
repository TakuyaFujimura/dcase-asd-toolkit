# Copyright 2024 Takuya Fujimura

import logging
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import hmean

logger = logging.getLogger(__name__)


def myround(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def get_table(metric: str, infer_dir: Path, machines: List[str]) -> pd.DataFrame:
    df_list = []
    col_list = []
    backend_list = None
    for m in machines:
        df = pd.read_csv(infer_dir / m / "test_result.csv")
        df_list.append(df[[metric]])
        col_list.append(m)
        if backend_list is None:
            backend_list = df.backend.values
        else:
            assert np.all(backend_list == df.backend.values)
    table_df = pd.concat(df_list, axis=1)
    table_df.columns = col_list
    table_df["hmean"] = table_df.apply(lambda x: hmean(x), axis=1)
    table_df = table_df.applymap(lambda x: myround(x * 100))
    table_df.index = backend_list
    table_df = table_df.reset_index().rename(columns={"index": "backend"})
    return table_df


def table_main(metric_list: List[str], infer_dir: Path, machines: List[str]):
    for metric in metric_list:
        save_path = infer_dir / f"{metric}.csv"
        logger.info(f"------------table of {metric}------------")
        logger.info(f"save_path: {save_path}")
        logger.info(f"machines: {machines}")
        logger.info("-----------------------------------------------")
        table_df = get_table(metric=metric, infer_dir=infer_dir, machines=machines)
        table_df.to_csv(save_path, index=False)


import logging
from typing import Any, Dict, cast

import hydra
import lightning.pytorch as pl
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from asdlib.utils.config_class import MainTableConfig


def hydra_to_pydantic(config: DictConfig) -> MainTableConfig:
    """Converts Hydra config to Pydantic config."""
    config_dict = cast(Dict[str, Any], OmegaConf.to_object(config))
    return MainTableConfig(**config_dict)


@hydra.main(version_base=None, config_path="../../config/umap", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg)
    logger.info(f"Start scoring: {HydraConfig().get().run.dir}")
    pl.seed_everything(seed=0, workers=True)

    output_dir = (
        cfg.result_dir
        / cfg.name
        / f"{cfg.version}_{cfg.seed}"
        / "output"
        / cfg.ckpt_ver
    )
