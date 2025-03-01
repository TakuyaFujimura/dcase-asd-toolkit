# Copyright 2024 Takuya Fujimura

import logging
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Dict, List, cast

import hydra
import lightning.pytorch as pl
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from scipy.stats import hmean

from asdlib.bin.utils.path import check_file_exists
from asdlib.utils.config_class import MainTableConfig
from asdlib.utils.dcase_utils import MACHINE_DICT

logger = logging.getLogger(__name__)


def hydra_to_pydantic(config: DictConfig) -> MainTableConfig:
    """Converts Hydra config to Pydantic config."""
    config_dict = cast(Dict[str, Any], OmegaConf.to_object(config))
    return MainTableConfig(**config_dict)


def myround(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def get_table(metric: str, output_dir: Path, machine_list: List[str]) -> pd.DataFrame:
    df_list = []
    col_list = []

    for i, m in enumerate(machine_list):
        df = pd.read_csv(output_dir / m / "test_evaluate.csv")
        df_list.append(df[[metric]])
        col_list.append(m)
        if i == 0:
            backend_list = df.backend.values
        elif np.all(backend_list == df.backend.values):
            pass
        else:
            raise ValueError(
                "Different backends are provided. "
                + "This script assumes the same backends for all machines."
            )
    table_df = pd.concat(df_list, axis=1)
    table_df.columns = col_list
    table_df["hmean"] = table_df.apply(lambda x: hmean(x), axis=1)
    table_df = table_df.applymap(lambda x: myround(x * 100))  # type: ignore
    table_df.index = backend_list
    table_df = table_df.reset_index().rename(columns={"index": "backend"})
    return table_df


class MetricMachineDict(BaseModel):
    metric: str
    machines: List[str]


def get_metric_machine_dict(cfg: MainTableConfig) -> Dict[str, MetricMachineDict]:
    metric_machine_dict = {}
    yy = cfg.dcase[-2:]
    if cfg.dcase in ["dcase2021", "dcase2022"]:
        for split in ["dev", "eval"]:
            metric_machine_dict[f"official{yy}-{split}"] = MetricMachineDict(
                metric=f"official{yy}-{split}",
                machines=MACHINE_DICT[f"{cfg.dcase}-{split}"],
            )

    elif cfg.dcase in ["dcase2023", "dcase2024"]:
        for split in ["dev", "eval"]:
            metric_machine_dict[f"official{yy}-{split}"] = MetricMachineDict(
                metric=f"official{yy}",
                machines=MACHINE_DICT[f"{cfg.dcase}-{split}"],
            )

    return metric_machine_dict


@hydra.main(version_base=None, config_path="../../config/table", config_name="config")
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

    check_file_exists(dir_path=output_dir, file_name="*.csv", overwrite=cfg.overwrite)

    metric_machine_dict = get_metric_machine_dict(cfg=cfg)

    for name in metric_machine_dict:
        logger.info(f"Making table of {name}")
        table_df = get_table(
            metric=metric_machine_dict[name].metric,
            output_dir=output_dir,
            machine_list=metric_machine_dict[name].machines,
        )
        table_df.to_csv(output_dir / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
