# Copyright 2024 Takuya Fujimura

import logging
from pathlib import Path
from typing import Any, Dict, Optional, cast

import hydra
import lightning.pytorch as pl
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from scipy.stats import hmean

from asdkit.utils.asdkit_utils.path import check_file_exists, get_version_dir
from asdkit.utils.config_class import MainTableConfig
from asdkit.utils.dcase_utils import MACHINE_DICT

logger = logging.getLogger(__name__)


def hydra_to_pydantic(config: DictConfig) -> MainTableConfig:
    """Converts Hydra config to Pydantic config."""
    config_dict = cast(Dict[str, Any], OmegaConf.to_object(config))
    return MainTableConfig(**config_dict)


def get_table_df(
    output_dir: Path, dcase: str, split: str, metric: str
) -> Optional[pd.DataFrame]:
    # init
    evaluate_df_list = []
    machine_list = MACHINE_DICT[f"{dcase}-{split}"]
    if dcase in ["dcase2020", "dcase2021", "dcase2022"]:
        metric = f"{split}_{metric}"
    elif dcase in ["dcase2023", "dcase2024"]:
        metric = f"0_{metric}"
    else:
        raise ValueError(f"Unknown dcase: {dcase}")

    # Loop of machines
    for i, m in enumerate(machine_list):
        evaluate_df = pd.read_csv(output_dir / m / "test_evaluate.csv")
        # check backend
        if i == 0:
            backend_array = evaluate_df.backend.values
        elif set(backend_array) != set(evaluate_df.backend.values):
            backend_array = np.intersect1d(backend_array, evaluate_df.backend.values)
            logger.warning(
                "Different backends found across machines. Using common backends."
            )
        evaluate_df_list += [evaluate_df]

    # Get table
    backend_array = np.sort(backend_array)
    table_df_list = []
    for df in evaluate_df_list:
        ordered_df = df[df["backend"].isin(backend_array)]
        ordered_df = ordered_df.set_index("backend").loc[backend_array].reset_index()
        table_df_list += [ordered_df[metric]]
    table_df = pd.concat(table_df_list, axis=1)
    table_df.columns = machine_list

    # Calculate mean or hmean
    if dcase == "dcase2020":
        table_df["total"] = table_df.apply(np.mean, axis=1)
    else:
        table_df["total"] = table_df.apply(lambda x: hmean(x), axis=1)

    table_df.index = backend_array  # type: ignore
    table_df = table_df.reset_index().rename(columns={"index": "backend"})
    return table_df


@hydra.main(version_base=None, config_path="../../config/table", config_name="main")
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg)
    logger.info(f"Start making table: {HydraConfig().get().run.dir}")
    pl.seed_everything(seed=0, workers=True)

    output_dir = get_version_dir(cfg) / "output" / cfg.infer_ver
    check_file_exists(dir_path=output_dir, file_name="*.csv", overwrite=cfg.overwrite)

    for metric in cfg.metrics:
        if metric == "official":
            metric = f"official{cfg.dcase[-2:]}"
        for split in ["dev", "eval"]:
            table_df = get_table_df(
                output_dir=output_dir,
                dcase=cfg.dcase,
                split=split,
                metric=metric,
            )
            if table_df is not None:
                table_df.to_csv(output_dir / f"{split}_{metric}.csv", index=False)


if __name__ == "__main__":
    main()
