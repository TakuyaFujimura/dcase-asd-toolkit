# Copyright 2024 Takuya Fujimura

import logging
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import hydra
import lightning.pytorch as pl
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from scipy.stats import hmean

from asdit.bin.utils.evaluate import (
    combine_section_metric,
    get_official_metriclist,
    get_official_sectionlist,
    hmean_is_available,
)
from asdit.bin.utils.path import check_file_exists, get_version_dir
from asdit.utils.config_class import MainTableConfig
from asdit.utils.dcase_utils import MACHINE_DICT

logger = logging.getLogger(__name__)


def hydra_to_pydantic(config: DictConfig) -> MainTableConfig:
    """Converts Hydra config to Pydantic config."""
    config_dict = cast(Dict[str, Any], OmegaConf.to_object(config))
    return MainTableConfig(**config_dict)


def myround(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def complete_hmean_cfg(
    hmean_cfg_dict: Dict[str, List[str]], dcase: str, split: str
) -> Dict[str, List[str]]:
    sectionlist = get_official_sectionlist(dcase=dcase, split=split)

    hmean_cfg_dict_new = {}
    for hmean_name, metriclist in hmean_cfg_dict.items():
        hmean_cfg_dict_new[hmean_name] = combine_section_metric(
            sectionlist=sectionlist, metriclist=metriclist
        )
    return hmean_cfg_dict_new


def get_table_df(
    dcase: str, split: str, output_dir: Path, hmean_name: str, metriclist: List[str]
) -> Optional[pd.DataFrame]:
    df_list = []
    machinelist = MACHINE_DICT[f"{dcase}-{split}"]
    for i, m in enumerate(machinelist):
        evaluate_df = pd.read_csv(output_dir / m / "test_evaluate.csv")

        # check backend
        if i == 0:
            backend_list = evaluate_df.backend.values
        elif np.any(backend_list != evaluate_df.backend.values):
            raise ValueError(
                "Different backends are provided. "
                + "This script assumes the same backends for all machines."
            )
            # TODO:
            # For example, when machine-A has backend1 and machine-B has backend1 and backend2,
            # the current script will raise an error.
            # However, we can handle this case.

        sectionlist = get_official_sectionlist(dcase=dcase, split=split, machine=m)
        hmean_cols = combine_section_metric(
            sectionlist=sectionlist, metriclist=metriclist
        )

        # check hmean
        if not hmean_is_available(
            evaluate_df=evaluate_df, hmean_name=hmean_name, hmean_cols=hmean_cols
        ):
            logger.warning(
                f"Skipped {hmean_name} because {hmean_cols} is not available."
            )
            return None

        # get hmean
        df_list += [evaluate_df[hmean_cols].apply(lambda x: hmean(x), axis=1)]

    # get table
    table_df = pd.concat(df_list, axis=1)
    table_df.columns = machinelist
    table_df["hmean"] = table_df.apply(lambda x: hmean(x), axis=1)
    table_df.index = backend_list  # type: ignore
    table_df = table_df.reset_index().rename(columns={"index": "backend"})
    return table_df


@hydra.main(version_base=None, config_path="../../config/table", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg)
    logger.info(f"Start making table: {HydraConfig().get().run.dir}")
    pl.seed_everything(seed=0, workers=True)

    output_dir = get_version_dir(cfg=cfg) / "output" / cfg.infer_ver

    check_file_exists(dir_path=output_dir, file_name="*.csv", overwrite=cfg.overwrite)

    if any([key.startswith("official") for key in cfg.hmean_cfg_dict]):
        raise ValueError("name starting with 'official' is reserved. Please rename.")

    cfg.hmean_cfg_dict[f"official{cfg.dcase[-2:]}"] = get_official_metriclist(
        dcase=cfg.dcase
    )

    for split in ["dev", "eval"]:
        for hmean_name, metriclist in cfg.hmean_cfg_dict.items():
            table_df = get_table_df(
                dcase=cfg.dcase,
                split=split,
                output_dir=output_dir,
                hmean_name=hmean_name,
                metriclist=metriclist,
            )
            if table_df is not None:
                table_df.to_csv(output_dir / f"{hmean_name}-{split}.csv", index=False)


if __name__ == "__main__":
    main()
