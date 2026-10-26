# Copyright 2024 Takuya Fujimura

import logging

import hydra
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from asdkit.utils.asdkit_utils.path import check_file_exists, get_version_dir
from asdkit.utils.asdkit_utils.table import build_metric_table
from asdkit.utils.common.hydra import hydra_to_pydantic
from asdkit.utils.config_class import MainTableConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config/table", config_name="main")
def main(hydra_cfg: DictConfig) -> None:
    cfg: MainTableConfig = hydra_to_pydantic(hydra_cfg=hydra_cfg, cls=MainTableConfig)
    logger.info(f"Start making table: {HydraConfig().get().run.dir}")
    pl.seed_everything(seed=0, workers=True)

    output_dir = get_version_dir(cfg) / "output" / cfg.infer_ver
    check_file_exists(dir_path=output_dir, file_name="*.csv", overwrite=cfg.overwrite)

    for metric in cfg.metrics:
        if metric == "official":
            metric = f"official{cfg.dcase[-2:]}"
        for subset in ["dev", "eval"]:
            table_df = build_metric_table(
                output_dir=output_dir, dcase=cfg.dcase, subset=subset, metric=metric
            )
            if table_df is not None:
                table_df.to_csv(output_dir / f"{subset}_{metric}.csv", index=False)


if __name__ == "__main__":
    main()
