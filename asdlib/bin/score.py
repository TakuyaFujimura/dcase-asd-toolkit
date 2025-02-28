# Copyright 2024 Takuya Fujimura

import logging
from typing import Any, Dict, cast

import hydra
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from asdlib.bin.utils.path import check_file_exists, get_output_dir
from asdlib.bin.utils.score import add_score, get_dicts
from asdlib.utils.config_class import MainScoreConfig

logger = logging.getLogger(__name__)


def hydra_to_pydantic(config: DictConfig) -> MainScoreConfig:
    """Converts Hydra config to Pydantic config."""
    config_dict = cast(Dict[str, Any], OmegaConf.to_object(config))
    return MainScoreConfig(**config_dict)


@hydra.main(version_base=None, config_path="../../config/score", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg)
    logger.info(f"Start scoring: {HydraConfig().get().run.dir}")
    pl.seed_everything(seed=0, workers=True)

    output_dir = get_output_dir(cfg=cfg)
    check_file_exists(
        dir_path=output_dir, file_name="*_score.csv", overwrite=cfg.overwrite
    )

    extract_df_dict, score_df_dict = get_dicts(output_dir=output_dir)

    # Loop for backend
    for backend_cfg in cfg.backend:
        score_df_dict = add_score(
            backend_cfg=backend_cfg,
            extract_df_dict=extract_df_dict,
            score_df_dict=score_df_dict,
        )

    # Save
    for split, score_df in score_df_dict.items():
        score_df_path = output_dir / f"{split}_score.csv"
        score_df.to_csv(score_df_path, index=False)
        logging.info(f"Saved at {str(score_df_path)}")


if __name__ == "__main__":
    main()
