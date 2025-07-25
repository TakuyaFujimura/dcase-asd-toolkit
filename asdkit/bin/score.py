# Copyright 2024 Takuya Fujimura

import logging
from typing import Any, Dict, cast

import hydra
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from asdkit.utils.asdkit_utils.df_utils import sort_columns
from asdkit.utils.asdkit_utils.path import make_output_dir
from asdkit.utils.asdkit_utils.score import add_score, get_extract_score_dicts
from asdkit.utils.config_class import MainScoreConfig

logger = logging.getLogger(__name__)


def hydra_to_pydantic(config: DictConfig) -> MainScoreConfig:
    """Converts Hydra config to Pydantic config."""
    config_dict = cast(Dict[str, Any], OmegaConf.to_object(config))
    return MainScoreConfig(**config_dict)


@hydra.main(version_base=None, config_path="../../config/score", config_name="main")
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg)
    logger.info(f"Start scoring: {HydraConfig().get().run.dir}")
    pl.seed_everything(seed=0, workers=True)

    output_dir = make_output_dir(cfg, "*_score.csv")
    extract_dict_dict, score_df_dict = get_extract_score_dicts(
        output_dir=output_dir, extract_items=cfg.extract_items
    )

    # Loop for backend
    for backend_cfg in cfg.backend:
        score_df_dict = add_score(
            backend_cfg=backend_cfg,
            extract_dict_dict=extract_dict_dict,
            score_df_dict=score_df_dict,
        )

    # Save
    for split, score_df in score_df_dict.items():
        score_df = sort_columns(score_df)
        score_df_path = output_dir / f"{split}_score.csv"
        score_df.to_csv(score_df_path, index=False)
        logging.info(f"Saved at {str(score_df_path)}")


if __name__ == "__main__":
    main()
