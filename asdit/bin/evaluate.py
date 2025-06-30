# Copyright 2024 Takuya Fujimura

import logging
from typing import Any, Dict, cast

import hydra
import lightning.pytorch as pl
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from asdit.utils.asdit_utils.evaluate import (
    add_official,
    dcase_auc,
    get_as_name,
    get_auc_type_list,
)
from asdit.utils.asdit_utils.path import make_output_dir
from asdit.utils.config_class import MainEvaluateConfig

logger = logging.getLogger(__name__)


def hydra_to_pydantic(config: DictConfig) -> MainEvaluateConfig:
    """Converts Hydra config to Pydantic config."""
    config_dict = cast(Dict[str, Any], OmegaConf.to_object(config))
    return MainEvaluateConfig(**config_dict)


@hydra.main(version_base=None, config_path="../../config/evaluate", config_name="main")
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg)
    logger.info(f"Start evaluation: {HydraConfig().get().run.dir}")
    pl.seed_everything(seed=0, workers=True)

    output_dir = make_output_dir(cfg, "*_evaluate.csv")

    score_df = pd.read_csv(output_dir / "test_score.csv")

    as_name_list = get_as_name(score_df)
    auc_type_list = get_auc_type_list(score_df)
    evaluate_df = pd.DataFrame(index=as_name_list, columns=auc_type_list)
    for as_name in as_name_list:
        for auc_type in auc_type_list:
            evaluate_df.at[as_name, auc_type] = dcase_auc(
                anomaly_score=score_df[as_name].values,  # type: ignore
                section=score_df["section"].values,  # type: ignore
                is_normal=score_df["is_normal"].values,  # type: ignore
                is_target=score_df["is_target"].values,  # type: ignore
                auc_type=auc_type,
            )
    evaluate_df = add_official(
        evaluate_df=evaluate_df,
        dcase=cfg.dcase,
        machine=cfg.machine,
    )

    evaluate_df = evaluate_df.reset_index().rename(columns={"index": "backend"})

    evaluate_df_path = output_dir / "test_evaluate.csv"
    evaluate_df.to_csv(evaluate_df_path, index=False)
    logger.info(f"Saved at {evaluate_df_path}")


if __name__ == "__main__":
    main()
