# Copyright 2024 Takuya Fujimura

import logging

import hydra
import lightning.pytorch as pl
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from asdkit.utils.asdkit_utils.evaluate import evaluate_scores
from asdkit.utils.asdkit_utils.path import make_output_dir
from asdkit.utils.common.hydra import hydra_to_pydantic
from asdkit.utils.config_class import MainEvaluateConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config/evaluate", config_name="main")
def main(hydra_cfg: DictConfig) -> None:
    cfg: MainEvaluateConfig = hydra_to_pydantic(
        hydra_cfg=hydra_cfg, cls=MainEvaluateConfig
    )
    logger.info(f"Start evaluation: {HydraConfig().get().run.dir}")
    pl.seed_everything(seed=0, workers=True)

    output_dir = make_output_dir(cfg=cfg, check_file_name="*_evaluate.csv")

    score_df = pd.read_csv(output_dir / "test_score.csv")

    evaluate_df = evaluate_scores(
        score_df=score_df, dcase=cfg.dcase, machine=cfg.machine
    )

    evaluate_df_path = output_dir / "test_evaluate.csv"
    evaluate_df.to_csv(evaluate_df_path, index=False)
    logger.info(f"Saved at {evaluate_df_path}")


if __name__ == "__main__":
    main()
