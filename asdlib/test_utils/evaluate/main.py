# Copyright 2024 Takuya Fujimura

import logging
from pathlib import Path

import pandas as pd

from ...utils.config_class.main_test_config import MainTestConfig
from .evaluate import evaluate

logger = logging.getLogger(__name__)


def evaluate_main(cfg: MainTestConfig, machine_dir: Path):

    split = "test"

    logger.info(f"Start evaluating {machine_dir}")
    score_df_path = machine_dir / f"{split}_score.csv"
    score_df = pd.read_csv(score_df_path)
    result_df = evaluate(hmean_list=cfg.hmean_list, score_df=score_df)
    result_df_path = machine_dir / f"{split}_result.csv"
    result_df.to_csv(result_df_path, index=False)
    logger.info(f"Saved at {result_df_path}")
