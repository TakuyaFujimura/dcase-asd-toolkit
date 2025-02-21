# Copyright 2024 Takuya Fujimura

import logging
from pathlib import Path
from typing import List

import pandas as pd

from ...utils.config_class.main_test_config import MainTestConfig
from .evaluate import evaluate

logger = logging.getLogger(__name__)


def evaluate_main(cfg: MainTestConfig, infer_dir: Path, machines: List[str]):
    if cfg.evaluate_cfg is None:
        raise ValueError("evaluate_cfg is not set")

    split = "test"
    for m in machines:
        logger.info(f"Start evaluating {m}")
        machine_dir = infer_dir / m
        score_df_path = machine_dir / f"{split}_score.csv"
        score_df = pd.read_csv(score_df_path)
        result_df = evaluate(cfg.evaluate_cfg, score_df)
        result_df_path = machine_dir / f"{split}_result.csv"
        result_df.to_csv(result_df_path, index=False)
        logger.info(f"Saved at {result_df_path}")
