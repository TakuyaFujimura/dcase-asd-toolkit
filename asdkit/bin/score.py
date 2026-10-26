# Copyright 2024 Takuya Fujimura

import logging

import hydra
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

from asdkit.backends.base import BaseBackend
from asdkit.utils.asdkit_utils.extract import load_extract_data
from asdkit.utils.asdkit_utils.path import make_output_dir
from asdkit.utils.asdkit_utils.score import (
    load_score_df,
    make_backend_name,
    save_scores,
)
from asdkit.utils.common.hydra import hydra_to_pydantic, instantiate
from asdkit.utils.config_class import MainScoreConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config/score", config_name="main")
def main(hydra_cfg: DictConfig) -> None:
    cfg: MainScoreConfig = hydra_to_pydantic(hydra_cfg=hydra_cfg, cls=MainScoreConfig)
    logger.info(f"Start scoring: {HydraConfig().get().run.dir}")
    pl.seed_everything(seed=0, workers=True)

    cfg.overwrite = cfg.overwrite or cfg.overwrite_keep_existing
    output_dir = make_output_dir(cfg=cfg, check_file_name="*_score.csv")
    extract_data_dict = {
        split: load_extract_data(output_dir=output_dir, split=split)
        for split in ["train", "test"]
    }
    score_df_dict = {
        split: load_score_df(
            output_dir=output_dir,
            split=split,
            extract_dict=data.extract_dict,
            extract_items=cfg.extract_items,
            overwrite_keep_existing=cfg.overwrite_keep_existing,
        )
        for split, data in extract_data_dict.items()
    }

    # Score each backend
    for backend_cfg in tqdm(cfg.backend):
        assert "_target_" in backend_cfg
        backend: BaseBackend = instantiate(backend_cfg)
        backend_name = make_backend_name(backend_cfg)
        training = extract_data_dict["train"]
        backend.fit(train_dict=training.extract_dict, metadata=training.metadata)
        for split, data in extract_data_dict.items():
            if split == "train" and cfg.skip_train_score:
                continue
            score_dict = backend.anomaly_score(
                extract_dict=data.extract_dict, split=split, metadata=data.metadata
            )
            for key, score in score_dict.items():
                score_df_dict[split][f"AS-{backend_name}-{key}"] = score

    # Save
    for split, score_df in score_df_dict.items():
        if split == "train" and cfg.skip_train_score:
            continue
        save_scores(score_df=score_df, score_path=output_dir / f"{split}_score.csv")


if __name__ == "__main__":
    main()
