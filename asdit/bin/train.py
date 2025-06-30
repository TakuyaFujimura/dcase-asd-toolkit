import logging
from pathlib import Path
from typing import Any, List, Optional

import hydra
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from asdit.datasets import PLDataModule
from asdit.utils.asdit_utils.callbacks import ModelCheckpoint, NaNCheckCallback
from asdit.utils.asdit_utils.path import get_version_dir
from asdit.utils.common import instantiate_tgt
from asdit.utils.config_class import MainTrainConfig
from asdit.utils.dcase_utils import parse_sec_cfg

logger = logging.getLogger(__name__)


def hydra_to_pydantic(config: DictConfig) -> MainTrainConfig:
    """Converts Hydra config to Pydantic config."""
    config_dict = parse_sec_cfg(OmegaConf.to_object(config))
    return MainTrainConfig(**config_dict)


def make_trainer(
    cfg: MainTrainConfig, ckpt_dir: Path, resume_ckpt_path: Optional[str]
) -> pl.Trainer:
    # Callbacks
    callback_list: List[Any] = [NaNCheckCallback()]
    for _, callback_cfg in cfg.callback.callbacks.items():
        callback_list.append(
            ModelCheckpoint(
                **callback_cfg, dirpath=ckpt_dir, resume_ckpt_path=resume_ckpt_path
            )
        )
    callback_list.append(TQDMProgressBar(refresh_rate=cfg.callback.tqdm_refresh_rate))
    # Logger
    pl_logger = TensorBoardLogger(
        save_dir=f"{cfg.result_dir}/{cfg.name}",
        name=cfg.dcase,
        version=f"{cfg.version}",
        sub_dir=f"{cfg.seed}/model/{cfg.model_ver}",
    )
    # Trainer
    trainer = instantiate_tgt(
        {
            **cfg.trainer,
            "callbacks": callback_list,
            "logger": pl_logger,
        }
    )
    return trainer


def setup_datamodule(cfg: MainTrainConfig) -> pl.LightningDataModule:
    logger.info("Create datamodule")
    dm = PLDataModule(dm_cfg=cfg.datamodule)
    return dm


def setup_frontend(cfg: MainTrainConfig) -> pl.LightningModule:
    logger.info("Create frontend")
    frontend = instantiate_tgt(cfg.frontend)
    return frontend


@hydra.main(version_base=None, config_path="../../config/train", config_name="main")
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg)
    if not cfg.trainer.get("deterministic", False):
        logger.warning("Not deterministic!")
    logger.info(f"Start experiment: {HydraConfig().get().run.dir}")
    logger.info(f"version: {cfg.version}/{cfg.seed}")
    pl.seed_everything(cfg.seed, workers=True)
    # torch.autograd.set_detect_anomaly(False)

    ckpt_dir = get_version_dir(cfg=cfg) / "model" / cfg.model_ver / "checkpoints"
    if ckpt_dir.exists() and cfg.resume_ckpt_path is None:
        logger.warning("Already done. Skipping training...")
        return

    dm = setup_datamodule(cfg)
    frontend = setup_frontend(cfg)
    trainer = make_trainer(cfg, ckpt_dir, cfg.resume_ckpt_path)

    logger.info("Start Training")
    trainer.fit(
        frontend,
        dm.train_dataloader(),
        dm.val_dataloader(),
        ckpt_path=cfg.resume_ckpt_path,
    )


if __name__ == "__main__":
    main()
