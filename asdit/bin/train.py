import logging
from pathlib import Path
from typing import Any, Dict, List, cast

import hydra
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from asdit.bin.utils import NaNCheckCallback
from asdit.bin.utils.path import get_version_dir
from asdit.datasets import PLDataModule
from asdit.utils.common import instantiate_tgt
from asdit.utils.config_class import MainTrainConfig
from asdit.utils.dcase_utils import parse_sec_cfg

logger = logging.getLogger(__name__)


def hydra_to_pydantic(config: DictConfig) -> MainTrainConfig:
    """Converts Hydra config to Pydantic config."""
    config_dict = parse_sec_cfg(OmegaConf.to_object(config))
    return MainTrainConfig(**config_dict)


def make_trainer(cfg: MainTrainConfig, ckpt_dir: Path) -> pl.Trainer:
    # Callbacks
    callback_list: List[Any] = [NaNCheckCallback()]
    for key_, cfg_ in cfg.callback_opts.items():
        callback_list.append(ModelCheckpoint(**{**cfg_, "dirpath": ckpt_dir}))
    callback_list.append(TQDMProgressBar(refresh_rate=cfg.refresh_rate))
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
            "check_val_every_n_epoch": cfg.every_n_epochs_valid,
        }
    )
    return trainer


def setup_datamodule(cfg: MainTrainConfig) -> pl.LightningDataModule:
    logger.info("Create datamodule")
    dm = PLDataModule(
        dm_cfg=cfg.datamodule,
        label_dict_path=cfg.label_dict_path,
    )
    return dm


def setup_frontend(cfg: MainTrainConfig) -> pl.LightningModule:
    logger.info("Create frontend")
    frontend = instantiate_tgt(
        {
            "label_dict_path": cfg.label_dict_path,
            **cfg.frontend.model_dump(),
        }
    )
    return frontend


@hydra.main(version_base=None, config_path="../../config/train", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg)
    if not cfg.trainer.get("deterministic", False):
        raise ValueError("Not deterministic!!!")
    logger.info(f"Start experiment: {HydraConfig().get().run.dir}")
    logger.info(f"version: {cfg.version}/{cfg.seed}")
    pl.seed_everything(cfg.seed, workers=True)
    # torch.autograd.set_detect_anomaly(False)

    ckpt_dir = get_version_dir(cfg=cfg) / "model" / cfg.model_ver / "checkpoints"
    if ckpt_dir.exists():
        logger.warning("already done")
        return

    dm = setup_datamodule(cfg)
    frontend = setup_frontend(cfg)
    trainer = make_trainer(cfg, ckpt_dir)

    logger.info("Start Training")
    trainer.fit(frontend, dm.train_dataloader(), dm.val_dataloader())


if __name__ == "__main__":
    main()
