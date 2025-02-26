import logging
from pathlib import Path
from typing import Any, Dict, List, cast

import hydra
import lightning.pytorch as pl
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from asdlib.datasets import PLDataModule
from asdlib.utils.config_class import MainConfig
from asdlib.utils.pl_utils import NaNCheckCallback, instantiate_tgt

logger = logging.getLogger(__name__)


def make_trainer(cfg: MainConfig, ckpt_dir: Path) -> pl.Trainer:
    # Callbacks
    callback_list: List[Any] = [NaNCheckCallback()]
    for key_, cfg_ in cfg.callback_opts.items():
        callback_list.append(ModelCheckpoint(**{**cfg_, "dirpath": ckpt_dir}))
    callback_list.append(TQDMProgressBar(refresh_rate=cfg.refresh_rate))
    # Logger
    pl_logger = TensorBoardLogger(
        save_dir=cfg.result_dir, name=cfg.name, version=cfg.version, sub_dir=cfg.machine
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


def hydra_to_pydantic(config: DictConfig) -> MainConfig:
    """Converts Hydra config to Pydantic config."""
    config_dict = cast(Dict[str, Any], OmegaConf.to_object(config))
    return MainConfig(**config_dict)


def set_logging(dst_dir: Path):
    logging.basicConfig(
        filename=dst_dir / f"{Path(__file__).stem}.log",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def setup_datamodule(cfg: MainConfig) -> pl.LightningDataModule:
    logger.info("Create datamodule")
    dm = PLDataModule(
        dm_cfg=cfg.datamodule,
        label_dict_path=cfg.label_dict_path,
    )
    return dm


def setup_plmodel(cfg: MainConfig) -> pl.LightningModule:
    logger.info("Create plmodel")
    plmodel = instantiate_tgt(
        {
            "label_dict_path": cfg.label_dict_path,
            **cfg.model.model_dump(),
        }
    )
    return plmodel


@hydra.main(version_base=None, config_path="../../config/train", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg)
    # set_logging(cfg.result_dir)
    if not cfg.trainer.get("deterministic", False):
        raise ValueError("Not deterministic!!!")
    if cfg.tf32:
        torch.set_float32_matmul_precision("high")
        logger.info("Set float32_matmul_precision to high")
    logger.info(f"Start experiment: {HydraConfig().get().run.dir}")
    logger.info(f"version: {cfg.version}")
    pl.seed_everything(cfg.seed, workers=True)
    # torch.autograd.set_detect_anomaly(False)

    ckpt_dir = cfg.result_dir / cfg.name / cfg.version / cfg.machine / "checkpoints"
    if ckpt_dir.exists():
        logger.warning("already done")
        return

    dm = setup_datamodule(cfg)
    plmodel = setup_plmodel(cfg)
    trainer = make_trainer(cfg, ckpt_dir)

    logger.info("Start Training")
    trainer.fit(plmodel, dm.train_dataloader(), dm.val_dataloader())


if __name__ == "__main__":
    main()
