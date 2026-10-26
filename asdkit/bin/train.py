import copy
import logging
from pathlib import Path
from typing import List, Optional

import hydra
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import Callback, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

from asdkit.datasets import PLDataModule
from asdkit.utils.asdkit_utils.callbacks import ModelCheckpoint, NaNCheckCallback
from asdkit.utils.asdkit_utils.path import get_version_dir
from asdkit.utils.common import instantiate
from asdkit.utils.common.checkpoint import to_checkpoint_config
from asdkit.utils.common.hydra import hydra_to_pydantic
from asdkit.utils.config_class import MainTrainConfig
from asdkit.utils.dcase_utils import parse_sec_cfg

logger = logging.getLogger(__name__)


def create_training_callbacks(
    cfg: MainTrainConfig, ckpt_dir: Path, resume_ckpt_path: Optional[str]
) -> List[Callback]:
    # Callbacks
    callback_list: List[Callback] = [NaNCheckCallback()]
    for callback_cfg in cfg.callback.callbacks.values():
        callback_list.append(
            ModelCheckpoint(
                **callback_cfg, dirpath=ckpt_dir, resume_ckpt_path=resume_ckpt_path
            )
        )
    callback_list.append(TQDMProgressBar(refresh_rate=cfg.callback.tqdm_refresh_rate))
    # Other callbacks
    for callback_cfg in cfg.other_callbacks.values():
        callback_list.append(instantiate(callback_cfg))
    return callback_list


def make_trainer(
    cfg: MainTrainConfig, ckpt_dir: Path, resume_ckpt_path: Optional[str]
) -> pl.Trainer:
    callback_list = create_training_callbacks(
        cfg=cfg, ckpt_dir=ckpt_dir, resume_ckpt_path=resume_ckpt_path
    )
    # Logger
    pl_logger = TensorBoardLogger(
        save_dir=f"{cfg.result_dir}/{cfg.name}",
        name=cfg.dcase,
        version=f"{cfg.version}",
        sub_dir=f"{cfg.seed}/model/{cfg.model_ver}",
    )
    # Trainer
    return instantiate(
        cfg=cfg.trainer, callbacks=callback_list, logger=pl_logger, _convert_="all"
    )


def setup_datamodule(cfg: MainTrainConfig) -> pl.LightningDataModule:
    logger.info("Create datamodule")
    dm = PLDataModule(dm_cfg=cfg.datamodule)
    return dm


def setup_frontend(cfg: MainTrainConfig) -> pl.LightningModule:
    """Construct a frontend and retain its construction config for checkpoint restore.

    With checkpoint_weights_only enabled, convert configuration objects to basic
    types supported by torch.load(weights_only=True). Otherwise, preserve the
    original Python types with a deep copy for loading with weights_only=False.
    """
    logger.info("Create frontend")
    frontend_cfg = (
        to_checkpoint_config(cfg.frontend)
        if cfg.checkpoint_weights_only
        else copy.deepcopy(cfg.frontend)
    )
    frontend = instantiate(frontend_cfg)
    frontend.checkpoint_frontend_cfg = frontend_cfg
    return frontend


@hydra.main(version_base=None, config_path="../../config/train", config_name="main")
def main(hydra_cfg: DictConfig) -> None:
    cfg: MainTrainConfig = hydra_to_pydantic(
        hydra_cfg=hydra_cfg, cls=MainTrainConfig, transform=parse_sec_cfg
    )
    if not cfg.trainer.get("deterministic", False):
        logger.warning("Not deterministic!")
    logger.info(f"Start experiment: {HydraConfig().get().run.dir}")
    logger.info(f"version: {cfg.version}/{cfg.seed}")
    pl.seed_everything(cfg.seed, workers=True)

    ckpt_dir = get_version_dir(cfg=cfg) / "model" / cfg.model_ver / "checkpoints"
    if ckpt_dir.exists() and cfg.resume_ckpt_path is None:
        logger.warning("Already done. Skipping training...")
        return

    dm = setup_datamodule(cfg)
    frontend = setup_frontend(cfg)
    trainer = make_trainer(
        cfg=cfg, ckpt_dir=ckpt_dir, resume_ckpt_path=cfg.resume_ckpt_path
    )

    logger.info("Start Training")
    trainer.fit(
        frontend,
        dm.train_dataloader(),
        dm.val_dataloader(),
        ckpt_path=cfg.resume_ckpt_path,
    )


if __name__ == "__main__":
    main()
