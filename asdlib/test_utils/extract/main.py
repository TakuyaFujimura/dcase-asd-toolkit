# Copyright 2024 Takuya Fujimura

import importlib
import logging
from pathlib import Path
from typing import List

from ...pl_models import BasePLModel
from ...utils.config_class import MainConfig, MainTestConfig
from ...utils.io_utils import get_best_path, get_path_glob
from .dataloader import get_loader
from .extract import extract

logger = logging.getLogger(__name__)


def get_ckpt_path(cfg: MainTestConfig) -> Path:
    ckpt_dir = cfg.result_dir / cfg.name / cfg.version / cfg.machine / "checkpoints"
    if cfg.infer_ver == "best":
        ckpt_path = get_best_path(ckpt_dir)
    elif cfg.infer_ver == "last":
        ckpt_path = ckpt_dir / "last.ckpt"
    elif cfg.infer_ver.startswith("epoch"):
        epoch = int(cfg.infer_ver.split("_")[-1])
        ckpt_condition = str(ckpt_dir / f"interval_epoch={epoch-1}-*.ckpt")
        ckpt_path = Path(get_path_glob(ckpt_condition))
    return ckpt_path


def load_plmodel(cfg: MainTestConfig, past_cfg: MainConfig) -> BasePLModel:
    ckpt_path = get_ckpt_path(cfg)
    module_name = ".".join(past_cfg.model.tgt_class.split(".")[:-1])
    class_name = past_cfg.model.tgt_class.split(".")[-1]
    module = importlib.import_module(module_name)
    plmodel_cls = getattr(module, class_name)
    plmodel = plmodel_cls.load_from_checkpoint(ckpt_path)
    plmodel.to(cfg.device)
    plmodel.eval()
    logger.info("Model was successfully loaded from ckpt_path")
    return plmodel


def extract_main(
    cfg: MainTestConfig,
    past_cfg: MainConfig,
    machine_dir: Path,
    all_path: List[str],
) -> None:
    plmodel = load_plmodel(cfg=cfg, past_cfg=past_cfg)
    logger.info(f"Start extracting {machine_dir}")

    for split in ["train", "test"]:
        loader = get_loader(
            cfg=cfg,
            past_cfg=past_cfg,
            all_path=all_path,
            machine=machine_dir.parents[1].name,
            split=split,
        )
        df = extract(dataloader=loader, plmodel=plmodel, device=cfg.device)
        df_path = machine_dir / f"{split}_extraction.csv"
        df.to_csv(df_path, index=False)
        logger.info(f"Saved extraction result to {df_path}")
