# Copyright 2024 Takuya Fujimura

import logging
from pathlib import Path
from typing import List, Tuple

from lightning import LightningModule

from mypkgs.datasets.torch_dataset import parse_path_selector

from ...pl_models import BasicDisPLModel
from ...utils.config_class import MainConfig, MainTestConfig
from ...utils.io_utils import get_best_path, get_path_glob
from .dataloader import get_loader
from .extract import extract

logger = logging.getLogger(__name__)


def get_ckpt_path(cfg: MainTestConfig) -> Path:
    ckpt_dir = cfg.result_dir / cfg.name / cfg.version / "checkpoints"
    if cfg.infer_ver == "best":
        ckpt_path = get_best_path(ckpt_dir)
    elif cfg.infer_ver == "last":
        ckpt_path = ckpt_dir / "last.ckpt"
    elif cfg.infer_ver.startswith("epoch"):
        epoch = int(cfg.infer_ver.split("_")[-1])
        ckpt_condition = str(ckpt_dir / f"interval_epoch={epoch-1}-*.ckpt")
        ckpt_path = Path(get_path_glob(ckpt_condition))
    return ckpt_path


def load_plmodel(cfg: MainTestConfig, past_cfg: MainConfig) -> LightningModule:
    ckpt_path = get_ckpt_path(cfg)
    if past_cfg.model.tgt_class == "mypkgs.pl_models.BasicDisPLModel":
        plmodel = BasicDisPLModel.load_from_checkpoint(ckpt_path)
    else:
        raise NotImplementedError("Unexpected model class")
    plmodel.to(cfg.device)
    plmodel.eval()
    logger.info("Model was successfully loaded from ckpt_path")
    return plmodel


def get_all_path_machine(
    cfg: MainTestConfig, past_cfg: MainConfig
) -> Tuple[List[str], List[str]]:
    all_path: List[str] = []
    machines: List[str] = []
    for selector in cfg.path_selector_list:
        all_path += parse_path_selector(selector)

    for p in all_path:
        # p is in the format of "<data_dir>/<dcase>/all/raw/<machine>/train-test/hoge.wav
        split_p = p.split("/")
        machines.append(split_p[-3])
        data_dir = "/".join(split_p[:-6])
        dcase = split_p[-6]
        if data_dir != past_cfg.data_dir:
            logger.warning(f"Unmatched data_dir: {data_dir} vs {past_cfg.data_dir}")
        if dcase != past_cfg.dcase:
            logger.warning(f"Unmatched dcase: {dcase} vs {past_cfg.dcase}")

    machines = list(set(machines))
    logger.info(f"{len(machines)} machines: {machines}")
    return all_path, machines


def extract_main(cfg: MainTestConfig, past_cfg: MainConfig, infer_dir: Path):
    plmodel = load_plmodel(cfg=cfg, past_cfg=past_cfg)
    all_path, machines = get_all_path_machine(cfg=cfg, past_cfg=past_cfg)

    for m in machines:
        logger.info(f"Start extracting {m}")
        machine_dir = infer_dir / m
        machine_dir.mkdir(exist_ok=True, parents=True)

        for split in ["train", "test"]:
            loader = get_loader(
                cfg=cfg,
                past_cfg=past_cfg,
                all_path=all_path,
                machine=m,
                split=split,
            )
            df = extract(dataloader=loader, plmodel=plmodel, device=cfg.device)
            df_path = machine_dir / f"{split}_extraction.csv"
            df.to_csv(df_path, index=False)
            logger.info(f"Saved extraction result to {df_path}")
