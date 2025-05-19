import importlib
import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional, cast

from omegaconf import OmegaConf

from asdit.bin.utils.path import get_version_dir
from asdit.frontends.base_plmodel import BasePLFrontend
from asdit.utils.common import get_best_path, get_path_glob
from asdit.utils.config_class import DMConfig, MainExtractConfig, MainTrainConfig
from asdit.utils.dcase_utils.parse_sec import parse_sec_cfg

logger = logging.getLogger(__name__)


class ExtractDMConfigMaker:
    def __init__(
        self,
        data_dir: str,
        dcase: str,
        sec: float | Literal["all"],
        sr: int,
        machine: str,
        dataloader_cfg: Optional[Dict[str, Any]] = None,
        collator_cfg: Optional[Dict[str, Any]] = None,
        dataset_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.data_dir = data_dir
        self.dcase = dcase
        self.machine = machine
        if dataloader_cfg is None:
            dataloader_cfg = {}
        if collator_cfg is None:
            collator_cfg = {}
        if dataset_cfg is None:
            dataset_cfg = {}

        self.dataloader_cfg = {
            "batch_size": 64,
            "num_workers": 0,
            **dataloader_cfg,
            "shuffle": False,
            "pin_memory": False,
        }
        self.collator_cfg = {
            "tgt_class": "asdit.datasets.BasicCollator",
            "sec": sec,
            "sr": sr,
            **collator_cfg,
            "need_feat": False,
            "shuffle": False,
        }
        self.dataset_cfg = {
            "tgt_class": "asdit.datasets.BasicDataset",
            **dataset_cfg,
        }

    def get_config(self, split: str) -> DMConfig:

        path_selector_list = [
            f"{self.data_dir}/formatted/{self.dcase}/raw/{self.machine}/{split}/*.wav"
        ]

        dmcfg_dict: Dict[str, Any] = {
            "dataloader": self.dataloader_cfg,
            "dataset": {**self.dataset_cfg, "path_selector_list": path_selector_list},
            "collator": self.collator_cfg,
            "batch_sampler": None,
        }

        return DMConfig(**dmcfg_dict)


def get_past_cfg(cfg: MainExtractConfig) -> MainTrainConfig:
    assert cfg.model_ver is not None
    config_path = (
        get_version_dir(cfg=cfg) / "model" / cfg.model_ver / ".hydra/config.yaml"
    )

    past_cfg = MainTrainConfig(
        **parse_sec_cfg(OmegaConf.to_object(OmegaConf.load(config_path)))
    )
    return past_cfg


def get_ckpt_path(cfg: MainExtractConfig) -> Path:
    assert cfg.model_ver is not None
    assert cfg.ckpt_ver is not None
    ckpt_dir = get_version_dir(cfg=cfg) / "model" / cfg.model_ver / "checkpoints"
    if cfg.ckpt_ver in ["min", "max"]:
        ckpt_path = get_best_path(ckpt_dir, cfg.ckpt_ver)
    elif cfg.ckpt_ver == "last":
        ckpt_path = ckpt_dir / "last.ckpt"
    elif cfg.ckpt_ver.startswith("epoch"):
        epoch = int(cfg.ckpt_ver.split("_")[-1])
        ckpt_condition = str(ckpt_dir / f"interval_epoch={epoch-1}-*.ckpt")
        ckpt_path = Path(get_path_glob(ckpt_condition))
    return ckpt_path


def restore_plfrontend(
    cfg: MainExtractConfig, past_cfg: MainTrainConfig
) -> BasePLFrontend:
    ckpt_path = get_ckpt_path(cfg)
    logger.info(f"Loading model from {ckpt_path}")
    module_name = ".".join(past_cfg.frontend.tgt_class.split(".")[:-1])
    class_name = past_cfg.frontend.tgt_class.split(".")[-1]
    strict = not getattr(past_cfg.frontend, "partially_saved_param_list", False)
    module = importlib.import_module(module_name)
    frontend_cls = getattr(module, class_name)
    frontend = frontend_cls.load_from_checkpoint(ckpt_path, strict=strict)
    frontend.to(cfg.device)
    frontend.eval()
    logger.info("Model was successfully loaded from the ckpt_path")
    return frontend


def restore_dmconfigmaker(
    cfg: MainExtractConfig, past_cfg: MainTrainConfig
) -> ExtractDMConfigMaker:
    if past_cfg.datamodule.valid is not None:
        past_collator_cfg = past_cfg.datamodule.valid.collator
    else:
        past_collator_cfg = past_cfg.datamodule.train.collator
    dmconfigmaker = ExtractDMConfigMaker(
        data_dir=past_cfg.data_dir,
        dcase=past_cfg.dcase,
        sec=past_collator_cfg["sec"],
        sr=past_collator_cfg["sr"],
        machine=cfg.machine,
        **cfg.datamodule,
    )
    return dmconfigmaker
