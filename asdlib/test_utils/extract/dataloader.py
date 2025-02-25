# Copyright 2024 Takuya Fujimura

import logging
from typing import Any, Dict, List

from torch.utils.data import DataLoader

from ...datasets.pl_dataset import PLDataModule
from ...utils.config_class import DMConfig, MainConfig, MainTestConfig

logger = logging.getLogger(__name__)


def get_datamoduleconfig(
    cfg: MainTestConfig, past_cfg: MainConfig, machine_split_path_list: List[str]
) -> DMConfig:
    if past_cfg.datamodule.valid is not None:
        past_dmcfg = past_cfg.datamodule.valid
    else:
        past_dmcfg = past_cfg.datamodule.train

    dataloader_cfg = {
        "shuffle": False,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": False,
    }
    collator_cfg = {
        "tgt_class": "asdlib.datasets.BasicCollator",
        "sec": past_dmcfg.collator.sec,
        "sr": past_dmcfg.collator.sr,
        "need_feat": False,
        "shuffle": False,
    }
    dataset_cfg = {
        "tgt_class": "asdlib.datasets.BasicDataset",
        "path_selector_list": machine_split_path_list,
        "use_cache": False,
    }

    dmcfg_dict: Dict[str, Any] = {
        "dataloader": dataloader_cfg,
        "dataset": dataset_cfg,
        "collator": collator_cfg,
        "batch_sampler": None,
    }

    return DMConfig(**dmcfg_dict)


def get_machine_split_path_list(
    all_path: List[str], machine: str, split: str
) -> List[str]:
    machine_split_path_list = []
    for p in all_path:
        # p is in the format of "<data_dir>/formatted/<dcase>/raw/<machine>/train_or_test/hoge.wav
        split_p = p.split("/")
        if split_p[-3] == machine and split_p[-2] == split:
            machine_split_path_list.append(p)
    machine_split_path_list.sort()
    logger.info(f"{machine} {split}: {len(machine_split_path_list)}")
    return machine_split_path_list


def get_loader(
    cfg: MainTestConfig,
    past_cfg: MainConfig,
    all_path: List[str],
    machine: str,
    split: str,
) -> DataLoader:

    machine_split_path_list = get_machine_split_path_list(
        all_path=all_path, machine=machine, split=split
    )

    datamoduleconfig = get_datamoduleconfig(
        cfg=cfg, past_cfg=past_cfg, machine_split_path_list=machine_split_path_list
    )
    loader = PLDataModule.get_loader(
        datamoduleconfig=datamoduleconfig,
        label_dict_path=past_cfg.label_dict_path,  # TODO: make label using test
    )
    assert loader is not None

    return loader
