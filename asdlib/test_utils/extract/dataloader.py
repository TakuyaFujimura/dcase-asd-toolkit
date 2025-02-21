# Copyright 2024 Takuya Fujimura

import logging
from typing import List

from torch.utils.data import DataLoader

from ...datasets.pl_dataset import BasicDataModule
from ...utils.config_class import MainConfig, MainTestConfig
from ...utils.config_class.main_config import (
    BasicDMConfig,
    BasicDMSplitConfig,
    MainConfig,
)

logger = logging.getLogger(__name__)


def get_datasetconfig(
    cfg: MainTestConfig, past_cfg: MainConfig, machine_split_path_list: List[str]
) -> BasicDMConfig:
    datasetconfig = BasicDMSplitConfig(**past_cfg.datamodule).train
    datasetconfig.collator.shuffle = False
    datasetconfig.dataset.use_cache = False
    datasetconfig.dataset.path_selector_list = machine_split_path_list
    datasetconfig.batch_sampler = None
    datasetconfig.dataloader["shuffle"] = False
    datasetconfig.dataloader["batch_size"] = cfg.batch_size
    datasetconfig.dataloader["num_workers"] = cfg.num_workers
    datasetconfig.dataloader["pin_memory"] = False
    return datasetconfig


def get_machine_split_path_list(
    all_path: List[str], machine: str, split: str
) -> List[str]:
    machine_split_path_list = []
    for p in all_path:
        # p is in the format of "<data_dir>/<dcase>/all/raw/<machine>/train-test/hoge.wav
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

    if past_cfg.datamodule_type == "basic":
        datasetconfig = get_datasetconfig(
            cfg=cfg, past_cfg=past_cfg, machine_split_path_list=machine_split_path_list
        )
        loader = BasicDataModule.get_loader(
            datasetconfig=datasetconfig,
            label_dict_path={},
        )
    else:
        # if past_cfg.datamodule_type == "gen":
        #    collator_cfg = past_cfg.datamodule.valid.collator
        raise NotImplementedError("Unexpected datamodule type")

    return loader
