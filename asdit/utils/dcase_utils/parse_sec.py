import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)


dcase2sec_dict = {
    "dcase2020": 11.0,
    "dcase2021": 10.0,
    "dcase2022": 10.0,
    "dcase2023": 18.0,
    "dcase2024": 12.0,
}


def dcase2sec(dcase: str) -> float:
    if dcase in dcase2sec_dict:
        return dcase2sec_dict[dcase]
    else:
        raise ValueError(f"Unexpected sec: {dcase}")


def parse_sec_inplace(cfg: Any):
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if k == "sec" and isinstance(v, str) and v.startswith("dcase202"):
                cfg[k] = dcase2sec(v)
                logger.info(f"sec: {v} -> {cfg[k]}")
            elif isinstance(v, (dict, list)):
                parse_sec_inplace(v)
    elif isinstance(cfg, list):
        for v in cfg:
            parse_sec_inplace(v)


def parse_sec_cfg(cfg: dict) -> dict:
    cfg_new = copy.deepcopy(cfg)
    parse_sec_inplace(cfg_new)
    return cfg_new
