import logging
from pathlib import Path
from typing import Any, Dict, List, cast

import hydra
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, field_validator

import mypkgs
from mypkgs.datasets.torch_dataset import parse_path_selector
from mypkgs.labelers.base import LabelerBase
from mypkgs.utils.io_utils.json_util import write_json
from mypkgs.utils.pl_utils.idx_util import LabelInfo

logger = logging.getLogger(__name__)


class Config(BaseModel):
    seed: int = 0
    save_path: Path
    path_selector_list: List[str]
    labeler: dict
    overwrite: bool = False

    @field_validator("save_path", mode="before")
    def check_save_path(cls, v):
        if v.endswith(".json"):
            return v
        else:
            raise ValueError("save_path must be a json file.")


def hydra_to_pydantic(config: DictConfig) -> Config:
    """Converts Hydra config to Pydantic config."""
    config_dict = cast(Dict[str, Any], OmegaConf.to_object(config))
    return Config(**config_dict)


def set_logging(dst_dir: Path):
    logging.basicConfig(
        filename=dst_dir / f"{Path(__file__).stem}.log",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_path_list(cfg: Config) -> List[str]:
    path_list: List[str] = []
    logger.info("Start Loading Paths")
    for selector in cfg.path_selector_list:
        path_list += parse_path_selector(selector)
    return sorted(path_list)


def get_labelinfo_dict(path_list: List[str], labeler) -> dict:
    path2idx_dict = {}
    for path in path_list:
        path2idx_dict[path] = labeler.trans(path)
    labelinfo_dict = {
        "num_class": len(set(path2idx_dict.values())),
        # I don't use labeler.num_class because it won't be correct
        # when there is train/test mismatch in labeler.
        # i.e., labeler.fit(path_list_1).num_class != len(set([labeler.trans(p) for p in path_list_2]))
        "path2idx_dict": path2idx_dict,
    }
    return labelinfo_dict


@hydra.main(version_base=None, config_path="config/label", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg)
    if cfg.save_path.exists() and not cfg.overwrite:
        raise FileExistsError(
            f"{cfg.save_path} already exists. Please set overwrite=True"
        )
    cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(cfg.seed, workers=True)

    path_list = get_path_list(cfg=cfg)
    labeler: LabelerBase = instantiate(cfg.labeler)
    labeler.fit(all_path_list=path_list)
    labelinfo_dict = get_labelinfo_dict(path_list=path_list, labeler=labeler)
    write_json(json_path=cfg.save_path, data=labelinfo_dict)


if __name__ == "__main__":
    main()
