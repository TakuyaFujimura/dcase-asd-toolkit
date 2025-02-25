import logging
from pathlib import Path
from typing import Any, Dict, List, cast

import hydra
import lightning.pytorch as pl
import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, field_validator

from asdlib.datasets.collators import get_relative_dcase_path
from asdlib.datasets.torch_dataset import parse_path_selector
from asdlib.labelers.base import LabelerBase
from asdlib.utils.dcase_utils import get_dcase_info
from asdlib.utils.io_utils.json_util import write_json
from asdlib.utils.pl_utils.idx_util import LabelInfo

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
    split_idx_dict = {"train": [], "test": []}
    for path in tqdm.tqdm(path_list):
        path = get_relative_dcase_path(path=path)
        idx = labeler.trans(path)
        path2idx_dict[path] = idx
        split_idx_dict[get_dcase_info(path=path, label="split")] += [idx]  # type: ignore

    labelinfo_dict = {
        "num_class": len(set(path2idx_dict.values())),
        "num_class_train": len(set(split_idx_dict["train"])),
        "num_class_test": len(set(split_idx_dict["test"])),
        "path2idx_dict": path2idx_dict,
    }
    return labelinfo_dict


@hydra.main(version_base=None, config_path="../../config/label", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg)
    if cfg.save_path.exists() and not cfg.overwrite:
        raise FileExistsError(
            f"{cfg.save_path} already exists. Please set overwrite=True"
        )
    logger.info(f"{cfg.save_path.parent} is created.")
    cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(cfg.seed, workers=True)

    path_list = get_path_list(cfg=cfg)
    labeler: LabelerBase = instantiate(cfg.labeler)
    labeler.fit(all_path_list=path_list)
    labelinfo_dict = get_labelinfo_dict(path_list=path_list, labeler=labeler)
    write_json(json_path=cfg.save_path, data=labelinfo_dict)


if __name__ == "__main__":
    main()
