import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import hydra
import lightning.pytorch as pl
import tqdm
from omegaconf import DictConfig
from pydantic import BaseModel, field_validator

from asdkit.datasets.collators import get_dcase_path
from asdkit.datasets.torch_dataset import parse_path_selector
from asdkit.labelers.base import LabelerBase
from asdkit.utils.common import instantiate
from asdkit.utils.common.hydra import hydra_to_pydantic
from asdkit.utils.common.json_util import write_json
from asdkit.utils.dcase_utils import get_dcase_info

logger = logging.getLogger(__name__)


class Config(BaseModel):
    seed: int = 0
    save_path: Path
    path_selector_list: List[str]
    labeler: Dict[Any, Any]
    machinewise: bool = False
    overwrite: bool = False

    @field_validator("save_path", mode="before")
    def check_save_path(cls, v: Union[str, Path]) -> Path:
        path = Path(v)
        if path.suffix == ".json":
            return path
        else:
            raise ValueError("save_path must be a json file.")


def set_logging(dst_dir: Path) -> None:
    logging.basicConfig(
        filename=dst_dir / f"{Path(__file__).stem}.log",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_path_list(cfg: Config) -> List[str]:
    path_list: List[str] = []
    logger.info("Start Loading Paths")
    for selector in cfg.path_selector_list:
        path_list.extend(parse_path_selector(selector))
    return sorted(path_list)


def get_labelinfo_dict(
    path_list: Iterable[str], labeler: LabelerBase
) -> Dict[str, Any]:
    path2idx_dict: Dict[str, int] = {}
    split_class_ids_dict: Dict[str, List[int]] = {"train": [], "test": []}
    for path in tqdm.tqdm(path_list):
        path = get_dcase_path(path=path)
        idx = labeler.trans(path)
        path2idx_dict[path] = idx
        split = get_dcase_info(path=path, label="split")
        split_class_ids_dict[split].append(idx)  # type: ignore

    labelinfo_dict = {
        "num_class": len(set(path2idx_dict.values())),
        "num_class_train": len(set(split_class_ids_dict["train"])),
        "num_class_test": len(set(split_class_ids_dict["test"])),
        "path2idx_dict": path2idx_dict,
    }
    return labelinfo_dict


@hydra.main(version_base=None, config_path="../../config/label", config_name="main")
def main(hydra_cfg: DictConfig) -> None:
    cfg: Config = hydra_to_pydantic(hydra_cfg=hydra_cfg, cls=Config)
    pl.seed_everything(cfg.seed, workers=True)
    path_list = get_path_list(cfg=cfg)
    if not path_list:
        raise ValueError("No files matched path_selector_list.")

    # Group paths by machine if machinewise is True
    path_groups: Dict[str, List[str]] = {}
    for path in path_list:
        machine = (
            str(get_dcase_info(path=path, label="machine")) if cfg.machinewise else ""
        )
        path_groups.setdefault(machine, []).append(path)

    # Process each group of paths
    for machine, paths in path_groups.items():
        save_path = (
            cfg.save_path.with_name(f"{machine}_{cfg.save_path.name}")
            if cfg.machinewise
            else cfg.save_path
        )
        if save_path.exists() and not cfg.overwrite:
            logger.info(f"Skipping existing labels: {save_path}")
            continue
        logger.info(f"{save_path.parent} is created.")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        labeler: LabelerBase = instantiate(cfg.labeler)
        labeler.fit(all_path_list=paths)
        labelinfo_dict = get_labelinfo_dict(path_list=paths, labeler=labeler)
        write_json(json_path=save_path, data=labelinfo_dict)


if __name__ == "__main__":
    main()
