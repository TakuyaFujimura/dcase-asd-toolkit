import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def get_version_dir(cfg) -> Path:
    version_dir = cfg.result_dir / cfg.name / cfg.dcase / cfg.version / str(cfg.seed)
    return version_dir


def get_output_dir(cfg) -> Path:
    output_dir = get_version_dir(cfg=cfg) / "output" / cfg.infer_ver / cfg.machine
    return output_dir


def dir_has_file(dir_path: Path, file_name: str) -> bool:
    return len(list(dir_path.glob(file_name))) > 0


def check_file_exists(dir_path: Path, file_name: str, overwrite: bool) -> None:
    if dir_has_file(dir_path, file_name) and not overwrite:
        logger.warning(
            "[Skip this process] "
            f"{dir_path}/{file_name} already exists. "
            + "Set asdit_cfg.overwrite=True to overwrite it."
        )
        sys.exit(1)


def make_output_dir(cfg, check_file_name: str) -> Path:
    output_dir = get_output_dir(cfg)
    output_dir.mkdir(exist_ok=True, parents=True)
    check_file_exists(
        dir_path=output_dir, file_name=check_file_name, overwrite=cfg.overwrite
    )
    return output_dir
