from pathlib import Path

from asdit.utils.config_class import (
    MainEvaluateConfig,
    MainExtractConfig,
    MainScoreConfig,
    MainTableConfig,
    MainUmapConfig,
)
from asdit.utils.config_class.train_config import MainTrainConfig


def get_version_dir(
    cfg: (
        MainTrainConfig
        | MainExtractConfig
        | MainScoreConfig
        | MainEvaluateConfig
        | MainUmapConfig
        | MainTableConfig
    ),
) -> Path:
    version_dir = cfg.result_dir / cfg.name / cfg.dcase / f"{cfg.version}_{cfg.seed}"
    return version_dir


def get_output_dir(
    cfg: MainExtractConfig | MainScoreConfig | MainEvaluateConfig | MainUmapConfig,
) -> Path:
    output_dir = get_version_dir(cfg=cfg) / "output" / cfg.infer_ver / cfg.machine
    return output_dir


def dir_has_file(dir_path: Path, file_name: str) -> bool:
    return len(list(dir_path.glob(file_name))) > 0


def check_file_exists(dir_path: Path, file_name: str, overwrite: bool) -> None:
    if dir_has_file(dir_path, file_name) and not overwrite:
        raise FileExistsError(
            f"{dir_path}/{file_name} already exists. "
            + "Set config.overwrite=True to overwrite."
        )
