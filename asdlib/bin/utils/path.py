from pathlib import Path

from asdlib.utils.config_class import (
    MainEvaluateConfig,
    MainExtractConfig,
    MainScoreConfig,
    MainUmapConfig,
)


def get_output_dir(
    cfg: MainExtractConfig | MainScoreConfig | MainEvaluateConfig | MainUmapConfig,
) -> Path:
    output_dir = (
        cfg.result_dir
        / cfg.name
        / f"{cfg.version}_{cfg.seed}"
        / "output"
        / cfg.ckpt_ver
        / cfg.machine
    )
    return output_dir


def dir_has_file(dir_path: Path, file_name: str) -> bool:
    return len(list(dir_path.glob(file_name))) > 0


def check_file_exists(dir_path: Path, file_name: str, overwrite: bool) -> None:
    if dir_has_file(dir_path, file_name) and not overwrite:
        raise FileExistsError(
            f"{dir_path}/{file_name} already exists. "
            + "Set config.overwrite=True to overwrite."
        )
