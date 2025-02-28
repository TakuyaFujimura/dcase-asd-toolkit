from pathlib import Path

from asdlib.utils.config_class import (
    MainEvaluateConfig,
    MainExtractConfig,
    MainScoreConfig,
)


def get_output_dir(
    cfg: MainExtractConfig | MainScoreConfig | MainEvaluateConfig,
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


def check_file_exists(dir_path: Path, file_name: str, overwrite: bool) -> None:
    dir_has_file = len(list(dir_path.glob(file_name))) > 0
    if dir_has_file and not overwrite:
        raise FileExistsError(
            f"{dir_path}/{file_name} already exists. Set config.overwrite=True to overwrite."
        )
