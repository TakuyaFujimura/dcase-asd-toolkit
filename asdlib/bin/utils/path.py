from pathlib import Path

from asdlib.utils.config_class import MainExtractConfig


def get_output_dir(cfg: MainExtractConfig) -> Path:
    output_dir = (
        cfg.result_dir
        / cfg.name
        / f"{cfg.version}_{cfg.seed}"
        / "output"
        / cfg.ckpt_ver
        / cfg.machine
    )
    return output_dir
