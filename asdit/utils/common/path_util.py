import glob
import logging
from operator import gt, lt
from pathlib import Path

logger = logging.getLogger(__name__)


def get_path_glob(glob_condition: str) -> str:
    path_list = glob.glob(glob_condition)
    if len(path_list) != 1:
        logger.error(f"Error: {len(path_list)} files found for {glob_condition}")
        raise FileNotFoundError(
            f"Error: {len(path_list)} files found for {glob_condition}"
        )
    return path_list[0]


def get_best_path(ckpt_dir: Path, best_type: str) -> Path:
    if best_type == "min":
        best_loss: float = float("inf")
        compare_fn = lt
    elif best_type == "max":
        best_loss: float = float("-inf")
        compare_fn = gt
    else:
        logger.error(f"Invalid best_type: {best_type}")
        raise ValueError(f"Invalid best_type: {best_type}")

    best_path: Path = Path("tmp")
    best_epoch: int = -1

    for ckpt_path in ckpt_dir.glob("epoch=*.ckpt"):
        loss = float(ckpt_path.stem.split("=")[-1])
        epoch = int(ckpt_path.stem.split("=")[1].split("-")[0])

        if compare_fn(loss, best_loss) or (loss == best_loss and best_epoch < epoch):
            best_loss = loss
            best_path = ckpt_path
            best_epoch = epoch

    return best_path
