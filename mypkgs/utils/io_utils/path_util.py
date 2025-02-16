import glob
from pathlib import Path


def get_path_glob(glob_condition: str) -> str:
    path_list = glob.glob(glob_condition)
    if len(path_list) != 1:
        raise FileNotFoundError(
            f"Error: {len(path_list)} files found for {glob_condition}"
        )
    return path_list[0]


def get_best_path(ckpt_dir: Path) -> Path:
    min_loss: float = float("inf")
    min_path: Path = Path("tmp")
    min_epoch: int = -1

    for ckpt_path in ckpt_dir.glob("epoch=*.ckpt"):
        loss = float(ckpt_path.stem.split("=")[-1])
        epoch = int(ckpt_path.stem.split("=")[1].split("-")[0])
        if (loss < min_loss) or (loss == min_loss and min_epoch < epoch):
            min_loss = loss
            min_path = ckpt_path
            min_epoch = epoch

    return min_path
