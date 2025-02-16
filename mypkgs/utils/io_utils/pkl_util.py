import pickle
from pathlib import Path


def write_pkl(pkl_path: Path | str, data: dict) -> None:
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)


def read_pkl(pkl_path: Path | str) -> dict:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data
