from pathlib import Path

import numpy as np


def load_npz_dict(npz_path: Path | str) -> dict:
    extract_dict = {}
    with np.load(npz_path) as npz:
        for key, value in npz.items():
            extract_dict[key] = value
    return extract_dict
