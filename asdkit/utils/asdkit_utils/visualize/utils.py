from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


def load_npz_dict(npz_path: Union[Path, str]) -> Dict[str, Any]:
    extract_dict: Dict[str, Any] = {}
    with np.load(npz_path) as npz:
        for key, value in npz.items():
            extract_dict[key] = value
    return extract_dict
