from pathlib import Path
from typing import Dict, cast

from pydantic import BaseModel

from ..io_utils import read_json


class LabelInfo(BaseModel):
    path2idx_dict: Dict[str, int]
    num_class: int


def check_label_dict(label_dict: LabelInfo):
    if set(label_dict.path2idx_dict.values()) != set(range(label_dict.num_class)):
        raise ValueError("path2idx_dict is not correct")


def get_label_dict(label_dict_path: Dict[str, Path]) -> Dict[str, LabelInfo]:
    label_dict: Dict[str, LabelInfo] = {}
    for key, path in label_dict_path.items():
        label_dict[key] = LabelInfo(**cast(dict, read_json(path)))
        check_label_dict(label_dict[key])
    return label_dict
