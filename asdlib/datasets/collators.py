import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from ..utils.pl_utils import LabelInfo, get_label_dict

logger = logging.getLogger(__name__)


def get_relative_dcase_path(path: str) -> str:
    # <data_dir>/formatted/dcase2021/raw/fan/train/hoge.wav
    path_split = path.split("/")
    assert path_split[-2] in ["train", "test"]
    assert path_split[-4] == "raw"
    assert path_split[-5].startswith("dcase202")
    assert path_split[-6] == "formatted"
    return "/".join(path_split[-5:])


def get_dcase_info(path: str, label: str) -> str | int:
    # section_00_target_train_normal_0009_<attribute>.wav
    split_path = path.split("/")[-1].split("_")
    if label == "machine":
        return path.split("/")[-3]
    elif label == "section":
        return int(split_path[1])
    elif label == "is_target":
        assert split_path[2] in ["source", "target"]
        return int(split_path[2] == "target")
    elif label == "is_normal":
        assert split_path[4] in ["normal", "anomaly"]
        return int(split_path[4] == "normal")
    elif label == "attr":
        return "_".join(split_path[6:]).replace(".wav", "")
    else:
        raise ValueError(f"Unknown label: {label}")


def wave_pad(wave: torch.Tensor, crop_len: int, pad_type: str = "tile") -> torch.Tensor:
    if len(wave) < crop_len:
        if pad_type == "tile":
            wave = wave.tile(int(np.ceil(crop_len / len(wave))))
        else:
            raise NotImplementedError()
    return wave


def wave_rand_crop(wave: torch.Tensor, crop_len: int) -> torch.Tensor:
    assert len(wave) >= crop_len
    start_frame = np.random.randint(0, max(1, len(wave) - crop_len))
    return wave[start_frame : start_frame + crop_len]


class BasicCollator(object):
    """Wave form data's collator."""

    def __init__(
        self,
        label_dict_path: Dict[str, Path],
        sec: float,
        sr: int,
        shuffle: bool = True,
    ):
        self.label_dict: Dict[str, LabelInfo] = get_label_dict(label_dict_path)
        self.crop_len = int(sr * sec)
        self.shuffle = shuffle

        if sr != 16000:
            raise ValueError("Only 16kHz is supported.")

    def crop_wave(self, wave: torch.Tensor) -> torch.Tensor:
        assert len(wave.shape) == 1
        wave = wave_pad(wave, self.crop_len, "tile")
        if self.shuffle:
            wave = wave_rand_crop(wave, self.crop_len)
        else:
            wave = wave[: self.crop_len]
        return wave

    def format_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, list]:
        list_items: Dict[str, list] = {
            "path": [],
            "machine": [],
            "section": [],
            "attr": [],
            "is_normal": [],
            "is_target": [],
            "wave": [],
        }

        for b in batch:
            list_items["path"].append(b["path"])
            for l in ["machine", "section", "attr", "is_normal", "is_target"]:
                list_items[l].append(get_dcase_info(b["path"], l))
            list_items["wave"].append(self.crop_wave(b["wave"]))

        list_items["wave"] = torch.stack(list_items["wave"])  # type: ignore
        # list_items["path"] = np.array(list_items["path"])
        return list_items

    def add_labels(self, items: Dict[str, Any]) -> Dict[str, Any]:
        for key, label_info in self.label_dict.items():
            idx_tensor = torch.Tensor(
                [
                    label_info.path2idx_dict[get_relative_dcase_path(path=path)]
                    for path in items["path"]
                ]
            )
            onehot_tensor = torch.nn.functional.one_hot(
                idx_tensor.long(), num_classes=label_info.num_class
            ).float()
            items[f"idx_{key}"] = idx_tensor[:, None]
            items[f"onehot_{key}"] = onehot_tensor
        return items

    def __call__(self, unformatted_batch: List[Dict[str, Any]]):
        """Convert into batch tensors."""
        batch = self.format_batch(unformatted_batch)
        batch = self.add_labels(batch)
        return batch
