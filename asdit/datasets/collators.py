import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np
import torch

from asdit.utils.config_class import LabelInfo
from asdit.utils.dcase_utils import get_dcase_info, get_label_dict

logger = logging.getLogger(__name__)


def get_relative_dcase_path(path: str) -> str:
    # <data_dir>/formatted/dcase2021/raw/fan/train/hoge.wav
    path_split = path.split("/")
    assert path_split[-2] in ["train", "test", "supplemental"]
    assert path_split[-4] == "raw"
    assert path_split[-5].startswith("dcase202")
    # assert path_split[-6] == "formatted"
    return "/".join(path_split[-5:])


def wave_pad_crop(
    wave: torch.Tensor, crop_len: int, pad_mode: str = "tile", shuffle: bool = False
) -> torch.Tensor:
    """Pad and crop wave tensor to a fixed length."""
    assert len(wave.shape) == 1
    # Pad
    if len(wave) < crop_len:
        if pad_mode == "tile":
            wave = wave.tile(int(np.ceil(crop_len / len(wave))))
        elif pad_mode == "zero":
            total_pad = crop_len - len(wave)
            if shuffle:
                left_pad = np.random.randint(0, total_pad + 1)
                right_pad = total_pad - left_pad
            else:
                left_pad = 0
                right_pad = total_pad
            wave = torch.nn.functional.pad(wave, (left_pad, right_pad), mode="constant")
        else:
            raise NotImplementedError(f"Unsupported pad mode: {pad_mode}")

    # Crop
    assert len(wave) >= crop_len
    if shuffle:
        start_frame = np.random.randint(0, len(wave) - crop_len + 1)
    else:
        start_frame = 0

    return wave[start_frame : start_frame + crop_len]


class BaseDCASECollator(ABC):
    def __init__(self, label_dict_path: Dict[str, Path]):
        self.label_dict: Dict[str, LabelInfo] = get_label_dict(label_dict_path)

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

    @abstractmethod
    def format_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, list]:
        pass

    def __call__(self, unformatted_batch: List[Dict[str, Any]]):
        """Convert into batch tensors."""
        batch = self.format_batch(unformatted_batch)
        batch = self.add_labels(batch)
        return batch


class DCASEWaveCollator(BaseDCASECollator):
    """Wave form data's collator."""

    def __init__(
        self,
        label_dict_path: Dict[str, Path],
        sec: float | str,
        sr: int,
        shuffle: bool = True,
        pad_mode: str = "tile",
    ):
        super().__init__(label_dict_path=label_dict_path)
        self.shuffle = shuffle
        self.crop_len: int | Literal["all"]

        if sr != 16000:
            raise ValueError("Unexpected sampling rate")

        if sec == "all":
            self.crop_len = "all"
        elif isinstance(sec, str):
            raise ValueError(f"Unexpected sec: {sec}")
        else:
            self.crop_len = int(sr * sec)

        self.pad_mode = pad_mode

    def crop_wave(self, wave: torch.Tensor) -> torch.Tensor:
        assert len(wave.shape) == 1
        if self.crop_len == "all":
            return wave
        else:
            wave = wave_pad_crop(
                wave=wave,
                crop_len=self.crop_len,
                pad_mode=self.pad_mode,
                shuffle=self.shuffle,
            )
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
        return list_items


class DCASEAudioFeatCollator(BaseDCASECollator):
    """Wave form data's collator."""

    def __init__(
        self,
        label_dict_path: Dict[str, Path],
        shuffle: bool = False,
    ):
        super().__init__(label_dict_path=label_dict_path)
        self.shuffle = shuffle

    def format_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, list]:
        list_items: Dict[str, list] = {
            "path": [],
            "machine": [],
            "section": [],
            "attr": [],
            "is_normal": [],
            "is_target": [],
            "feat": [],
        }

        for b in batch:
            list_items["path"].append(b["path"])
            for l in ["machine", "section", "attr", "is_normal", "is_target"]:
                list_items[l].append(get_dcase_info(b["path"], l))
            list_items["feat"].append(b["feat"])

        list_items["feat"] = torch.stack(list_items["feat"])  # type: ignore
        return list_items
