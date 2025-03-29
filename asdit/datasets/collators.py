import logging
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
    assert path_split[-2] in ["train", "test"]
    assert path_split[-4] == "raw"
    assert path_split[-5].startswith("dcase202")
    assert path_split[-6] == "formatted"
    return "/".join(path_split[-5:])


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
        sec: float | str,
        sr: int,
        need_feat: bool = False,
        shuffle: bool = True,
    ):
        self.label_dict: Dict[str, LabelInfo] = get_label_dict(label_dict_path)
        self.shuffle = shuffle
        self.crop_len: int | Literal["all", "none"]

        if sr != 16000:
            raise ValueError("Unexpected sampling rate")

        if sec == "all":
            self.crop_len = "all"
        elif sec == "none":
            self.crop_len = "none"
            logging.info(
                "`wave` in batch will be ignored. If you need it, set `sec` parameter."
            )
        else:
            if sec in ["dcase2021", "dcase2022"]:
                sec = 10.0
            elif sec == "dcase2023":
                sec = 18.0
            elif sec == "dcase2024":
                sec = 12.0
            elif type(sec) is str:
                raise ValueError(f"Unexpected sec: {sec}")
            assert type(sec) is float
            self.crop_len = int(sr * sec)

        self.need_wave: bool = self.crop_len != "none"
        self.need_feat = need_feat
        if self.need_feat and self.need_wave:
            raise ValueError("Cannot use both wave and feat at the same time.")
        # If we use only `feat`, `sec` will not be used and
        # meaningless `sec` parameters will appear in hparams.yaml.
        # This is a source of confusion and the ValuerError is to prevent it.
        # Using `feat` and `wave` at the same time itself can be easily realized
        # but I didn't do that to simply avoid confusion.
        # (I think there is another better way...)

    def crop_wave(self, wave: torch.Tensor) -> torch.Tensor:
        assert len(wave.shape) == 1
        assert self.crop_len != "none"
        if self.crop_len == "all":
            return wave
        else:
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
        }
        if self.need_wave:
            list_items["wave"] = []
        if self.need_feat:
            list_items["feat"] = []

        for b in batch:
            list_items["path"].append(b["path"])
            for l in ["machine", "section", "attr", "is_normal", "is_target"]:
                list_items[l].append(get_dcase_info(b["path"], l))
            if self.need_wave:
                list_items["wave"].append(self.crop_wave(b["wave"]))
            if self.need_feat:
                list_items["feat"].append(b["feat"])

        if self.need_wave:
            list_items["wave"] = torch.stack(list_items["wave"])  # type: ignore
        if self.need_feat:
            list_items["feat"] = torch.stack(list_items["feat"])  # type: ignore

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
