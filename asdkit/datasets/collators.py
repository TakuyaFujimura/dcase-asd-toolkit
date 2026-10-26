import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch

from asdkit.utils.config_class import LabelInfo
from asdkit.utils.dcase_utils import get_dcase_info, get_label_dict
from asdkit.utils.dcase_utils.machines import MACHINE_DICT

logger = logging.getLogger(__name__)


def get_dcase_path(path: str) -> str:
    # <data_dir>/formatted/dcase2021/raw/fan/train/hoge.wav
    path_parts = path.split("/")
    assert path_parts[-2] in ["train", "test", "supplemental"]
    assert path_parts[-5].startswith("dcase202")
    # assert path_parts[-4] == "raw"
    # assert path_parts[-6] == "formatted"
    return "/".join(path_parts[-5:])


def wave_pad_crop(
    wave: torch.Tensor, crop_len: int, pad_mode: str = "tile", shuffle: bool = False
) -> torch.Tensor:
    """Pad and crop (T,) or (C, T) waveforms along the time axis."""
    assert wave.ndim in (1, 2)
    # Pad
    time_len = wave.shape[-1]
    if time_len < crop_len:
        if pad_mode == "tile":
            repeats = [1] * wave.ndim
            repeats[-1] = int(np.ceil(crop_len / time_len))
            wave = wave.tile(tuple(repeats))
        elif pad_mode == "zero":
            total_pad = crop_len - time_len
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
    time_len = wave.shape[-1]
    assert time_len >= crop_len
    if shuffle:
        start_frame = np.random.randint(0, time_len - crop_len + 1)
    else:
        start_frame = 0

    return wave[..., start_frame : start_frame + crop_len]


class BaseDCASECollator(ABC):
    def __init__(self, label_dict_path: Dict[str, Path]) -> None:
        self.label_dict: Dict[str, LabelInfo] = get_label_dict(label_dict_path)

    def add_labels(self, items: Dict[str, Any]) -> Dict[str, Any]:
        for key, label_info in self.label_dict.items():
            idx_tensor = torch.Tensor(
                [
                    label_info.path2idx_dict[get_dcase_path(path=path)]
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
    def format_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pass

    def __call__(self, unformatted_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert into batch tensors."""
        batch = self.format_batch(unformatted_batch)
        batch = self.add_labels(batch)
        return batch


class DCASEWaveCollator(BaseDCASECollator):
    """Collate mono waveforms into (B, T) batches with DCASE labels."""

    def __init__(
        self,
        label_dict_path: Dict[str, Path],
        sec: Union[float, str],
        sr: int,
        shuffle: bool = True,
        pad_mode: str = "tile",
        metadata_keys: Optional[List[str]] = None,
    ) -> None:
        super().__init__(label_dict_path=label_dict_path)
        self.shuffle = shuffle
        self.crop_len: Union[int, Literal["all"]]
        if metadata_keys is None:
            self.metadata_keys = [
                "machine",
                "section",
                "attr",
                "is_normal",
                "is_target",
            ]
        else:
            self.metadata_keys = metadata_keys

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

    def format_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        list_items: Dict[str, Any] = defaultdict(list)
        for b in batch:
            # Fill the list_items dictionary
            list_items["path"].append(b["path"])
            for key in self.metadata_keys:
                list_items[key].append(get_dcase_info(path=b["path"], label=key))
            list_items["wave"].append(self.crop_wave(wave=b["wave"]))

        list_items["wave"] = torch.stack(list_items["wave"])  # type: ignore
        return list_items


class DCASEWaveCollatorMultiCh(DCASEWaveCollator):
    """Collate (C, T) waveforms into (B, C, T) batches with DCASE labels."""

    def crop_wave(self, wave: torch.Tensor) -> torch.Tensor:
        assert wave.ndim == 2
        if self.crop_len == "all":
            return wave
        return wave_pad_crop(
            wave=wave,
            crop_len=self.crop_len,
            pad_mode=self.pad_mode,
            shuffle=self.shuffle,
        )


class DCASEEvalWaveCollator(DCASEWaveCollator):
    """Collator for challenge evaluation data."""

    def __init__(
        self,
        dcase: str,
        label_dict_path: Dict[str, Path],
        sec: Union[float, str],
        sr: int,
        shuffle: bool = True,
        pad_mode: str = "tile",
        metadata_keys: Optional[List[str]] = None,
        metadata_keys_eval_test: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            label_dict_path=label_dict_path,
            sec=sec,
            sr=sr,
            shuffle=shuffle,
            pad_mode=pad_mode,
            metadata_keys=metadata_keys,
        )
        self.dcase = dcase
        if metadata_keys_eval_test is None:
            self.metadata_keys_eval_test = ["machine", "section"]
        else:
            self.metadata_keys_eval_test = metadata_keys_eval_test

    def format_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        list_items: Dict[str, Any] = defaultdict(list)
        for b in batch:
            # Only machine and section labels are available during challenge evaluation.
            path_parts = b["path"].split("/")
            if (
                path_parts[-5] == self.dcase
                and path_parts[-2] == "test"
                and path_parts[-3] in MACHINE_DICT[f"{self.dcase}-eval"]
            ):
                key_list = self.metadata_keys_eval_test
            else:
                key_list = self.metadata_keys

            # Fill the list_items dictionary
            list_items["path"].append(b["path"])
            for key in key_list:
                list_items[key].append(get_dcase_info(path=b["path"], label=key))
            list_items["wave"].append(self.crop_wave(wave=b["wave"]))

        list_items["wave"] = torch.stack(list_items["wave"])  # type: ignore
        return list_items


class DCASEAudioFeatCollator(BaseDCASECollator):
    """Wave form data's collator."""

    def __init__(
        self,
        label_dict_path: Dict[str, Path],
        shuffle: bool = False,
        metadata_keys: Optional[List[str]] = None,
    ) -> None:
        super().__init__(label_dict_path=label_dict_path)
        self.shuffle = shuffle
        if metadata_keys is None:
            self.metadata_keys = [
                "machine",
                "section",
                "attr",
                "is_normal",
                "is_target",
            ]
        else:
            self.metadata_keys = metadata_keys

    def format_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        list_items: Dict[str, Any] = defaultdict(list)

        for b in batch:
            list_items["path"].append(b["path"])
            for key in self.metadata_keys:
                list_items[key].append(get_dcase_info(path=b["path"], label=key))
            list_items["feat"].append(b["feat"])

        list_items["feat"] = torch.stack(list_items["feat"])  # type: ignore
        return list_items
