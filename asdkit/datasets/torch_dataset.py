import glob
import logging
from typing import Any, Dict, List, Optional, Tuple

import soundfile as sf
import torch
import tqdm
from torch.utils.data import Dataset

from asdkit.models.audio_feature import ConsecutiveStftFrames
from asdkit.utils.common import read_json

logger = logging.getLogger(__name__)


def parse_path_selector(selector: str) -> List[str]:
    assert isinstance(selector, str)
    if selector.endswith(".json"):  # json format
        path_list = read_json(selector)
        logger.info(f"Loaded {len(path_list)} paths from {selector}")
    elif "*" in selector:  # glob format
        path_list = sorted(glob.glob(selector))
        logger.info(f"Loaded {len(path_list)} paths from {selector}")
    elif selector.endswith(".wav"):  # single file format
        path_list = [selector]
    else:
        raise ValueError(f"Unknown path_list format: {selector}")
    return path_list  # type: ignore


def torch_mono_wav_load(path: str) -> torch.Tensor:
    wave, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == 16000 and wave.shape[1] == 1
    return torch.from_numpy(wave[:, 0])


class WaveDataset(Dataset):
    def __init__(
        self,
        path_selector_list: List[str],
    ):
        super().__init__()
        self.path_list = []

        logger.info("Start Loading Paths")
        for selector in path_selector_list:
            self.path_list += parse_path_selector(selector)
        logger.info("Finished Loading Paths")

    def get_item(self, idx) -> dict:
        wave = torch_mono_wav_load(path=self.path_list[idx])
        items = {"wave": wave, "path": self.path_list[idx]}
        return items

    def __getitem__(self, idx):
        return self.get_item(idx)

    def __len__(self):
        return len(self.path_list)


class WaveDatasetMultiCh(WaveDataset):
    """Load (C, T) waveforms, keeping the channel axis even when C is one.

    None skips source channel validation or selects all channels, respectively.
    Selected channels use zero-based indices in the given order.
    """

    def __init__(
        self,
        path_selector_list: List[str],
        source_num_channels: Optional[int] = None,
        selected_channels: Optional[List[int]] = None,
        squeeze_channel: bool = False,
    ) -> None:
        super().__init__(path_selector_list=path_selector_list)
        self.source_num_channels = source_num_channels
        if selected_channels is not None:
            assert len(selected_channels) > 0
            assert all(channel >= 0 for channel in selected_channels)
        self.selected_channels = selected_channels
        self.squeeze_channel = squeeze_channel

    def get_item(self, idx: int) -> Dict[str, Any]:
        path = self.path_list[idx]
        wave, sr = sf.read(path, dtype="float32", always_2d=True)
        assert sr == 16000
        if self.source_num_channels is not None:
            assert wave.shape[1] == self.source_num_channels

        wave = torch.from_numpy(wave.T)
        if self.selected_channels is not None:
            wave = wave[self.selected_channels]

        if self.squeeze_channel:
            if wave.shape[0] != 1:
                raise ValueError(
                    "squeeze_channel=True requires exactly one output channel, "
                    f"got {wave.shape[0]}"
                )
            wave = wave.squeeze(0)

        return {"wave": wave, "path": path}


class ConsecutiveStftFramesDataset(Dataset):
    """Precompute consecutive STFT windows and index each window as one sample.

    This enables window-level shuffling across audio files, as in the DCASE
    baseline.
    """

    def __init__(
        self,
        path_selector_list: List[str],
        stft_cfg: Dict[str, Any],
        n_frames: int,
        frame_shift: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.audio_feat_extractor = ConsecutiveStftFrames(
            stft_cfg=stft_cfg, n_frames=n_frames, frame_shift=frame_shift
        )
        self.path_list: List[str] = []

        logger.info("Start Loading Paths")
        for selector in path_selector_list:
            self.path_list += parse_path_selector(selector)
        logger.info("Finished Loading Paths")

        self.audio_feat_tensor, self.path_idx_list = self.prepare_audio_feat()
        assert len(self.audio_feat_tensor) == len(self.path_idx_list)

    def prepare_audio_feat(self) -> Tuple[torch.Tensor, List[int]]:
        logger.info("Start Extracting Audio Features")
        audio_feat_list: List[torch.Tensor] = []
        path_idx_list: List[int] = []
        for i, path in enumerate(tqdm.tqdm(self.path_list)):
            wave = torch_mono_wav_load(path=path)[None]  # 1, T
            x = self.audio_feat_extractor(wave)  # 1, N, D
            x = x.squeeze(0)  # N, D: each window is one dataset sample.
            audio_feat_list += [x]
            path_idx_list += [i] * len(x)
        logger.info("Finished Extracting Audio Features")
        return torch.concat(audio_feat_list, dim=0), path_idx_list

    def get_item(self, idx: int) -> Dict[str, Any]:
        items = {
            "feat": self.audio_feat_tensor[idx],
            "path": self.path_list[self.path_idx_list[idx]],
        }
        return items

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.get_item(idx)

    def __len__(self) -> int:
        return len(self.audio_feat_tensor)
