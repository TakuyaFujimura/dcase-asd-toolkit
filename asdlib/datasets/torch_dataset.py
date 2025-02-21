import glob
import logging
from typing import List

import torchaudio
from torch.utils.data import Dataset

from ..utils.io_utils import read_json

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


class BasicDataset(Dataset):
    def __init__(
        self,
        path_selector_list: List[str],
        use_cache: bool = False,
    ):
        super().__init__()
        self.path_list = []

        logger.info(f"Start Loading Paths")
        for selector in path_selector_list:
            self.path_list += parse_path_selector(selector)
        logger.info(f"Finished Loading Paths")

        self.use_cache = use_cache
        assert not self.use_cache
        # if self.use_cache:
        #     self.caches: List[dict] = []
        #     for idx in tqdm.tqdm(range(len(self.path_list))):
        #         self.caches.append(self.get_item(idx))

    def get_item(self, idx) -> dict:
        wave, sr = torchaudio.load(self.path_list[idx])
        assert sr == 16000 and wave.shape[0] == 1
        items = {
            "wave": wave[0],
            "path": self.path_list[idx],
        }
        return items

    def __getitem__(self, idx):
        return self.get_item(idx)
        # if self.use_cache:
        #     return self.caches[idx]
        # else:
        #     return self.get_item(idx)

    def __len__(self):
        return len(self.path_list)
