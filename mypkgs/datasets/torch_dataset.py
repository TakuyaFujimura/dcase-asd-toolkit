from pathlib import Path

import torchaudio
from torch.utils.data import Dataset

from ..utils.io_utils import read_json


class BasicDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        path_list_json: Path,
        use_cache: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.path_list = read_json(path_list_json)
        self.use_cache = use_cache
        assert not self.use_cache
        # if self.use_cache:
        #     self.caches: List[dict] = []
        #     for idx in tqdm.tqdm(range(len(self.path_list))):
        #         self.caches.append(self.get_item(idx))

    def get_item(self, idx) -> dict:
        wave, sr = torchaudio.load(self.data_dir / self.path_list[idx])
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
