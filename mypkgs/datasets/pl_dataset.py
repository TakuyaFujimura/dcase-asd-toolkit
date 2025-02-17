from pathlib import Path
from typing import Dict

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from ..utils.config_class import BasicDMConfig, BasicDMSplitConfig
from ..utils.pl_utils import instantiate_tgt
from .collators import BasicCollator
from .torch_dataset import BasicDataset


class BasicDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dm_cfg: BasicDMSplitConfig,
        label_dict_path: Dict[str, Path],
    ):
        super().__init__()
        self.dm_cfg = dm_cfg
        self.label_dict_path = label_dict_path

    def _get_loader(self, datasetconfig: BasicDMConfig):
        dataset = BasicDataset(
            data_dir=datasetconfig.dataset.data_dir,
            path_list_json=datasetconfig.dataset.path_list_json,
            use_cache=datasetconfig.dataset.use_cache,
        )
        if datasetconfig.batch_sampler is None:
            batch_sampler = None
        else:
            batch_sampler = instantiate_tgt(
                {"dataset": dataset, **datasetconfig.batch_sampler}
            )
        collator = BasicCollator(
            label_dict_path=self.label_dict_path,
            **datasetconfig.collator.model_dump(),
        )
        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            **datasetconfig.dataloader,
        )

    def train_dataloader(self):
        return self._get_loader(self.dm_cfg.train)

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None
