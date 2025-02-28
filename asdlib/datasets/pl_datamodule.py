from pathlib import Path
from typing import Dict, Optional

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from asdlib.utils.common import instantiate_tgt
from asdlib.utils.config_class import DMConfig, DMSplitConfig


class PLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dm_cfg: DMSplitConfig,
        label_dict_path: Dict[str, Path],
    ):
        super().__init__()
        self.dm_cfg = dm_cfg
        self.label_dict_path = label_dict_path

    @staticmethod
    def get_loader(
        datamoduleconfig: Optional[DMConfig], label_dict_path: Dict[str, Path]
    ) -> DataLoader | None:

        if datamoduleconfig is None:
            return None

        dataset = instantiate_tgt(datamoduleconfig.dataset.model_dump())

        if datamoduleconfig.batch_sampler is None:
            batch_sampler = None
        else:
            batch_sampler = instantiate_tgt(
                {"dataset": dataset, **datamoduleconfig.batch_sampler}
            )

        collator = instantiate_tgt(
            {
                "label_dict_path": label_dict_path,
                **datamoduleconfig.collator.model_dump(),
            }
        )
        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            **datamoduleconfig.dataloader,
        )

    def train_dataloader(self):
        return self.get_loader(
            datamoduleconfig=self.dm_cfg.train,
            label_dict_path=self.label_dict_path,
        )

    def val_dataloader(self):
        if self.dm_cfg.valid is not None:
            # check validation configuration
            if self.dm_cfg.valid.dataloader.get("shuffle", True):
                raise ValueError("Validation dataloader should not shuffle")
            if self.dm_cfg.valid.batch_sampler is not None:
                raise ValueError("Validation dataloader should not use batch_sampler")
            if getattr(self.dm_cfg.valid.collator, "shuffle", True):
                raise ValueError("Validation collator should not shuffle")

        return self.get_loader(
            datamoduleconfig=self.dm_cfg.valid,
            label_dict_path=self.label_dict_path,
        )

    def test_dataloader(self):
        return None
