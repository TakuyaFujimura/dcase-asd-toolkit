import logging
from pathlib import Path
from typing import Dict, Optional

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from asdit.utils.common import instantiate_tgt
from asdit.utils.config_class import DMConfig, DMSplitConfig

logger = logging.getLogger(__name__)


class PLDataModule(pl.LightningDataModule):
    def __init__(self, dm_cfg: DMSplitConfig):
        super().__init__()
        self.dm_cfg = dm_cfg

    @staticmethod
    def get_loader(dm_config: Optional[DMConfig]) -> DataLoader | None:

        if dm_config is None:
            return None

        dataset = instantiate_tgt(dm_config.dataset)

        if dm_config.batch_sampler is None:
            batch_sampler = None
        else:
            batch_sampler = instantiate_tgt(
                {"dataset": dataset, **dm_config.batch_sampler}
            )

        collator = instantiate_tgt(dm_config.collator)
        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            **dm_config.dataloader,
        )

    def train_dataloader(self):
        return self.get_loader(dm_config=self.dm_cfg.train)

    def val_dataloader(self):
        if self.dm_cfg.valid is not None:
            # check validation configuration
            if self.dm_cfg.valid.dataloader.get("shuffle", True):
                logger.warning("Validation dataloader set shuffle=True, which is not recommended")
            if self.dm_cfg.valid.batch_sampler is not None:
                logger.warning("Validation batch_sampler is not None, which is not recommended")
            if self.dm_cfg.valid.collator.get("shuffle", True):
                logger.warning("Validation collator set shuffle=True, which is not recommended")

        return self.get_loader(dm_config=self.dm_cfg.valid)

    def test_dataloader(self):
        return None
