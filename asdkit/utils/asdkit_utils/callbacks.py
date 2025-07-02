import logging
from typing import Optional

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import ModelCheckpoint as PLModelCheckpoint

logger = logging.getLogger(__name__)


class ModelCheckpoint(PLModelCheckpoint):
    def __init__(self, *args, resume_ckpt_path: Optional[str], **kwargs):
        """
        The ModelCheckpoint skips on_validation_end during the first epoch when resuming training.
        This is intended to avoid several errors.

        For example, if no validation set is used, the following error may occur without this class:
        ---------------------------
        lightning.fabric.utilities.exceptions.MisconfigurationException: `ModelCheckpoint(monitor='train/main')` could not find the monitored key in the returned metrics: ['epoch', 'step']. HINT: Did you call `log('train/main', value)` in the `LightningModule`?
        ----------------------------
        """
        super().__init__(*args, **kwargs)
        if resume_ckpt_path is None:
            self.skip_epoch = None
        else:
            ckpt = torch.load(resume_ckpt_path, map_location="cpu")
            self.skip_epoch = ckpt["epoch"]

    def on_validation_end(self, trainer, pl_module):
        if self.skip_epoch is None:
            super().on_validation_end(trainer, pl_module)
        elif trainer.current_epoch != self.skip_epoch:
            super().on_validation_end(trainer, pl_module)
        else:
            return  # skip saving


class NaNCheckCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._check_for_nan(outputs):
            logging.warning("NaN detected in training batch, stopping training.")
            trainer.should_stop = True

    @staticmethod
    def _check_for_nan(outputs):
        if isinstance(outputs, torch.Tensor):
            return torch.isnan(outputs).any().item()
        elif isinstance(outputs, dict):
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and torch.isnan(value).any().item():
                    return True
        return False
