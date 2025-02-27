import logging

import torch
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)


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
