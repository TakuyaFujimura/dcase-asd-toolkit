from typing import Dict

import torch
from torch import nn

from .utils import get_dec, get_reverse_perm


class Mixup(nn.Module):
    def __init__(self, prob: float):
        super().__init__()
        if not (0 <= prob <= 1):
            raise ValueError(f"prob should be in [0, 1], but got {prob}.")
        self.prob = prob

    @staticmethod
    def process(
        lam: torch.Tensor, perm: torch.Tensor, dec: torch.Tensor, data: torch.Tensor
    ) -> torch.Tensor:
        lam = lam.reshape([-1] + [1] * (len(data.shape) - 1))
        dec = dec.reshape([-1] + [1] * (len(data.shape) - 1))
        data_mix = lam * data + (1 - lam) * data[perm]
        result = dec * data_mix + (1 - dec) * data
        return result

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        is_applied: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if not self.training or not is_applied:
            return batch

        new_batch: Dict[str, torch.Tensor] = {}

        wave = batch["wave"]
        lam = torch.rand(len(wave), device=wave.device)
        perm = get_reverse_perm(len(wave), wave.device)
        dec = get_dec(len(wave), self.prob, wave.device)

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                assert (
                    key.startswith("idx_") or key.startswith("onehot_") or key == "wave"
                )
                new_batch[key] = self.process(lam, perm, dec, batch[key])
            else:
                assert key in [
                    "path",
                    "machine",
                    "section",
                    "attr",
                    "is_normal",
                    "is_target",
                ]
                new_batch[key] = batch[key]

        return new_batch
