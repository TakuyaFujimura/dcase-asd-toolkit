from typing import Dict, List, Optional

import torch
from torch import nn

from asdit.utils.common import re_match_any

from .utils import get_dec, get_rand_perm


class Mixup(nn.Module):
    def __init__(self, prob: float, target_keys: Optional[List[str]] = None):
        super().__init__()
        if not (0 <= prob <= 1):
            raise ValueError(f"prob should be in [0, 1], but got {prob}.")
        self.prob = prob
        if target_keys is None:
            target_keys = ["onehot_.*", "wave"]
        self.target_keys = target_keys

    @staticmethod
    def process(
        lam: torch.Tensor, perm: torch.Tensor, dec: torch.Tensor, data: torch.Tensor
    ) -> torch.Tensor:
        lam = lam.reshape([-1] + [1] * (len(data.shape) - 1))
        dec = dec.reshape([-1] + [1] * (len(data.shape) - 1))
        data_mix = lam * data + (1 - lam) * data[perm]
        result = dec * data_mix + (1 - dec) * data
        return result

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        if not self.training:
            return batch

        new_batch: Dict[str, torch.Tensor] = {}

        wave = batch["wave"]
        lam = torch.rand(len(wave), device=wave.device)
        perm = get_rand_perm(len(wave), wave.device)
        dec = get_dec(len(wave), self.prob, wave.device)

        for key in batch:
            if re_match_any(patterns=self.target_keys, string=key):
                new_batch[key] = self.process(lam, perm, dec, batch[key])
            else:
                new_batch[key] = batch[key]

        return new_batch
