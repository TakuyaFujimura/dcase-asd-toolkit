from typing import Dict, List

import torch
from torch import nn

from .utils import get_dec, get_perm


class Mixup(nn.Module):
    def __init__(self, prob: float, label_candidate: List[str]):
        super().__init__()
        self.prob = prob
        assert 0 <= prob <= 1
        self.label_candidate = label_candidate
        assert "wave" in label_candidate

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
        perm = get_perm(len(wave), wave.device)
        dec = get_dec(len(wave), self.prob, wave.device)

        for key in batch:
            if key in self.label_candidate:
                new_batch[key] = self.process(lam, perm, dec, batch[key])
            else:
                new_batch[key] = batch[key]

        return new_batch
