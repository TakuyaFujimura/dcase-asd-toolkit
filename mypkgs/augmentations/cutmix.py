import copy
from typing import Dict, List

import torch
from torch import nn

from .utils import get_dec, get_perm


class Cutmix(nn.Module):
    def __init__(self, prob: float):
        super().__init__()
        self.prob = prob
        assert 0 <= prob <= 1

    @staticmethod
    def process_wave(
        lam: torch.Tensor, perm: torch.Tensor, dec: torch.Tensor, data: torch.Tensor
    ):
        assert len(data.shape) == 2
        all_len = data.shape[-1]
        crop_len = ((1 - lam) * all_len).type(torch.int32)
        data_mix = copy.deepcopy(data)
        for i in range(len(data)):
            if dec[i] == 1.0:
                start = torch.randint(0, max(1, int(all_len - crop_len[i])), (1,))[0]
                data_mix[i, start : start + crop_len[i]] = data[
                    perm[i], start : start + crop_len[i]
                ]
        return data_mix

    @staticmethod
    def process_label(
        lam: torch.Tensor, perm: torch.Tensor, dec: torch.Tensor, data: torch.Tensor
    ):
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
            if isinstance(batch[key], torch.Tensor):
                assert (
                    key.startswith("idx_") or key.startswith("onehot_") or key == "wave"
                )
                if key == "wave":
                    new_batch[key] = self.process_wave(lam, perm, dec, batch[key])
                else:
                    new_batch[key] = self.process_label(lam, perm, dec, batch[key])
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
