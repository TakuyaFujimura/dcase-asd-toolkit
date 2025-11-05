from typing import Dict, List, Tuple

import torch
from torch import nn

from asdkit.utils.common import re_match_any

from .utils import get_dec, get_rand_perm


class FeatEx(nn.Module):
    def __init__(self, prob: float, subspace_embed_size: int):
        """
        K. Wilkinghoff, "Self-supervised learning for anomalous sound detection," Proc. ICASSP, 2024.

        Args:
            prob (float): Probability of applying featex.
            subspace_embed_size (int): Size of each sub-embedding in the concatenated embedding.
        """
        super().__init__()
        if not (0 <= prob <= 1):
            raise ValueError(f"prob should be in [0, 1], but got {prob}.")
        self.prob = prob
        self.subspace_embed_size = subspace_embed_size
        self.target_keys = ["onehot_.*"]

    def process_emb(
        self, perm_list: List[torch.Tensor], dec: torch.Tensor, embed: torch.Tensor
    ) -> torch.Tensor:
        # embed: (B, ..., D)
        if torch.all(dec == 0):
            return embed
        else:
            dec = dec.reshape([-1] + [1] * (len(embed.shape) - 1))
            embed_ex_list = []
            for i, perm in enumerate(perm_list):
                embed_split = embed[
                    ...,
                    i * self.subspace_embed_size : (i + 1) * self.subspace_embed_size,
                ]
                embed_ex_list.append(embed_split[perm])
            embed_ex = torch.cat(embed_ex_list, dim=1)
            return dec * embed_ex + (1 - dec) * embed

    def process_label(
        self, perm_list: List[torch.Tensor], dec: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        dec = dec.reshape([-1] + [1] * (len(label.shape) - 1))
        label_org_list = [label]
        label_org_list += [torch.zeros_like(label) for _ in perm_list]
        label_org = torch.cat(label_org_list, dim=-1)
        if torch.all(dec == 0):
            return label_org
        else:
            label_ex_list = [torch.zeros_like(label)]
            label_ex_list += [label[perm] for perm in perm_list]
            label_ex = torch.cat(label_ex_list, dim=-1)
            label_ex /= len(perm_list)
            return dec * label_ex + (1 - dec) * label_org

    def forward(
        self, embed: torch.Tensor, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # check embed shape
        if embed.shape[1] % self.subspace_embed_size != 0:
            raise ValueError(
                f"embed_size {embed.shape[1]} should be divisible by subspace_embed_size {self.subspace_embed_size}"
            )
        embed_num = embed.shape[1] // self.subspace_embed_size

        # decide whether to apply featex
        if self.training:
            dec = get_dec(len(embed), self.prob, embed.device)
        else:
            dec = torch.zeros(len(embed), device=embed.device)
        dec = dec.float()

        # generate random permutations
        perm_list = [torch.arange(len(embed), device=embed.device)]
        perm_list += [
            get_rand_perm(len(embed), embed.device) for _ in range(embed_num - 1)
        ]

        # process embeddings
        new_embed = self.process_emb(perm_list, dec, embed)
        new_batch: Dict[str, torch.Tensor] = {}

        # process labels
        for key in batch:
            if re_match_any(patterns=self.target_keys, string=key):
                new_batch[key] = self.process_label(perm_list, dec, batch[key])
            else:
                new_batch[key] = batch[key]

        return new_embed, new_batch
