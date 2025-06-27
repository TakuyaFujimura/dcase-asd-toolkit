import re

import torch

# def get_reverse_perm(batch_size: int, device: torch.device) -> torch.Tensor:
#     perm = torch.arange(batch_size - 1, -1, -1).to(device)
#     return perm


def get_rand_perm(batch_size: int, device: torch.device) -> torch.Tensor:
    perm = torch.randperm(batch_size, device=device)
    return perm


def get_dec(batch_size: int, prob: float, device: torch.device) -> torch.Tensor:
    dec = torch.rand(batch_size, device=device) < prob
    dec = dec.float()
    return dec


def re_match_any(patterns: list[str], string: str) -> bool:
    """
    Check if any pattern matches the string.
    """
    return any(re.fullmatch(p, string) for p in patterns)
