from pathlib import Path

import torch
from torch import nn

from ..modules import AttnStatPool, add_lora
from .tools import resume


class BEATsLoRA(nn.Module):
    def __init__(
        self,
        sr: int,
        embed_size: int = 128,
        ckpt_path: str = "pretrained_models/beats/BEATs_iter3.pt",
        lora_cfg: dict = {"r": 64, "target_modules": ["q_proj", "v_proj"]},
        last_layer: str = "linear",
        update_cfg: dict = {},
    ):
        super().__init__()
        if sr != 16000:
            raise ValueError("The sampling rate should be 16000")
        model = resume(ckpt_path=Path(ckpt_path), update_cfg=update_cfg)
        self.model = add_lora(model, lora_cfg)
        self.embed_size = embed_size
        self.last_layer = last_layer
        if self.last_layer == "linear":
            self.network = nn.Linear(768, self.embed_size)
        elif self.last_layer == "attn_stat_pool":
            self.network = nn.Sequential(
                AttnStatPool(embed_size=768), nn.Linear(768, self.embed_size)
            )

        else:
            raise NotImplementedError(f"last_layer={self.last_layer} is not supported.")

    def forward(self, x_time):
        """
        Args
            x_time: (B, L)
        """
        z = self.model.extract_features(x_time)[0]  # (B, L, 768)

        if self.last_layer == "linear":
            z = self.network(z)  # (B, L, D)
            z = torch.mean(z, dim=1)  # (B, D)
        elif self.last_layer == "attn_stat_pool":
            z = self.network(z)  # (B, D)
        else:
            raise NotImplementedError(f"last_layer={self.last_layer} is not supported.")

        return z
