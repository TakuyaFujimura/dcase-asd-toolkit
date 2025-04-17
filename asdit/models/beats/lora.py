from pathlib import Path

import torch
from torch import nn

from ..modules import add_lora
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
            self.network = nn.Linear(768, self.embed_size, bias=True)
        else:
            raise NotImplementedError(f"last_layer={self.last_layer} is not supported.")

    def forward(self, x_time):
        """
        Args
            x_time: (B, L)
        """
        z = self.model.extract_features(x_time)[0]  # (B, L, C)

        if self.last_layer == "linear":
            z = self.network(z)  # (B, L, emb_base_size)
            z = torch.mean(z, dim=1)  # (B, C)
        else:
            raise NotImplementedError(f"last_layer={self.last_layer} is not supported.")

        return z
