from typing import Any, Dict, Tuple

import torch
from torch import nn


def add_lora(model: nn.Module, lora_cfg: Dict[str, Any]) -> nn.Module:
    """
    Args:
        model: Model to adapt.
        lora_cfg: PEFT LoRA configuration.

    Returns:
        Model containing the adapters.
    """
    import peft

    model = peft.get_peft_model(model, peft.LoraConfig(**lora_cfg))  # type: ignore
    model.print_trainable_parameters()
    return model


def calc_filtered_size_1d(input_length: int, p: int, k: int, s: int, d: int = 1) -> int:
    """
    Args:
        input_length: Input length along one spatial axis.
        p: Padding per side.
        k: Kernel size.
        s: Stride.
        d: Dilation.

    Returns:
        Output spatial length.
    """

    return int((input_length + (2 * p) - (d * (k - 1)) - 1) / s) + 1


def calc_filtered_size_3d(
    input_size: Tuple[int, int, int], c: int, p: int, k: int, s: int, d: int = 1
) -> Tuple[int, int, int]:
    """
    Args:
        input_size: Input ``(channels, height, width)``.
        c: Output channel count.
        p: Padding per side.
        k: Kernel size.
        s: Stride.
        d: Dilation.

    Returns:
        Output ``(channels, height, width)``.
    """
    assert len(input_size) == 3
    output_size = (
        c,
        calc_filtered_size_1d(input_length=input_size[1], p=p, k=k, s=s, d=d),
        calc_filtered_size_1d(input_length=input_size[2], p=p, k=k, s=s, d=d),
    )
    return output_size


class SEBlock(nn.Module):
    def __init__(self, num_channels: int, ratio: int = 16) -> None:
        """
        Args:
            num_channels: Input channel count.
            ratio: Reduction ratio for the gating bottleneck.
        """
        super().__init__()
        self.num_channels = num_channels
        self.ratio = ratio
        self.layer1 = nn.Sequential(
            nn.Linear(
                in_features=self.num_channels,
                out_features=self.num_channels // self.ratio,
                bias=False,
            ),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(
                in_features=self.num_channels // self.ratio,
                out_features=self.num_channels,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features shaped ``[batch, channels, ...]`` with spatial axes.

        Returns:
            Gated features with the same shape as ``x``.
        """
        w = torch.mean(x, dim=list(range(2, len(x.shape))))  # GAP
        w = self.layer1(w)
        w = self.layer2(w)
        w = w.view([x.shape[0], x.shape[1]] + ([1] * (len(x.shape) - 2)))
        return w * x


class AttnStatPool(nn.Module):
    def __init__(self, embedding_dim: int, hidden_ratio: int = 16) -> None:
        """
        Args:
            embedding_dim: Number of input embedding features.
            hidden_ratio: Reduction ratio for attention channels.
        """
        super().__init__()
        hidden_size = embedding_dim // hidden_ratio
        self.attention = nn.Sequential(
            nn.Conv1d(embedding_dim * 3, hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.Conv1d(hidden_size, embedding_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.last_layer = nn.Sequential(
            nn.BatchNorm1d(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Frame embeddings shaped ``[batch, time, embedding_dim]``.

        Returns:
            Pooled embeddings shaped ``[batch, embedding_dim]``.
        """
        x = x.permute(0, 2, 1)  # B, C, L

        t = x.size()[-1]
        global_x = torch.cat(
            (
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(
                    1, 1, t
                ),
            ),
            dim=1,
        )  # B, C*3, L
        w = self.attention(global_x)  # B, C, L
        mu = torch.sum(x * w, dim=2)  # B, C
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))  # B, C
        x = torch.cat((mu, sg), 1)  # B, C*2
        x = self.last_layer(x)  # B, C
        return x
