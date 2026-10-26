import logging
from typing import Dict, List, Literal, Optional

from torch import Tensor, nn

from asdkit.models.modules import SEBlock, calc_filtered_size_1d

logger = logging.getLogger(__name__)


class Conv1dEncoderLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        use_bias: bool,
        emb_base_size: int,
        conv_param_list: Optional[List[Dict[str, int]]] = None,
        aggregate: Literal["mlp", "gap"] = "mlp",
    ) -> None:
        """Build a fixed-width network for FFT magnitudes.

        Args:
            input_size: Number of FFT bins, excluding batch and channel axes.
            use_bias: Enable Conv/Linear biases and BatchNorm affine together.
                SEBlock retains its own internal bias policy.
            emb_base_size: Internal channel/hidden width and output vector width.
            conv_param_list: Three convolution settings with k, s and optional d.
            aggregate: Flatten fixed-size bins (mlp) or average remaining bins (gap).
        """
        super().__init__()
        self.embedding_dim = emb_base_size
        if conv_param_list is None:
            conv_param_list = [
                {"k": 256, "s": 64},
                {"k": 64, "s": 32},
                {"k": 16, "s": 4},
            ]
        assert len(conv_param_list) == 3

        logger.info("===Conv1dEncoderLayer==========")
        for i, param_dict in enumerate(conv_param_list):
            input_size = calc_filtered_size_1d(
                input_length=input_size,
                p=0,
                k=param_dict["k"],
                s=param_dict["s"],
                d=param_dict.get("d", 1),
            )
            logger.info(f"{i}th: [{input_size}]")
        logger.info("===============================")

        conv_layers: List[nn.Module] = []
        for index, params in enumerate(conv_param_list):
            conv_layers.extend(
                [
                    nn.Conv1d(
                        in_channels=1 if index == 0 else emb_base_size,
                        out_channels=emb_base_size,
                        kernel_size=params["k"],
                        stride=params["s"],
                        dilation=params.get("d", 1),
                        bias=use_bias,
                    ),
                    nn.ReLU(),
                    SEBlock(num_channels=emb_base_size, ratio=16),
                ]
            )
        self.conv1d_layer = nn.Sequential(*conv_layers)

        if aggregate == "mlp":
            self.aggregate_layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size * emb_base_size, emb_base_size, bias=use_bias),
                nn.BatchNorm1d(emb_base_size, affine=use_bias),
                nn.ReLU(),
            )
        elif aggregate == "gap":
            self.aggregate_layer = nn.Sequential(
                nn.AdaptiveAvgPool1d((1)),
                nn.Flatten(),
                nn.Linear(emb_base_size, emb_base_size, bias=use_bias),
                nn.BatchNorm1d(emb_base_size, affine=use_bias),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError(
                f"aggregate should be mlp or gap, but got {aggregate}"
            )

        self.last_layer = nn.Sequential(
            nn.Linear(emb_base_size, emb_base_size, bias=use_bias),
            nn.BatchNorm1d(emb_base_size, affine=use_bias),
            nn.ReLU(),
            nn.Linear(emb_base_size, emb_base_size, bias=use_bias),
            nn.BatchNorm1d(emb_base_size, affine=use_bias),
            nn.ReLU(),
            nn.Linear(emb_base_size, emb_base_size, bias=use_bias),
            nn.BatchNorm1d(emb_base_size, affine=use_bias),
            nn.ReLU(),
            nn.Linear(emb_base_size, emb_base_size, bias=use_bias),
            nn.BatchNorm1d(emb_base_size, affine=use_bias),
            nn.ReLU(),
            nn.Linear(emb_base_size, emb_base_size, bias=use_bias),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: FFT magnitudes shaped (B, 1, input_size).

        Returns:
            Vectors shaped (B, emb_base_size).
        """
        x = self.conv1d_layer(x)
        x = self.aggregate_layer(x)
        x = self.last_layer(x)
        return x
