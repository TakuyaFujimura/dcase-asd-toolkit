import logging
from typing import List, Literal, Optional

from torch import nn

from asdkit.models.modules import SEBlock, calc_filtered_size_1d

logger = logging.getLogger(__name__)


class Conv1dEncoderLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        use_bias: bool,
        emb_base_size: int,
        conv_param_list: Optional[List[dict]] = None,
        aggregate: Literal["mlp", "gap"] = "mlp",
    ):
        super().__init__()
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

        self.conv1d_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=emb_base_size,
                kernel_size=conv_param_list[0]["k"],
                stride=conv_param_list[0]["s"],
                dilation=conv_param_list[0].get("d", 1),
                bias=use_bias,
            ),
            nn.ReLU(),
            SEBlock(emb_base_size, ratio=16),
            nn.Conv1d(
                in_channels=emb_base_size,
                out_channels=emb_base_size,
                kernel_size=conv_param_list[1]["k"],
                stride=conv_param_list[1]["s"],
                dilation=conv_param_list[1].get("d", 1),
                bias=use_bias,
            ),
            nn.ReLU(),
            SEBlock(emb_base_size, ratio=16),
            nn.Conv1d(
                in_channels=emb_base_size,
                out_channels=emb_base_size,
                kernel_size=conv_param_list[2]["k"],
                stride=conv_param_list[2]["s"],
                dilation=conv_param_list[2].get("d", 1),
                bias=use_bias,
            ),
            nn.ReLU(),
            SEBlock(num_channels=emb_base_size, ratio=16),
        )

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

    def forward(self, x):
        """
        x: (B, C, L)
        """
        x = self.conv1d_layer(x)
        x = self.aggregate_layer(x)
        x = self.last_layer(x)
        return x
