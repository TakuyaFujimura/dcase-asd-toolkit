# Copyright 2024 Takuya Fujimura
import logging

import torch
from torch import nn

from ..modules import SEBlock, calc_filtered_size_3d
from ..stft import STFT

logger = logging.getLogger(__name__)


class Conv2dEncoderLayer_ResNetBlock(nn.Module):
    def __init__(
        self,
        input_size: tuple,
        out_channel: int,
        kernel: int,
        stride: int,
        use_bias: bool,
        additional_layer: str = "SEBlock",
    ):
        """
        Args
            input_size: tuple ([C, H, W]) or int (C)
        """
        super().__init__()
        self.out_channel = out_channel
        self.stride = stride
        self.kernel = kernel
        if self.kernel != 3:
            raise NotImplementedError()

        self.bn = nn.BatchNorm2d(num_features=input_size[0])

        if stride == 1:
            self.skip_connect = None
        elif stride == 2:
            self.skip_connect = nn.Sequential(
                nn.MaxPool2d(kernel_size=self.kernel, stride=self.stride, padding=1),
                nn.Conv2d(
                    in_channels=input_size[0],
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=use_bias,
                ),
            )
        else:
            raise NotImplementedError()

        self.layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=input_size[0],
                out_channels=self.out_channel,
                kernel_size=self.kernel,
                stride=self.stride,
                padding=1,
                bias=use_bias,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.out_channel,
                out_channels=self.out_channel,
                kernel_size=self.kernel,
                stride=1,
                padding=1,
                bias=use_bias,
            ),
        )
        if additional_layer == "SEBlock":
            self.additional_layer = SEBlock(out_channel, ratio=16)
        else:
            raise NotImplementedError()

    def calc_outsize(self, input_size: tuple) -> tuple:
        intermediate_size = calc_filtered_size_3d(
            input_size=input_size, c=self.out_channel, p=1, k=self.kernel, s=self.stride
        )
        intermediate_size = calc_filtered_size_3d(
            input_size=intermediate_size, c=self.out_channel, p=1, k=self.kernel, s=1
        )
        return intermediate_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        xr = self.layer(x)
        xr = self.additional_layer(xr)
        if self.skip_connect is not None:
            x = self.skip_connect(x) + xr
        else:
            x = x + xr
        return x


class Conv2dEncoderLayer_FirstConv(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel: int,
        stride: int,
        use_bias: bool,
    ):
        super().__init__()
        self.out_channel = out_channel
        self.kernel = kernel
        self.stride = stride
        self.pool_kernel = 3
        self.pool_stride = 2

        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=self.out_channel,
                kernel_size=self.kernel,
                stride=self.stride,
                bias=use_bias,
                padding=0,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.pool_stride),
        )

    def calc_convsize(self, input_size: tuple) -> tuple:
        return calc_filtered_size_3d(
            input_size, self.out_channel, 0, self.kernel, self.stride
        )

    def calc_poolsize(self, input_size: tuple) -> tuple:
        return calc_filtered_size_3d(
            input_size, input_size[0], 0, self.pool_kernel, self.pool_stride
        )

    def calc_output_size(self, input_size: tuple) -> tuple:
        return self.calc_poolsize(self.calc_convsize(input_size))

    def forward(self, x):
        return self.layer(x)


class Conv2dEncoderLayer(nn.Module):
    def __init__(
        self,
        input_size: tuple,
        use_bias: bool,
        emb_base_size: int,
        resnet_additional_layer: str,
    ):
        super().__init__()
        if emb_base_size % 8 != 0:
            raise ValueError("emb_base_size must be divisible by 8")
        if len(input_size) != 2:
            raise ValueError(
                "input_size must be in the form of (H, W), i.e., no channel"
            )

        self.bn_freq = nn.BatchNorm1d(num_features=input_size[0])
        self.layers = nn.ModuleList()
        logger.info("===Conv2dEncoderLayer============")
        intermediate_size = self.setup_first_layer(
            input_size=input_size, emb_base_size=emb_base_size, use_bias=use_bias
        )
        intermediate_size = self.setup_second_layer(
            input_size=intermediate_size,
            emb_base_size=emb_base_size,
            use_bias=use_bias,
            resnet_additional_layer=resnet_additional_layer,
        )
        intermediate_size = self.setup_3_4_5th_layer(
            input_size=intermediate_size,
            emb_base_size=emb_base_size,
            use_bias=use_bias,
            resnet_additional_layer=resnet_additional_layer,
        )
        logger.info("=================================")
        self.bn = nn.BatchNorm1d(num_features=emb_base_size * intermediate_size[1])
        self.linear = nn.Linear(
            in_features=emb_base_size * intermediate_size[1],
            out_features=emb_base_size,
            bias=use_bias,
        )

    def setup_first_layer(
        self, input_size: tuple, emb_base_size: int, use_bias: bool
    ) -> tuple:
        layer = Conv2dEncoderLayer_FirstConv(
            in_channel=1,
            out_channel=emb_base_size // 8,
            kernel=7,
            stride=2,
            use_bias=use_bias,
        )
        output_size = layer.calc_output_size((1, input_size[0], input_size[1]))
        self.layers.append(layer)
        logger.info(f"1st: {output_size}")
        return output_size

    def setup_second_layer(
        self,
        input_size: tuple,
        emb_base_size: int,
        use_bias: bool,
        resnet_additional_layer: str,
    ) -> tuple:
        res1 = Conv2dEncoderLayer_ResNetBlock(
            input_size=input_size,
            out_channel=emb_base_size // 8,
            kernel=3,
            stride=1,
            use_bias=use_bias,
            additional_layer=resnet_additional_layer,
        )
        output_size = res1.calc_outsize(input_size)
        res2 = Conv2dEncoderLayer_ResNetBlock(
            input_size=output_size,
            out_channel=emb_base_size // 8,
            kernel=3,
            stride=1,
            use_bias=use_bias,
            additional_layer=resnet_additional_layer,
        )
        output_size = res2.calc_outsize(output_size)
        self.layers.append(nn.Sequential(res1, res2))
        logger.info(f"2nd: {output_size}")
        return output_size

    def setup_3_4_5th_layer(
        self,
        input_size: tuple,
        emb_base_size: int,
        use_bias: bool,
        resnet_additional_layer: str,
    ) -> tuple:
        intermediate_size = input_size
        for i, c in enumerate(
            [
                emb_base_size // 4,
                emb_base_size // 2,
                emb_base_size,
            ]
        ):
            res1 = Conv2dEncoderLayer_ResNetBlock(
                input_size=intermediate_size,
                out_channel=c,
                kernel=3,
                stride=2,
                use_bias=use_bias,
                additional_layer=resnet_additional_layer,
            )
            intermediate_size = res1.calc_outsize(intermediate_size)
            res2 = Conv2dEncoderLayer_ResNetBlock(
                input_size=intermediate_size,
                out_channel=c,
                kernel=3,
                stride=1,
                use_bias=use_bias,
                additional_layer=resnet_additional_layer,
            )
            intermediate_size = res2.calc_outsize(intermediate_size)
            self.layers.append(nn.Sequential(res1, res2))
            logger.info(f"{i+3}th: {intermediate_size}")
        return intermediate_size

    def forward(self, x):
        """
        Args
            x: spectrogram (B, F, T)
        """
        x = self.bn_freq(x).unsqueeze(1)  # B, 1, F, T
        for l in self.layers:
            x = l(x)  # B, C, F, T
        x = torch.max(x, dim=-1).values  # B, C, F
        x = self.bn(torch.flatten(x, start_dim=1))  # B, C*F
        x = self.linear(x)  # B, emb_base_size
        return x


# -------------------------------------------------------------------------------- #


class STFTEncoderLayer(nn.Module):
    def __init__(
        self,
        sec: int,
        sr: int,
        stft_cfg: dict,
        use_bias: bool,
        emb_base_size: int,
        resnet_additional_layer: str = "SEBlock",
    ):
        super().__init__()
        self.stft = STFT(**stft_cfg)
        spectrogram_size = self.stft(torch.randn(sec * sr)).shape
        if min(spectrogram_size) < 36:
            raise ValueError("input sequence or n_fft is too short")
        self.layer = Conv2dEncoderLayer(
            input_size=spectrogram_size,
            use_bias=use_bias,
            emb_base_size=emb_base_size,
            resnet_additional_layer=resnet_additional_layer,
        )
