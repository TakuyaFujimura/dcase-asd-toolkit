import logging
from typing import List

from torch import nn

from asdlib.models.audio_feature.stft import FFT
from asdlib.models.modules import SEBlock, calc_filtered_size_1d

logger = logging.getLogger(__name__)


class Conv1dEncoderLayer(nn.Module):
    def __init__(
        self,
        in_channel: int,
        length: int,
        use_bias: bool = False,
        emb_base_size: int = 128,
        conv_param_list: List[dict] = [
            {"k": 256, "s": 64},
            {"k": 64, "s": 32},
            {"k": 16, "s": 4},
        ],
    ):
        super().__init__()
        assert len(conv_param_list) == 3

        logger.info("===Conv1dEncoderLayer==========")
        for i, param_dict in enumerate(conv_param_list):
            length = calc_filtered_size_1d(
                input_length=length,
                p=0,
                k=param_dict["k"],
                s=param_dict["s"],
                d=param_dict.get("d", 1),
            )
            logger.info(f"{i}th: [{length}]")
        logger.info("===============================")

        self.layer = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channel,
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
            nn.Flatten(),
            nn.Linear(length * emb_base_size, emb_base_size, bias=use_bias),
            nn.BatchNorm1d(emb_base_size),
            nn.ReLU(),
            nn.Linear(emb_base_size, emb_base_size, bias=use_bias),
            nn.BatchNorm1d(emb_base_size),
            nn.ReLU(),
            nn.Linear(emb_base_size, emb_base_size, bias=use_bias),
            nn.BatchNorm1d(emb_base_size),
            nn.ReLU(),
            nn.Linear(emb_base_size, emb_base_size, bias=use_bias),
            nn.BatchNorm1d(emb_base_size),
            nn.ReLU(),
            nn.Linear(emb_base_size, emb_base_size, bias=use_bias),
            nn.BatchNorm1d(emb_base_size),
            nn.ReLU(),
            nn.Linear(emb_base_size, emb_base_size, bias=use_bias),
        )

    def forward(self, x):
        """
        x: (B, C, L)
        """
        x = self.layer(x)
        return x


class FFTEncoderLayer(nn.Module):
    def __init__(self, sec, sr, use_bias=False, emb_base_size=128):
        super().__init__()
        fft_len = (sec * sr) // 2 + 1
        conv_param_list = [{"k": 256, "s": 64}, {"k": 64, "s": 32}, {"k": 16, "s": 4}]
        self.layer = Conv1dEncoderLayer(
            1, fft_len, use_bias, emb_base_size, conv_param_list
        )
        self.fft = FFT()

    def forward(self, x):
        """
        x: (B, L)
        """
        x = self.fft(x)
        x = x.unsqueeze(1)
        return self.layer(x)
