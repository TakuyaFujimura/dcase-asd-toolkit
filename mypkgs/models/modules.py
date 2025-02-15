import torch
from torch import nn


def calc_filtered_size_3d(
    input_size: tuple, c: int, p: int, k: int, s: int, d: int = 1
) -> tuple:
    assert len(input_size) == 3
    output_size = (
        c,
        calc_filtered_size_1d(input_length=input_size[1], p=p, k=k, s=s, d=d),
        calc_filtered_size_1d(input_length=input_size[2], p=p, k=k, s=s, d=d),
    )
    return output_size


def calc_filtered_size_1d(input_length: int, p: int, k: int, s: int, d: int = 1) -> int:
    """Calculate the output size of the filtering operation (e.g., Conv1d, MaxPool1d, etc).

    Args:
        input_length (int): input length
        c (int): channel
        p (int): padding
        k (int): kernel size
        s (int): stride
        d (int, optional): dilation. Defaults to 1.

    Returns:
        tuple: output length
    """

    return int((input_length + (2 * p) - (d * (k - 1)) - 1) / s) + 1


class SEBlock(nn.Module):
    def __init__(self, num_channels: int, ratio: int = 16):
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

    def forward(self, x):
        w = torch.mean(x, dim=list(range(2, len(x.shape))))  # GAP
        w = self.layer1(w)
        w = self.layer2(w)
        w = w.view([x.shape[0], x.shape[1]] + ([1] * (len(x.shape) - 2)))
        return w * x


# class SE3DBlock(nn.Module):
#     def __init__(self, feat_size, ratio=4):
#         super().__init__()
#         self.ratio = ratio
#         self.layer_list = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Linear(num_feat, num_feat // self.ratio, bias=False),
#                     nn.ReLU(),
#                     nn.Linear(num_feat // self.ratio, num_feat, bias=False),
#                     nn.Sigmoid(),
#                 )
#                 for num_feat in feat_size
#             ]
#         )

#     def squeeze_excitation(self, x, dim):
#         squeeze_dims = [i for i in range(1, 4) if i != dim]
#         a = torch.mean(x, dim=squeeze_dims)
#         a = self.layer_list[dim - 1](a)
#         w = torch.sigmoid(a)
#         unsqueezed_shape = [x.shape[0]]
#         unsqueezed_shape += [1 if i != dim else x.shape[i] for i in range(1, 4)]
#         w = w.reshape(unsqueezed_shape)
#         return w * x

#     def forward(self, x):
#         """
#         Args
#             x: (B, 1, H, W)
#         """
#         for i in range(1, 4):
#             x = x + self.squeeze_excitation(x, i)
#         return x
