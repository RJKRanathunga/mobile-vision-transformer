import torch
from torch import nn

def make_divisible(value: int, divisor: int = 8) -> int:
    """Make a channel count divisible by the given divisor"""
    return int((value+divisor-1)//divisor)*divisor

class MV2Block(nn.Module):
    EXPAND_RATIO = 4

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
    ) -> None:
        super().__init__()

        if stride not in (1,2):
            raise ValueError("stride must be either 1 or 2")

        hidden_channels = make_divisible(
            round(in_channels * self.EXPAND_RATIO),
            divisor=8
        )

        self.in_channel = in_channels
        self.out_channel = out_channels
        self.stride = stride
        self.hidden_channels = hidden_channels

        self.expansion = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_channels, # make it depthwise convolution
                bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        self.use_residual = (
            stride == 1 and in_channels == out_channels
        ) # add input to output (input+output) of the block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block = nn.Sequential(
            self.expansion,
            self.depthwise,
            self.projection
        )

        y = block(x)
        if self.use_residual:
            return x + y
        else:
            return y