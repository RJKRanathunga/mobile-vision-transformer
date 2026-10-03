from typing import Any

import torch
from torch import nn

class FirstConvBlock(nn.Module):
    """
    First convolutional blok of the MVT backbone.

    Architecture:
        Conv2d(3 -> 16, kernel=3, stride=2, padding=1)
        BatchNorm2d(16)
        ReLU
    """

    def __init__(self) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)