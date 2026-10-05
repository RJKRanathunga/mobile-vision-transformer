import torch
from torch import nn

from mvt.models.backbone.first_cnn import FirstConvBlock
from mvt.models.backbone.mobilenet_v2 import MV2Block
from mvt.models.backbone.siam_movit import SiamMoViTBlock


class MVTBackbone(nn.Module):
    """
    MVT-Small backbone.

    Processes the search and template images using shared weights.

    Input:
        search:   [B, 3, 256, 256]
        template: [B, 3, 128, 128]

    Output:
        search:   [B, 128, 16, 16]
        template: [B, 128, 8, 8]
    """

    def __init__(self) -> None:
        super().__init__()

        # =========================================================
        # Initial convolution
        #
        # 3 -> 16
        # stride = 2
        # =========================================================

        self.first_conv = FirstConvBlock()

        # =========================================================
        # Layer 1
        #
        # MV2:
        # 16 -> 32
        # stride = 1
        # =========================================================

        self.layer1 = nn.Sequential(
            MV2Block(
                in_channels=16,
                out_channels=32,
                stride=1,
            )
        )

        # =========================================================
        # Layer 2
        #
        # MV2 x 3
        #
        # First block:
        #   32 -> 64, stride=2
        #
        # Remaining blocks:
        #   64 -> 64, stride=1
        # =========================================================

        self.layer2 = nn.Sequential(
            MV2Block(
                in_channels=32,
                out_channels=64,
                stride=2,
            ),
            MV2Block(
                in_channels=64,
                out_channels=64,
                stride=1,
            ),
            MV2Block(
                in_channels=64,
                out_channels=64,
                stride=1,
            ),
        )

        # =========================================================
        # Layer 3
        #
        # MV2:
        #   64 -> 96
        #   stride=2
        #
        # followed by Siam-MoViT:
        #   C=96
        #   D=144
        #   FFN=288
        #   Transformer blocks=2
        # =========================================================

        self.layer3_mv2 = MV2Block(
            in_channels=64,
            out_channels=96,
            stride=2,
        )

        self.layer3_movit = SiamMoViTBlock(
            in_channels=96,
            transformer_dim=144,
            ffn_dim=288,
            num_transformer_blocks=2,
            patch_h=2,
            patch_w=2,
            num_heads=4,
        )

        # =========================================================
        # Layer 4
        #
        # MV2:
        #   96 -> 128
        #   stride=2
        #
        # followed by Siam-MoViT:
        #   C=128
        #   D=192
        #   FFN=384
        #   Transformer blocks=4
        # =========================================================

        self.layer4_mv2 = MV2Block(
            in_channels=96,
            out_channels=128,
            stride=2,
        )

        self.layer4_movit = SiamMoViTBlock(
            in_channels=128,
            transformer_dim=192,
            ffn_dim=384,
            num_transformer_blocks=4,
            patch_h=2,
            patch_w=2,
            num_heads=4,
        )

    def forward(
        self,
        search: torch.Tensor,
        template: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # =========================================================
        # Initial convolution
        #
        # SAME weights for search and template.
        # =========================================================

        search = self.first_conv(search)
        template = self.first_conv(template)

        # =========================================================
        # Layer 1
        # =========================================================

        search = self.layer1(search)
        template = self.layer1(template)

        # =========================================================
        # Layer 2
        # =========================================================

        search = self.layer2(search)
        template = self.layer2(template)

        # =========================================================
        # Layer 3
        # =========================================================

        search = self.layer3_mv2(search)
        template = self.layer3_mv2(template)

        search, template = self.layer3_movit(
            search,
            template,
        )

        # =========================================================
        # Layer 4
        # =========================================================

        search = self.layer4_mv2(search)
        template = self.layer4_mv2(template)

        search, template = self.layer4_movit(
            search,
            template,
        )

        return search, template