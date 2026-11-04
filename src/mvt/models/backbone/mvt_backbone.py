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

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Match the initialization used by the upstream MobileViT encoder."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def extract_stem_features(self, image: torch.Tensor) -> torch.Tensor:
        """Run the shared convolutional stem through MobileNetV2 layer 2."""
        image = self.first_conv(image)
        image = self.layer1(image)
        return self.layer2(image)

    def extract_template(self, template: torch.Tensor) -> torch.Tensor:
        """Extract the cacheable template feature used by upstream inference."""
        return self.extract_stem_features(template)

    def forward_with_template_features(
        self,
        search: torch.Tensor,
        template_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run joint layers using a template cached after backbone layer 2."""
        search = self.extract_stem_features(search)
        template = template_features

        search = self.layer3_mv2(search)
        template = self.layer3_mv2(template)
        search, template = self.layer3_movit(search, template)

        search = self.layer4_mv2(search)
        template = self.layer4_mv2(template)
        return self.layer4_movit(search, template)

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

        template_features = self.extract_template(template)
        return self.forward_with_template_features(search, template_features)
