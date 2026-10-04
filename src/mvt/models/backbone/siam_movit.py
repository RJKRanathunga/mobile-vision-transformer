import math

import torch
import torch.nn.functional as F
from torch import nn

from mvt.layers.transformer import TransformerEncoderBlock


class SiamMoViTBlock(nn.Module):
    """
    Args:
        in_channels: Number of input feature channels (C).
        transformer_dim: Transformer embedding dimension (D).
        ffn_dim: Hidden dimension of the Transformer FFN.
        num_transformer_blocks: Number of Transformer encoder blocks.
        patch_h: Patch height.
        patch_w: Patch width.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        num_transformer_blocks: int,
        patch_h: int = 2,
        patch_w: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()

        if transformer_dim % num_heads != 0:
            raise ValueError(
                "transformer_dim must be divisible by num_heads"
            )

        if patch_h <= 0 or patch_w <= 0:
            raise ValueError("patch_h and patch_w must be positive")

        self.in_channels = in_channels
        self.transformer_dim = transformer_dim
        self.ffn_dim = ffn_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = patch_h * patch_w
        self.num_heads = num_heads

        # ---------------------------------------------------------
        # Local representation
        #
        # C -> C -> D
        #
        #   3x3 Conv + BN + activation
        #   1x1 Conv
        # ---------------------------------------------------------

        self.local_rep = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=in_channels,
                out_channels=transformer_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

        # ---------------------------------------------------------
        # Global representation
        #
        # TransformerEncoder x L
        # followed by LayerNorm.
        # ---------------------------------------------------------

        self.global_rep = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embed_dim=transformer_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                )
                for _ in range(num_transformer_blocks)
            ],
            nn.LayerNorm(transformer_dim),
        )

        # ---------------------------------------------------------
        # Projection
        #
        # D -> C
        # ---------------------------------------------------------

        self.conv_proj = nn.Sequential(
            nn.Conv2d(
                in_channels=transformer_dim,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # ---------------------------------------------------------
        # Fusion
        #
        # [original C, transformed C] -> 2C -> C
        # ---------------------------------------------------------

        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        search: torch.Tensor,
        template: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # ---------------------------------------------------------
        # Save original feature maps for fusion
        # ---------------------------------------------------------

        residual_search = search
        residual_template = template

        # ---------------------------------------------------------
        # 1. Local representation
        #
        # The SAME self.local_rep is used for both branches.
        # There are NOT separate template/search convolutions.
        # ---------------------------------------------------------

        search = self.local_rep(search)
        template = self.local_rep(template)

        # ---------------------------------------------------------
        # 2. Convert feature maps to patch tokens
        # ---------------------------------------------------------

        search_patches, search_info = self._unfold(search)
        template_patches, template_info = self._unfold(template)

        # ---------------------------------------------------------
        # 3. Concatenate search and template tokens
        # torch.cat((patches_x, patches_z), 1)
        # Dimension 1 is the token dimension.
        # ---------------------------------------------------------

        concatenated_patches = torch.cat(
            (search_patches, template_patches),
            dim=1,
        )

        # ---------------------------------------------------------
        # 4. Global representation
        #
        # Transformer operates jointly on search + template.
        # ---------------------------------------------------------

        for transformer_layer in self.global_rep:
            concatenated_patches = transformer_layer(
                concatenated_patches
            )

        # ---------------------------------------------------------
        # 5. Split search and template tokens
        # ---------------------------------------------------------

        num_search_patches = search_info["total_patches"]

        search_patches = concatenated_patches[
            :,
            :num_search_patches,
            :,
        ]

        template_patches = concatenated_patches[
            :,
            num_search_patches:,
            :,
        ]

        # ---------------------------------------------------------
        # 6. Convert tokens back to feature maps
        # ---------------------------------------------------------

        search = self._fold(
            search_patches,
            search_info,
        )

        template = self._fold(
            template_patches,
            template_info,
        )

        # ---------------------------------------------------------
        # 7. Project D -> C
        # ---------------------------------------------------------

        search = self.conv_proj(search)
        template = self.conv_proj(template)

        # ---------------------------------------------------------
        # 8. Fuse with original feature maps
        # ---------------------------------------------------------

        search = self.fusion(
            torch.cat(
                (residual_search, search),
                dim=1,
            )
        )

        template = self.fusion(
            torch.cat(
                (residual_template, template),
                dim=1,
            )
        )

        return search, template

    def _unfold(
        self,
        feature_map: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:

        patch_h = self.patch_h
        patch_w = self.patch_w
        patch_area = self.patch_area

        batch_size, channels, orig_h, orig_w = feature_map.shape

        # ---------------------------------------------------------
        # The official implementation supports feature maps whose
        # dimensions are not divisible by the patch size by using
        # bilinear interpolation.
        # ---------------------------------------------------------

        new_h = math.ceil(orig_h / patch_h) * patch_h
        new_w = math.ceil(orig_w / patch_w) * patch_w

        interpolate = False

        if new_h != orig_h or new_w != orig_w:
            feature_map = F.interpolate(
                feature_map,
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )
            interpolate = True

        # Number of patches along each spatial dimension.

        num_patch_h = new_h // patch_h
        num_patch_w = new_w // patch_w

        num_patches = num_patch_h * num_patch_w

        # ---------------------------------------------------------
        # Official implementation:
        #
        # [B, C, H, W]
        #
        # -> [B*C*Nh, Ph, Nw, Pw]
        # ---------------------------------------------------------

        reshaped = feature_map.reshape(
            batch_size * channels * num_patch_h,
            patch_h,
            num_patch_w,
            patch_w,
        )

        # [B*C*Nh, Ph, Nw, Pw]
        # -> [B*C*Nh, Nw, Ph, Pw]

        transposed = reshaped.transpose(1, 2)

        # [B*C*Nh, Nw, Ph, Pw]
        # -> [B, C, N, P]

        reshaped = transposed.reshape(
            batch_size,
            channels,
            num_patches,
            patch_area,
        )

        # [B, C, N, P]
        # -> [B, P, N, C]

        transposed = reshaped.transpose(1, 3)

        # [B, P, N, C]
        # -> [B*P, N, C]

        patches = transposed.reshape(
            batch_size * patch_area,
            num_patches,
            channels,
        )

        info = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return patches, info

    def _fold(
        self,
        patches: torch.Tensor,
        info: dict,
    ) -> torch.Tensor:

        if patches.dim() != 3:
            raise ValueError(
                "Expected patches with shape [B*P, N, C], "
                f"got {tuple(patches.shape)}"
            )

        # ---------------------------------------------------------
        # [B*P, N, C]
        # -> [B, P, N, C]
        # ---------------------------------------------------------

        patches = patches.contiguous().view(
            info["batch_size"],
            self.patch_area,
            info["total_patches"],
            -1,
        )

        batch_size, _, num_patches, channels = patches.shape

        num_patch_h = info["num_patches_h"]
        num_patch_w = info["num_patches_w"]

        # ---------------------------------------------------------
        # [B, P, N, C]
        # -> [B, C, N, P]
        # ---------------------------------------------------------

        patches = patches.transpose(1, 3)

        # ---------------------------------------------------------
        # [B, C, N, P]
        # -> [B*C*Nh, Nw, Ph, Pw]
        # ---------------------------------------------------------

        feature_map = patches.reshape(
            batch_size * channels * num_patch_h,
            num_patch_w,
            self.patch_h,
            self.patch_w,
        )

        # ---------------------------------------------------------
        # [B*C*Nh, Nw, Ph, Pw]
        # -> [B*C*Nh, Ph, Nw, Pw]
        # ---------------------------------------------------------

        feature_map = feature_map.transpose(1, 2)

        # ---------------------------------------------------------
        # [B*C*Nh, Ph, Nw, Pw]
        # -> [B, C, H, W]
        # ---------------------------------------------------------

        feature_map = feature_map.reshape(
            batch_size,
            channels,
            num_patch_h * self.patch_h,
            num_patch_w * self.patch_w,
        )

        # Return to original spatial dimensions if interpolation
        # was required during unfolding.

        if info["interpolate"]:
            feature_map = F.interpolate(
                feature_map,
                size=info["orig_size"],
                mode="bilinear",
                align_corners=False,
            )

        return feature_map