import torch
from torch import nn


def conv_block(
    in_channels: int,
    out_channels: int,
) -> nn.Sequential:

    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class CenterPredictor(nn.Module):
    """
    Center-based bounding box prediction head.

    Input:
        [B, 256, 16, 16]

    Outputs:

        center score map:
            [B, 1, 16, 16]

        size map:
            [B, 2, 16, 16]

        offset map:
            [B, 2, 16, 16]

        bbox:
            [B, 4]

        bbox format:
            [cx, cy, w, h]
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 256,
        feat_size: int = 16,
    ) -> None:
        super().__init__()

        self.feat_size = feat_size

        # =========================================================
        # Center branch
        #
        # 256 -> 128 -> 64 -> 32 -> 1
        # =========================================================

        self.conv1_ctr = conv_block(
            in_channels,
            hidden_channels,
        )

        self.conv2_ctr = conv_block(
            hidden_channels,
            hidden_channels // 2,
        )

        self.conv3_ctr = conv_block(
            hidden_channels // 2,
            hidden_channels // 4,
        )

        self.conv4_ctr = conv_block(
            hidden_channels // 4,
            hidden_channels // 8,
        )

        self.conv5_ctr = nn.Conv2d(
            hidden_channels // 8,
            1,
            kernel_size=1,
        )

        # =========================================================
        # Offset branch
        #
        # 256 -> 128 -> 64 -> 32 -> 2
        # =========================================================

        self.conv1_offset = conv_block(
            in_channels,
            hidden_channels,
        )

        self.conv2_offset = conv_block(
            hidden_channels,
            hidden_channels // 2,
        )

        self.conv3_offset = conv_block(
            hidden_channels // 2,
            hidden_channels // 4,
        )

        self.conv4_offset = conv_block(
            hidden_channels // 4,
            hidden_channels // 8,
        )

        self.conv5_offset = nn.Conv2d(
            hidden_channels // 8,
            2,
            kernel_size=1,
        )

        # =========================================================
        # Size branch
        #
        # 256 -> 128 -> 64 -> 32 -> 2
        # =========================================================

        self.conv1_size = conv_block(
            in_channels,
            hidden_channels,
        )

        self.conv2_size = conv_block(
            hidden_channels,
            hidden_channels // 2,
        )

        self.conv3_size = conv_block(
            hidden_channels // 2,
            hidden_channels // 4,
        )

        self.conv4_size = conv_block(
            hidden_channels // 4,
            hidden_channels // 8,
        )

        self.conv5_size = nn.Conv2d(
            hidden_channels // 8,
            2,
            kernel_size=1,
        )

        # Official implementation uses Xavier initialization.
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    @staticmethod
    def _sigmoid(
        x: torch.Tensor,
    ) -> torch.Tensor:

        return torch.clamp(
            torch.sigmoid(x),
            min=1e-4,
            max=1.0 - 1e-4,
        )

    def get_score_map(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        # =========================================================
        # Center
        # =========================================================

        center = self.conv1_ctr(x)
        center = self.conv2_ctr(center)
        center = self.conv3_ctr(center)
        center = self.conv4_ctr(center)

        score_map_ctr = self.conv5_ctr(center)

        # =========================================================
        # Offset
        # =========================================================

        offset = self.conv1_offset(x)
        offset = self.conv2_offset(offset)
        offset = self.conv3_offset(offset)
        offset = self.conv4_offset(offset)

        score_map_offset = self.conv5_offset(offset)

        # =========================================================
        # Size
        # =========================================================

        size = self.conv1_size(x)
        size = self.conv2_size(size)
        size = self.conv3_size(size)
        size = self.conv4_size(size)

        score_map_size = self.conv5_size(size)

        return (
            self._sigmoid(score_map_ctr),
            self._sigmoid(score_map_size),
            score_map_offset,
        )

    def cal_bbox(
        self,
        score_map_ctr: torch.Tensor,
        size_map: torch.Tensor,
        offset_map: torch.Tensor,
    ) -> torch.Tensor:

        # Find highest-scoring center location.
        _, idx = torch.max(
            score_map_ctr.flatten(1),
            dim=1,
            keepdim=True,
        )

        # Convert flattened index to (x, y).
        idx_y = idx // self.feat_size
        idx_x = idx % self.feat_size

        # Gather size and offset at that location.
        gather_idx = idx.unsqueeze(1).expand(
            idx.shape[0],
            2,
            1,
        )

        size = size_map.flatten(2).gather(
            dim=2,
            index=gather_idx,
        )

        offset = offset_map.flatten(2).gather(
            dim=2,
            index=gather_idx,
        ).squeeze(-1)

        # ---------------------------------------------------------
        # Bounding box:
        #
        # cx = (grid_x + offset_x) / feature_size
        # cy = (grid_y + offset_y) / feature_size
        # w  = predicted width
        # h  = predicted height
        # ---------------------------------------------------------

        bbox = torch.cat(
            [
                (
                    idx_x.to(torch.float32)
                    + offset[:, :1]
                ) / self.feat_size,

                (
                    idx_y.to(torch.float32)
                    + offset[:, 1:]
                ) / self.feat_size,

                size.squeeze(-1),
            ],
            dim=1,
        )

        return bbox

    def forward(
        self,
        x: torch.Tensor,
        gt_score_map: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        score_map_ctr, size_map, offset_map = (
            self.get_score_map(x)
        )

        if gt_score_map is None:
            bbox = self.cal_bbox(
                score_map_ctr,
                size_map,
                offset_map,
            )
        else:
            bbox = self.cal_bbox(
                gt_score_map.unsqueeze(1),
                size_map,
                offset_map,
            )

        return (
            score_map_ctr,
            bbox,
            size_map,
            offset_map,
        )