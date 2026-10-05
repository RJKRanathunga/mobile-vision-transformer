import torch
from torch import nn

from mvt.losses.box_ops import (
    box_cxcywh_to_xyxy,
    box_xywh_to_xyxy,
)
from mvt.losses.focal_loss import FocalLoss
from mvt.losses.giou_loss import giou_loss
from mvt.losses.target import generate_heatmap


class MVTLoss(nn.Module):
    """
    Complete training objective used by MVT.

        L = L_focal + 5 * L1 + 2 * L_GIoU
    """

    def __init__(
        self,
        focal_weight: float = 1.0,
        l1_weight: float = 5.0,
        giou_weight: float = 2.0,
        heatmap_size: int = 16,
    ) -> None:
        super().__init__()

        self.focal_loss = FocalLoss()

        self.focal_weight = focal_weight
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight

        self.heatmap_size = heatmap_size

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        gt_boxes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            predictions:
                Output dictionary from MVT.

            gt_boxes:
                Ground-truth boxes in normalized xywh format.

                Shape:
                    [B, 4]

        Returns:
            Dictionary containing individual losses and total loss.
        """

        score_map = predictions["score_map"]
        pred_boxes = predictions["pred_boxes"]

        # --------------------------------------------------
        # 1. Generate Gaussian target heatmap
        # --------------------------------------------------

        gt_heatmap = generate_heatmap(
            gt_boxes,
            heatmap_size=self.heatmap_size,
        )

        # --------------------------------------------------
        # 2. Focal / localization loss
        # --------------------------------------------------

        focal = self.focal_loss(
            score_map,
            gt_heatmap,
        )

        # --------------------------------------------------
        # 3. Bounding-box conversion
        # --------------------------------------------------

        # Model:
        # [cx, cy, w, h]

        # GT:
        # [x, y, w, h]

        pred_boxes = pred_boxes.view(-1, 4)

        gt_boxes = gt_boxes[:, None, :]
        gt_boxes = gt_boxes.expand(
            -1,
            predictions["pred_boxes"].shape[1],
            -1,
        )
        gt_boxes = gt_boxes.reshape(-1, 4)

        pred_boxes_xyxy = box_cxcywh_to_xyxy(
            pred_boxes
        )

        gt_boxes_xyxy = box_xywh_to_xyxy(
            gt_boxes
        )

        gt_boxes_xyxy = gt_boxes_xyxy.clamp(
            min=0.0,
            max=1.0,
        )

        # --------------------------------------------------
        # 4. L1 loss
        # --------------------------------------------------

        l1 = torch.nn.functional.l1_loss(
            pred_boxes_xyxy,
            gt_boxes_xyxy,
        )

        # --------------------------------------------------
        # 5. GIoU loss
        # --------------------------------------------------

        giou, iou = giou_loss(
            pred_boxes_xyxy,
            gt_boxes_xyxy,
        )

        # --------------------------------------------------
        # 6. Weighted total
        # --------------------------------------------------

        total = (
            self.focal_weight * focal
            + self.l1_weight * l1
            + self.giou_weight * giou
        )

        return {
            "loss": total,
            "focal": focal,
            "l1": l1,
            "giou": giou,
            "iou": iou.mean(),
            "gt_heatmap": gt_heatmap,
        }