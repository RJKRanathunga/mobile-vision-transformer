import torch


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute area of boxes in xyxy format.

    Args:
        boxes: [N, 4]

    Returns:
        areas: [N]
    """

    width = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    height = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)

    return width * height


def box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute pairwise IoU for corresponding boxes.

    Args:
        boxes1: [N, 4] in xyxy format.
        boxes2: [N, 4] in xyxy format.

    Returns:
        iou: [N]
        union: [N]
    """

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    top_left = torch.maximum(
        boxes1[:, :2],
        boxes2[:, :2],
    )

    bottom_right = torch.minimum(
        boxes1[:, 2:],
        boxes2[:, 2:],
    )

    wh = (bottom_right - top_left).clamp(min=0)

    intersection = wh[:, 0] * wh[:, 1]

    union = area1 + area2 - intersection

    iou = intersection / union.clamp(min=1e-12)

    return iou, union


def generalized_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized IoU.

    Args:
        boxes1: [N, 4] in xyxy format.
        boxes2: [N, 4] in xyxy format.

    Returns:
        giou: [N]
        iou: [N]
    """

    if not torch.all(boxes1[:, 2:] >= boxes1[:, :2]):
        raise ValueError("boxes1 contains invalid boxes.")

    if not torch.all(boxes2[:, 2:] >= boxes2[:, :2]):
        raise ValueError("boxes2 contains invalid boxes.")

    iou, union = box_iou(boxes1, boxes2)

    enclosing_top_left = torch.minimum(
        boxes1[:, :2],
        boxes2[:, :2],
    )

    enclosing_bottom_right = torch.maximum(
        boxes1[:, 2:],
        boxes2[:, 2:],
    )

    enclosing_wh = (
        enclosing_bottom_right - enclosing_top_left
    ).clamp(min=0)

    enclosing_area = (
        enclosing_wh[:, 0] * enclosing_wh[:, 1]
    )

    giou = (
        iou
        - (enclosing_area - union)
        / enclosing_area.clamp(min=1e-12)
    )

    return giou, iou


def giou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GIoU loss.

    Returns:
        loss: scalar
        iou: [N]
    """

    giou, iou = generalized_box_iou(
        boxes1,
        boxes2,
    )

    loss = (1.0 - giou).mean()

    return loss, iou