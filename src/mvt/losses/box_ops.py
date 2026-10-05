import torch


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from:

        (center_x, center_y, width, height)

    to:

        (x1, y1, x2, y2)

    Args:
        boxes: [..., 4]

    Returns:
        Converted boxes: [..., 4]
    """

    cx, cy, w, h = boxes.unbind(dim=-1)

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    return torch.stack(
        [x1, y1, x2, y2],
        dim=-1,
    )


def box_xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from:

        (x, y, width, height)

    to:

        (x1, y1, x2, y2)

    where (x, y) is the top-left corner.

    Args:
        boxes: [..., 4]

    Returns:
        Converted boxes: [..., 4]
    """

    x, y, w, h = boxes.unbind(dim=-1)

    x2 = x + w
    y2 = y + h

    return torch.stack(
        [x, y, x2, y2],
        dim=-1,
    )