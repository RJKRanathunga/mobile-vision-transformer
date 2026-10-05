import numpy as np
import torch


def generate_heatmap(
    bboxes: torch.Tensor,
    heatmap_size: int = 16,
    min_overlap: float = 0.7,
) -> torch.Tensor:
    """
    Generate CenterNet-style Gaussian heatmaps.

    Args:
        bboxes:
            [B, 4] normalized xywh boxes.

        heatmap_size:
            Spatial size of the output heatmap.

        min_overlap:
            Minimum overlap used to determine Gaussian radius.

    Returns:
        [B, 1, heatmap_size, heatmap_size]
    """

    device = bboxes.device

    heatmap = torch.zeros(
        bboxes.shape[0],
        heatmap_size,
        heatmap_size,
        device=device,
        dtype=bboxes.dtype,
    )

    # Convert normalized box coordinates into
    # heatmap coordinates.
    boxes = bboxes * heatmap_size

    wh = boxes[:, 2:]

    centers = boxes[:, :2] + wh / 2.0

    centers_int = centers.round().long()

    radius = get_gaussian_radius(
        wh,
        min_overlap,
    )

    radius = torch.clamp(
        radius,
        min=0,
    ).long()

    for i in range(bboxes.shape[0]):
        draw_gaussian(
            heatmap[i],
            center=centers_int[i],
            radius=int(radius[i].item()),
        )

    return heatmap.unsqueeze(1)


def get_gaussian_radius(
    box_size: torch.Tensor,
    min_overlap: float,
) -> torch.Tensor:
    """
    Calculate Gaussian radius using the CenterNet equations.

    Args:
        box_size: [N, 2], where columns are width and height.

    Returns:
        [N] radius values.
    """

    width = box_size[:, 0]
    height = box_size[:, 1]

    # Case 1
    a1 = torch.ones_like(width)
    b1 = height + width
    c1 = (
        width
        * height
        * (1.0 - min_overlap)
        / (1.0 + min_overlap)
    )

    sq1 = torch.sqrt(
        torch.clamp(
            b1**2 - 4.0 * a1 * c1,
            min=0,
        )
    )

    r1 = (b1 + sq1) / 2.0

    # Case 2
    a2 = torch.full_like(width, 4.0)
    b2 = 2.0 * (height + width)
    c2 = (1.0 - min_overlap) * width * height

    sq2 = torch.sqrt(
        torch.clamp(
            b2**2 - 4.0 * a2 * c2,
            min=0,
        )
    )

    r2 = (b2 + sq2) / 2.0

    # Case 3
    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height

    sq3 = torch.sqrt(
        torch.clamp(
            b3**2 - 4.0 * a3 * c3,
            min=0,
        )
    )

    r3 = (b3 + sq3) / (2.0 * a3)

    return torch.minimum(
        r1,
        torch.minimum(r2, r3),
    )


def gaussian_2d(
    radius: int,
    sigma: float | None = None,
) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel.
    """

    if sigma is None:
        diameter = 2 * radius + 1
        sigma = diameter / 6.0

    y, x = np.ogrid[
        -radius:radius + 1,
        -radius:radius + 1,
    ]

    gaussian = np.exp(
        -(x * x + y * y)
        / (2.0 * sigma * sigma)
    )

    gaussian[
        gaussian < np.finfo(gaussian.dtype).eps
        * gaussian.max()
    ] = 0

    return gaussian


def draw_gaussian(
    heatmap: torch.Tensor,
    center: torch.Tensor,
    radius: int,
) -> None:
    """
    Draw a Gaussian onto one heatmap.

    Args:
        heatmap: [H, W]
        center: [2] containing x, y
        radius: Gaussian radius
    """

    diameter = 2 * radius + 1

    gaussian = gaussian_2d(
        radius,
        sigma=diameter / 6.0,
    )

    gaussian = torch.as_tensor(
        gaussian,
        dtype=heatmap.dtype,
        device=heatmap.device,
    )

    x = int(center[0].item())
    y = int(center[1].item())

    height, width = heatmap.shape

    left = min(x, radius)
    right = min(width - x, radius + 1)

    top = min(y, radius)
    bottom = min(height - y, radius + 1)

    if left <= 0 and right <= 0:
        return

    if top <= 0 and bottom <= 0:
        return

    heatmap_slice = heatmap[
        y - top:y + bottom,
        x - left:x + right,
    ]

    gaussian_slice = gaussian[
        radius - top:radius + bottom,
        radius - left:radius + right,
    ]

    if heatmap_slice.numel() == 0:
        return

    heatmap[
        y - top:y + bottom,
        x - left:x + right,
    ] = torch.maximum(
        heatmap_slice,
        gaussian_slice,
    )