import torch


def get_jittered_box(
    box: torch.Tensor,
    center_jitter: float,
    scale_jitter: float,
) -> torch.Tensor:
    """
    Jitter a bounding box following the official MVT preprocessing.

    Args:
        box: [x, y, w, h]
        center_jitter: Center jitter factor.
        scale_jitter: Scale jitter factor.

    Returns:
        Jittered box [x, y, w, h].
    """

    jittered_size = box[2:4] * torch.exp(
        torch.randn(2) * scale_jitter
    )

    max_offset = (
        torch.sqrt(jittered_size.prod())
        * center_jitter
    )

    original_center = (
        box[:2] + 0.5 * box[2:4]
    )

    jittered_center = (
        original_center
        + max_offset * (torch.rand(2) - 0.5)
    )

    jittered_box = torch.cat(
        (
            jittered_center - 0.5 * jittered_size,
            jittered_size,
        )
    )

    return jittered_box