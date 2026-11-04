import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def sample_target(
    image: np.ndarray,
    target_bbox: torch.Tensor | list[float] | tuple[float, float, float, float],
    search_area_factor: float,
    output_size: int | None,
    mask: torch.Tensor | None = None,
):
    """
    Extract a square crop centered on the target bounding box.

    Args:
        image:
            Input image as H x W x 3 numpy array.

        target_bbox:
            Bounding box in [x, y, w, h] format.

        search_area_factor:
            Determines the size of the crop relative to the target.

        output_size:
            Size of the output square crop.

    Returns:
        crop:
            Cropped and resized image.

        resize_factor:
            Factor used to resize the crop.

        attention_mask:
            Boolean mask indicating padded regions.
    """

    if isinstance(target_bbox, torch.Tensor):
        x, y, w, h = target_bbox.tolist()
    else:
        x, y, w, h = target_bbox

    # Size of the square crop.
    crop_size = math.ceil(
        math.sqrt(w * h) * search_area_factor
    )

    if crop_size < 1:
        raise ValueError("Target bounding box is too small.")

    # Crop coordinates.
    x1 = round(x + 0.5 * w - 0.5 * crop_size)
    x2 = x1 + crop_size

    y1 = round(y + 0.5 * h - 0.5 * crop_size)
    y2 = y1 + crop_size

    # Amount of padding required if crop extends outside image.
    x1_pad = max(0, -x1)
    x2_pad = max(x2 - image.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - image.shape[0] + 1, 0)

    # Crop the valid part of the image.
    image_crop = image[
        y1 + y1_pad : y2 - y2_pad,
        x1 + x1_pad : x2 - x2_pad,
        :,
    ]
    if mask is not None:
        mask_crop = mask[
            y1 + y1_pad : y2 - y2_pad,
            x1 + x1_pad : x2 - x2_pad,
        ]

    # Pad areas outside the original image.
    image_crop = cv2.copyMakeBorder(
        image_crop,
        y1_pad,
        y2_pad,
        x1_pad,
        x2_pad,
        cv2.BORDER_CONSTANT,
    )
    if mask is not None:
        mask_crop = F.pad(
            mask_crop,
            pad=(x1_pad, x2_pad, y1_pad, y2_pad),
            mode="constant",
            value=0,
        )

    # ---------------------------------------------------------
    # Attention mask
    # ---------------------------------------------------------

    height, width, _ = image_crop.shape

    # Upstream creates this as floating point before interpolation. Using
    # uint8 changes boundary pixels after OpenCV's bilinear resize.
    attention_mask = np.ones((height, width))

    end_x = -x2_pad if x2_pad > 0 else None
    end_y = -y2_pad if y2_pad > 0 else None

    attention_mask[
        y1_pad:end_y,
        x1_pad:end_x,
    ] = False

    # ---------------------------------------------------------
    # Resize
    # ---------------------------------------------------------

    if output_size is None:
        if mask is None:
            return image_crop, attention_mask.astype(np.bool_), 1.0
        return image_crop, 1.0, attention_mask.astype(np.bool_), mask_crop

    resize_factor = output_size / crop_size

    image_crop = cv2.resize(
        image_crop,
        (output_size, output_size),
    )

    attention_mask = cv2.resize(
        attention_mask,
        (output_size, output_size),
    ).astype(np.bool_)

    if mask is None:
        return image_crop, resize_factor, attention_mask

    mask_crop = F.interpolate(
        mask_crop[None, None],
        (output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return image_crop, resize_factor, attention_mask, mask_crop


def transform_image_to_crop(
    box: torch.Tensor,
    crop_box: torch.Tensor,
    resize_factor: float,
    crop_size: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Transform a bounding box from original-image coordinates
    to coordinates inside the cropped image.

    All boxes use [x, y, w, h].
    """

    crop_box_center = (
        crop_box[:2] + 0.5 * crop_box[2:4]
    )

    box_center = (
        box[:2] + 0.5 * box[2:4]
    )

    # Transform center into crop coordinates.
    output_center = (
        (crop_size - 1) / 2
        + (box_center - crop_box_center) * resize_factor
    )

    # Transform width and height.
    output_wh = box[2:4] * resize_factor

    output_box = torch.cat(
        (
            output_center - 0.5 * output_wh,
            output_wh,
        )
    )

    if normalize:
        output_box = output_box / crop_size[0]

    return output_box


def jittered_center_crop(
    frames,
    box_extract,
    box_gt,
    search_area_factor,
    output_size,
    masks=None,
):
    """
    Crop frames around box_extract and transform box_gt
    into the cropped-image coordinate system.
    """

    crops = []
    resize_factors = []
    attention_masks = []
    mask_crops = [] if masks is not None else None
    boxes = []

    if masks is None:
        samples = zip(frames, box_extract, box_gt, [None] * len(frames))
    else:
        samples = zip(frames, box_extract, box_gt, masks)

    for frame, extract_box, gt_box, mask in samples:
        result = sample_target(
            frame, extract_box, search_area_factor, output_size, mask
        )
        if mask is None:
            crop, resize_factor, attention_mask = result
        else:
            crop, resize_factor, attention_mask, mask_crop = result
            mask_crops.append(mask_crop)

        crops.append(crop)
        resize_factors.append(resize_factor)
        attention_masks.append(attention_mask)

        crop_size = torch.tensor(
            [output_size, output_size],
            dtype=torch.float32,
        )

        transformed_box = transform_image_to_crop(
            gt_box,
            extract_box,
            resize_factor,
            crop_size,
            normalize=True,
        )

        boxes.append(transformed_box)

    return (
        crops,
        boxes,
        attention_masks,
        mask_crops,
    )
