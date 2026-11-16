"""Tracking-pair transforms based on the original MVT / OSTrack pipeline."""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch


def _flip(image: np.ndarray, box: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
    out = box.clone()
    out[0] = 1.0 - box[0] - box[2]  # normalized xywh
    return np.ascontiguousarray(image[:, ::-1]), out


def _to_tensor(image: np.ndarray, brightness: float = 1.0) -> torch.Tensor:
    # RGB HWC uint8 -> CHW float; upstream mean=0 and std=1.
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().mul_(brightness / 255.0).clamp_(0.0, 1.0)


def augment_pair(
    template_image: np.ndarray,
    template_box: torch.Tensor,
    search_image: np.ndarray,
    search_box: torch.Tensor,
    *,
    training: bool,
    rng: random.Random,
) -> dict[str, torch.Tensor]:
    """Apply shared grayscale/flip, independent crop flips and brightness jitter.

    Flips are performed after cropping, with normalized boxes transformed in
    the same way. Validation is intentionally free from color/flip augmentation.
    """
    if training:
        if rng.random() < 0.05:
            for name, im in (("template", template_image), ("search", search_image)):
                gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                if name == "template":
                    template_image = rgb
                else:
                    search_image = rgb

        shared_flip = rng.random() < 0.5
        if shared_flip:
            template_image, template_box = _flip(template_image, template_box)
            search_image, search_box = _flip(search_image, search_box)
        if rng.random() < 0.5:
            template_image, template_box = _flip(template_image, template_box)
        if rng.random() < 0.5:
            search_image, search_box = _flip(search_image, search_box)

    template_brightness = rng.uniform(0.8, 1.2) if training else 1.0
    search_brightness = rng.uniform(0.8, 1.2) if training else 1.0
    return {
        "template_image": _to_tensor(template_image, template_brightness),
        "template_bbox": template_box.float(),
        "search_image": _to_tensor(search_image, search_brightness),
        "search_bbox": search_box.float(),
    }
