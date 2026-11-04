import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mvt.checkpoint import load_mvt_checkpoint
from mvt.data.input import MVTInputProcessor
from mvt.data.processing_utils import sample_target
from mvt.models.mvt import MVT


@dataclass(frozen=True)
class TrackerConfig:
    template_factor: float = 2.0
    template_size: int = 128
    search_factor: float = 4.0
    search_size: int = 256
    backbone_stride: int = 16
    clip_margin: int = 10


def hann1d(size: int) -> torch.Tensor:
    if size < 1:
        raise ValueError("Hann window size must be positive")
    positions = torch.arange(1, size + 1, dtype=torch.float32)
    return 0.5 * (
        1.0
        - torch.cos((2.0 * math.pi / (size + 1)) * positions)
    )


def hann2d(height: int, width: int) -> torch.Tensor:
    return (
        hann1d(height).reshape(1, 1, height, 1)
        * hann1d(width).reshape(1, 1, 1, width)
    )


def clip_box(
    box: list[float],
    height: int,
    width: int,
    margin: int = 0,
) -> list[float]:
    x1, y1, box_width, box_height = box
    x2 = x1 + box_width
    y2 = y1 + box_height
    x1 = min(max(0.0, x1), width - margin)
    x2 = min(max(float(margin), x2), float(width))
    y1 = min(max(0.0, y1), height - margin)
    y2 = min(max(float(margin), y2), float(height))
    return [
        x1,
        y1,
        max(float(margin), x2 - x1),
        max(float(margin), y2 - y1),
    ]


class MVTTracker:
    """Stateful MVT inference pipeline matching the upstream tracker."""

    def __init__(
        self,
        model: MVT | None = None,
        *,
        checkpoint: str | Path | Mapping[str, Any] | None = None,
        config: TrackerConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config or TrackerConfig()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = model or MVT()
        if checkpoint is not None:
            load_mvt_checkpoint(self.model, checkpoint)
        self.model.to(self.device).eval()

        feature_size = (
            self.config.search_size // self.config.backbone_stride
        )
        if feature_size != self.model.head.feat_size:
            raise ValueError(
                "tracker search size/stride does not match the model head"
            )
        self.output_window = hann2d(feature_size, feature_size).to(self.device)
        self.state: list[float] | None = None
        self.template_features: torch.Tensor | None = None

    def _to_batch(self, image: np.ndarray) -> torch.Tensor:
        return (
            MVTInputProcessor.to_tensor(np.ascontiguousarray(image))
            .unsqueeze(0)
            .to(self.device)
        )

    def initialize(
        self,
        image: np.ndarray,
        init_bbox: list[float] | tuple[float, float, float, float],
    ) -> dict[str, list[float]]:
        template_crop, _, _ = sample_target(
            image,
            list(init_bbox),
            self.config.template_factor,
            self.config.template_size,
        )
        with torch.inference_mode():
            self.template_features = self.model.extract_template_features(
                self._to_batch(template_crop)
            )
        self.state = [float(value) for value in init_bbox]
        return {"target_bbox": self.state.copy()}

    def track(self, image: np.ndarray) -> dict[str, list[float]]:
        if self.state is None or self.template_features is None:
            raise RuntimeError("initialize() must be called before track()")

        height, width = image.shape[:2]
        search_crop, resize_factor, _ = sample_target(
            image,
            self.state,
            self.config.search_factor,
            self.config.search_size,
        )

        with torch.inference_mode():
            outputs = self.model.forward_with_template_features(
                self._to_batch(search_crop),
                self.template_features,
            )
            response = outputs["score_map"] * self.output_window
            boxes = self.model.head.cal_bbox(
                response,
                outputs["size_map"],
                outputs["offset_map"],
            ).view(-1, 4)
            crop_box = (
                boxes.mean(dim=0)
                * self.config.search_size
                / resize_factor
            ).tolist()

        mapped = self.map_box_back(crop_box, resize_factor)
        self.state = clip_box(
            mapped,
            height,
            width,
            margin=self.config.clip_margin,
        )
        return {"target_bbox": self.state.copy()}

    def map_box_back(
        self,
        predicted_box: list[float],
        resize_factor: float,
    ) -> list[float]:
        if self.state is None:
            raise RuntimeError("tracker state is not initialized")
        previous_cx = self.state[0] + 0.5 * self.state[2]
        previous_cy = self.state[1] + 0.5 * self.state[3]
        cx, cy, width, height = predicted_box
        half_side = 0.5 * self.config.search_size / resize_factor
        real_cx = cx + previous_cx - half_side
        real_cy = cy + previous_cy - half_side
        return [
            real_cx - 0.5 * width,
            real_cy - 0.5 * height,
            width,
            height,
        ]
