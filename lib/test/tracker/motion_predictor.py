"""Constant velocity/acceleration motion prior for MVT inference."""
from collections import deque
import math
import numpy as np
import torch


class MotionPredictor:
    def __init__(self, mode="baseline"):
        if mode not in ("baseline", "velocity", "acceleration"):
            raise ValueError("Invalid motion mode: " + str(mode))
        self.mode = mode
        self.history = deque(maxlen=3)

    def reset(self, bbox):
        self.history.clear()
        if self.mode != "baseline":
            self.update(bbox)

    def update(self, bbox):
        if self.mode == "baseline":
            return
        x, y, w, h = bbox
        self.history.append(np.array([x + w / 2, y + h / 2], dtype=np.float64))

    def predict(self):
        if self.mode == "baseline" or len(self.history) < 2:
            return None
        p1, p2 = self.history[-1], self.history[-2]
        v = p1 - p2
        if self.mode == "acceleration" and len(self.history) == 3:
            a = v - (p2 - self.history[-3])
            return p1 + v + a
        return p1 + v


def motion_window(center, previous_bbox, resize_factor, search_size,
                  score_map, offset_map, sigma):
    """Gaussian at predicted full-frame center, evaluated at decoded centers.

    Match MVT map_box_back and CENTER cal_bbox: cell index + predicted offset.
    Sigma is in feature cells. If the forecast is outside the search crop,
    return None so the original Hann response is used unchanged.
    """
    if center is None or not np.isfinite(center).all():
        return None
    h, w = score_map.shape[-2:]
    x, y, bw, bh = previous_bbox
    side = float(search_size) / float(resize_factor)
    if not math.isfinite(side) or side <= 0:
        return None
    mx = (float(center[0]) - (x + bw / 2 - side / 2)) * w / side
    my = (float(center[1]) - (y + bh / 2 - side / 2)) * h / side
    if not (0 <= mx < w and 0 <= my < h):
        return None
    # Original head uses (index + offset) / feature size, not cell centers.
    gy, gx = torch.meshgrid(
        torch.arange(h, device=score_map.device, dtype=torch.float32),
        torch.arange(w, device=score_map.device, dtype=torch.float32),
        indexing="ij")
    cx = gx + offset_map[:, 0].float()
    cy = gy + offset_map[:, 1].float()
    prior = torch.exp(-((cx - mx)**2 + (cy - my)**2) / (2 * sigma**2))
    return prior.unsqueeze(1).to(dtype=score_map.dtype)
