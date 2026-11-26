"""Causal, hysteresis-based crossing detection for an infinite image line."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CrossingEvent:
    frame_number: int
    timestamp: float
    direction: str
    center_x: float
    center_y: float
    signed_distance: float


class VirtualTripwire:
    def __init__(self, point_a, point_b, margin=10.0):
        self.a = tuple(map(float, point_a))
        self.b = tuple(map(float, point_b))
        if len(self.a) != 2 or len(self.b) != 2:
            raise ValueError('Endpoints must be XY pairs')
        self.dx = self.b[0] - self.a[0]
        self.dy = self.b[1] - self.a[1]
        self.length = math.hypot(self.dx, self.dy)
        self.margin = float(margin)
        if not math.isfinite(self.length) or self.length <= 0:
            raise ValueError('Tripwire endpoints must be finite and distinct')
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError('Margin must be finite and nonnegative')
        self.reset()

    def reset(self):
        self.previous_side = None
        self.last_frame = 0
        self.events = []

    @staticmethod
    def get_center(bbox):
        x, y, w, h = map(float, bbox)
        if not all(map(math.isfinite, (x, y, w, h))) or w <= 0 or h <= 0:
            raise ValueError('Invalid XYWH box')
        return x + w / 2, y + h / 2

    def signed_distance(self, center):
        x, y = center
        return (self.dx * (y - self.a[1]) - self.dy * (x - self.a[0])) / self.length

    def update(self, bbox, frame_number, fps):
        if frame_number <= self.last_frame or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Frame numbers must increase and FPS must be positive')
        if frame_number != self.last_frame + 1:
            self.previous_side = None
        self.last_frame = frame_number
        try:
            center = self.get_center(bbox)
        except (ValueError, TypeError, OverflowError):
            self.previous_side = None
            return None
        d = self.signed_distance(center)
        side = 1 if d > self.margin else -1 if d < -self.margin else 0
        if side == 0:
            return None
        previous = self.previous_side
        self.previous_side = side
        if previous is None or previous == side:
            return None
        event = CrossingEvent(frame_number, (frame_number - 1) / fps,
                              'positive_to_negative' if side == -1 else 'negative_to_positive',
                              *center, d)
        self.events.append(event)
        return event
