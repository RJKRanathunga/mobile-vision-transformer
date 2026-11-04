import numpy as np
import pytest
import torch
from torch import nn

from mvt.evaluation import MVTTracker, TrackerConfig, clip_box, hann2d


class _FakeHead:
    feat_size = 16

    @staticmethod
    def cal_bbox(response, size_map, offset_map):
        batch_size = response.shape[0]
        return torch.tensor(
            [[0.5, 0.5, 0.25, 0.5]],
            device=response.device,
        ).expand(batch_size, -1)


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.head = _FakeHead()
        self.template_calls = 0

    def extract_template_features(self, template):
        self.template_calls += 1
        return template.mean(dim=(2, 3), keepdim=True)

    def forward_with_template_features(self, search, template_features):
        batch_size = search.shape[0]
        device = search.device
        return {
            "score_map": torch.ones(batch_size, 1, 16, 16, device=device),
            "size_map": torch.zeros(batch_size, 2, 16, 16, device=device),
            "offset_map": torch.zeros(batch_size, 2, 16, 16, device=device),
        }


def test_hann_window_matches_upstream_shape_and_symmetry():
    window = hann2d(16, 16)
    assert window.shape == (1, 1, 16, 16)
    assert torch.allclose(window, window.flip(-1))
    assert torch.allclose(window, window.flip(-2))


def test_tracker_caches_template_and_maps_box_back():
    model = _FakeModel()
    tracker = MVTTracker(
        model=model,
        config=TrackerConfig(),
        device="cpu",
    )
    image = np.zeros((100, 120, 3), dtype=np.uint8)

    tracker.initialize(image, [40.0, 40.0, 20.0, 20.0])
    result = tracker.track(image)

    assert model.template_calls == 1
    assert result["target_bbox"] == pytest.approx(
        [40.0, 30.0, 20.0, 40.0]
    )


def test_clip_box_matches_upstream_margin_behavior():
    assert clip_box([-5.0, 95.0, 2.0, 20.0], 100, 120, margin=10) == [
        0.0,
        90,
        10.0,
        10.0,
    ]
