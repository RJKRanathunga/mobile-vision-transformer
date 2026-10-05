import torch

from mvt.models.head.mvt_head import CenterPredictor


def test_center_predictor_shapes():

    head = CenterPredictor(
        in_channels=256,
        hidden_channels=256,
        feat_size=16,
    )

    x = torch.randn(
        2, 256, 16, 16
    )

    score_map, bbox, size_map, offset_map = head(x)

    assert score_map.shape == (
        2, 1, 16, 16
    )

    assert bbox.shape == (
        2, 4
    )

    assert size_map.shape == (
        2, 2, 16, 16
    )

    assert offset_map.shape == (
        2, 2, 16, 16
    )


def test_size_map_is_sigmoid_bounded():

    head = CenterPredictor()

    x = torch.randn(
        2, 256, 16, 16
    )

    _, _, size_map, _ = head(x)

    assert torch.all(size_map > 0)
    assert torch.all(size_map < 1)


def test_center_predictor_backward():

    head = CenterPredictor()

    x = torch.randn(
        2,
        256,
        16,
        16,
        requires_grad=True,
    )

    score_map, bbox, size_map, offset_map = head(x)

    loss = (
        score_map.mean()
        + bbox.mean()
        + size_map.mean()
        + offset_map.mean()
    )

    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()