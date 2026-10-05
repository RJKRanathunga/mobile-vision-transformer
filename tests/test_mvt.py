import torch

from mvt.models.mvt import MVT


def test_mvt_output_shapes():

    model = MVT()

    search = torch.randn(
        2,
        3,
        256,
        256,
    )

    template = torch.randn(
        2,
        3,
        128,
        128,
    )

    outputs = model(
        search,
        template,
    )

    assert outputs["pred_boxes"].shape == (
        2,
        1,
        4,
    )

    assert outputs["score_map"].shape == (
        2,
        1,
        16,
        16,
    )

    assert outputs["size_map"].shape == (
        2,
        2,
        16,
        16,
    )

    assert outputs["offset_map"].shape == (
        2,
        2,
        16,
        16,
    )


def test_mvt_backward():

    model = MVT()

    search = torch.randn(
        1,
        3,
        256,
        256,
        requires_grad=True,
    )

    template = torch.randn(
        1,
        3,
        128,
        128,
        requires_grad=True,
    )

    outputs = model(
        search,
        template,
    )

    loss = (
        outputs["score_map"].mean()
        + outputs["pred_boxes"].mean()
        + outputs["size_map"].mean()
        + outputs["offset_map"].mean()
    )

    loss.backward()

    assert search.grad is not None
    assert template.grad is not None

    assert torch.isfinite(search.grad).all()
    assert torch.isfinite(template.grad).all()