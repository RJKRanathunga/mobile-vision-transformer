import torch

from mvt.losses.box_ops import (
    box_cxcywh_to_xyxy,
    box_xywh_to_xyxy,
)
from mvt.losses.focal_loss import FocalLoss
from mvt.losses.giou_loss import giou_loss
from mvt.losses.target import generate_heatmap


def test_box_cxcywh_to_xyxy():
    boxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.4],
    ])

    result = box_cxcywh_to_xyxy(boxes)

    expected = torch.tensor([
        [0.4, 0.3, 0.6, 0.7],
    ])

    assert torch.allclose(result, expected)


def test_box_xywh_to_xyxy():
    boxes = torch.tensor([
        [0.2, 0.3, 0.4, 0.5],
    ])

    result = box_xywh_to_xyxy(boxes)

    expected = torch.tensor([
        [0.2, 0.3, 0.6, 0.8],
    ])

    assert torch.allclose(result, expected)


def test_giou_identical_boxes():
    boxes = torch.tensor([
        [0.2, 0.2, 0.6, 0.6],
    ])

    loss, iou = giou_loss(
        boxes,
        boxes,
    )

    assert torch.allclose(
        loss,
        torch.tensor(0.0),
        atol=1e-6,
    )

    assert torch.allclose(
        iou,
        torch.tensor([1.0]),
        atol=1e-6,
    )


def test_focal_loss_backward():
    prediction = torch.full(
        (2, 1, 16, 16),
        0.5,
        requires_grad=True,
    )

    target = torch.zeros(
        2,
        1,
        16,
        16,
    )

    target[:, :, 8, 8] = 1.0

    loss = FocalLoss()(
        prediction,
        target,
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_heatmap_shape():
    boxes = torch.tensor([
        [0.4, 0.4, 0.2, 0.2],
        [0.1, 0.2, 0.3, 0.4],
    ])

    heatmap = generate_heatmap(
        boxes,
        heatmap_size=16,
    )

    assert heatmap.shape == (
        2,
        1,
        16,
        16,
    )

    assert torch.all(heatmap >= 0)
    assert torch.all(heatmap <= 1)


def test_heatmap_has_peak():
    boxes = torch.tensor([
        [0.4, 0.4, 0.2, 0.2],
    ])

    heatmap = generate_heatmap(
        boxes,
        heatmap_size=16,
    )

    # Center:
    #
    # x = (0.4 + 0.1) * 16 = 8
    # y = (0.4 + 0.1) * 16 = 8
    #
    # Therefore the Gaussian peak should be around (8, 8).

    assert heatmap[0, 0, 8, 8] == 1.0