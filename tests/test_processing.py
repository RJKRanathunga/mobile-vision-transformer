import numpy as np
import torch

from mvt.data.processing_utils import (
    sample_target,
    transform_image_to_crop,
)

from mvt.data.processing import get_jittered_box


def test_sample_target_search_size():
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    bbox = torch.tensor(
        [250.0, 180.0, 80.0, 100.0]
    )

    crop, resize_factor, attention_mask = sample_target(
        image,
        bbox,
        search_area_factor=4.0,
        output_size=256,
    )

    assert crop.shape == (256, 256, 3)
    assert attention_mask.shape == (256, 256)
    assert np.isfinite(resize_factor)


def test_sample_target_template_size():
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    bbox = torch.tensor(
        [250.0, 180.0, 80.0, 100.0]
    )

    crop, resize_factor, attention_mask = sample_target(
        image,
        bbox,
        search_area_factor=2.0,
        output_size=128,
    )

    assert crop.shape == (128, 128, 3)
    assert attention_mask.shape == (128, 128)


def test_transform_image_to_crop():
    bbox = torch.tensor(
        [100.0, 100.0, 50.0, 50.0]
    )

    crop_box = torch.tensor(
        [50.0, 50.0, 150.0, 150.0]
    )

    result = transform_image_to_crop(
        bbox,
        crop_box,
        resize_factor=1.0,
        crop_size=torch.tensor([200.0, 200.0]),
        normalize=True,
    )

    assert result.shape == (4,)
    assert torch.isfinite(result).all()


def test_jittered_box_shape():
    box = torch.tensor(
        [100.0, 100.0, 50.0, 80.0]
    )

    jittered = get_jittered_box(
        box,
        center_jitter=3.0,
        scale_jitter=0.25,
    )

    assert jittered.shape == (4,)
    assert torch.isfinite(jittered).all()


def test_zero_jitter():
    box = torch.tensor(
        [100.0, 100.0, 50.0, 80.0]
    )

    jittered = get_jittered_box(
        box,
        center_jitter=0.0,
        scale_jitter=0.0,
    )

    assert torch.allclose(jittered, box)