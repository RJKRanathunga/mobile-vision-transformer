import numpy as np
import torch
from PIL import Image

from mvt.data.input import MVTInputProcessor


def test_load_image(tmp_path):
    processor = MVTInputProcessor()

    image = Image.new("RGB", (300, 200))
    image_path = tmp_path / "test.jpg"
    image.save(image_path)

    loaded_image = processor.load_image(image_path)

    assert isinstance(loaded_image, np.ndarray)
    assert loaded_image.shape == (200, 300, 3)
    assert loaded_image.dtype == np.uint8


def test_load_image_is_rgb(tmp_path):
    processor = MVTInputProcessor()

    # Create a grayscale image.
    image = Image.new("L", (300, 200))
    image_path = tmp_path / "grayscale.jpg"
    image.save(image_path)

    loaded_image = processor.load_image(image_path)

    # load_image() should always convert images to RGB.
    assert loaded_image.shape == (200, 300, 3)


def test_to_tensor():
    processor = MVTInputProcessor()

    image = np.zeros(
        (128, 128, 3),
        dtype=np.uint8,
    )

    tensor = processor.to_tensor(image)

    assert tensor.shape == (3, 128, 128)
    assert tensor.dtype == torch.float32
    assert torch.all(tensor >= 0)
    assert torch.all(tensor <= 1)


def test_to_tensor_scales_pixel_values():
    processor = MVTInputProcessor()

    image = np.array(
        [
            [
                [0, 128, 255],
            ]
        ],
        dtype=np.uint8,
    )

    tensor = processor.to_tensor(image)

    expected = torch.tensor(
        [
            [[0.0]],
            [[128.0 / 255.0]],
            [[1.0]],
        ]
    )

    assert torch.allclose(tensor, expected)