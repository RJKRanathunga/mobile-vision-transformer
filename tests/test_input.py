import numpy as np
import torch
from PIL import Image

from mvt.data.input import MVTInputProcessor


def test_template_processing(tmp_path):
    processor = MVTInputProcessor()

    image = Image.new("RGB", (300, 200))
    image_path = tmp_path / "template.jpg"
    image.save(image_path)

    image = processor.load_image(image_path)

    bbox = torch.tensor(
        [100.0, 60.0, 50.0, 80.0]
    )

    crop, bbox_crop, attention_mask = processor.process_template(
        image,
        bbox,
    )

    assert crop.shape == (128, 128, 3)
    assert bbox_crop.shape == (4,)
    assert attention_mask.shape == (128, 128)


def test_search_processing(tmp_path):
    processor = MVTInputProcessor()

    image = Image.new("RGB", (500, 400))
    image_path = tmp_path / "search.jpg"
    image.save(image_path)

    image = processor.load_image(image_path)

    bbox = torch.tensor(
        [200.0, 150.0, 80.0, 100.0]
    )

    crop, bbox_crop, attention_mask = processor.process_search(
        image,
        bbox_extract=bbox,
        bbox_gt=bbox,
    )

    assert crop.shape == (256, 256, 3)
    assert bbox_crop.shape == (4,)
    assert attention_mask.shape == (256, 256)


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