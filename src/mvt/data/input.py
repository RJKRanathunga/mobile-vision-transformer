from pathlib import Path

import torch
from PIL import Image

from mvt.data.processing_utils import jittered_center_crop


TEMPLATE_SIZE = 128
SEARCH_SIZE = 256

TEMPLATE_FACTOR = 2.0
SEARCH_FACTOR = 4.0


class MVTInputProcessor:
    """
    Prepare template and search regions for MVT.

    The input images are original video frames and the corresponding
    bounding boxes are used to extract the template/search regions.
    """

    def load_image(self, image_path: str | Path):
        """
        Load an image as an RGB numpy array.
        """

        image = Image.open(image_path).convert("RGB")

        return __import__("numpy").array(image)

    def process_template(
        self,
        image,
        bbox: torch.Tensor,
    ):
        """
        Extract a 128x128 template crop.
        """

        crops, boxes, attention_masks = jittered_center_crop(
            frames=[image],
            box_extract=[bbox],
            box_gt=[bbox],
            search_area_factor=TEMPLATE_FACTOR,
            output_size=TEMPLATE_SIZE,
        )

        return crops[0], boxes[0], attention_masks[0]

    def process_search(
        self,
        image,
        bbox_extract: torch.Tensor,
        bbox_gt: torch.Tensor,
    ):
        """
        Extract a 256x256 search crop.

        bbox_extract determines where the crop is taken.
        bbox_gt is the actual target box that gets transformed.
        """

        crops, boxes, attention_masks = jittered_center_crop(
            frames=[image],
            box_extract=[bbox_extract],
            box_gt=[bbox_gt],
            search_area_factor=SEARCH_FACTOR,
            output_size=SEARCH_SIZE,
        )

        return crops[0], boxes[0], attention_masks[0]

    @staticmethod
    def to_tensor(image):
        """
        Convert HxWx3 uint8 image to 3xHxW float tensor in [0, 1].
        """

        tensor = torch.from_numpy(image).float()

        tensor = tensor.permute(2, 0, 1)

        tensor /= 255.0

        return tensor