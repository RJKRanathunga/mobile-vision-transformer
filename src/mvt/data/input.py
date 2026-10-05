from pathlib import Path

import numpy as np
import torch
from PIL import Image


class MVTInputProcessor:

    @staticmethod
    def load_image(
        image_path: str | Path,
    ) -> np.ndarray:
        """
        Load an image as RGB H x W x 3 numpy array.
        """

        image = Image.open(image_path).convert("RGB")

        return np.array(image)

    @staticmethod
    def to_tensor(
        image: np.ndarray,
    ) -> torch.Tensor:
        """
        Convert H x W x 3 uint8 image to
        3 x H x W float tensor in [0, 1].
        """

        tensor = torch.from_numpy(image).float()

        tensor = tensor.permute(2, 0, 1)

        tensor /= 255.0

        return tensor