from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

TEMPLATE_SIZE = (128, 128)
SEARCH_SIZE = (256, 256)

class MVTInputProcessor:
    """
    Prepare template and search region images for MVT model.

    Template: 128 x 128 x 3
    Search region: 256 x 256 x 3
    """

    def __init__(self):
        self.template_transform = transforms.Compose([
            transforms.Resize(TEMPLATE_SIZE),
            transforms.ToTensor()
        ])

        self.search_transform = transforms.Compose([
            transforms.Resize(SEARCH_SIZE),
            transforms.ToTensor()
        ])

    def load_template(self, image_path: str | Path) -> torch.Tensor:
        """
        Load and preprocess a template image.
        Returns:
            Tensor with shape [1,3,128,128]
        """
        image = Image.open(image_path).convert("RGB")

        tensor = self.template_transform(image)

        # Add batch dimension (3, 128, 128) -> (1, 3, 128, 128)
        tensor = tensor.unsqueeze(0)
        return tensor

    def load_search(self, image_path: str | Path) -> torch.Tensor:
        """
        Load and preprocess a search region image
        Returns:
            Tensor with shape [1,3,256,256]
        """
        image = Image.open(image_path).convert("RGB")

        tensor = self.search_transform(image)

        return tensor.unsqueeze(0)

    def load_pair(self, template_path: str | Path, search_path: str | Path):
        template = self.load_template(template_path)
        search = self.load_search(search_path)
        return template, search