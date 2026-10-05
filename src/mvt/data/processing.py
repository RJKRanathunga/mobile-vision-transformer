import torch

from mvt.data.processing_utils import jittered_center_crop


class MVTProcessing:
    """
    Official-style MVT training preprocessing.

    A template and a search frame are cropped around their target
    bounding boxes. The search crop uses bounding-box jittering.
    """

    TEMPLATE_SIZE = 128
    SEARCH_SIZE = 256

    TEMPLATE_FACTOR = 2.0
    SEARCH_FACTOR = 4.0

    TEMPLATE_CENTER_JITTER = 0.0
    TEMPLATE_SCALE_JITTER = 0.0

    SEARCH_CENTER_JITTER = 3.0
    SEARCH_SCALE_JITTER = 0.25

    @staticmethod
    def get_jittered_box(
        box: torch.Tensor,
        center_jitter: float,
        scale_jitter: float,
    ) -> torch.Tensor:
        """
        Jitter a bounding box following the official MVT implementation.
        """

        jittered_size = box[2:4] * torch.exp(
            torch.randn(2) * scale_jitter
        )

        max_offset = (
            torch.sqrt(jittered_size.prod())
            * center_jitter
        )

        jittered_center = (
            box[:2]
            + 0.5 * box[2:4]
            + max_offset * (torch.rand(2) - 0.5)
        )

        return torch.cat(
            (
                jittered_center - 0.5 * jittered_size,
                jittered_size,
            )
        )

    def __call__(
        self,
        template_image,
        template_bbox: torch.Tensor,
        search_image,
        search_bbox: torch.Tensor,
    ):
        """
        Process one template/search pair.

        Args:
            template_image:
                Original template frame as H x W x 3 numpy array.

            template_bbox:
                Template GT bbox [x, y, w, h].

            search_image:
                Original search frame as H x W x 3 numpy array.

            search_bbox:
                Search GT bbox [x, y, w, h].

        Returns:
            Dictionary containing processed images, normalized boxes,
            and attention masks.
        """

        # -----------------------------------------------------
        # Template
        # -----------------------------------------------------

        template_extract_bbox = self.get_jittered_box(
            template_bbox,
            center_jitter=self.TEMPLATE_CENTER_JITTER,
            scale_jitter=self.TEMPLATE_SCALE_JITTER,
        )

        (
            template_crops,
            template_boxes,
            template_masks,
        ) = jittered_center_crop(
            frames=[template_image],
            box_extract=[template_extract_bbox],
            box_gt=[template_bbox],
            search_area_factor=self.TEMPLATE_FACTOR,
            output_size=self.TEMPLATE_SIZE,
        )

        # -----------------------------------------------------
        # Search
        # -----------------------------------------------------

        search_extract_bbox = self.get_jittered_box(
            search_bbox,
            center_jitter=self.SEARCH_CENTER_JITTER,
            scale_jitter=self.SEARCH_SCALE_JITTER,
        )

        (
            search_crops,
            search_boxes,
            search_masks,
        ) = jittered_center_crop(
            frames=[search_image],
            box_extract=[search_extract_bbox],
            box_gt=[search_bbox],
            search_area_factor=self.SEARCH_FACTOR,
            output_size=self.SEARCH_SIZE,
        )

        return {
            "template_image": template_crops[0],
            "template_bbox": template_boxes[0],
            "template_attention_mask": template_masks[0],
            "search_image": search_crops[0],
            "search_bbox": search_boxes[0],
            "search_attention_mask": search_masks[0],
        }
