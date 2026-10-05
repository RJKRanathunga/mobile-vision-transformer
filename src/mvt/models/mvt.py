import torch
from torch import nn

from mvt.models.backbone.mvt_backbone import MVTBackbone
from mvt.models.neck.mvt_neck import MVTNeck, FeatureFusor
from mvt.models.head.mvt_head import CenterPredictor


class MVT(nn.Module):
    """
    Mobile Vision Transformer-based Visual Object Tracker.

    MVT-Small configuration:

    Search image:
        [B, 3, 256, 256]

    Template image:
        [B, 3, 128, 128]

    Backbone output:
        search:   [B, 128, 16, 16]
        template: [B, 128, 8, 8]

    Feature fusor output:
        [B, 256, 16, 16]

    Prediction:
        score_map:  [B, 1, 16, 16]
        bbox:       [B, 4]
        size_map:   [B, 2, 16, 16]
        offset_map: [B, 2, 16, 16]
    """

    def __init__(self) -> None:
        super().__init__()

        # =========================================================
        # Backbone
        #
        # Processes search and template using shared weights.
        # =========================================================

        self.backbone = MVTBackbone()

        # =========================================================
        # Neck
        #
        # Applies separate BatchNorm adjustment to search/template.
        # =========================================================

        self.neck = MVTNeck(
            num_channels=128,
        )

        # =========================================================
        # Feature fusor
        #
        # Template:
        #     [B, 128, 8, 8]
        #
        # Search:
        #     [B, 128, 16, 16]
        #
        # PWCA:
        #     [B, 64, 16, 16]
        #
        # 1x1 adjustment:
        #     [B, 256, 16, 16]
        # =========================================================

        self.feature_fusor = FeatureFusor(
            input_channels=128,
            correlation_channels=64,
            output_channels=256,
        )

        # =========================================================
        # Center prediction head
        # =========================================================

        self.head = CenterPredictor(
            in_channels=256,
            hidden_channels=256,
            feat_size=16,
        )

    def forward(
        self,
        search: torch.Tensor,
        template: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        # =========================================================
        # 1. Backbone
        # =========================================================

        search_features, template_features = self.backbone(
            search,
            template,
        )

        # =========================================================
        # 2. Neck
        # =========================================================

        search_features, template_features = self.neck(
            search_features,
            template_features,
        )

        # =========================================================
        # 3. Feature fusion
        #
        # The template acts as the correlation kernel and the
        # search feature is where the correlation map is produced.
        # =========================================================

        fused_features = self.feature_fusor(
            template_features,
            search_features,
        )

        # =========================================================
        # 4. Prediction head
        # =========================================================

        score_map, bbox, size_map, offset_map = self.head(
            fused_features,
        )

        # =========================================================
        # 5. Return predictions
        # =========================================================

        batch_size = search.shape[0]

        return {
            "pred_boxes": bbox.view(
                batch_size,
                1,
                4,
            ),
            "score_map": score_map,
            "size_map": size_map,
            "offset_map": offset_map,
        }