import torch

from mvt.models.backbone.mvt_backbone import MVTBackbone


def test_mvt_backbone():
    backbone = MVTBackbone()

    search = torch.randn(1, 3, 256, 256)
    template = torch.randn(1, 3, 128, 128)

    search_features, template_features = backbone(
        search,
        template,
    )

    assert search_features.shape == (1, 128, 16, 16)
    assert template_features.shape == (1, 128, 8, 8)


def test_mvt_backbone_shapes():
    backbone = MVTBackbone()

    search = torch.randn(1, 3, 256, 256)
    template = torch.randn(1, 3, 128, 128)

    # First conv
    search = backbone.first_conv(search)
    template = backbone.first_conv(template)

    assert search.shape == (1, 16, 128, 128)
    assert template.shape == (1, 16, 64, 64)

    # Layer 1
    search = backbone.layer1(search)
    template = backbone.layer1(template)

    assert search.shape == (1, 32, 128, 128)
    assert template.shape == (1, 32, 64, 64)

    # Layer 2
    search = backbone.layer2(search)
    template = backbone.layer2(template)

    assert search.shape == (1, 64, 64, 64)
    assert template.shape == (1, 64, 32, 32)

    # Layer 3 MV2
    search = backbone.layer3_mv2(search)
    template = backbone.layer3_mv2(template)

    assert search.shape == (1, 96, 32, 32)
    assert template.shape == (1, 96, 16, 16)

    # Layer 3 Siam-MoViT
    search, template = backbone.layer3_movit(
        search,
        template,
    )

    assert search.shape == (1, 96, 32, 32)
    assert template.shape == (1, 96, 16, 16)

    # Layer 4 MV2
    search = backbone.layer4_mv2(search)
    template = backbone.layer4_mv2(template)

    assert search.shape == (1, 128, 16, 16)
    assert template.shape == (1, 128, 8, 8)

    # Layer 4 Siam-MoViT
    search, template = backbone.layer4_movit(
        search,
        template,
    )

    assert search.shape == (1, 128, 16, 16)
    assert template.shape == (1, 128, 8, 8)