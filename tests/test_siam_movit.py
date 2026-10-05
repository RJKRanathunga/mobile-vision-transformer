import torch

from mvt.models.backbone.siam_movit import SiamMoViTBlock


def test_siam_movit_layer3():
    block = SiamMoViTBlock(
        in_channels=96,
        transformer_dim=144,
        ffn_dim=288,
        num_transformer_blocks=2,
        patch_h=2,
        patch_w=2,
        num_heads=4,
    )

    search = torch.randn(1, 96, 32, 32)
    template = torch.randn(1, 96, 16, 16)

    search_out, template_out = block(search, template)

    assert search_out.shape == search.shape
    assert template_out.shape == template.shape


def test_siam_movit_layer4():
    block = SiamMoViTBlock(
        in_channels=128,
        transformer_dim=192,
        ffn_dim=384,
        num_transformer_blocks=4,
        patch_h=2,
        patch_w=2,
        num_heads=4,
    )

    search = torch.randn(1, 128, 16, 16)
    template = torch.randn(1, 128, 8, 8)

    search_out, template_out = block(search, template)

    assert search_out.shape == search.shape
    assert template_out.shape == template.shape