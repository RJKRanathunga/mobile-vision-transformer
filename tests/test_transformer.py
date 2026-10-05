import torch

from mvt.layers.transformer import TransformerEncoderBlock


def test_transformer_encoder_block_shape():

    block = TransformerEncoderBlock(
        embed_dim=144,
        ffn_dim=288,
        num_heads=4,
    )

    x = torch.randn(
        2,
        32,
        144,
    )

    output = block(x)

    assert output.shape == (
        2,
        32,
        144,
    )

def test_mvt_layer3_transformer():

    block = TransformerEncoderBlock(
        embed_dim=144,
        ffn_dim=288,
        num_heads=4,
        attn_dropout=0.1,
        dropout=0.1,
        ffn_dropout=0.0,
    )

    x = torch.randn(
        2,
        32,
        144,
    )

    output = block(x)

    assert output.shape == x.shape


def test_mvt_layer4_transformer():

    block = TransformerEncoderBlock(
        embed_dim=192,
        ffn_dim=384,
        num_heads=4,
        attn_dropout=0.1,
        dropout=0.1,
        ffn_dropout=0.0,
    )

    x = torch.randn(
        2,
        20,
        192,
    )

    output = block(x)

    assert output.shape == x.shape