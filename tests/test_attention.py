import torch

from mvt.layers.attention import MultiHeadAttention


def test_multi_head_attention_shape():

    attention = MultiHeadAttention(
        embed_dim=144,
        num_heads=4,
    )

    x = torch.randn(
        2,
        32,
        144,
    )

    output = attention(x)

    assert output.shape == (
        2,
        32,
        144,
    )

def test_multi_head_cross_attention_shape():

    attention = MultiHeadAttention(
        embed_dim=144,
        num_heads=4,
    )

    query = torch.randn(
        2,
        32,
        144,
    )

    key_value = torch.randn(
        2,
        16,
        144,
    )

    output = attention(
        x_q=query,
        x_kv=key_value,
    )

    assert output.shape == (
        2,
        32,
        144,
    )