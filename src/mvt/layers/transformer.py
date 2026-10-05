import torch
from torch import nn

from mvt.layers.attention import MultiHeadAttention


class TransformerEncoderBlock(nn.Module):
    """
    Pre-norm Transformer encoder block used by MVT.

    Input:
        [B, N, D]

    Output:
        [B, N, D]
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible "
                f"by num_heads ({num_heads})"
            )

        # =========================================================
        # Multi-head attention
        # =========================================================

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            bias=True,
        )

        self.attention_dropout = nn.Dropout(
            dropout
        )

        # =========================================================
        # Feed-forward network
        # =========================================================

        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(
                embed_dim,
                ffn_dim,
                bias=True,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                ffn_dropout
            ),

            nn.Linear(
                ffn_dim,
                embed_dim,
                bias=True,
            ),

            nn.Dropout(
                dropout
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_prev: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # =========================================================
        # Multi-head attention
        # =========================================================

        residual = x

        x = self.norm1(x)

        x = self.attention(
            x_q=x,
            x_kv=x_prev,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )

        x = self.attention_dropout(x)

        x = x + residual

        # =========================================================
        # Feed-forward network
        # =========================================================

        x = x + self.ffn(
            self.norm2(x)
        )

        return x
