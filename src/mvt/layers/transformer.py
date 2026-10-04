import torch
from torch import nn

class TransformerEncoderBlock(nn.Module):
    """
    Pre-norm Transformer encoder block used by MVT.

    Input: [B,N,D]
    Output: [B,N,D]
    Where:
        B = batch size
        N = number of tokens
        D = embedding dimension
    """

    def __init__(
            self,
            embed_dim: int,
            ffn_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            ffn_dropout: float = 0.0
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # -------------------------
        # Multi Head Self-Attention
        # LayerNorm
        # -> MultiheadAttention
        # -> Dropout
        # -> Residual
        # ------------------------

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.attention_dropout = nn.Dropout(dropout)

        # -----------------------------
        # Free forward Network
        # LayerNorm
        # -> Linear(D -> FFN)
        # -> ReLU
        # -> Dropout
        # -> Linear(FFN -> D)
        # -> Dropout
        # -> Residual
        # ---------------------------

        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B,N,D]

        Returns:
            Tensor of shape [B,N,D]
        """

        # ---------------------------
        # Multi-head self-attention
        # ------------------------------
        residual = x
        x = self.norm1(x)

        attention_output, _ = self.attention(x,x,x,need_weight=False)
        attention_output = self.attention_dropout(attention_output)
        x = residual + attention_output

        # -----------------------
        # Feed-forward network
        # ----------------------
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x