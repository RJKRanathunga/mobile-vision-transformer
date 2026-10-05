import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention / cross-attention used by MVT.
    Input:
        x_q:
            [B, S, D]

        x_kv:
            [B, T, D]
            If None, self-attention is performed.

    Output:
        [B, S, D]
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible "
                f"by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Official MVT implementation uses:
        #
        # Linear(D -> 3D)
        #
        # which contains Q, K and V projections.
        self.qkv_proj = nn.Linear(
            embed_dim,
            3 * embed_dim,
            bias=bias,
        )

        self.attn_dropout = nn.Dropout(attn_dropout)

        # Final projection after concatenating attention heads.
        self.out_proj = nn.Linear(
            embed_dim,
            embed_dim,
            bias=bias,
        )

        # 1 / sqrt(head_dim)
        self.scaling = self.head_dim ** -0.5

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # =========================================================
        # Input
        #
        # x_q:
        #   [B, S, D]
        #
        # x_kv:
        #   [B, T, D]
        #
        # For self-attention:
        #   x_kv = None
        # =========================================================

        batch_size, source_length, _ = x_q.shape

        # =========================================================
        # Self-attention
        # =========================================================

        if x_kv is None:

            # [B, S, D] -> [B, S, 3D]
            qkv = self.qkv_proj(x_q)

            # [B, S, 3D] -> [B, S, 3, H, head_dim]
            qkv = qkv.reshape(
                batch_size,
                source_length,
                3,
                self.num_heads,
                self.head_dim,
            )

            # [B, S, 3, H, C] -> [B, H, 3, S, C]
            qkv = qkv.transpose(1, 3).contiguous()

            # Each:
            # [B, H, S, head_dim]
            query = qkv[:, :, 0]
            key = qkv[:, :, 1]
            value = qkv[:, :, 2]

            target_length = source_length

        # =========================================================
        # Cross-attention
        #
        # Not currently needed by Siam-MoViT, but this matches
        # the interface of the official implementation.
        # =========================================================

        else:

            target_length = x_kv.shape[1]

            # -----------------------------------------------------
            # Q
            # -----------------------------------------------------

            query = F.linear(
                x_q,
                weight=self.qkv_proj.weight[
                    :self.embed_dim
                ],
                bias=(
                    self.qkv_proj.bias[:self.embed_dim]
                    if self.qkv_proj.bias is not None
                    else None
                ),
            )

            # [B, S, D] -> [B, S, H, head_dim] -> [B, H, S, head_dim]

            query = query.reshape(
                batch_size,
                source_length,
                self.num_heads,
                self.head_dim,
            )

            query = query.transpose(1, 2).contiguous()

            # -----------------------------------------------------
            # K and V
            # -----------------------------------------------------

            kv = F.linear(
                x_kv,
                weight=self.qkv_proj.weight[
                    self.embed_dim:
                ],
                bias=(
                    self.qkv_proj.bias[self.embed_dim:]
                    if self.qkv_proj.bias is not None
                    else None
                ),
            )

            # [B, T, 2D] -> [B, T, 2, H, head_dim]
            kv = kv.reshape(
                batch_size,
                target_length,
                2,
                self.num_heads,
                self.head_dim,
            )

            # [B, T, 2, H, C] -> [B, H, 2, T, C]

            kv = kv.transpose(1, 3).contiguous()

            key = kv[:, :, 0]
            value = kv[:, :, 1]

        # =========================================================
        # Scale queries
        # =========================================================

        query = query * self.scaling

        # =========================================================
        # Q K^T
        #
        # query:
        #   [B, H, S, C]
        #
        # key:
        #   [B, H, T, C]
        #
        # key.T:
        #   [B, H, C, T]
        #
        # result:
        #   [B, H, S, T]
        # =========================================================

        key = key.transpose(-1, -2)

        attention = torch.matmul(
            query,
            key,
        )

        # =========================================================
        # Attention mask
        # =========================================================

        if attn_mask is not None:

            expected_shape = (
                batch_size,
                source_length,
                target_length,
            )

            if tuple(attn_mask.shape) != expected_shape:
                raise ValueError(
                    "attn_mask must have shape "
                    f"{expected_shape}, "
                    f"got {tuple(attn_mask.shape)}"
                )

            # [B, S, T] -> [B, 1, S, T]
            attention = attention + attn_mask.unsqueeze(1)

        # =========================================================
        # Key padding mask
        # =========================================================

        if key_padding_mask is not None:

            expected_shape = (
                batch_size,
                target_length,
            )

            if tuple(key_padding_mask.shape) != expected_shape:
                raise ValueError(
                    "key_padding_mask must have shape "
                    f"{expected_shape}, "
                    f"got {tuple(key_padding_mask.shape)}"
                )

            attention = attention.masked_fill(
                key_padding_mask
                .unsqueeze(1)
                .unsqueeze(2)
                .bool(),
                float("-inf"),
            )

        # =========================================================
        # Softmax
        #
        # Official implementation performs softmax in float32
        # before converting back to the original dtype.
        # =========================================================

        attention_dtype = attention.dtype

        attention = self.softmax(
            attention.float()
        ).to(attention_dtype)

        # =========================================================
        # Attention dropout
        # =========================================================

        attention = self.attn_dropout(attention)

        # =========================================================
        # Weighted sum
        #
        # [B, H, S, T] x [B, H, T, C] = [B, H, S, C]
        # =========================================================

        output = torch.matmul(
            attention,
            value,
        )

        # =========================================================
        # Concatenate heads
        #
        # [B, H, S, C] -> [B, S, H, C] -> [B, S, D]
        # =========================================================

        output = output.transpose(1, 2).reshape(
            batch_size,
            source_length,
            self.embed_dim,
        )

        # =========================================================
        # Output projection
        # =========================================================

        output = self.out_proj(output)

        return output