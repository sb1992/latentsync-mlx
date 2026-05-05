import math

import mlx.core as mx
import mlx.nn as nn

from .attention import Attention, FeedForward


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 24):
        super().__init__()
        position = mx.arange(max_len).reshape(-1, 1)
        div_term = mx.exp(mx.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = mx.zeros((1, max_len, dim))
        pe[0, :, 0::2] = mx.sin(position * div_term)
        pe[0, :, 1::2] = mx.cos(position * div_term)
        self.pe = pe

    def __call__(self, x):
        return x + self.pe[:, : x.shape[1]]


class VersatileAttention(Attention):
    """Temporal self-attention that operates across the frame dimension."""

    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        temporal_position_encoding: bool = False,
        temporal_position_encoding_max_len: int = 24,
        **kwargs,
    ):
        super().__init__(
            query_dim=query_dim, heads=heads, dim_head=dim_head,
        )
        self.pos_encoder = (
            PositionalEncoding(query_dim, max_len=temporal_position_encoding_max_len)
            if temporal_position_encoding
            else None
        )

    def __call__(self, x, video_length=None, **kwargs):
        S = x.shape[1]
        # (B*F, S, C) → (B*S, F, C): attend across frames for each spatial position
        x = x.reshape(-1, video_length, S, x.shape[-1])
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(-1, video_length, x.shape[-1])

        if self.pos_encoder is not None:
            x = self.pos_encoder(x)

        x = super().__call__(x)

        # (B*S, F, C) → (B*F, S, C)
        x = x.reshape(-1, S, video_length, x.shape[-1])
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(-1, S, x.shape[-1])

        return x


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_head: int,
        temporal_position_encoding: bool = False,
        temporal_position_encoding_max_len: int = 24,
    ):
        super().__init__()
        self.attention_blocks = [
            VersatileAttention(
                query_dim=dim,
                heads=num_heads,
                dim_head=dim_head,
                temporal_position_encoding=temporal_position_encoding,
                temporal_position_encoding_max_len=temporal_position_encoding_max_len,
            )
            for _ in range(2)  # Temporal_Self x2
        ]
        self.norms = [nn.LayerNorm(dim) for _ in range(2)]

        self.ff = FeedForward(dim)
        self.ff_norm = nn.LayerNorm(dim)

    def __call__(self, x, video_length=None):
        for attn, norm in zip(self.attention_blocks, self.norms):
            x = attn(norm(x), video_length=video_length) + x
        x = self.ff(self.ff_norm(x)) + x
        return x


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        dim_head: int,
        num_layers: int = 2,
        norm_num_groups: int = 32,
        temporal_position_encoding: bool = True,
        temporal_position_encoding_max_len: int = 24,
    ):
        super().__init__()
        inner_dim = num_heads * dim_head

        self.norm = nn.GroupNorm(norm_num_groups, in_channels, pytorch_compatible=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = [
            TemporalTransformerBlock(
                dim=inner_dim,
                num_heads=num_heads,
                dim_head=dim_head,
                temporal_position_encoding=temporal_position_encoding,
                temporal_position_encoding_max_len=temporal_position_encoding_max_len,
            )
            for _ in range(num_layers)
        ]
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def __call__(self, x):
        assert x.ndim == 5
        B, F, H, W, C = x.shape
        residual = x

        x = x.reshape(B * F, H, W, C)
        x = self.norm(x)
        x = x.reshape(B * F, H * W, C)
        x = self.proj_in(x)

        for block in self.transformer_blocks:
            x = block(x, video_length=F)

        x = self.proj_out(x)
        x = x.reshape(B, F, H, W, C)

        return x + residual


class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int = 8,
        num_transformer_block: int = 2,
        temporal_position_encoding: bool = True,
        temporal_position_encoding_max_len: int = 24,
        temporal_attention_dim_div: int = 1,
    ):
        super().__init__()
        dim_head = in_channels // num_attention_heads // temporal_attention_dim_div
        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_heads=num_attention_heads,
            dim_head=dim_head,
            num_layers=num_transformer_block,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
        )

    def __call__(self, x, **kwargs):
        return self.temporal_transformer(x)
