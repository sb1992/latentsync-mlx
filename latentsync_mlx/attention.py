import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class Attention(nn.Module):
    """Multi-head attention with optional cross-attention."""

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int = None,
        heads: int = 8,
        dim_head: int = 64,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim or query_dim

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def __call__(self, x, encoder_hidden_states=None):
        B, S, _ = x.shape
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else x

        q = self.to_q(x)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        q = q.reshape(B, S, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        Sk = k.shape[1]
        k = k.reshape(B, Sk, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, Sk, self.heads, self.dim_head).transpose(0, 2, 1, 3)

        scores = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, S, -1)

        return self.to_out(out)


class FeedForward(nn.Module):
    """GeGLU feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, dim)

    def __call__(self, x):
        return self.linear3(self.linear1(x) * nn.gelu(self.linear2(x)))


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_head: int,
        cross_attention_dim: int = None,
        add_audio_layer: bool = False,
    ):
        super().__init__()
        self.add_audio_layer = add_audio_layer

        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(
            query_dim=dim, heads=num_heads, dim_head=dim_head,
        )

        if add_audio_layer:
            self.norm2 = nn.LayerNorm(dim)
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_heads,
                dim_head=dim_head,
            )

        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def __call__(self, x, encoder_hidden_states=None, video_length=None):
        x = self.attn1(self.norm1(x)) + x

        if self.add_audio_layer and encoder_hidden_states is not None:
            enc = encoder_hidden_states
            if enc.ndim == 4:
                B_orig = enc.shape[0]
                enc = enc.reshape(B_orig * enc.shape[1], enc.shape[2], enc.shape[3])
            x = self.attn2(self.norm2(x), encoder_hidden_states=enc) + x

        x = self.ff(self.norm3(x)) + x
        return x


class Transformer3DModel(nn.Module):
    """Transformer for 5D video tensors (B, F, H, W, C)."""

    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        dim_head: int,
        num_layers: int = 1,
        cross_attention_dim: int = None,
        norm_num_groups: int = 32,
        add_audio_layer: bool = False,
    ):
        super().__init__()
        inner_dim = num_heads * dim_head

        self.norm = nn.GroupNorm(norm_num_groups, in_channels, pytorch_compatible=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = [
            BasicTransformerBlock(
                dim=inner_dim,
                num_heads=num_heads,
                dim_head=dim_head,
                cross_attention_dim=cross_attention_dim,
                add_audio_layer=add_audio_layer,
            )
            for _ in range(num_layers)
        ]
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def __call__(self, x, encoder_hidden_states=None):
        assert x.ndim == 5, f"Expected 5D input, got {x.ndim}D"
        B, F, H, W, C = x.shape
        residual = x

        x = x.reshape(B * F, H, W, C)
        x = self.norm(x)
        x = x.reshape(B * F, H * W, C)
        x = self.proj_in(x)

        for block in self.transformer_blocks:
            x = block(x, encoder_hidden_states=encoder_hidden_states, video_length=F)

        x = self.proj_out(x)
        x = x.reshape(B, F, H, W, C)

        return x + residual
