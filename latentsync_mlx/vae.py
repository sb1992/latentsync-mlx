"""SD 1.5 VAE for LatentSync. Adapted from mlx-examples/stable_diffusion/vae.py"""

import math
from typing import List

import mlx.core as mx
import mlx.nn as nn

from .resnet import upsample_nearest


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, groups: int = 32):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(groups, in_channels, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            self.conv_shortcut = nn.Linear(in_channels, out_channels)

    def __call__(self, x):
        residual = x
        x = nn.silu(self.norm1(x))
        x = self.conv1(x)
        x = nn.silu(self.norm2(x))
        x = self.conv2(x)
        if hasattr(self, "conv_shortcut"):
            residual = self.conv_shortcut(residual)
        return x + residual


class VAEAttention(nn.Module):
    def __init__(self, dims: int, norm_groups: int = 32):
        super().__init__()
        self.group_norm = nn.GroupNorm(norm_groups, dims, pytorch_compatible=True)
        self.query_proj = nn.Linear(dims, dims)
        self.key_proj = nn.Linear(dims, dims)
        self.value_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(self, x):
        B, H, W, C = x.shape
        y = self.group_norm(x)
        q = self.query_proj(y).reshape(B, H * W, C)
        k = self.key_proj(y).reshape(B, H * W, C)
        v = self.value_proj(y).reshape(B, H * W, C)

        scale = 1 / math.sqrt(C)
        scores = (q * scale) @ k.transpose(0, 2, 1)
        attn = mx.softmax(scores, axis=-1)
        y = (attn @ v).reshape(B, H, W, C)
        return x + self.out_proj(y)


class EncoderDecoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, groups=32,
                 add_downsample=False, add_upsample=False):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels, groups)
            for i in range(num_layers)
        ]
        if add_downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=0)
        if add_upsample:
            self.upsample = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)

    def __call__(self, x):
        for r in self.resnets:
            x = r(x)
        if hasattr(self, "downsample"):
            x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
            x = self.downsample(x)
        if hasattr(self, "upsample"):
            x = self.upsample(upsample_nearest(x))
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, block_out_channels=(128, 256, 512, 512),
                 layers_per_block=3, groups=32):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], 3, stride=1, padding=1)

        self.mid_blocks = [
            ResnetBlock2D(block_out_channels[-1], block_out_channels[-1], groups),
            VAEAttention(block_out_channels[-1], groups),
            ResnetBlock2D(block_out_channels[-1], block_out_channels[-1], groups),
        ]

        channels = list(reversed(block_out_channels))
        channels = [channels[0]] + channels
        self.up_blocks = [
            EncoderDecoderBlock2D(
                ic, oc, num_layers=layers_per_block, groups=groups,
                add_upsample=i < len(block_out_channels) - 1,
            )
            for i, (ic, oc) in enumerate(zip(channels, channels[1:]))
        ]

        self.conv_norm_out = nn.GroupNorm(groups, block_out_channels[0], pytorch_compatible=True)
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def __call__(self, x):
        x = self.conv_in(x)
        for mb in self.mid_blocks:
            x = mb(x)
        for ub in self.up_blocks:
            x = ub(x)
        x = nn.silu(self.conv_norm_out(x))
        return self.conv_out(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=8, block_out_channels=(128, 256, 512, 512),
                 layers_per_block=2, groups=32):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, stride=1, padding=1)

        channels = [block_out_channels[0]] + list(block_out_channels)
        self.down_blocks = [
            EncoderDecoderBlock2D(
                ic, oc, num_layers=layers_per_block, groups=groups,
                add_downsample=i < len(block_out_channels) - 1,
            )
            for i, (ic, oc) in enumerate(zip(channels, channels[1:]))
        ]

        self.mid_blocks = [
            ResnetBlock2D(block_out_channels[-1], block_out_channels[-1], groups),
            VAEAttention(block_out_channels[-1], groups),
            ResnetBlock2D(block_out_channels[-1], block_out_channels[-1], groups),
        ]

        self.conv_norm_out = nn.GroupNorm(groups, block_out_channels[-1], pytorch_compatible=True)
        self.conv_out = nn.Conv2d(block_out_channels[-1], out_channels, 3, padding=1)

    def __call__(self, x):
        x = self.conv_in(x)
        for db in self.down_blocks:
            x = db(x)
        for mb in self.mid_blocks:
            x = mb(x)
        x = nn.silu(self.conv_norm_out(x))
        return self.conv_out(x)


class Autoencoder(nn.Module):
    def __init__(self, scaling_factor: float = 0.18215, shift_factor: float = 0.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_proj = nn.Linear(8, 8)
        self.post_quant_proj = nn.Linear(4, 4)

    def encode(self, x):
        x = self.encoder(x)
        x = self.quant_proj(x)
        mean, logvar = mx.split(x, 2, axis=-1)
        return mean, logvar

    def decode(self, z):
        z = self.post_quant_proj(z)
        return self.decoder(z)

    def encode_for_lipsync(self, x):
        """Encode and return scaled latent (no sampling)."""
        mean, _ = self.encode(x)
        return (mean - self.shift_factor) * self.scaling_factor

    def decode_for_lipsync(self, z):
        """Decode from scaled latent."""
        z = z / self.scaling_factor + self.shift_factor
        return self.decode(z)
