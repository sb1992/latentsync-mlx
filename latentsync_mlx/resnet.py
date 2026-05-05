import mlx.core as mx
import mlx.nn as nn


def upsample_nearest(x, scale: int = 2):
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
    x = x.reshape(B, H * scale, W * scale, C)
    return x


class InflatedConv2d(nn.Conv2d):
    """Conv2d that handles 5D video tensors via reshape.

    Input:  (B, F, H, W, C) in NHWC
    Output: (B, F, H', W', C') in NHWC

    Reshapes to (B*F, H, W, C), runs Conv2d, reshapes back.
    """

    def __call__(self, x):
        if x.ndim == 5:
            B, F, H, W, C = x.shape
            x = x.reshape(B * F, H, W, C)
            x = super().__call__(x)
            _, H2, W2, C2 = x.shape
            x = x.reshape(B, F, H2, W2, C2)
            return x
        return super().__call__(x)


class InflatedGroupNorm(nn.GroupNorm):
    """GroupNorm that handles 5D video tensors via reshape.

    Input:  (B, F, H, W, C)
    Output: (B, F, H, W, C)
    """

    def __call__(self, x):
        if x.ndim == 5:
            B, F, H, W, C = x.shape
            x = x.reshape(B * F, H, W, C)
            x = super().__call__(x)
            x = x.reshape(B, F, H, W, C)
            return x
        return super().__call__(x)


class Upsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = InflatedConv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x):
        if x.ndim == 5:
            B, F, H, W, C = x.shape
            x = x.reshape(B * F, H, W, C)
            x = upsample_nearest(x)
            x = x.reshape(B, F, H * 2, W * 2, C)
        else:
            x = upsample_nearest(x)
        x = self.conv(x)
        return x


class Downsample3D(nn.Module):
    def __init__(self, channels: int, padding: int = 1):
        super().__init__()
        self.conv = InflatedConv2d(
            channels, channels, kernel_size=3, stride=2, padding=padding
        )

    def __call__(self, x):
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        temb_channels: int = 512,
        groups: int = 32,
        time_embedding_norm: str = "default",
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embedding_norm = time_embedding_norm

        self.norm1 = InflatedGroupNorm(groups, in_channels, pytorch_compatible=True)
        self.conv1 = InflatedConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = InflatedGroupNorm(groups, out_channels, pytorch_compatible=True)
        self.conv2 = InflatedConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            self.conv_shortcut = nn.Linear(in_channels, out_channels)

    def __call__(self, x, temb=None):
        residual = x

        x = self.norm1(x)
        x = nn.silu(x)
        x = self.conv1(x)

        if temb is not None and hasattr(self, "time_emb_proj"):
            temb = self.time_emb_proj(nn.silu(temb))
            if x.ndim == 5:
                x = x + temb[:, None, None, None, :]
            else:
                x = x + temb[:, None, None, :]

        x = self.norm2(x)
        x = nn.silu(x)
        x = self.conv2(x)

        if hasattr(self, "conv_shortcut"):
            residual = self.conv_shortcut(residual)

        return x + residual
