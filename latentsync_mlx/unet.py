import math
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .resnet import InflatedConv2d, InflatedGroupNorm, ResnetBlock3D, Upsample3D, Downsample3D
from .attention import Transformer3DModel
from .motion_module import VanillaTemporalModule


class CrossAttnDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 2,
        num_heads: int = 8,
        cross_attention_dim: int = 384,
        norm_num_groups: int = 32,
        add_downsample: bool = True,
        use_motion_module: bool = False,
        motion_module_kwargs: dict = None,
        add_audio_layer: bool = False,
    ):
        super().__init__()
        self.has_cross_attention = True

        self.resnets = []
        self.attentions = []
        self.motion_modules = []

        for i in range(num_layers):
            ic = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock3D(
                    in_channels=ic,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=norm_num_groups,
                )
            )
            dim_head = out_channels // num_heads
            self.attentions.append(
                Transformer3DModel(
                    in_channels=out_channels,
                    num_heads=num_heads,
                    dim_head=dim_head,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=norm_num_groups,
                    add_audio_layer=add_audio_layer,
                )
            )
            if use_motion_module:
                mkw = motion_module_kwargs or {}
                self.motion_modules.append(VanillaTemporalModule(in_channels=out_channels, **mkw))

        if add_downsample:
            self.downsample = Downsample3D(out_channels)

    def __call__(self, x, temb=None, encoder_hidden_states=None):
        output_states = []
        for i in range(len(self.resnets)):
            x = self.resnets[i](x, temb)
            x = self.attentions[i](x, encoder_hidden_states=encoder_hidden_states)
            if len(self.motion_modules) > i:
                x = self.motion_modules[i](x)
            output_states.append(x)

        if hasattr(self, "downsample"):
            x = self.downsample(x)
            output_states.append(x)

        return x, output_states


class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 2,
        norm_num_groups: int = 32,
        add_downsample: bool = True,
        use_motion_module: bool = False,
        motion_module_kwargs: dict = None,
    ):
        super().__init__()
        self.has_cross_attention = False

        self.resnets = []
        self.motion_modules = []

        for i in range(num_layers):
            ic = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock3D(
                    in_channels=ic,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=norm_num_groups,
                )
            )
            if use_motion_module:
                mkw = motion_module_kwargs or {}
                self.motion_modules.append(VanillaTemporalModule(in_channels=out_channels, **mkw))

        if add_downsample:
            self.downsample = Downsample3D(out_channels)

    def __call__(self, x, temb=None, encoder_hidden_states=None):
        output_states = []
        for i in range(len(self.resnets)):
            x = self.resnets[i](x, temb)
            if len(self.motion_modules) > i:
                x = self.motion_modules[i](x)
            output_states.append(x)

        if hasattr(self, "downsample"):
            x = self.downsample(x)
            output_states.append(x)

        return x, output_states


class CrossAttnUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_out_channels: int,
        temb_channels: int,
        num_layers: int = 2,
        num_heads: int = 8,
        cross_attention_dim: int = 384,
        norm_num_groups: int = 32,
        add_upsample: bool = True,
        use_motion_module: bool = False,
        motion_module_kwargs: dict = None,
        add_audio_layer: bool = False,
    ):
        super().__init__()
        self.has_cross_attention = True

        self.resnets = []
        self.attentions = []
        self.motion_modules = []

        res_channels = [out_channels] * (num_layers - 1) + [in_channels]
        in_channels_list = [prev_out_channels] + [out_channels] * (num_layers - 1)

        for i in range(num_layers):
            ic = in_channels_list[i] + res_channels[i]
            self.resnets.append(
                ResnetBlock3D(
                    in_channels=ic,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=norm_num_groups,
                )
            )
            dim_head = out_channels // num_heads
            self.attentions.append(
                Transformer3DModel(
                    in_channels=out_channels,
                    num_heads=num_heads,
                    dim_head=dim_head,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=norm_num_groups,
                    add_audio_layer=add_audio_layer,
                )
            )
            if use_motion_module:
                mkw = motion_module_kwargs or {}
                self.motion_modules.append(VanillaTemporalModule(in_channels=out_channels, **mkw))

        if add_upsample:
            self.upsample = Upsample3D(out_channels)

    def __call__(self, x, temb=None, encoder_hidden_states=None, res_hidden_states=None):
        for i in range(len(self.resnets)):
            res = res_hidden_states.pop()
            x = mx.concatenate([x, res], axis=-1)
            x = self.resnets[i](x, temb)
            x = self.attentions[i](x, encoder_hidden_states=encoder_hidden_states)
            if len(self.motion_modules) > i:
                x = self.motion_modules[i](x)

        if hasattr(self, "upsample"):
            x = self.upsample(x)

        return x


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_out_channels: int,
        temb_channels: int,
        num_layers: int = 2,
        norm_num_groups: int = 32,
        add_upsample: bool = True,
        use_motion_module: bool = False,
        motion_module_kwargs: dict = None,
    ):
        super().__init__()
        self.has_cross_attention = False

        self.resnets = []
        self.motion_modules = []

        res_channels = [out_channels] * (num_layers - 1) + [in_channels]
        in_channels_list = [prev_out_channels] + [out_channels] * (num_layers - 1)

        for i in range(num_layers):
            ic = in_channels_list[i] + res_channels[i]
            self.resnets.append(
                ResnetBlock3D(
                    in_channels=ic,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=norm_num_groups,
                )
            )
            if use_motion_module:
                mkw = motion_module_kwargs or {}
                self.motion_modules.append(VanillaTemporalModule(in_channels=out_channels, **mkw))

        if add_upsample:
            self.upsample = Upsample3D(out_channels)

    def __call__(self, x, temb=None, encoder_hidden_states=None, res_hidden_states=None):
        for i in range(len(self.resnets)):
            res = res_hidden_states.pop()
            x = mx.concatenate([x, res], axis=-1)
            x = self.resnets[i](x, temb)
            if len(self.motion_modules) > i:
                x = self.motion_modules[i](x)

        if hasattr(self, "upsample"):
            x = self.upsample(x)

        return x


class UNetMidBlock3DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        num_heads: int = 8,
        cross_attention_dim: int = 384,
        norm_num_groups: int = 32,
        use_motion_module: bool = False,
        motion_module_kwargs: dict = None,
        add_audio_layer: bool = False,
    ):
        super().__init__()
        dim_head = in_channels // num_heads

        self.resnet_0 = ResnetBlock3D(
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            groups=norm_num_groups,
        )
        self.attention = Transformer3DModel(
            in_channels=in_channels,
            num_heads=num_heads,
            dim_head=dim_head,
            cross_attention_dim=cross_attention_dim,
            norm_num_groups=norm_num_groups,
            add_audio_layer=add_audio_layer,
        )
        self.resnet_1 = ResnetBlock3D(
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            groups=norm_num_groups,
        )

        if use_motion_module:
            mkw = motion_module_kwargs or {}
            self.motion_module = VanillaTemporalModule(in_channels=in_channels, **mkw)

    def __call__(self, x, temb=None, encoder_hidden_states=None):
        x = self.resnet_0(x, temb)
        x = self.attention(x, encoder_hidden_states=encoder_hidden_states)
        if hasattr(self, "motion_module"):
            x = self.motion_module(x)
        x = self.resnet_1(x, temb)
        return x


class UNet3DConditionModel(nn.Module):
    """LatentSync UNet: SD 1.5 UNet extended with 13-channel input and audio cross-attention."""

    def __init__(
        self,
        in_channels: int = 13,
        out_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        cross_attention_dim: int = 384,
        attention_head_dim: int = 8,
        norm_num_groups: int = 32,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock3D", "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D", "DownBlock3D",
        ),
        up_block_types: Tuple[str, ...] = (
            "UpBlock3D", "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D", "CrossAttnUpBlock3D",
        ),
        use_motion_module: bool = True,
        motion_module_resolutions: Tuple[int, ...] = (1, 2, 4, 8),
        motion_module_kwargs: dict = None,
        add_audio_layer: bool = True,
    ):
        super().__init__()
        self.add_audio_layer = add_audio_layer
        time_embed_dim = block_out_channels[0] * 4

        self.conv_in = InflatedConv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.timesteps = nn.SinusoidalPositionalEncoding(
            block_out_channels[0],
            max_freq=1,
            min_freq=math.exp(
                -math.log(10000) + 2 * math.log(10000) / block_out_channels[0]
            ),
            scale=1.0,
            cos_first=True,
            full_turns=False,
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(block_out_channels[0], time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        mkw = motion_module_kwargs or {
            "num_attention_heads": 8,
            "num_transformer_block": 1,
            "temporal_position_encoding": True,
            "temporal_position_encoding_max_len": 24,
            "temporal_attention_dim_div": 1,
        }

        self.down_blocks = []
        output_channel = block_out_channels[0]
        for i, down_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final = i == len(block_out_channels) - 1
            res = 2 ** i
            use_mm = use_motion_module and (res in motion_module_resolutions)

            if "CrossAttn" in down_type:
                self.down_blocks.append(CrossAttnDownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    num_layers=layers_per_block,
                    num_heads=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=norm_num_groups,
                    add_downsample=not is_final,
                    use_motion_module=use_mm,
                    motion_module_kwargs=mkw,
                    add_audio_layer=add_audio_layer,
                ))
            else:
                self.down_blocks.append(DownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    num_layers=layers_per_block,
                    norm_num_groups=norm_num_groups,
                    add_downsample=not is_final,
                    use_motion_module=use_mm,
                    motion_module_kwargs=mkw,
                ))

        self.mid_block = UNetMidBlock3DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            num_heads=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            norm_num_groups=norm_num_groups,
            use_motion_module=use_motion_module and False,  # mid_block motion disabled in config
            motion_module_kwargs=mkw,
            add_audio_layer=add_audio_layer,
        )

        reversed_channels = list(reversed(block_out_channels))
        self.up_blocks = []
        output_channel = reversed_channels[0]
        for i, up_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_channels[i]
            input_channel = reversed_channels[min(i + 1, len(reversed_channels) - 1)]
            is_final = i == len(block_out_channels) - 1
            res = 2 ** (3 - i)
            use_mm = use_motion_module and (res in motion_module_resolutions)

            if "CrossAttn" in up_type:
                self.up_blocks.append(CrossAttnUpBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_out_channels=prev_output_channel,
                    temb_channels=time_embed_dim,
                    num_layers=layers_per_block + 1,
                    num_heads=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=norm_num_groups,
                    add_upsample=not is_final,
                    use_motion_module=use_mm,
                    motion_module_kwargs=mkw,
                    add_audio_layer=add_audio_layer,
                ))
            else:
                self.up_blocks.append(UpBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_out_channels=prev_output_channel,
                    temb_channels=time_embed_dim,
                    num_layers=layers_per_block + 1,
                    norm_num_groups=norm_num_groups,
                    add_upsample=not is_final,
                    use_motion_module=use_mm,
                    motion_module_kwargs=mkw,
                ))

        self.conv_norm_out = InflatedGroupNorm(norm_num_groups, block_out_channels[0], pytorch_compatible=True)
        self.conv_out = InflatedConv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def __call__(self, x, timestep, encoder_hidden_states=None):
        # x: (B, F, H, W, 13) in NHWC
        # timestep: scalar or (B,)
        # encoder_hidden_states: (B, F, S, 384) or (2*B, F, S, 384) for CFG

        temb = self.timesteps(timestep).astype(x.dtype)
        temb = self.time_embedding(temb)

        x = self.conv_in(x)

        down_block_res_samples = [x]
        for block in self.down_blocks:
            x, res_samples = block(x, temb=temb, encoder_hidden_states=encoder_hidden_states)
            down_block_res_samples.extend(res_samples)

        x = self.mid_block(x, temb=temb, encoder_hidden_states=encoder_hidden_states)

        for block in self.up_blocks:
            x = block(x, temb=temb, encoder_hidden_states=encoder_hidden_states,
                      res_hidden_states=down_block_res_samples)

        x = self.conv_norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)

        return x
