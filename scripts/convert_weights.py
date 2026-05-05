"""Convert LatentSync PyTorch checkpoint to MLX format.

Key transformations:
- Conv2d weights: PyTorch (O, I, kH, kW) → MLX (O, kH, kW, I)
- 1x1 conv weights squeezed to linear
- GeGLU feed-forward split into linear1/linear2
- Key name remapping to match MLX module structure
"""

import argparse
import torch
import mlx.core as mx
import numpy as np
from pathlib import Path


def convert_conv_weight(w):
    """Convert PyTorch Conv2d weight (O, I, kH, kW) → MLX (O, kH, kW, I)."""
    if w.ndim == 4:
        return w.permute(0, 2, 3, 1).contiguous().numpy()
    return w.numpy()


def map_key(key: str) -> str:
    """Map PyTorch LatentSync weight key to MLX module key."""

    # time_embedding
    key = key.replace("time_embedding.linear_1", "time_embedding.layers.0")
    key = key.replace("time_embedding.linear_2", "time_embedding.layers.2")

    # conv_in / conv_out with InflatedConv2d (inherits Conv2d)
    # No key change needed for conv_in/conv_out

    # down_blocks / up_blocks
    # PyTorch: down_blocks.0.resnets.0 → MLX: down_blocks.0.resnets.0
    # PyTorch: down_blocks.0.attentions.0 → MLX: down_blocks.0.attentions.0
    # These match, no remapping needed

    # Downsamplers/upsamplers
    key = key.replace("downsamplers.0.conv", "downsample.conv")
    key = key.replace("upsamplers.0.conv", "upsample.conv")

    # Mid block
    key = key.replace("mid_block.resnets.0", "mid_block.resnet_0")
    key = key.replace("mid_block.resnets.1", "mid_block.resnet_1")
    key = key.replace("mid_block.attentions.0", "mid_block.attention")

    # Attention layers: to_out.0 → to_out (unwrap Sequential wrapper)
    key = key.replace(".to_out.0.", ".to_out.")

    # Feed-forward: ff.net.0.proj → split into linear1/linear2; ff.net.2 → linear3
    key = key.replace("ff.net.2", "ff.linear3")
    # ff.net.0.proj needs special handling (GeGLU split) — done in convert_weight()

    # Transformer blocks: proj_in/proj_out
    # These are already named correctly

    # Motion modules
    # temporal_transformer → temporal_transformer (matches)

    # GroupNorm: conv_norm_out → conv_norm_out (matches)

    return key


def convert_unet_weights(ckpt_path: str, output_path: str):
    """Convert LatentSync UNet checkpoint to MLX safetensors."""
    print(f"Loading PyTorch checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    mlx_weights = {}

    for key, value in state_dict.items():
        value = value.float()
        mapped_key = map_key(key)

        # Handle GeGLU split: ff.net.0.proj.weight/bias → linear1 + linear2
        if "ff.net.0.proj" in mapped_key:
            k1 = mapped_key.replace("ff.net.0.proj", "ff.linear1")
            k2 = mapped_key.replace("ff.net.0.proj", "ff.linear2")
            v1, v2 = value.chunk(2, dim=0) if "weight" in key else value.chunk(2, dim=0)
            mlx_weights[k1] = mx.array(v1.numpy())
            mlx_weights[k2] = mx.array(v2.numpy())
            continue

        # Handle conv weights
        if value.ndim == 4 and value.shape[2] == 1 and value.shape[3] == 1:
            # 1x1 conv → squeeze to linear (O, I)
            np_val = value.squeeze(-1).squeeze(-1).numpy()
        elif value.ndim == 4:
            # Regular conv → transpose to NHWC: (O, I, kH, kW) → (O, kH, kW, I)
            np_val = convert_conv_weight(value)
        else:
            np_val = value.numpy()

        mlx_weights[mapped_key] = mx.array(np_val)

    print(f"Converted {len(mlx_weights)} weight tensors")
    print(f"Saving to: {output_path}")
    mx.save_safetensors(output_path, mlx_weights)
    print("Done!")

    return mlx_weights


def convert_vae_weights(vae_repo: str = "stabilityai/sd-vae-ft-mse", output_path: str = None):
    """Convert SD 1.5 VAE weights to MLX format."""
    from diffusers import AutoencoderKL

    print(f"Loading VAE from: {vae_repo}")
    vae = AutoencoderKL.from_pretrained(vae_repo)
    state_dict = vae.state_dict()

    mlx_weights = {}
    for key, value in state_dict.items():
        value = value.float()
        mapped_key = key

        # Map structure
        mapped_key = mapped_key.replace("mid_block.resnets.0", "mid_blocks.0")
        mapped_key = mapped_key.replace("mid_block.attentions.0", "mid_blocks.1")
        mapped_key = mapped_key.replace("mid_block.resnets.1", "mid_blocks.2")
        mapped_key = mapped_key.replace("downsamplers.0.conv", "downsample")
        mapped_key = mapped_key.replace("upsamplers.0.conv", "upsample")
        mapped_key = mapped_key.replace("to_q", "query_proj")
        mapped_key = mapped_key.replace("to_k", "key_proj")
        mapped_key = mapped_key.replace("to_v", "value_proj")
        mapped_key = mapped_key.replace("to_out.0", "out_proj")
        mapped_key = mapped_key.replace("quant_conv", "quant_proj")
        mapped_key = mapped_key.replace("post_quant_conv", "post_quant_proj")

        if value.ndim == 4 and value.shape[2] == 1 and value.shape[3] == 1:
            np_val = value.squeeze(-1).squeeze(-1).numpy()
        elif value.ndim == 4:
            np_val = convert_conv_weight(value)
        else:
            np_val = value.numpy()

        mlx_weights[mapped_key] = mx.array(np_val)

    if output_path:
        print(f"Saving VAE to: {output_path}")
        mx.save_safetensors(output_path, mlx_weights)

    return mlx_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LatentSync weights to MLX")
    parser.add_argument("--unet-ckpt", type=str, help="Path to latentsync_unet.pt")
    parser.add_argument("--unet-output", type=str, default="checkpoints/latentsync_unet_mlx.safetensors")
    parser.add_argument("--vae-output", type=str, default="checkpoints/vae_mlx.safetensors")
    parser.add_argument("--convert-vae", action="store_true")
    args = parser.parse_args()

    if args.unet_ckpt:
        convert_unet_weights(args.unet_ckpt, args.unet_output)

    if args.convert_vae:
        convert_vae_weights(output_path=args.vae_output)
