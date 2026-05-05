"""LatentSync MLX inference script.

Usage:
  # v1.5 (256px):
  PYTHONPATH=. python scripts/inference_mlx.py \
    --video_path assets/demo1_video.mp4 \
    --audio_path assets/demo1_audio.wav \
    --video_out_path /tmp/latentsync-test/mlx_v15_demo1.mp4

  # v1.6 (512px):
  PYTHONPATH=. python scripts/inference_mlx.py \
    --resolution 512 \
    --video_path assets/demo1_video.mp4 \
    --audio_path assets/demo1_audio.wav \
    --video_out_path /tmp/latentsync-test/mlx_v16_demo1.mp4
"""

import argparse
import torch
import mlx.core as mx
import mlx.nn as nn

from latentsync.whisper.audio2feature import Audio2Feature
from latentsync.utils.image_processor import ImageProcessor, load_fixed_mask
from latentsync_mlx.unet import UNet3DConditionModel
from latentsync_mlx.vae import Autoencoder
from latentsync_mlx.sampler import DDIMSampler
from latentsync_mlx.pipeline import LipsyncPipelineMLX


def main():
    parser = argparse.ArgumentParser(description="LatentSync MLX inference")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=256, choices=[256, 512])
    parser.add_argument("--unet_weights", type=str, default="checkpoints/latentsync_unet_mlx.safetensors")
    parser.add_argument("--vae_weights", type=str, default="checkpoints/vae_mlx.safetensors")
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--num_frames", type=int, default=16, help="Frames per chunk (16 or 8)")
    parser.add_argument("--temp_dir", type=str, default="temp_mlx")
    parser.add_argument("--no-float16", action="store_true", help="Disable float16 (use float32)")
    parser.add_argument("--cache-gb", type=float, default=None,
                        help="MLX cache limit in GB (auto-detected from system RAM if omitted)")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = mx.float32 if args.no_float16 else mx.float16
    print(f"Resolution: {args.resolution}px | Device (preprocessing): {device} | Precision: {dtype}")

    # --- Load MLX UNet ---
    print("Loading MLX UNet...")
    unet = UNet3DConditionModel()
    weights = mx.load(args.unet_weights)
    weights = {k: v.astype(dtype) for k, v in weights.items()}
    unet.load_weights(list(weights.items()))
    mx.eval(unet.parameters())

    # --- Load MLX VAE ---
    print("Loading MLX VAE...")
    vae = Autoencoder()
    vae_weights = mx.load(args.vae_weights)
    vae_weights = {k: v.astype(dtype) for k, v in vae_weights.items()}
    vae.load_weights(list(vae_weights.items()))
    mx.eval(vae.parameters())

    # --- Load Whisper (PyTorch) ---
    print("Loading Whisper audio encoder...")
    audio_encoder = Audio2Feature(
        model_path="checkpoints/whisper/tiny.pt",
        device=torch.device(device),
        num_frames=16,
        audio_feat_length=[2, 2],
    )

    # --- Load ImageProcessor (PyTorch) ---
    print(f"Loading ImageProcessor at {args.resolution}px...")
    mask_image = load_fixed_mask(args.resolution)
    image_processor = ImageProcessor(
        resolution=args.resolution,
        device=device,
        mask_image=mask_image,
    )

    # --- Build pipeline ---
    sampler = DDIMSampler()
    pipeline = LipsyncPipelineMLX(
        unet=unet,
        vae=vae,
        sampler=sampler,
        audio_encoder=audio_encoder,
        image_processor=image_processor,
        resolution=args.resolution,
        dtype=dtype,
        cache_limit_gb=args.cache_gb,
    )

    # --- Run ---
    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
        num_frames=args.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        temp_dir=args.temp_dir,
    )


if __name__ == "__main__":
    main()
