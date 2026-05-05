# LatentSync MLX

An [Apple MLX](https://github.com/ml-explore/mlx) port of [ByteDance's LatentSync](https://github.com/bytedance/LatentSync) — audio-conditioned latent diffusion for lip sync, running natively on Apple Silicon.

This port runs the UNet denoising loop and VAE encode/decode on MLX (the hot path), while keeping preprocessing (Whisper audio encoding, InsightFace face detection) in PyTorch. On an M-series Mac, MLX inference is **~2.3x faster** than PyTorch MPS.

## Architecture

LatentSync is built on Stable Diffusion 1.5 with modifications for lip sync:

- **13-channel UNet input**: 4 noise + 1 mask + 4 masked frame + 4 reference frame
- **Audio cross-attention** (dim=384): Whisper tiny embeddings condition the denoising
- **Temporal motion modules**: self-attention across frames with sinusoidal positional encoding
- **Fake 3D convolutions**: InflatedConv3d reshapes `(B,F,H,W,C)` → `(B*F,H,W,C)`, runs Conv2d, reshapes back
- **DDIM sampling**: 20 steps, guidance_scale 1.5, scaled_linear beta schedule

### MLX-specific adaptations

| PyTorch | MLX |
|---------|-----|
| Conv2d weights `(O, I, kH, kW)` | Transposed to `(O, kH, kW, I)` |
| 1x1 conv as linear `(O, I, 1, 1)` | Squeezed to `(O, I)` |
| GeGLU `ff.net.0.proj` (2x hidden) | Split into `linear1` + `linear2` |
| `to_out.0` (Sequential wrapper) | Unwrapped to `to_out` |
| NCHW data layout | NHWC (MLX native) |

## Setup

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- The original [LatentSync](https://github.com/bytedance/LatentSync) repo (for preprocessing dependencies)

### Installation

```bash
# Clone this repo
git clone https://github.com/user/latentsync-mlx.git
cd latentsync-mlx

# Install dependencies
pip install -r requirements.txt

# Clone upstream LatentSync for preprocessing
git clone https://github.com/bytedance/LatentSync.git upstream
pip install -e upstream
```

### Download and convert weights

```bash
# Download the LatentSync v1.5 checkpoint
huggingface-cli download ByteDance/LatentSync-1.5 --local-dir checkpoints

# Or v1.6 (same architecture, trained at 512px)
# huggingface-cli download ByteDance/LatentSync-1.6 --local-dir checkpoints

# Convert UNet weights to MLX format
python scripts/convert_weights.py \
  --unet-ckpt checkpoints/latentsync_unet.pt \
  --unet-output checkpoints/latentsync_unet_mlx.safetensors

# Convert VAE weights
python scripts/convert_weights.py --convert-vae \
  --vae-output checkpoints/vae_mlx.safetensors
```

## Usage

```bash
# v1.5 (256px)
PYTHONPATH=. python scripts/inference.py \
  --video_path assets/demo1_video.mp4 \
  --audio_path assets/demo1_audio.wav \
  --video_out_path output.mp4

# v1.6 (512px) — same weights, higher resolution
PYTHONPATH=. python scripts/inference.py \
  --resolution 512 \
  --video_path assets/demo1_video.mp4 \
  --audio_path assets/demo1_audio.wav \
  --video_out_path output_512.mp4
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--resolution` | 256 | Face crop resolution (256 for v1.5, 512 for v1.6) |
| `--inference_steps` | 20 | DDIM denoising steps |
| `--guidance_scale` | 1.5 | Classifier-free guidance scale |
| `--seed` | 1247 | Random seed |
| `--num_frames` | 16 | Frames per denoising chunk (lower = less memory) |
| `--cache-gb` | auto | MLX buffer cache limit (auto-detected from system RAM) |
| `--no-float16` | off | Use float32 instead of float16 |
| `--unet_weights` | `checkpoints/latentsync_unet_mlx.safetensors` | MLX UNet weights |
| `--vae_weights` | `checkpoints/vae_mlx.safetensors` | MLX VAE weights |

## Validation

MLX output validated against PyTorch MPS baseline across all 3 upstream demo samples:

| Demo | Frames | PSNR (mean) | PSNR (range) |
|------|--------|-------------|--------------|
| demo1 | 242 | 39.1 dB | 37.6 – 39.6 |
| demo2 | 498 | 40.6 dB | 36.7 – 41.2 |
| demo3 | 466 | 40.4 dB | 39.6 – 41.0 |

PSNR > 37 dB across all frames — differences are floating-point precision, not algorithmic.

### Performance (Apple Silicon)

| Backend | Time per 16-frame chunk | Relative |
|---------|------------------------|----------|
| PyTorch MPS | ~32s | 1.0x |
| MLX | ~14s | **2.3x faster** |

### Memory optimization

The pipeline auto-detects system RAM and configures the MLX buffer cache to balance speed vs. memory usage. Override with `--cache-gb`:

```bash
# Force 2 GB cache (conservative, for loaded 16 GB machines)
PYTHONPATH=. python scripts/inference.py --cache-gb 2 ...

# Force 0 cache (minimum memory footprint, slower)
PYTHONPATH=. python scripts/inference.py --cache-gb 0 ...
```

| System RAM | Auto Cache | Peak Memory (v1.5, 256px) | Speed |
|---|---|---|---|
| 8 GB | 0 GB | ~10.5 GB | ~21s/chunk |
| 16 GB | 1.6 GB | ~15 GB | ~19s/chunk |
| 24 GB | 4.8 GB | ~18 GB | ~18s/chunk |
| 32+ GB | 8 GB | ~22 GB | ~14s/chunk |

The pipeline also:
- Frees the Whisper encoder and face detector after preprocessing
- Clears the MLX buffer cache between denoising chunks
- Frees the UNet and VAE after inference before face restoration
- Slices VAE encode/decode to reduce per-frame peak memory

## File structure

```
latentsync_mlx/
  __init__.py
  unet.py          # UNet3DConditionModel (13-ch input, audio cross-attn)
  vae.py           # SD 1.5 VAE (Encoder + Decoder)
  attention.py     # Attention, FeedForward (GeGLU), Transformer3DModel
  motion_module.py # Temporal self-attention with positional encoding
  resnet.py        # InflatedConv2d, ResnetBlock3D, Up/Downsample3D
  sampler.py       # DDIM scheduler
  pipeline.py      # Hybrid inference pipeline
  convert_weights.py
scripts/
  inference.py     # CLI entry point
  convert_weights.py
```

## Known limitations

- **v1.6 mask artifacts**: The upstream binary mask (designed for 256px) produces visible boundary artifacts at 512px. This is an upstream LatentSync issue, not specific to the MLX port.
- **Preprocessing in PyTorch**: Whisper and InsightFace remain in PyTorch. These are not in the hot path and add minimal overhead.

## Credits

This is a port of [LatentSync](https://github.com/bytedance/LatentSync) by ByteDance to Apple MLX.

**Original paper**: [LatentSync: Audio Conditioned Latent Diffusion Models for Lip Sync](https://arxiv.org/abs/2412.09262)

```bibtex
@article{li2024latentsync,
  title={LatentSync: Audio Conditioned Latent Diffusion Models for Lip Sync},
  author={Li, Chunyu and Zhang, Chao and Nie, Weikai and Liang, Weilin and Xia, Jiawei and Yang, Liang and Zhu, Yi and Zuo, Zhendong},
  journal={arXiv preprint arXiv:2412.09262},
  year={2024}
}
```

## License

Apache License 2.0 — same as the original LatentSync.
