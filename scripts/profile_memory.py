"""Profile memory usage at each pipeline stage."""
import os
import resource

def rss_gb():
    """Current process RSS in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)

def report(label):
    import mlx.core as mx
    import torch
    mlx_active = mx.get_active_memory() / 1e9
    mlx_peak = mx.get_peak_memory() / 1e9
    try:
        mps_alloc = torch.mps.current_allocated_memory() / 1e9
        mps_driver = torch.mps.driver_allocated_memory() / 1e9
    except:
        mps_alloc = mps_driver = 0
    rss = rss_gb()
    print(f"[{label}]")
    print(f"  MLX active: {mlx_active:.2f} GB | MLX peak: {mlx_peak:.2f} GB")
    print(f"  MPS alloc:  {mps_alloc:.2f} GB | MPS driver: {mps_driver:.2f} GB")
    print(f"  Process RSS: {rss:.2f} GB")
    print()

print("=== Memory Profile ===\n")

# --- Baseline ---
report("Before imports")

import mlx.core as mx
import mlx.nn as nn
import torch
import numpy as np
report("After import torch+mlx")

mx.reset_peak_memory()
device = "mps"
dtype = mx.float16

# --- Load UNet ---
from latentsync_mlx.unet import UNet3DConditionModel
from latentsync_mlx.vae import Autoencoder
from latentsync_mlx.sampler import DDIMSampler

unet = UNet3DConditionModel()
weights = mx.load("checkpoints/latentsync_unet_mlx.safetensors")
weights = {k: v.astype(dtype) for k, v in weights.items()}
unet.load_weights(list(weights.items()))
mx.eval(unet.parameters())
del weights
report("After loading UNet (MLX)")

# --- Load VAE ---
vae = Autoencoder()
vw = mx.load("checkpoints/vae_mlx.safetensors")
vw = {k: v.astype(dtype) for k, v in vw.items()}
vae.load_weights(list(vw.items()))
mx.eval(vae.parameters())
del vw
report("After loading VAE (MLX)")

# --- Load Whisper ---
from latentsync.whisper.audio2feature import Audio2Feature
audio_encoder = Audio2Feature(
    model_path="checkpoints/whisper/tiny.pt",
    device=torch.device(device), num_frames=16, audio_feat_length=[2, 2],
)
report("After loading Whisper (PyTorch)")

# --- Load ImageProcessor ---
from latentsync.utils.image_processor import ImageProcessor, load_fixed_mask
mask_image = load_fixed_mask(256)
image_processor = ImageProcessor(resolution=256, device=device, mask_image=mask_image)
report("After loading ImageProcessor + InsightFace (PyTorch)")

# --- Stage 1: Audio encoding ---
whisper_feature = audio_encoder.audio2feat("assets/demo1_audio.wav")
whisper_chunks = audio_encoder.feature2chunks(feature_array=whisper_feature, fps=25)
report("After audio encoding")

# Free audio encoder
del audio_encoder, whisper_feature
import gc; gc.collect()
torch.mps.empty_cache()
report("After freeing audio encoder")

# --- Stage 2: Video/face processing ---
from latentsync.utils.util import read_video
video_frames = read_video("assets/demo1_video.mp4", use_decord=False)
report("After reading video frames")

from latentsync_mlx.pipeline import LipsyncPipelineMLX
import math, tqdm

# Affine transform
faces, boxes, affine_matrices = [], [], []
for frame in video_frames[:len(whisper_chunks)]:
    face, box, aff = image_processor.affine_transform(frame)
    faces.append(face)
    boxes.append(box)
    affine_matrices.append(aff)
faces = torch.stack(faces)
report("After face detection + affine transform")

# Free face detector
if hasattr(image_processor, 'face_detector'):
    del image_processor.face_detector
    image_processor.face_detector = None
gc.collect()
torch.mps.empty_cache()
report("After freeing face detector")

# --- Stage 3: One denoising chunk ---
sampler = DDIMSampler()
mx.random.seed(1247)
sampler.set_timesteps(20)
mx.reset_peak_memory()

num_frames = 16
latent_h = latent_w = 256 // 8
single_noise = mx.random.normal((1, 1, latent_h, latent_w, 4)).astype(dtype)

chunk_faces = faces[:num_frames]
ref_pv, masked_pv, masks_pt = image_processor.prepare_masks_and_masked_images(
    chunk_faces, affine_transform=False
)
ref_pv_mlx = mx.array(ref_pv.numpy().transpose(0, 2, 3, 1)).astype(dtype)
masked_pv_mlx = mx.array(masked_pv.numpy().transpose(0, 2, 3, 1)).astype(dtype)
masks_mlx = mx.array(masks_pt.numpy().transpose(0, 2, 3, 1)).astype(dtype)

# VAE encode
latents = []
for i in range(masked_pv_mlx.shape[0]):
    batch = masked_pv_mlx[i:i+1]
    mean, _ = vae.encode(batch)
    latents.append((mean - vae.shift_factor) * vae.scaling_factor)
    mx.eval(latents[-1])
masked_latents = mx.concatenate(latents, axis=0)
report("After VAE encode (1 chunk)")

# Prepare full UNet input
ref_latents_list = []
for i in range(ref_pv_mlx.shape[0]):
    batch = ref_pv_mlx[i:i+1]
    mean, _ = vae.encode(batch)
    ref_latents_list.append((mean - vae.shift_factor) * vae.scaling_factor)
    mx.eval(ref_latents_list[-1])
ref_latents = mx.concatenate(ref_latents_list, axis=0)

mask_resized = torch.nn.functional.interpolate(
    masks_pt, size=(latent_h, latent_w)
).numpy().transpose(0, 2, 3, 1)
mask_latents = mx.array(mask_resized).astype(dtype)

# Add batch dim + CFG doubling
masked_latents_in = mx.concatenate([masked_latents[None]] * 2, axis=0)
ref_latents_in = mx.concatenate([ref_latents[None]] * 2, axis=0)
mask_latents_in = mx.concatenate([mask_latents[None]] * 2, axis=0)

audio_embeds = torch.stack(whisper_chunks[:num_frames])
audio_embeds_mlx = mx.array(audio_embeds.float().numpy()).astype(dtype)[None]
null_audio = mx.zeros_like(audio_embeds_mlx)
audio_embeds_in = mx.concatenate([null_audio, audio_embeds_mlx], axis=0)

noise = mx.broadcast_to(single_noise, (1, num_frames, latent_h, latent_w, 4)) * sampler.init_noise_sigma
report("Before UNet forward pass")

# Single UNet forward pass
latent_input = mx.concatenate([noise] * 2, axis=0)
latent_input = sampler.scale_model_input(latent_input, sampler.timesteps[0])
unet_input = mx.concatenate([latent_input, mask_latents_in, masked_latents_in, ref_latents_in], axis=-1)
timestep_mx = mx.array([sampler.timesteps[0]])

noise_pred = unet(unet_input, timestep_mx, encoder_hidden_states=audio_embeds_in)
mx.eval(noise_pred)
report("After 1 UNet forward pass (PEAK)")

print("\n=== Summary ===")
print(f"MLX peak during UNet forward: {mx.get_peak_memory() / 1e9:.2f} GB")
