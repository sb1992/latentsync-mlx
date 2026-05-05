"""LatentSync MLX inference pipeline.

Face detection and Whisper audio encoding stay in PyTorch/ONNX (not hot path).
The UNet denoising loop and VAE decode run in MLX (hot path).
"""

import gc
import os
import math
import shutil
import subprocess

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import torch
import cv2
import tqdm
from einops import rearrange

from .unet import UNet3DConditionModel
from .vae import Autoencoder
from .sampler import DDIMSampler


def write_video_frames(path: str, frames: np.ndarray, fps: int = 25):
    h, w = frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


class LipsyncPipelineMLX:
    """Hybrid pipeline: PyTorch for preprocessing, MLX for inference."""

    def __init__(
        self,
        unet: UNet3DConditionModel,
        vae: Autoencoder,
        sampler: DDIMSampler,
        audio_encoder,
        image_processor,
        resolution: int = 256,
        dtype=None,
    ):
        self.unet = unet
        self.vae = vae
        self.sampler = sampler
        self.audio_encoder = audio_encoder
        self.image_processor = image_processor
        self.resolution = resolution
        self.vae_scale_factor = 8
        self.dtype = dtype or mx.float32

    def _affine_transform_video(self, video_frames):
        faces, boxes, affine_matrices = [], [], []
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)
        return torch.stack(faces), boxes, affine_matrices

    def _loop_video(self, whisper_chunks, video_frames):
        if len(whisper_chunks) > len(video_frames):
            faces, boxes, affine_matrices = self._affine_transform_video(video_frames)
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_vf, loop_faces, loop_boxes, loop_aff = [], [], [], []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_vf.append(video_frames)
                    loop_faces.append(faces)
                    loop_boxes += boxes
                    loop_aff += affine_matrices
                else:
                    loop_vf.append(video_frames[::-1])
                    loop_faces.append(faces.flip(0))
                    loop_boxes += boxes[::-1]
                    loop_aff += affine_matrices[::-1]
            video_frames = np.concatenate(loop_vf)[:len(whisper_chunks)]
            faces = torch.cat(loop_faces)[:len(whisper_chunks)]
            boxes = loop_boxes[:len(whisper_chunks)]
            affine_matrices = loop_aff[:len(whisper_chunks)]
        else:
            video_frames = video_frames[:len(whisper_chunks)]
            faces, boxes, affine_matrices = self._affine_transform_video(video_frames)
        return video_frames, faces, boxes, affine_matrices

    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        num_frames: int = 16,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        seed: int = 1247,
        temp_dir: str = "temp_mlx",
    ):
        import soundfile as sf
        from latentsync.utils.util import read_audio, read_video

        print("=== LatentSync MLX Pipeline ===")

        # --- Stage 1: Audio encoding (PyTorch) ---
        print("Encoding audio with Whisper...")
        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(
            feature_array=whisper_feature, fps=video_fps
        )
        audio_samples = read_audio(audio_path)

        del self.audio_encoder
        self.audio_encoder = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("Freed audio encoder.")

        # --- Stage 2: Video/face processing (PyTorch/ONNX) ---
        print("Processing video frames...")
        video_frames = read_video(video_path, use_decord=False)
        video_frames, faces, boxes, affine_matrices = self._loop_video(
            whisper_chunks, video_frames
        )

        if hasattr(self.image_processor, 'face_detector'):
            del self.image_processor.face_detector
            self.image_processor.face_detector = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("Freed face detector.")

        # --- Stage 3: MLX denoising ---
        print(f"Running MLX denoising ({num_inference_steps} steps, {len(whisper_chunks)} frames)...")

        mx.random.seed(seed)
        self.sampler.set_timesteps(num_inference_steps)

        do_cfg = guidance_scale > 1.0
        num_inferences = math.ceil(len(whisper_chunks) / num_frames)
        latent_h = self.resolution // self.vae_scale_factor
        latent_w = self.resolution // self.vae_scale_factor

        # Same noise for all frames, matching PyTorch: randn(1,4,1,h,w).repeat(1,1,F,1,1)
        single_noise = mx.random.normal((1, 1, latent_h, latent_w, 4)).astype(self.dtype)
        noise_shape = (1, len(whisper_chunks), latent_h, latent_w, 4)
        all_latents = mx.broadcast_to(single_noise, noise_shape) * self.sampler.init_noise_sigma

        synced_frames_list = []

        for chunk_idx in tqdm.tqdm(range(num_inferences), desc="MLX inference"):
            start = chunk_idx * num_frames
            end = min(start + num_frames, len(whisper_chunks))
            chunk_faces = faces[start:end]  # (F, C, H, W) torch

            # Use PyTorch ImageProcessor for mask/normalize (matches MPS baseline exactly)
            ref_pv, masked_pv, masks_pt = self.image_processor.prepare_masks_and_masked_images(
                chunk_faces, affine_transform=False
            )
            # ref_pv: (F, C, H, W) [-1,1], masked_pv: (F, C, H, W) [-1,1], masks_pt: (F, 1, H, W) [0,1]

            # Convert to MLX NHWC
            ref_pv_mlx = mx.array(ref_pv.numpy().transpose(0, 2, 3, 1)).astype(self.dtype)
            masked_pv_mlx = mx.array(masked_pv.numpy().transpose(0, 2, 3, 1)).astype(self.dtype)
            masks_mlx = mx.array(masks_pt.numpy().transpose(0, 2, 3, 1)).astype(self.dtype)

            # VAE encode
            masked_latents = self._vae_encode(masked_pv_mlx)  # (F, h, w, 4)
            ref_latents = self._vae_encode(ref_pv_mlx)

            # Resize mask to latent space
            F_count = masked_latents.shape[0]
            masks_np = masks_pt.numpy()  # (F, 1, H, W)
            mask_resized = torch.nn.functional.interpolate(
                masks_pt, size=(latent_h, latent_w)
            ).numpy().transpose(0, 2, 3, 1)  # (F, h, w, 1)
            mask_latents = mx.array(mask_resized).astype(self.dtype)

            # Add batch dim: (F,...) → (1,F,...)
            masked_latents = masked_latents[None]
            ref_latents = ref_latents[None]
            mask_latents = mask_latents[None]

            # CFG doubling
            if do_cfg:
                masked_latents = mx.concatenate([masked_latents] * 2, axis=0)
                ref_latents = mx.concatenate([ref_latents] * 2, axis=0)
                mask_latents = mx.concatenate([mask_latents] * 2, axis=0)

            # Audio embeddings
            audio_embeds = torch.stack(whisper_chunks[start:end])
            audio_embeds_mlx = mx.array(audio_embeds.float().numpy()).astype(self.dtype)
            audio_embeds_mlx = audio_embeds_mlx[None]  # (1, F, S, 384)
            if do_cfg:
                null_audio = mx.zeros_like(audio_embeds_mlx)
                audio_embeds_mlx = mx.concatenate([null_audio, audio_embeds_mlx], axis=0)

            latents = all_latents[:, start:end]

            # Denoising loop
            for t in self.sampler.timesteps:
                latent_input = mx.concatenate([latents] * 2, axis=0) if do_cfg else latents
                latent_input = self.sampler.scale_model_input(latent_input, t)

                # 13 channels: noise(4) + mask(1) + masked(4) + ref(4)
                unet_input = mx.concatenate(
                    [latent_input, mask_latents, masked_latents, ref_latents], axis=-1
                )

                timestep_mx = mx.array([t])
                noise_pred = self.unet(
                    unet_input, timestep_mx, encoder_hidden_states=audio_embeds_mlx
                )

                if do_cfg:
                    pred_uncond, pred_audio = mx.split(noise_pred, 2, axis=0)
                    noise_pred = pred_uncond + guidance_scale * (pred_audio - pred_uncond)

                latents = self.sampler.step(noise_pred, t, latents)
                mx.eval(latents)

            # VAE decode
            decoded = self._vae_decode(latents.reshape(-1, latent_h, latent_w, 4))

            # Paste surrounding pixels back: generated in mouth, original elsewhere
            # masks_mlx has 1 where we keep original, 0 where we regenerate (mouth)
            inv_mask = 1.0 - masks_mlx  # 1 in mouth region
            combined = decoded[:F_count] * inv_mask + ref_pv_mlx * masks_mlx
            synced_frames_list.append(combined)

        del self.unet
        self.unet = None
        gc.collect()
        mx.clear_cache()
        print("Freed UNet and cleared MLX cache.")

        # --- Stage 4: Restore faces to original frames ---
        print("Restoring faces to video...")
        all_synced = mx.concatenate(synced_frames_list, axis=0)  # (total_F, H, W, C) in [-1,1]

        # Convert to PyTorch (C, H, W) for restore_img
        all_synced_np = np.array(all_synced.astype(mx.float32))  # NHWC, ensure float32 for torch
        all_synced_torch = torch.from_numpy(all_synced_np).permute(0, 3, 1, 2)  # NCHW

        del self.vae
        self.vae = None
        gc.collect()
        mx.clear_cache()
        print("Freed VAE and cleared MLX cache.")

        restored = self._restore_video(
            all_synced_torch, video_frames, boxes, affine_matrices
        )

        # --- Stage 5: Write output ---
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        video_temp = os.path.join(temp_dir, "video.mp4")
        audio_temp = os.path.join(temp_dir, "audio.wav")

        write_video_frames(video_temp, restored, fps=video_fps)

        audio_remain_len = int(restored.shape[0] / video_fps * audio_sample_rate)
        audio_np = audio_samples[:audio_remain_len].cpu().numpy()
        sf.write(audio_temp, audio_np, audio_sample_rate)

        cmd = (
            f"ffmpeg -y -loglevel error -nostdin "
            f"-i {video_temp} -i {audio_temp} "
            f"-c:v libx264 -crf 18 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        )
        subprocess.run(cmd, shell=True)
        print(f"Output saved to: {video_out_path}")

    def _vae_encode(self, images, batch_size=1):
        """Encode NHWC images to latents, sliced to reduce peak memory."""
        latents = []
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i + batch_size]
            mean, _ = self.vae.encode(batch)
            latents.append((mean - self.vae.shift_factor) * self.vae.scaling_factor)
            mx.eval(latents[-1])
        return mx.concatenate(latents, axis=0)

    def _vae_decode(self, latents, batch_size=1):
        """Decode latents to NHWC images, sliced to reduce peak memory."""
        scaled = latents / self.vae.scaling_factor + self.vae.shift_factor
        decoded = []
        for i in range(0, scaled.shape[0], batch_size):
            batch = scaled[i:i + batch_size]
            decoded.append(self.vae.decode(batch))
            mx.eval(decoded[-1])
        return mx.concatenate(decoded, axis=0)

    def _restore_video(self, faces_torch, video_frames, boxes, affine_matrices):
        """Restore synced faces back into original video frames."""
        import torchvision.transforms as transforms

        device = self.image_processor.restorer.device
        video_frames = video_frames[:len(faces_torch)]
        out_frames = []
        print(f"Restoring {len(faces_torch)} faces...")
        for idx, face in enumerate(tqdm.tqdm(faces_torch)):
            x1, y1, x2, y2 = boxes[idx]
            h, w = int(y2 - y1), int(x2 - x1)
            face = transforms.functional.resize(
                face, size=(h, w),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
            aff = affine_matrices[idx]
            frame = self.image_processor.restorer.restore_img(
                video_frames[idx], face.to(device), aff
            )
            out_frames.append(frame)
        return np.stack(out_frames)
