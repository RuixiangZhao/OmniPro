"""Phi-4-multimodal-instruct probe model for OmniProact-Bench.

Paper: "Phi-4-Mini Technical Report: Compact yet Powerful Multimodal
Language Models" (Microsoft, 2025).

Uses transformers native support (>=4.48.2):
  processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
  inputs = processor(text=prompt, images=frames, audios=[(audio, sr)], return_tensors='pt')
  outputs = model.generate(**inputs)

Video is fed as multi-frame images; audio extracted via ffmpeg.

Requires:
  - Model weights (HF: microsoft/Phi-4-multimodal-instruct)
  - flash-attn, soundfile, scipy, pillow
  - No transformers version shim needed (native support in 4.48+)
"""

import logging
import os
import subprocess
import tempfile
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import BaseModel

logger = logging.getLogger(__name__)


def _extract_frames(video_path: str, num_frames: int = 8):
    """Extract uniformly sampled frames from video using decord."""
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()  # (N, H, W, C)
    return [Image.fromarray(f) for f in frames]


def _extract_audio(video_path: str, sr: int = 16000, max_seconds: int = 30):
    """Extract audio from video via ffmpeg, return (ndarray, sample_rate)."""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(sr), "-ac", "1",
            "-f", "s16le", "-loglevel", "error",
            "-"
        ]
        result = subprocess.run(
            cmd, capture_output=True, check=True, timeout=60,
        )
        audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        # Truncate to max_seconds
        max_samples = max_seconds * sr
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        return audio, sr
    except Exception as e:
        logger.warning(f"Audio extraction failed: {e}")
        return None, sr


class Phi4Multimodal(BaseModel):
    """Phi-4-multimodal-instruct for probe evaluation (audio+visual)."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        fps: float = 1.0,
        max_new_tokens: int = 512,
        use_audio: bool = True,
        num_frames: int = 8,
    ):
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

        logger.info(f"Loading Phi-4-multimodal from {model_path} ...")

        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device if device not in (None, "auto") else "cuda:0",
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation="flash_attention_2",
        )
        self._generation_config = GenerationConfig.from_pretrained(model_path)
        self._device = device if device not in (None, "auto") else "cuda:0"
        self._max_new_tokens = max_new_tokens
        self._use_audio = use_audio
        self._num_frames = num_frames

        logger.info("Phi-4-multimodal loaded.")

    def name(self) -> str:
        suffix = "" if self._use_audio else "-NoAudio"
        return f"Phi-4-multimodal{suffix}"

    @torch.no_grad()
    def generate(self, instruction: str, video_path: str, **kwargs) -> str:
        # Extract video frames
        try:
            frames = _extract_frames(video_path, self._num_frames)
        except Exception as e:
            logger.warning(f"Frame extraction failed: {e}")
            return ""

        # Build image placeholders
        image_tags = "".join(
            f"<|image_{i+1}|>" for i in range(len(frames))
        )

        # Extract audio if enabled
        audio_data = None
        audio_tag = ""
        if self._use_audio:
            audio, sr = _extract_audio(video_path)
            if audio is not None and len(audio) > 0:
                audio_data = [(audio, sr)]
                audio_tag = "<|audio_1|>"

        # Build prompt
        prompt = f"<|user|>{image_tags}{audio_tag}{instruction}<|end|><|assistant|>"

        # Process inputs
        proc_kwargs = {
            "text": prompt,
            "images": frames,
            "return_tensors": "pt",
        }
        if audio_data:
            proc_kwargs["audios"] = audio_data

        inputs = self._processor(**proc_kwargs).to(self._device)

        # Generate
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            generation_config=self._generation_config,
        )

        # Decode only generated tokens
        output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        response = self._processor.batch_decode(
            output_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response.strip()


class Phi4Multimodal_NoAudio(Phi4Multimodal):
    """Phi-4-multimodal without audio (video-only)."""

    def __init__(self, model_path: str, device: str = "cuda:0",
                 fps: float = 1.0, max_new_tokens: int = 512):
        super().__init__(
            model_path=model_path,
            device=device,
            fps=fps,
            max_new_tokens=max_new_tokens,
            use_audio=False,
        )
