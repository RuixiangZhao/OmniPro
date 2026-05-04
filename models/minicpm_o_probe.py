"""MiniCPM-o 4.5 probe model for OmniProact-Bench.

Uses the standard non-streaming ``model.chat()`` interface for offline
probe evaluation. This file is completely independent of the online/duplex
adapter in ``minicpm_o.py`` — both files can coexist without interference.

Video handling:
  - ``minicpmo.utils.get_video_frame_audio_segments`` extracts 1-fps PIL
    frames + 16 kHz audio segments from a video file.
  - For probe mode we feed all frames (+ optionally audio) into a single
    ``model.chat()`` call.

Requires:
  - transformers == 4.51.0  (MiniCPM-o pinned requirement)
  - minicpmo-utils >= 1.0.2 (provides minicpmo.utils)
  - flash-attn, decord, librosa, moviepy, Pillow
  - Model weights: openbmb/MiniCPM-o-4.5 (~19 GB bf16)
"""

import logging
import subprocess
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import BaseModel

logger = logging.getLogger(__name__)


def _patch_whisper_attention():
    """Same patch as minicpm_o.py for WhisperAttention compatibility."""
    try:
        from transformers.models.whisper.modeling_whisper import WhisperAttention
        import inspect
        sig = inspect.signature(WhisperAttention.forward)
        if "past_key_values" not in sig.parameters:
            _orig_forward = WhisperAttention.forward

            def _patched_forward(self, *args, past_key_values=None, **kwargs):
                if "past_key_value" not in kwargs:
                    kwargs["past_key_value"] = past_key_values
                result = _orig_forward(self, *args, **kwargs)
                if isinstance(result, tuple) and len(result) > 2:
                    return result[:2]
                return result

            WhisperAttention.forward = _patched_forward
            logger.info("Patched WhisperAttention.forward for MiniCPM-o probe")
    except Exception as e:
        logger.warning(f"WhisperAttention patch failed: {e}")


def _extract_audio_np(video_path: str, sr: int = 16000) -> Optional[np.ndarray]:
    """Extract full audio track as float32 numpy array via ffmpeg."""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(sr), "-ac", "1",
            "-f", "s16le", "-loglevel", "error", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        return audio if len(audio) > 0 else None
    except Exception as e:
        logger.warning(f"Audio extraction failed: {e}")
        return None


class MiniCPMOProbe(BaseModel):
    """MiniCPM-o 4.5 for probe evaluation (audio + visual)."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        fps: float = 1.0,
        max_new_tokens: int = 512,
        use_audio: bool = True,
    ):
        from transformers import AutoModel, AutoTokenizer

        if device in (None, "auto"):
            device = "cuda:0"

        _patch_whisper_attention()

        logger.info(f"Loading MiniCPM-o (probe mode) from {model_path} ...")

        self._model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )
        self._model.eval().to(device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._use_audio = use_audio

        logger.info("MiniCPM-o (probe mode) loaded.")

    def name(self) -> str:
        suffix = "" if self._use_audio else "-NoAudio"
        return f"MiniCPM-o-4.5{suffix}"

    @torch.no_grad()
    def generate(self, instruction: str, video_path: str, **kwargs) -> str:
        """Single-turn video QA via model.chat()."""
        try:
            from minicpmo.utils import get_video_frame_audio_segments
        except ImportError:
            logger.error("minicpmo.utils not found. Install: pip install minicpmo-utils")
            return ""

        # Extract frames (PIL) and audio segments (numpy) at 1 fps
        try:
            video_frames, audio_segments, _ = get_video_frame_audio_segments(
                video_path, stack_frames=1, use_ffmpeg=True, adjust_audio_length=True,
            )
        except Exception as e:
            logger.warning(f"Video loading failed for {video_path}: {e}")
            return ""

        # Build message content: frames + optional audio + text question
        content = []

        if self._use_audio and audio_segments:
            # Interleave frames and audio (official omni format)
            for i in range(len(video_frames)):
                content.append(video_frames[i])
                if i < len(audio_segments):
                    content.append(audio_segments[i])
        else:
            # Vision-only: just frames
            content.extend(video_frames)

        content.append(instruction)

        msgs = [{"role": "user", "content": content}]

        # Use omni_mode only when audio is included
        use_omni = self._use_audio and bool(audio_segments)

        try:
            response = self._model.chat(
                msgs=msgs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                use_image_id=False,
                max_slice_nums=1,
                use_tts_template=False,
                enable_thinking=False,
                generate_audio=False,
                omni_mode=use_omni,
                tokenizer=self._tokenizer,
            )
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return ""

        if isinstance(response, tuple):
            response = response[0]
        return response.strip() if isinstance(response, str) else str(response).strip()


class MiniCPMOProbe_NoAudio(MiniCPMOProbe):
    """MiniCPM-o 4.5 probe without audio (vision-only)."""

    def __init__(self, model_path: str, device: str = "cuda:0",
                 fps: float = 1.0, max_new_tokens: int = 512):
        super().__init__(
            model_path=model_path,
            device=device,
            fps=fps,
            max_new_tokens=max_new_tokens,
            use_audio=False,
        )
