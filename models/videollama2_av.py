"""VideoLLaMA2.1-7B-AV probe model for OmniProact-Bench.

Paper: "VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio
Understanding in Video-LLMs" (2024).
Repo: https://github.com/DAMO-NLP-SG/VideoLLaMA2 (audio_visual branch)

Uses the official inference API:
  model, processor, tokenizer = model_init(model_path)
  video_tensor = processor['video'](video_path, va=True)  # video + audio
  output = mm_infer(video_tensor, instruction, model, tokenizer, modal='video')

Requires:
  - third_party/VideoLLaMA2/ (audio_visual branch) in the project root
  - Model weights (HF: DAMO-NLP-SG/VideoLLaMA2.1-7B-AV)
  - transformers~=4.40, soundfile, librosa, torchaudio
"""

import logging
import os
import sys
from typing import Optional

import torch

from .base import BaseModel

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VIDEOLLAMA2_ROOT = os.path.join(_PROJECT_ROOT, "third_party", "VideoLLaMA2")


def _ensure_importable():
    if _VIDEOLLAMA2_ROOT not in sys.path:
        sys.path.insert(0, _VIDEOLLAMA2_ROOT)


class VideoLLaMA2AV(BaseModel):
    """VideoLLaMA2.1-7B-AV for probe evaluation (audio+visual)."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        fps: float = 1.0,
        max_new_tokens: int = 512,
        use_audio: bool = True,
    ):
        _ensure_importable()

        from videollama2 import model_init
        from videollama2.utils import disable_torch_init

        disable_torch_init()

        logger.info(f"Loading VideoLLaMA2-AV from {model_path} ...")
        model, processor, tokenizer = model_init(model_path)
        
        self._model = model
        self._processor = processor
        self._tokenizer = tokenizer
        self._device = device if device not in (None, "auto") else "cuda:0"
        self._max_new_tokens = max_new_tokens
        self._use_audio = use_audio

        logger.info("VideoLLaMA2-AV loaded.")

    def name(self) -> str:
        suffix = "" if self._use_audio else "-NoAudio"
        return f"VideoLLaMA2.1-7B-AV{suffix}"

    @torch.no_grad()
    def generate(self, instruction: str, video_path: str, **kwargs) -> str:
        from videollama2 import mm_infer

        try:
            preprocess = self._processor["video"]
            video_tensor = preprocess(video_path, va=self._use_audio)
        except Exception as e:
            logger.warning(f"VideoLLaMA2-AV video preprocessing failed: {e}")
            return ""

        try:
            output = mm_infer(
                video_tensor,
                instruction,
                model=self._model,
                tokenizer=self._tokenizer,
                modal="video",
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                max_new_tokens=self._max_new_tokens,
            )
        except Exception as e:
            logger.warning(f"VideoLLaMA2-AV inference failed: {e}")
            return ""

        return output.strip() if output else ""


class VideoLLaMA2AV_NoAudio(VideoLLaMA2AV):
    """VideoLLaMA2.1-7B-AV without audio (video-only)."""

    def __init__(self, model_path: str, device: str = "cuda:0",
                 fps: float = 1.0, max_new_tokens: int = 512):
        super().__init__(
            model_path=model_path,
            device=device,
            fps=fps,
            max_new_tokens=max_new_tokens,
            use_audio=False,
        )
