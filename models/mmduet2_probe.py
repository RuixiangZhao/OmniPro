"""MMDuet2 probe model for OmniProact-Bench.

Paper: "MMDuet2: Enhancing Proactive Interaction of Video MLLMs with
Multi-Turn Reinforcement Learning" (ICLR 2026).

Architecture: Qwen2.5-VL-3B + MMDuet2 LoRA. For probe mode, we use the
base Qwen2.5-VL's single-turn video QA (same as Phase 2 of the online
adapter). Vision-only, no audio.

Requires:
  - transformers == 4.49.x (MMDuet2 pinned, via shim)
  - third_party/MMDuet2/proactive_eval/ (custom Qwen2.5-VL code)
  - flash-attn
  - Model weights: wangyueqian/MMDuet2 (from HF cache)
"""

import logging
import os
import sys
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import BaseModel

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MMDUET2_EVAL_ROOT = os.path.join(
    _PROJECT_ROOT, "third_party", "MMDuet2", "proactive_eval"
)


def _import_mmduet2():
    if _MMDUET2_EVAL_ROOT not in sys.path:
        sys.path.insert(0, _MMDUET2_EVAL_ROOT)
    from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    return Qwen2_5_VLForConditionalGeneration, AutoProcessor, process_vision_info


class MMDuet2Probe(BaseModel):
    """MMDuet2 for probe evaluation (vision-only, single-turn Qwen2.5-VL)."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        fps: float = 1.0,
        max_new_tokens: int = 512,
        num_frames: int = 16,
    ):
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HOME", "/path/to/huggingface_cache")

        (Qwen2_5_VLForConditionalGeneration, AutoProcessor,
         process_vision_info) = _import_mmduet2()

        if device in (None, "auto"):
            device = "cuda:0"

        logger.info(f"Loading MMDuet2 (probe) from {model_path} ...")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            local_files_only=True,
        ).eval().to(device)
        self._processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True,
        )
        self._process_vision_info = process_vision_info
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._num_frames = num_frames

        logger.info("MMDuet2 (probe) loaded.")

    def name(self) -> str:
        return "MMDuet2"

    @torch.no_grad()
    def generate(self, instruction: str, video_path: str, **kwargs) -> str:
        """Single-turn video QA using Qwen2.5-VL processor + generate."""
        # Build messages with video
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": [
                {"type": "video", "video": video_path,
                 "max_pixels": 360 * 420, "fps": 1.0},
            ]},
        ]

        try:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            img_inputs, vid_inputs = self._process_vision_info(messages)
            inputs = self._processor(
                text=[text],
                images=img_inputs if img_inputs else None,
                videos=vid_inputs if vid_inputs else None,
                padding=True,
                return_tensors="pt",
            ).to(self._device)

            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
            )
            output_ids = output[:, inputs.input_ids.size(1):]
            response = self._processor.batch_decode(
                output_ids, skip_special_tokens=True,
            )[0].strip()

            if response.upper().startswith("NO REPLY") or response.upper().startswith("NO"):
                # Model decided not to answer — return empty
                return ""
            return response

        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return ""
