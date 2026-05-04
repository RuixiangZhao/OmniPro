"""MMDuet2 streaming adapter for online evaluation.

Paper: "MMDuet2: Enhancing Proactive Interaction of Video MLLMs with
Multi-Turn Reinforcement Learning" (ICLR 2026).
Repo: https://github.com/yellow-binary-tree/MMDuet2

Two-phase inference strategy:
  Phase 1 (timing): Use the official NO REPLY system prompt with KV cache
      streaming. The model decides when to speak. Fast (~2 tokens for
      "NO REPLY" on silent frames). This leverages MMDuet2's RL-trained
      proactive timing ability.
  Phase 2 (content): When Phase 1 emits (non-NO-REPLY), do a separate
      single-turn call with the session prompt (task instruction) + the
      current frame. This uses the Qwen2.5-VL base model's instruction-
      following ability to produce correctly formatted output.

This decouples timing (MMDuet2's strength) from content formatting
(base model's strength), avoiding the conflict between the NO REPLY
mechanism and structured output instructions.

Key optimizations:
  - Cached image_processor outputs (O(1) per frame)
  - flash_attention_2 (avoids O(N²) causal mask at 40K+ tokens)
  - Phase 2 is single-turn so it's fast (~0.5s)

Requires:
  - third_party/MMDuet2/ in the project root
  - Model weights at model_path (HF: wangyueqian/MMDuet2)
  - PYTHONPATH includes /path/to/mmduet2_pkgs (transformers 4.49 shim)
  - flash-attn >= 2.3
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from .streaming_base import StreamingModel

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MMDUET2_EVAL_ROOT = os.path.join(
    _PROJECT_ROOT, "third_party", "MMDuet2", "proactive_eval"
)

# Official system prompt — keeps the NO REPLY mechanism for timing decisions
_SYSTEM_PROMPT = (
    "You are a helpful assistant. Your task is to answer questions based on "
    "continuously incoming video frames. Your responses should include "
    "information from the video since your last reply (if any). If the "
    "information in this segment of the video cannot answer the question, "
    'output "NO REPLY".'
)


def _import_mmduet2():
    if _MMDUET2_EVAL_ROOT not in sys.path:
        sys.path.insert(0, _MMDUET2_EVAL_ROOT)
    from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    return Qwen2_5_VLForConditionalGeneration, AutoProcessor, process_vision_info


class MMDuet2Streaming(StreamingModel):
    """MMDuet2 two-phase: timing via NO REPLY + content via session prompt."""

    accepts_audio: bool = False
    native_fps: float = 1.0

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        attn_implementation: str = "flash_attention_2",
        max_new_tokens: int = 128,
        max_context_tokens: int = 100000,
    ):
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

        (Qwen2_5_VLForConditionalGeneration, AutoProcessor,
         process_vision_info) = _import_mmduet2()

        logger.info(f"Loading MMDuet2 from {model_path} ...")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation, local_files_only=True,
        ).eval().to(device)
        self._processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True,
        )
        self._process_vision_info = process_vision_info
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._max_context_tokens = max_context_tokens

        # Token constants for manual placeholder expansion
        self._image_token = self._processor.image_token
        self._merge_length = self._processor.image_processor.merge_size ** 2

        # Per-sample state (Phase 1 streaming)
        self._history: List[Dict] = []
        self._past_key_values = None
        self._cached_pixel_values: Optional[torch.Tensor] = None
        self._cached_image_grid_thw: Optional[torch.Tensor] = None
        self._pending_text: Optional[str] = None
        self._instruction: str = ""
        self._response_history: List[Dict] = []
        self._token_count: int = 0

        logger.info("MMDuet2 loaded.")

    def name(self) -> str:
        return "MMDuet2"

    def begin(self, instruction: str, task_meta: Dict[str, Any]) -> None:
        self._reset()
        self._instruction = instruction
        self._pending_text = instruction
        self._history = [{"role": "system", "content": _SYSTEM_PROMPT}]

    def end(self) -> None:
        self._reset()

    @torch.no_grad()
    def observe(
        self,
        frame,
        t_sec: float,
        history: List[Dict[str, Any]],
        audio_chunk: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        if frame is None:
            return None

        if isinstance(frame, np.ndarray):
            image = Image.fromarray(frame)
        elif isinstance(frame, Image.Image):
            image = frame
        else:
            return None

        # ── Phase 1: timing decision (streaming with KV cache) ──
        should_speak, phase1_reply = self._phase1_timing(image)

        if not should_speak:
            return None

        # ── Phase 2: content generation (single-turn, session prompt) ──
        reply_text = self._phase2_content(image)

        self._response_history.append({"t_sec": t_sec, "raw": reply_text})
        logger.info(f"MMDuet2 emit at t={t_sec:.1f}s: {reply_text!r:.200s}")
        return reply_text

    def _phase1_timing(self, image: Image.Image) -> tuple:
        """Phase 1: use NO REPLY streaming to decide if model should speak.

        Maintains KV cache across frames for efficiency. Returns
        (should_speak: bool, raw_reply: str).
        """
        if self._token_count > self._max_context_tokens:
            self._segment_reset()

        # Build user turn
        content: list = [{"type": "image", "image": image}]
        if self._pending_text:
            content.append({"type": "text", "text": self._pending_text})
            self._pending_text = None
        query = {"role": "user", "content": content}
        self._history.append(query)

        # Incremental image processing
        new_img_inputs, _ = self._process_vision_info([query])
        if new_img_inputs:
            img_result = self._processor.image_processor(
                images=new_img_inputs, videos=None,
            )
            new_pv = img_result["pixel_values"]
            new_grid = img_result["image_grid_thw"]
            if not isinstance(new_pv, torch.Tensor):
                new_pv = torch.tensor(new_pv)
            if not isinstance(new_grid, torch.Tensor):
                new_grid = torch.tensor(new_grid)
            if self._cached_pixel_values is None:
                self._cached_pixel_values = new_pv
                self._cached_image_grid_thw = new_grid
            else:
                self._cached_pixel_values = torch.cat(
                    [self._cached_pixel_values, new_pv], dim=0
                )
                self._cached_image_grid_thw = torch.cat(
                    [self._cached_image_grid_thw, new_grid], dim=0
                )

        # Build and tokenize
        text = self._processor.apply_chat_template(
            self._history, tokenize=False, add_generation_prompt=True,
        )
        if self._cached_image_grid_thw is not None:
            idx = 0
            while self._image_token in text:
                n_pads = (
                    self._cached_image_grid_thw[idx].prod().item()
                    // self._merge_length
                )
                text = text.replace(
                    self._image_token, "<|placeholder|>" * n_pads, 1,
                )
                idx += 1
            text = text.replace("<|placeholder|>", self._image_token)

        text_inputs = self._processor.tokenizer(
            [text], return_tensors="pt", padding=True,
        )
        input_ids = text_inputs.input_ids.to(self._device)
        attention_mask = text_inputs.attention_mask.to(self._device)
        self._token_count = input_ids.size(1)

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self._cached_pixel_values is not None:
            inputs["pixel_values"] = self._cached_pixel_values.to(self._device)
            inputs["image_grid_thw"] = self._cached_image_grid_thw.to(self._device)

        # Apply keep masks
        if (hasattr(self._model, 'model')
                and hasattr(self._model.model, 'all_keep_masks')
                and self._model.model.all_keep_masks):
            keep_mask = torch.ones_like(input_ids, dtype=torch.bool)
            old_keep_mask = torch.cat(
                self._model.model.all_keep_masks, dim=1
            )
            keep_mask[:, :old_keep_mask.size(1)] = old_keep_mask
            inputs["input_ids"] = input_ids[keep_mask].unsqueeze(0)
            inputs["attention_mask"] = attention_mask[keep_mask].unsqueeze(0)

        # Generate
        model_output = self._model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            past_key_values=self._past_key_values,
            return_dict_in_generate=True,
            drop_method="none",
            drop_threshold=1.0,
            drop_absolute=True,
        )
        self._past_key_values = model_output.past_key_values

        output_ids = model_output.sequences[:, inputs["input_ids"].size(1):]
        reply_text = self._processor.batch_decode(
            output_ids, skip_special_tokens=True,
        )[0].strip()

        # Record in history for KV cache continuity
        self._history.append({"role": "assistant", "content": reply_text})

        should_speak = not reply_text.upper().startswith("NO")
        return should_speak, reply_text

    def _phase2_content(self, image: Image.Image) -> str:
        """Phase 2: single-turn call with session prompt for formatted output.

        Uses the task instruction as system prompt so the model follows
        the output format. No KV cache — fresh single-turn inference.
        Saves/restores model global state to avoid polluting Phase 1.
        """
        # Save model global state (Phase 1's keep_masks, drop_ratios, etc.)
        saved_keep_masks = list(self._model.model.all_keep_masks) \
            if hasattr(self._model.model, 'all_keep_masks') else []
        saved_drop_ratios = list(self._model.model.all_drop_ratios) \
            if hasattr(self._model.model, 'all_drop_ratios') else []
        saved_last_frames = self._model.model.all_last_frames \
            if hasattr(self._model.model, 'all_last_frames') else None
        saved_rope_deltas = self._model.rope_deltas \
            if hasattr(self._model, 'rope_deltas') else None

        try:
            messages = [
                {"role": "system", "content": self._instruction},
                {"role": "user", "content": [{"type": "image", "image": image}]},
            ]

            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

            img_inputs, _ = self._process_vision_info(messages)
            inputs = self._processor(
                text=[text],
                images=img_inputs if img_inputs else None,
                padding=True,
                return_tensors="pt",
            ).to(self._device)

            model_output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
            )

            output_ids = model_output[:, inputs.input_ids.size(1):]
            reply = self._processor.batch_decode(
                output_ids, skip_special_tokens=True,
            )[0].strip()

            if reply.upper().startswith("NO"):
                return "(no content)"
            return reply

        finally:
            # Restore model global state
            if hasattr(self._model.model, 'all_keep_masks'):
                self._model.model.all_keep_masks = saved_keep_masks
            if hasattr(self._model.model, 'all_drop_ratios'):
                self._model.model.all_drop_ratios = saved_drop_ratios
            if hasattr(self._model.model, 'all_last_frames'):
                self._model.model.all_last_frames = saved_last_frames
            if hasattr(self._model, 'rope_deltas'):
                self._model.rope_deltas = saved_rope_deltas

    def _segment_reset(self):
        logger.info(
            f"MMDuet2 segment reset at {self._token_count} tokens"
        )
        self._past_key_values = None
        self._cached_pixel_values = None
        self._cached_image_grid_thw = None
        self._token_count = 0
        self._model_reset()

        self._history = [{"role": "system", "content": _SYSTEM_PROMPT}]
        summary_lines = [
            f"[{h['t_sec']:.0f}s] {h['raw']}"
            for h in self._response_history[-10:]
        ]
        summary = "\n".join(summary_lines) if summary_lines else "(none)"
        self._pending_text = (
            f"{self._instruction}\n\n"
            f"Previous responses:\n{summary}\n\n"
            f"Continue monitoring from here."
        )

    def _model_reset(self):
        if hasattr(self._model, 'model'):
            m = self._model.model
            if hasattr(m, 'all_keep_masks'):
                m.all_keep_masks = []
            if hasattr(m, 'all_drop_ratios'):
                m.all_drop_ratios = []
        if hasattr(self._model, 'reset_status'):
            self._model.reset_status()

    def _reset(self):
        self._history = []
        self._past_key_values = None
        self._cached_pixel_values = None
        self._cached_image_grid_thw = None
        self._pending_text = None
        self._instruction = ""
        self._response_history = []
        self._token_count = 0
        self._model_reset()
