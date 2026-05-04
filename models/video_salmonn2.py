"""Video-SALMONN 2+ probe model for OmniProact-Bench.

Paper: "video-SALMONN 2: Caption-Enhanced Audio-Visual Large Language
Models" (2025). Built on Qwen2.5-VL with Whisper audio encoder.
Repo: https://github.com/bytedance/video-SALMONN-2

Uses the official inference pipeline:
  1. prepare_dataset() → DataArguments + processors
  2. test_data._get_item(input_dict) → tokenized inputs
  3. model.generate(**inputs) → output tokens
  4. tokenizer.decode() → text

Requires:
  - third_party/video-SALMONN-2/video_SALMONN2_plus/ in the project root
  - Model weights (HF: tsinghua-ee/video-SALMONN2_plus_7B_full)
  - liger_kernel, torchcodec, flash-attn
"""

import logging
import os
import sys
from typing import Optional

import torch

from .base import BaseModel

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SALMONN_ROOT = os.path.join(
    _PROJECT_ROOT, "third_party", "video-SALMONN-2", "video_SALMONN2_plus"
)


def _ensure_importable():
    if _SALMONN_ROOT not in sys.path:
        sys.path.insert(0, _SALMONN_ROOT)


class VideoSALMONN2Plus(BaseModel):
    """Video-SALMONN 2+ 7B for probe evaluation (audio+visual)."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        fps: float = 1.0,
        max_new_tokens: int = 512,
        use_audio: bool = True,
    ):
        _ensure_importable()

        # Apply Liger kernel optimizations before model loading
        from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
        from qwenvl.model import modeling_qwen2_5_vl
        modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = liger_multimodal_rotary_pos_emb
        modeling_qwen2_5_vl.Qwen2RMSNorm = LigerRMSNorm
        modeling_qwen2_5_vl.Qwen2MLP = LigerSwiGLUMLP

        from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus as ModelClass
        from qwenvl.data.dataset import make_supervised_data_module
        from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
        from qwenvl.train.argument import DataArguments
        from transformers import AutoTokenizer, WhisperFeatureExtractor

        logger.info(f"Loading Video-SALMONN 2+ from {model_path} ...")

        # Prepare data arguments and processors
        data_args = DataArguments()
        data_args.video_max_frames = 768
        data_args.video_min_frames = 16
        data_args.base_interval = 0.1
        data_args.max_pixels = 61250
        data_args.video_max_frame_pixels = 61250
        data_args.run_test = True
        data_args.image_processor = Qwen2VLImageProcessorFast.from_pretrained(
            model_path, local_files_only=True,
        )
        data_args.audio_processor = WhisperFeatureExtractor(
            feature_size=data_args.feature_size,
            sampling_rate=data_args.sampling_rate,
            hop_length=data_args.hop_length,
            chunk_length=data_args.chunk_length,
        )
        data_args.model_type = "qwen2.5vl"

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=131072,
            padding_side="right",
            use_fast=False,
            local_files_only=True,
        )

        # Load model
        self._model = ModelClass.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            local_files_only=True,
        )
        # Ensure valid device (probe framework may pass 'auto')
        if device in (None, "auto"):
            device = "cuda:0"
        self._model.to(device)

        # Create dataset module for preprocessing
        data_module = make_supervised_data_module(
            tokenizer=self._tokenizer, data_args=data_args,
        )
        self._test_data = data_module["train_dataset"]

        self._device = device
        self._max_new_tokens = max_new_tokens
        self._use_audio = use_audio

        logger.info("Video-SALMONN 2+ loaded.")

    def name(self) -> str:
        suffix = "" if self._use_audio else "-NoAudio"
        return f"Video-SALMONN2plus-7B{suffix}"

    @torch.no_grad()
    def generate(self, instruction: str, video_path: str, **kwargs) -> str:
        input_dict = {
            "video": video_path,
            "use_audio": self._use_audio,
            "conversations": [
                {"from": "human", "value": f"<video>\n{instruction}"},
                {"from": "gpt", "value": ""},
            ],
        }

        try:
            inputs = self._test_data._get_item(input_dict)
        except Exception as e:
            logger.warning(f"Video-SALMONN 2+ preprocessing failed: {e}")
            return ""

        # Clean non-tensor fields and move to GPU
        for key in ("video", "image", "prompt", "ref", "audio",
                     "use_audio", "should_use"):
            inputs.pop(key, None)
        inputs = {
            k: v.to(self._device)
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor)
        }

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
        )

        output_trimmed = outputs[0, len(inputs["input_ids"][0]):]
        text = self._tokenizer.decode(
            output_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return text.strip()


class VideoSALMONN2Plus_NoAudio(VideoSALMONN2Plus):
    """Video-SALMONN 2+ 7B without audio (video-only)."""

    def __init__(self, model_path: str, device: str = "cuda:0",
                 fps: float = 1.0, max_new_tokens: int = 512):
        super().__init__(
            model_path=model_path,
            device=device,
            fps=fps,
            max_new_tokens=max_new_tokens,
            use_audio=False,
        )


class VideoSALMONN2Plus_AudioOnly(VideoSALMONN2Plus):
    """Video-SALMONN 2+ 7B audio-only (no video frames).

    Uses the <audio> tag instead of <video> and passes the video file path
    as the 'audio' field so only the audio track is extracted and fed to the
    Whisper encoder.  No visual tokens are generated.
    """

    def __init__(self, model_path: str, device: str = "cuda:0",
                 fps: float = 1.0, max_new_tokens: int = 512):
        super().__init__(
            model_path=model_path,
            device=device,
            fps=fps,
            max_new_tokens=max_new_tokens,
            use_audio=True,  # need audio encoder loaded
        )

    def name(self) -> str:
        return "Video-SALMONN2plus-7B-AudioOnly"

    @torch.no_grad()
    def generate(self, instruction: str, video_path: str, **kwargs) -> str:
        # Audio-only: use <audio> tag and pass the video file as audio source.
        # The dataset._get_item handles "audio" key separately from "video".
        input_dict = {
            "audio": video_path,
            "conversations": [
                {"from": "human", "value": f"<audio>\n{instruction}"},
                {"from": "gpt", "value": ""},
            ],
        }

        try:
            inputs = self._test_data._get_item(input_dict)
        except Exception as e:
            logger.warning(f"Video-SALMONN 2+ AudioOnly preprocessing failed: {e}")
            return ""

        # Clean non-tensor fields and move to GPU
        for key in ("video", "image", "prompt", "ref", "audio",
                     "use_audio", "should_use"):
            inputs.pop(key, None)
        inputs = {
            k: v.to(self._device)
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor)
        }

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
        )

        output_trimmed = outputs[0, len(inputs["input_ids"][0]):]
        text = self._tokenizer.decode(
            output_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return text.strip()
