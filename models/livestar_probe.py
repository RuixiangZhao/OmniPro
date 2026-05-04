"""LiveStar-8B probe model for OmniProact-Bench.

Paper: "LiveStar: Live Streaming Assistant for Real-World Online Video
Understanding" (NeurIPS 2025).

Architecture: InternViT-300M + InternLM2-7B (same as InternVL family).
Uses model.chat() for video QA. Vision-only (no audio).

Requires:
  - transformers == 4.37.2 (LiveStar pinned, via shim)
  - Model weights: /path/to/huggingface_cache/LiveStar_8B
  - flash-attn, decord, torchvision, pillow
"""

import logging
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from .base import BaseModel

logger = logging.getLogger(__name__)

# ── Image preprocessing (same as livestar.py online adapter) ──────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        tar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - tar)
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio
        elif diff == best_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=1, image_size=448, use_thumbnail=True):
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    tar = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_w, orig_h, image_size)
    target_w = image_size * tar[0]
    target_h = image_size * tar[1]
    blocks = tar[0] * tar[1]
    resized = image.resize((target_w, target_h))
    tiles = []
    for i in range(blocks):
        box = (
            (i % (target_w // image_size)) * image_size,
            (i // (target_w // image_size)) * image_size,
            ((i % (target_w // image_size)) + 1) * image_size,
            ((i // (target_w // image_size)) + 1) * image_size,
        )
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def _load_video_frames(video_path: str, num_segments: int = 16,
                       input_size: int = 448, max_num: int = 1
                       ) -> Tuple[torch.Tensor, List[int]]:
    """Load video frames with LiveStar preprocessing."""
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    seg_size = float(max_frame) / num_segments
    frame_indices = np.array([
        int(seg_size / 2 + seg_size * idx) for idx in range(num_segments)
    ])
    frame_indices = np.clip(frame_indices, 0, max_frame)

    transform = _build_transform(input_size)
    pixel_values_list = []
    num_patches_list = []

    for fi in frame_indices:
        img = Image.fromarray(vr[fi].asnumpy()).convert('RGB')
        tiles = _dynamic_preprocess(img, image_size=input_size,
                                    use_thumbnail=True, max_num=max_num)
        pv = torch.stack([transform(t) for t in tiles])
        num_patches_list.append(pv.shape[0])
        pixel_values_list.append(pv)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


class LiveStarProbe(BaseModel):
    """LiveStar-8B for probe evaluation (vision-only)."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        fps: float = 1.0,
        max_new_tokens: int = 512,
        num_frames: int = 16,
    ):
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading LiveStar-8B (probe) from {model_path} ...")

        if device in (None, "auto"):
            device = "cuda:0"

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self._model = (
            AutoModel.from_pretrained(model_path, trust_remote_code=True)
            .half().to(device).to(torch.bfloat16).eval()
        )
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._num_frames = num_frames

        logger.info("LiveStar-8B (probe) loaded.")

    def name(self) -> str:
        return "LiveStar-8B"

    @torch.no_grad()
    def generate(self, instruction: str, video_path: str, **kwargs) -> str:
        try:
            pixel_values, num_patches_list = _load_video_frames(
                video_path, num_segments=self._num_frames,
                input_size=448, max_num=1,
            )
        except Exception as e:
            logger.warning(f"Video loading failed: {e}")
            return ""

        pixel_values = pixel_values.to(torch.bfloat16).to(self._device)

        # Build question with frame tags
        video_prefix = ''.join(
            [f'Frame-{i+1}: <image>\n' for i in range(len(num_patches_list))]
        )
        question = video_prefix + instruction

        generation_config = dict(
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
        )

        try:
            response = self._model.chat(
                self._tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False,
            )
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return ""

        return response.strip() if isinstance(response, str) else str(response).strip()
