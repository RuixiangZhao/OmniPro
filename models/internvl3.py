"""InternVL3.5-8B probe model for OmniProact-Bench.

Paper: "InternVL3: Exploring Advanced Training and Test-Time Recipes
for Open-Source Multimodal Models" (OpenGVLab, 2025).

Uses the official trust_remote_code interface:
  model = AutoModel.from_pretrained(path, trust_remote_code=True)
  tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
  response = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=...)

Video frames are extracted with decord, preprocessed with dynamic resolution
patching (official InternVL recipe), then fed via model.chat().

Requires:
  - transformers >= 4.52.1
  - flash-attn, decord, torchvision, pillow
  - Model weights: OpenGVLab/InternVL3_5-8B (~16GB)
"""

import logging
import math
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from .base import BaseModel

logger = logging.getLogger(__name__)

# ── Image preprocessing (official InternVL recipe) ──────────────────────────

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
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


# ── Video loading ────────────────────────────────────────────────────────────

def _load_video_frames(
    video_path: str,
    num_segments: int = 16,
    input_size: int = 448,
    max_num_per_frame: int = 1,
) -> Tuple[torch.Tensor, List[int]]:
    """Load video frames with dynamic preprocess (official InternVL recipe).

    Returns:
        pixel_values: (total_patches, C, H, W) tensor
        num_patches_list: list of patch counts per frame
    """
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    # Uniform sampling
    seg_size = float(max_frame) / num_segments
    frame_indices = np.array([
        int(seg_size / 2 + seg_size * idx) for idx in range(num_segments)
    ])
    frame_indices = np.clip(frame_indices, 0, max_frame)

    transform = _build_transform(input_size)
    pixel_values_list = []
    num_patches_list = []

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        tiles = _dynamic_preprocess(
            img, image_size=input_size, use_thumbnail=True, max_num=max_num_per_frame)
        pv = torch.stack([transform(tile) for tile in tiles])
        num_patches_list.append(pv.shape[0])
        pixel_values_list.append(pv)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


# ── Model class ──────────────────────────────────────────────────────────────

class InternVL3(BaseModel):
    """InternVL3.5-8B for probe evaluation (vision-only)."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        fps: float = 1.0,
        max_new_tokens: int = 512,
        num_frames: int = 16,
    ):
        from transformers import AutoTokenizer, AutoModel

        logger.info(f"Loading InternVL3.5-8B from {model_path} ...")

        self._model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval()

        # Place on device
        if device not in (None, "auto"):
            self._model = self._model.to(device)
            self._device = device
        else:
            self._model = self._model.cuda()
            self._device = "cuda:0"

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False)

        self._max_new_tokens = max_new_tokens
        self._num_frames = num_frames

        logger.info(f"InternVL3.5-8B loaded on {self._device}.")

    def name(self) -> str:
        return "InternVL3.5-8B"

    @torch.no_grad()
    def generate(self, instruction: str, video_path: str, **kwargs) -> str:
        try:
            pixel_values, num_patches_list = _load_video_frames(
                video_path,
                num_segments=self._num_frames,
                input_size=448,
                max_num_per_frame=1,  # 1 patch per frame to control memory
            )
        except Exception as e:
            logger.warning(f"Video loading failed for {video_path}: {e}")
            return ""

        pixel_values = pixel_values.to(torch.bfloat16).to(self._device)

        # Build video prefix: Frame1: <image>\nFrame2: <image>\n...
        video_prefix = ''.join(
            [f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))]
        )
        question = video_prefix + instruction

        generation_config = dict(max_new_tokens=self._max_new_tokens, do_sample=False)

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

        return response.strip()
