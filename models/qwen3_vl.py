"""Qwen3-VL model wrapper for evaluation."""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .base import BaseModel


class Qwen3VL(BaseModel):
    """Qwen3-VL evaluation model."""

    def __init__(self, model_path: str, device: str = "auto",
                 fps: float = 1.0, max_pixels: int = 360 * 420,
                 max_new_tokens: int = 256):
        self.model_path = model_path
        self._name = model_path.rstrip("/").split("/")[-1]
        self.fps = fps
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def name(self) -> str:
        return self._name

    @torch.no_grad()
    def generate(self, instruction: str, video_path: str, **kwargs) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "fps": self.fps},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        images, videos, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
        )

        generated_ids_trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]

        output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output[0].strip()
