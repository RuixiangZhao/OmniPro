"""Qwen3-Omni-30B-A3B model wrapper for evaluation (MoE, audio-capable).

Mirrors the Qwen2.5-Omni wrapper. The only material differences are the new
class names exposed by transformers (`Qwen3OmniMoe...`) and the `generate()`
return signature — Qwen3-Omni returns `(text_ids, audio)` when
`thinker_return_dict_in_generate=True` is set, but we stick with the plain
tensor output by disabling audio generation via `return_audio=False`, same as
the 2.5 wrapper.
"""

import torch
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

from .base import BaseModel


def _check_video_has_audio(video_path: str) -> bool:
    try:
        from decord import AudioReader
        ar = AudioReader(video_path)
        return len(ar) > 0
    except Exception:
        return False


class Qwen3Omni(BaseModel):
    """Qwen3-Omni-30B-A3B-Instruct evaluation wrapper (MoE, audio on by default)."""

    def __init__(self, model_path: str, device: str = "auto",
                 fps: float = 1.0, max_pixels: int = 360 * 420,
                 max_new_tokens: int = 256, use_audio: bool = True):
        self.model_path = model_path
        self._name = model_path.rstrip("/").split("/")[-1]
        if not use_audio:
            self._name += "-NoAudio"
        self.fps = fps
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.use_audio = use_audio

        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map=device,
            attn_implementation="flash_attention_2",
        )
        # Disable the audio "talker" head so generate() returns plain tensors.
        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

    def name(self) -> str:
        return self._name

    @torch.no_grad()
    def generate(self, instruction: str, video_path: str, **kwargs) -> str:
        from qwen_omni_utils import process_mm_info

        use_audio = self.use_audio and _check_video_has_audio(video_path)

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": self.max_pixels,
                        "fps": self.fps,
                    },
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )

        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=use_audio,
        )

        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        gen_out = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            use_audio_in_video=use_audio,
            return_audio=False,
        )
        # Qwen3-Omni may return either a plain tensor or a (text_ids, audio)
        # tuple depending on transformers version / return_audio flag. Always
        # coerce to the text tensor.
        text_ids = gen_out[0] if isinstance(gen_out, tuple) else gen_out

        output_text = self.processor.batch_decode(
            text_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()


class Qwen3Omni_NoAudio(Qwen3Omni):
    """Qwen3-Omni with audio disabled (visual-only)."""
    def __init__(self, model_path: str, device: str = "auto", **kwargs):
        super().__init__(model_path, device, use_audio=False, **kwargs)


class Qwen3Omni_AudioOnly(Qwen3Omni):
    """Qwen3-Omni with video disabled (audio-only).

    Passes the video file as an 'audio' content item so only the audio track
    is extracted. No visual tokens are generated.
    """

    def __init__(self, model_path: str, device: str = "auto", **kwargs):
        super().__init__(model_path, device, use_audio=True, **kwargs)
        self._name = self._name.replace("-NoAudio", "") + "-AudioOnly"

    def name(self) -> str:
        return self._name

    @torch.no_grad()
    def generate(self, instruction: str, video_path: str, **kwargs) -> str:
        from qwen_omni_utils import process_mm_info

        if not _check_video_has_audio(video_path):
            return ""

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": video_path},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )

        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=False,
        )

        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        gen_out = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            use_audio_in_video=False,
            return_audio=False,
        )
        text_ids = gen_out[0] if isinstance(gen_out, tuple) else gen_out

        output_text = self.processor.batch_decode(
            text_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()
