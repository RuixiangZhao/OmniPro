"""Qwen2.5-Omni model wrapper for evaluation (supports audio)."""

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from .base import BaseModel


def _check_video_has_audio(video_path: str) -> bool:
    """Check if a video file has an audio track."""
    try:
        from decord import AudioReader
        ar = AudioReader(video_path)
        return len(ar) > 0
    except Exception:
        return False


class Qwen2_5Omni(BaseModel):
    """Qwen2.5-Omni evaluation model with audio support."""

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

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device,
            attn_implementation="flash_attention_2",
        )
        self.model.disable_talker()

        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

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

        text_ids = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            use_audio_in_video=use_audio,
            return_audio=False,
        )

        output_text = self.processor.batch_decode(
            text_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()


class Qwen2_5Omni_NoAudio(Qwen2_5Omni):
    """Qwen2.5-Omni with audio disabled (visual-only)."""
    def __init__(self, model_path: str, device: str = "auto", **kwargs):
        super().__init__(model_path, device, use_audio=False, **kwargs)


class Qwen2_5Omni_AudioOnly(Qwen2_5Omni):
    """Qwen2.5-Omni with video disabled (audio-only).

    Instead of passing the video clip as a 'video' content item, we pass it
    as an 'audio' content item so the model receives only the audio track.
    """

    def __init__(self, model_path: str, device: str = "auto", **kwargs):
        super().__init__(model_path, device, use_audio=True, **kwargs)
        self._name = self._name.replace("-NoAudio", "") + "-AudioOnly"

    def name(self) -> str:
        return self._name

    @torch.no_grad()
    def generate(self, instruction: str, video_path: str, **kwargs) -> str:
        from qwen_omni_utils import process_mm_info

        # Check that the video has an audio track
        if not _check_video_has_audio(video_path):
            # No audio → model has zero input signal; return empty
            return ""

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio": video_path,   # extract audio from the video file
                    },
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

        text_ids = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            use_audio_in_video=False,
            return_audio=False,
        )

        output_text = self.processor.batch_decode(
            text_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()
