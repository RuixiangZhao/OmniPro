"""MiniCPM-o 4.5 streaming model adapter for online evaluation.

Uses the Duplex API (``as_duplex()``) for true incremental streaming:
each ``observe()`` call feeds one second of audio + one video frame via
``streaming_prefill``, then checks ``streaming_generate`` to see if the
model wants to speak.

Usage:
    model = MiniCPMOStreaming(
        model_path="/path/to/MiniCPM-o-4_5",
        device="cuda:0",
    )
    # Then pass to OnlineEvaluator.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from .streaming_base import StreamingModel

logger = logging.getLogger(__name__)


def _patch_whisper_attention():
    """Monkey-patch WhisperAttention.forward to accept ``past_key_values``
    (plural) in addition to ``past_key_value`` (singular).

    MiniCPM-o's custom MiniCPMWhisperEncoderLayer passes ``past_key_values=None``
    to ``self.self_attn()``, but transformers 4.51's WhisperAttention.forward()
    only has ``past_key_value`` (singular). Since the value is always None this
    is harmless — we just need the signature to not raise TypeError.

    Additionally, the model code expects exactly 2 return values
    ``(hidden_states, attn_weights)`` from self_attn, but the original forward
    may return a 3rd element (past_key_value cache). We strip the extra element.
    """
    try:
        from transformers.models.whisper.modeling_whisper import WhisperAttention
        import inspect
        sig = inspect.signature(WhisperAttention.forward)
        if "past_key_values" not in sig.parameters:
            _orig_forward = WhisperAttention.forward

            def _patched_forward(self, *args, past_key_values=None, **kwargs):
                # Map plural → singular (always None anyway)
                if "past_key_value" not in kwargs:
                    kwargs["past_key_value"] = past_key_values
                result = _orig_forward(self, *args, **kwargs)
                # MiniCPMWhisperEncoderLayer expects exactly 2 values:
                # (hidden_states, attn_weights). Strip any extra cache element.
                if isinstance(result, tuple) and len(result) > 2:
                    return result[:2]
                return result

            WhisperAttention.forward = _patched_forward
            logger.info("Patched WhisperAttention.forward for MiniCPM-o compatibility")
    except Exception as e:
        logger.warning(f"Failed to patch WhisperAttention: {e}")


class MiniCPMOStreaming(StreamingModel):
    """MiniCPM-o 4.5 in Duplex mode for online evaluation.

    Key behaviour:
    - ``begin()``:  ``model.as_duplex().prepare(system_prompt)``
    - ``observe()``: ``streaming_prefill(audio, frame)`` then ``streaming_generate()``
    - ``end()``:    signal session stop and clean up
    """

    accepts_audio: bool = True
    native_fps: float = 1.0

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        max_new_speak_tokens: int = 1024,
        generate_audio: bool = False,
        torch_dtype=None,
    ):
        """
        Args:
            model_path: HuggingFace-style path or local directory.
            device: CUDA device string.
            max_new_speak_tokens: Max tokens generated per streaming_generate
                call (controls response length per tick).
            generate_audio: Whether to produce TTS audio output.  ``False``
                for evaluation (we only need text).
            torch_dtype: Override torch dtype (default bfloat16).
        """
        from transformers import AutoModel, AutoTokenizer

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        # Fix WhisperAttention parameter name mismatch
        _patch_whisper_attention()

        logger.info(f"Loading MiniCPM-o from {model_path} ...")
        self._base_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch_dtype,
        )
        self._base_model.eval().to(device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self._device = device
        self._max_new_speak_tokens = max_new_speak_tokens
        self._generate_audio = generate_audio

        # Duplex wrapper — created once, reused across samples
        self._duplex = self._base_model.as_duplex(
            generate_audio=generate_audio,
        )
        logger.info("MiniCPM-o loaded and converted to duplex mode.")

    def name(self) -> str:
        return "MiniCPM-o-4.5-Duplex"

    # ── StreamingModel interface ────────────────────────────────────────

    def begin(self, instruction: str, task_meta: Dict[str, Any]) -> None:
        """Start a new duplex streaming session with *instruction* as
        system prompt."""
        self._duplex.prepare(
            prefix_system_prompt=instruction,
        )

    def observe(
        self,
        frame,
        t_sec: float,
        history: List[Dict[str, Any]],
        audio_chunk: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Feed one frame + optional audio chunk; return text or None."""
        # Build frame_list (PIL Image)
        frame_list = []
        if frame is not None:
            if isinstance(frame, np.ndarray):
                frame_list.append(Image.fromarray(frame))
            elif isinstance(frame, Image.Image):
                frame_list.append(frame)

        # Step 1: incremental prefill (audio + vision → KV cache)
        self._duplex.streaming_prefill(
            audio_waveform=audio_chunk,
            frame_list=frame_list if frame_list else None,
            max_slice_nums=1,
        )

        # Step 2: let the model decide whether to speak
        result = self._duplex.streaming_generate(
            max_new_speak_tokens_per_chunk=self._max_new_speak_tokens,
            decode_mode="sampling",
        )

        if result.get("is_listen", True):
            return None  # STANDBY — model chose to keep listening

        text = result.get("text", "").strip()
        return text if text else None

    def end(self) -> None:
        """Signal end of session and clean up."""
        try:
            self._duplex.set_session_stop()
        except Exception:
            pass


class MiniCPMOStreamingNoAudio(MiniCPMOStreaming):
    """Vision-only variant: drops the audio chunk before calling prefill.

    Useful for the IEA/SCA audio-ablation study where we want to isolate
    how much of the benchmark score comes from visual grounding alone.
    """

    accepts_audio: bool = False  # Evaluator will skip audio extraction.

    def name(self) -> str:
        return "MiniCPM-o-4.5-Duplex-NoAudio"

    def observe(
        self,
        frame,
        t_sec: float,
        history: List[Dict[str, Any]],
        audio_chunk: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        # Force drop audio even if the evaluator somehow passes one.
        return super().observe(frame, t_sec, history, audio_chunk=None)


class MiniCPMOStreamingAudioOnly(MiniCPMOStreaming):
    """Audio-only variant: drops the video frame before calling prefill.

    Sends audio_waveform but no frame_list, so the model relies solely
    on audio input to decide when and what to emit.
    """

    accepts_audio: bool = True  # Evaluator will extract audio chunks.

    def name(self) -> str:
        return "MiniCPM-o-4.5-Duplex-AudioOnly"

    def observe(
        self,
        frame,
        t_sec: float,
        history: List[Dict[str, Any]],
        audio_chunk: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        # Force drop video frame, keep audio only.
        return super().observe(None, t_sec, history, audio_chunk=audio_chunk)

