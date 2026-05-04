"""StreamingModel base class for true-online evaluation.

Unlike `BaseModel.generate(instruction, video_path)` which feeds the whole
clip at once, a streaming model is driven by the evaluator frame-by-frame and
decides itself when to emit a response.

Minimum contract:
    begin(instruction, task_meta) -> None
    observe(frame, t_sec, history) -> Optional[str]
    end() -> None

The evaluator:
  * calls `begin` once per sample,
  * calls `observe` every 1 second (1 fps) with the newest frame and the full
    list of the model's own previous outputs,
  * collects any non-None return values with the timestamp,
  * calls `end` when the video is exhausted.

`history` is a list of dicts:
    [{"t_sec": 46.0, "raw": "A whistle just started blowing."}, ...]
Models output free-form natural language; downstream parsers extract
structured fields (position / count / state) where needed.

Concrete streaming models (VideoLLM-online, Flash-VStream, Streamo ...)
should subclass this and implement the three methods. Non-streaming wrappers
(Qwen / Gemini) won't implement this — they use BaseModel + ProbeEvaluator.
"""

from abc import abstractmethod
from typing import List, Optional, Dict, Any

import numpy as np

from .base import BaseModel


class StreamingModel(BaseModel):
    """Per-frame streaming model interface for online evaluation."""

    # --- optional: concrete models can override to signal capabilities ---
    accepts_audio: bool = False
    native_fps: float = 1.0

    @abstractmethod
    def begin(self, instruction: str, task_meta: Dict[str, Any]) -> None:
        """Open a new streaming session for one sample.

        Args:
            instruction: The full online prompt for this sample (already task-
                specific, with don't-repeat constraints and output grammar).
            task_meta: Extra info the model may use (task name, allowed states
                for RSM, etc). Evaluator passes the whole sample dict.
        """

    @abstractmethod
    def observe(
        self,
        frame,
        t_sec: float,
        history: List[Dict[str, Any]],
        audio_chunk: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Feed one frame (+ optional audio); model decides whether to emit.

        Args:
            frame: A single video frame (numpy HWC uint8 or torch tensor — up
                to the concrete model to spec). Can be None if the model
                prefers to pull frames itself from the stream.
            t_sec: Timestamp of this frame in the video (seconds, float).
            history: All prior non-None outputs from this session, in order:
                [{"t_sec": float, "raw": str}, ...]
            audio_chunk: Optional 1-second audio waveform as a numpy array
                (16 kHz mono, shape ``(16000,)``). ``None`` when the model
                does not accept audio or the video has no audio track.

        Returns:
            None  -> STANDBY (no output this tick).
            str   -> Raw response string (evaluator stores it with t_sec).
        """

    @abstractmethod
    def end(self) -> None:
        """Close the current streaming session and free any caches."""

    # generate() isn't meaningful for streaming models but BaseModel requires
    # it; provide a default that errors clearly.
    def generate(self, instruction: str, video_path: str, **kwargs) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} is a StreamingModel; use OnlineEvaluator, "
            f"not ProbeEvaluator / generate()."
        )
