"""Online (true-streaming) evaluator.

Drives a StreamingModel at 1 fps through each sample's full video and
collects whatever the model chooses to emit. No pre/post probing, no forced
querying — the model alone decides when to speak.

Per-sample loop:
    model.begin(instruction, sample)
    frames, audio_chunks = pre_extract(video_path)   # one-time I/O
    history = []
    for t in 0..duration (step 1s):
        resp = model.observe(frames[t], t_sec=t, history, audio_chunks[t])
        if resp is non-empty:
            history.append({"t_sec": t, "raw": resp})
    model.end()
    save {id, task, predictions: history, ground_truth, ...}
"""

import os
import subprocess
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from models.streaming_base import StreamingModel
from utils.prompts import build_online_session_prompt
from utils.online_parser import is_standby, parse_streaming_output
from utils.io import save_prediction


# Tasks with structured output fields (position / count / state) that
# benefit from having `parsed` inlined into the jsonl for easy inspection.
# Free-text tasks (IEA/SCA/EN/SSI) are kept raw — their `parsed.description`
# would just duplicate `raw`.
_STRUCTURED_OUTPUT_TASKS = {
    "explicit_target_grounding",   # position
    "snapshot_counting",      # count
    "cumulative_counting",         # count
    "dedup_counting",               # count
    "realtime_state_monitor",      # state
}


# ── Pre-extraction helpers ───────────────────────────────────────────────

def _extract_audio_chunks(
    video_path: str,
    duration: float,
    sr: int = 16000,
) -> Dict[int, np.ndarray]:
    """Extract audio from video and split into 1-second chunks.

    Uses ffmpeg to decode the full audio track to 16 kHz mono PCM, then
    slices into per-second numpy arrays.

    Returns:
        dict mapping integer second ``t`` to ``np.ndarray`` of shape
        ``(sr,)``.  Empty dict if the video has no audio track.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",                          # no video
            "-acodec", "pcm_s16le",         # 16-bit PCM
            "-ar", str(sr),                 # resample
            "-ac", "1",                     # mono
            "-loglevel", "error",
            tmp_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)

        # Read raw PCM bytes → float32 numpy
        raw = np.fromfile(tmp_path, dtype=np.int16).astype(np.float32) / 32768.0

        chunks: Dict[int, np.ndarray] = {}
        total_sec = int(min(duration, len(raw) / sr))
        for t in range(total_sec + 1):
            start = t * sr
            end = min((t + 1) * sr, len(raw))
            if start >= len(raw):
                break
            chunk = raw[start:end]
            # Pad last chunk to full second if needed
            if len(chunk) < sr:
                chunk = np.pad(chunk, (0, sr - len(chunk)))
            chunks[t] = chunk
        return chunks

    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # No audio track or ffmpeg not available
        return {}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _extract_all_frames(
    video_path: str,
    duration: float,
    fps: float = 1.0,
) -> Dict[int, np.ndarray]:
    """Pre-extract all frames at the target fps from the video.

    Opens the VideoReader once, reads all needed frame indices, and
    returns a dict mapping integer second ``t`` to HWC uint8 numpy array.

    Returns:
        dict mapping integer second to frame array.  Empty dict on failure.
    """
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        video_fps = vr.get_avg_fps() or 1.0
        total_frames = len(vr)

        t_end = int(round(duration))
        # Build list of (t_sec, frame_idx) pairs
        indices = []
        t_secs = []
        for t in range(t_end + 1):
            idx = min(int(round(t * video_fps)), total_frames - 1)
            idx = max(idx, 0)
            indices.append(idx)
            t_secs.append(t)

        if not indices:
            return {}

        # Batch read all frames at once (much faster than one-by-one)
        frames_batch = vr.get_batch(indices).asnumpy()  # (N, H, W, C)

        return {t: frames_batch[i] for i, t in enumerate(t_secs)}

    except Exception as e:
        print(f"[WARN] Failed to pre-extract frames from {video_path}: {e}")
        return {}


import re


def _merge_continuation_fragments(history, gap_sec: float = 2.0):
    """Merge consecutive fragments that likely belong to one utterance.

    Streaming models may split a single utterance across consecutive ticks.
    Without keyword prefixes to anchor new emits, we use a time-gap heuristic:
    if two emits arrive within ``gap_sec`` of each other, treat the later
    one as a continuation of the former (concatenate into one emit, keep
    the earlier timestamp).
    """
    if not history:
        return history

    merged = [dict(history[0])]
    last_t = history[0]["t_sec"]
    for h in history[1:]:
        if h["t_sec"] - last_t <= gap_sec:
            merged[-1]["raw"] = merged[-1]["raw"] + " " + h["raw"]
        else:
            merged.append(dict(h))
        last_t = h["t_sec"]
    return merged


class OnlineEvaluator:
    """True-online evaluation: 1 fps frame stream, model-driven emits."""

    def __init__(
        self,
        model: StreamingModel,
        fps: float = 1.0,
    ):
        if not isinstance(model, StreamingModel):
            raise TypeError(
                f"OnlineEvaluator requires a StreamingModel, got "
                f"{type(model).__name__}"
            )
        self.model = model
        self.fps = fps

    def evaluate(
        self,
        dataset: List[dict],
        output_dir: str,
        resume: bool = True,
        **kwargs,
    ) -> List[dict]:
        os.makedirs(output_dir, exist_ok=True)

        if resume:
            from utils.io import get_completed_ids
            completed = get_completed_ids(output_dir)
            before = len(dataset)
            dataset = [s for s in dataset if s["id"] not in completed]
            if before > len(dataset):
                print(f"[Online] Resuming: {before - len(dataset)} "
                      f"already done, {len(dataset)} to go")

        results = []
        for sample in tqdm(dataset, desc="Online eval"):
            try:
                pred = self._run_sample(sample)
                save_prediction(pred, output_dir)
                results.append(pred)
            except Exception as e:
                import traceback
                print(f"[ERROR] {sample['id']}: {e}")
                traceback.print_exc()
        return results

    # ------------------------------------------------------------------
    def _run_sample(self, sample: dict) -> dict:
        task = sample["task"]
        video_path = sample["video_path"]
        duration = float(sample.get("duration", 0.0))
        question_time_sec = _mmss_to_sec(sample.get("question_time", "00:00"))
        session_prompt = build_online_session_prompt(task, sample)

        # 1 fps tick range — inclusive of final second
        t_start = int(round(question_time_sec))
        t_end = int(round(duration))
        tick_interval = 1.0 / max(self.fps, 1e-6)

        # ── Pre-extract all frames and audio ONCE ──
        t0 = time.time()
        frames = _extract_all_frames(video_path, duration, self.fps)
        audio_chunks: Dict[int, np.ndarray] = {}
        if getattr(self.model, "accepts_audio", False):
            audio_chunks = _extract_audio_chunks(video_path, duration)
        preprocess_time = time.time() - t0

        self.model.begin(instruction=session_prompt, task_meta=sample)
        history = []  # [{t_sec, raw}]
        try:
            t = float(t_start)
            while t <= t_end:
                frame = frames.get(int(t))  # pre-extracted, no I/O
                audio_chunk = audio_chunks.get(int(t))
                raw = self.model.observe(
                    frame=frame, t_sec=t,
                    history=list(history),
                    audio_chunk=audio_chunk,
                )
                if not is_standby(raw):
                    history.append({"t_sec": t, "raw": raw})
                t += tick_interval
        finally:
            self.model.end()

        # Merge continuation fragments
        history = _merge_continuation_fragments(history)

        # Store raw emits; inline `parsed` for structured tasks so the
        # extracted field (position / count / state) is visible in the
        # jsonl without re-running the parser.
        inline_parsed = task in _STRUCTURED_OUTPUT_TASKS
        predictions = []
        for h in history:
            entry = {"t_sec": h["t_sec"], "raw": h["raw"]}
            if inline_parsed:
                p = parse_streaming_output(h["raw"], task)
                # Keep only the task-specific structured field(s), not the
                # redundant raw/payload copies already present at top level.
                keep = {}
                for k in ("position", "count", "state", "valid"):
                    if k in p:
                        keep[k] = p[k]
                entry["parsed"] = keep
            predictions.append(entry)

        return {
            "id": sample["id"],
            "task": task,
            "video_id": sample.get("video_id"),
            "question": sample.get("question"),
            "ground_truth": sample.get("ground_truth", []),
            "predictions": predictions,
            "model": self.model.name(),
            "eval_mode": "online",
            "eval_config": {
                "fps": self.fps,
                "t_start": t_start,
                "t_end": t_end,
                "preprocess_sec": round(preprocess_time, 1),
            },
        }


def _mmss_to_sec(s):
    if isinstance(s, (int, float)):
        return float(s)
    if not s:
        return 0.0
    parts = str(s).split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return float(s)
