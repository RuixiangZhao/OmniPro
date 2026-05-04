"""Dummy streaming models that cheat using ground truth.

Purpose: validate OnlineEvaluator + online_metrics end-to-end without needing
a real streaming VLM. Two flavors:

  - DummyStreamingPerfect  : emits the exact right thing at each trigger_sec
  - DummyStreamingNoisy    : adds realistic failure modes (miss 20%, shift
    ±2s, wrong content 20%)

Evaluator treats them as StreamingModel. They pull GT from task_meta passed
to .begin().
"""

import random
from typing import Dict, List, Optional

import numpy as np

from .streaming_base import StreamingModel


def _gt_sec(gt):
    if "trigger_time_sec" in gt:
        return float(gt["trigger_time_sec"])
    parts = str(gt.get("trigger_time", "00:00")).split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0.0


def _format_emit(task: str, gt: Dict) -> str:
    """Produce an emit string in the free-text format the current online
    prompts ask for (no TRIGGER:/UPDATE:/... keywords).
    """
    if task in ("instant_event_alert", "semantic_condition_alert"):
        return gt.get("event_description") or gt.get("response") or "event"
    if task == "explicit_target_grounding":
        desc = gt.get("event_description") or "target event"
        pos = gt.get("position") or "center"
        return f"{desc}. Position: {pos}"
    if task in ("snapshot_counting", "cumulative_counting",
                "dedup_counting"):
        return str(gt.get("count", 0))
    if task == "realtime_state_monitor":
        return gt.get("state_to") or gt.get("state") or "unknown"
    if task in ("event_narration", "sequential_step_instruction"):
        return (gt.get("response") or gt.get("event_description")
                or "something happened")
    return "unknown"


class DummyStreamingPerfect(StreamingModel):
    """Emits a perfect response exactly at each GT trigger_sec. Used to
    assert that a perfect model scores F1=1, content=1 under the metric."""

    def __init__(self, **kwargs):
        self._task = None
        self._emit_plan: List[tuple] = []  # list of (t_sec, raw)
        self._emitted = set()

    def name(self) -> str:
        return "DummyStreamingPerfect"

    def begin(self, instruction: str, task_meta: Dict) -> None:
        self._task = task_meta["task"]
        self._emit_plan = []
        for gt in task_meta.get("ground_truth", []):
            t = int(round(_gt_sec(gt)))
            raw = _format_emit(self._task, gt)
            self._emit_plan.append((t, raw))
        self._emitted = set()

    def observe(self, frame, t_sec: float,
                history: List[Dict],
                audio_chunk: Optional[np.ndarray] = None) -> Optional[str]:
        # emit if there is a planned entry at this tick
        for idx, (pt, raw) in enumerate(self._emit_plan):
            if idx in self._emitted:
                continue
            if abs(pt - t_sec) < 0.5:
                self._emitted.add(idx)
                return raw
        return None

    def end(self) -> None:
        self._emit_plan = []
        self._emitted = set()


class DummyStreamingNoisy(StreamingModel):
    """Adds realistic failure modes to a perfect stream:
       - miss_rate fraction of GTs are dropped
       - each emit's time shifted uniformly in [-shift_max, +shift_max] s
       - wrong_content_rate fraction get payload scrambled
    """

    def __init__(self, miss_rate=0.2, shift_max=2.0, wrong_content_rate=0.2,
                 seed=0, **kwargs):
        self.miss_rate = miss_rate
        self.shift_max = shift_max
        self.wrong_content_rate = wrong_content_rate
        self.seed = seed
        self._plan: List[tuple] = []
        self._emitted = set()
        self._task = None

    def name(self) -> str:
        return (f"DummyStreamingNoisy(miss={self.miss_rate},"
                f"shift={self.shift_max},wrong={self.wrong_content_rate})")

    def begin(self, instruction: str, task_meta: Dict) -> None:
        rng = random.Random(self.seed + hash(task_meta.get("id", "")) % 10**9)
        self._task = task_meta["task"]
        self._plan = []
        for gt in task_meta.get("ground_truth", []):
            if rng.random() < self.miss_rate:
                continue  # dropped
            shift = rng.uniform(-self.shift_max, self.shift_max)
            t = max(0, int(round(_gt_sec(gt) + shift)))
            if rng.random() < self.wrong_content_rate:
                # break content: flip count / position / state
                bad_gt = dict(gt)
                if "count" in bad_gt and bad_gt["count"] is not None:
                    bad_gt["count"] = int(bad_gt["count"]) + 1
                if bad_gt.get("position"):
                    bad_gt["position"] = "top-left" if bad_gt["position"] != "top-left" else "bottom-right"
                if bad_gt.get("state_to"):
                    bad_gt["state_to"] = "__wrong__"
                raw = _format_emit(self._task, bad_gt)
            else:
                raw = _format_emit(self._task, gt)
            self._plan.append((t, raw))
        self._plan.sort()
        self._emitted = set()

    def observe(self, frame, t_sec: float,
                history: List[Dict],
                audio_chunk: Optional[np.ndarray] = None) -> Optional[str]:
        for idx, (pt, raw) in enumerate(self._plan):
            if idx in self._emitted:
                continue
            if abs(pt - t_sec) < 0.5:
                self._emitted.add(idx)
                return raw
        return None

    def end(self) -> None:
        self._plan = []
        self._emitted = set()
