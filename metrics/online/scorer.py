"""Online-mode metrics.

Two orthogonal dimensions per task:
  (1) Temporal: greedy one-to-one matching between GT triggers and model
      emits within a ±tolerance window. Yields Precision, Recall, F1.
  (2) Content: for each TP pair, task-specific correctness of the payload.

Report both dimensions separately; do NOT fold into a single number.

A model emit is classified:
  - TP: matched to some GT within tolerance AND content correct
  - TP_time: matched in time but content wrong (time-only hit)
  - FP: matched no GT -> false alert
  - FN: GT with no matching emit -> miss
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from utils.online_parser import parse_streaming_output


TASK_CONTENT_KIND = {
    # Alert tasks — content is not evaluated; any time-matched emit is correct.
    "instant_event_alert":         "time_only",
    "semantic_condition_alert":    "time_only",
    # Structured tasks — rule-based exact matching.
    "explicit_target_grounding":   "position",
    "snapshot_counting":      "count",
    "cumulative_counting":         "count_at_time",
    "dedup_counting":              "count_at_time",
    "realtime_state_monitor":      "state",
    # Free-text narration tasks — need GPT judge (fallback: non-empty check).
    "event_narration":             "gpt_judge",
    "sequential_step_instruction": "gpt_judge",
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


def _gt_trigger_sec(gt: Dict) -> float:
    if "trigger_time_sec" in gt:
        return float(gt["trigger_time_sec"])
    return _mmss_to_sec(gt.get("trigger_time", "00:00"))


# ---------------------------------------------------------------------
# Greedy 1-to-1 matching (global minimum Δt)
# ---------------------------------------------------------------------
def match_emits_to_gt(
    emits: List[Dict],
    gts: List[Dict],
    tolerance: float = 3.0,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedy 1-to-1 matching: repeatedly pick the globally closest (emit, gt)
    pair with |Δt| <= tolerance, remove both, continue.

    Returns:
        matches: list of (emit_idx, gt_idx, abs_dt)
        unmatched_emits: list of emit_idx (false positives)
        unmatched_gts:   list of gt_idx   (false negatives)
    """
    emit_times = [float(e["t_sec"]) for e in emits]
    gt_times = [_gt_trigger_sec(g) for g in gts]

    pairs = []
    for i, et in enumerate(emit_times):
        for j, gt in enumerate(gt_times):
            dt = abs(et - gt)
            if dt <= tolerance:
                pairs.append((dt, i, j))
    pairs.sort()

    used_e, used_g = set(), set()
    matches = []
    for dt, i, j in pairs:
        if i in used_e or j in used_g:
            continue
        matches.append((i, j, dt))
        used_e.add(i)
        used_g.add(j)

    unmatched_emits = [i for i in range(len(emits)) if i not in used_e]
    unmatched_gts = [j for j in range(len(gts)) if j not in used_g]
    return matches, unmatched_emits, unmatched_gts


# ---------------------------------------------------------------------
# Content scoring
# ---------------------------------------------------------------------
def _get_parsed(emit: Dict, task: str) -> Dict:
    """Get parsed fields from an emit, parsing on-the-fly if needed."""
    if "parsed" in emit and emit["parsed"]:
        return emit["parsed"]
    # New format: only {t_sec, raw} — parse now
    return parse_streaming_output(emit.get("raw", ""), task)


def _score_content(task: str, emit: Dict, gt: Dict,
                   gts_all: List[Dict]) -> Optional[bool]:
    """Return True/False for TP if scorable, None if unscorable (e.g. GPT
    judge needed but not provided).

    Strategy per task kind:
      - time_only  : content is ignored → always True when time matches
      - position   : 9-region exact match
      - count      : integer exact match
      - count_at_time : integer equals cumulative count at emit.t_sec
      - state      : state name exact match (to state_to)
      - gpt_judge  : requires external judge; returns None here (caller
                     can re-score with a GPT judge later)
    """
    parsed = _get_parsed(emit, task)
    kind = TASK_CONTENT_KIND.get(task, "gpt_judge")

    # time_only: content is not part of the metric at all.
    if kind == "time_only":
        return True

    # gpt_judge tasks: cannot auto-judge; leave for caller to fill in.
    if kind == "gpt_judge":
        return None

    if not parsed.get("valid", False):
        return False

    if kind == "position":
        pp = (parsed.get("position") or "").lower().strip()
        gp = (gt.get("position") or "").lower().strip()
        if not pp or not gp:
            return False
        return pp == gp

    if kind == "count":
        pc = parsed.get("count")
        gc = gt.get("count")
        if pc is None or gc is None:
            return False
        return int(pc) == int(gc)

    if kind == "count_at_time":
        pc = parsed.get("count")
        if pc is None:
            return False
        emit_t = float(emit["t_sec"])
        sorted_gts = sorted(
            [g for g in gts_all if "count" in g],
            key=_gt_trigger_sec,
        )
        expected = None
        for g in sorted_gts:
            if _gt_trigger_sec(g) <= emit_t + 1e-6:
                expected = g.get("count")
        if expected is None:
            return False
        return int(pc) == int(expected)

    if kind == "state":
        ps = (parsed.get("state") or "").lower().strip()
        gs_to = (gt.get("state_to") or gt.get("state") or "").lower().strip()
        if not ps or not gs_to:
            return False
        return ps == gs_to

    return False


# ---------------------------------------------------------------------
# Top-level per-sample + aggregation
# ---------------------------------------------------------------------
def evaluate_sample(
    sample_pred: Dict,
    tolerance: float = 3.0,
) -> Dict:
    """Compute temporal + content breakdown for one sample."""
    task = sample_pred["task"]
    emits = sample_pred.get("predictions", [])
    gts = sample_pred.get("ground_truth", [])

    matches, un_e, un_g = match_emits_to_gt(emits, gts, tolerance=tolerance)

    tp_time = len(matches)
    fp = len(un_e)
    fn = len(un_g)

    tp_content = 0
    content_scored = 0
    per_match = []
    for i, j, dt in matches:
        ok = _score_content(task, emits[i], gts[j], gts)
        if ok is None:
            per_match.append({"emit_idx": i, "gt_idx": j, "dt": dt,
                              "content_ok": None})
            continue
        content_scored += 1
        if ok:
            tp_content += 1
        per_match.append({"emit_idx": i, "gt_idx": j, "dt": dt,
                          "content_ok": bool(ok)})

    # time-only P/R/F1
    time_p = tp_time / (tp_time + fp) if (tp_time + fp) > 0 else 0.0
    time_r = tp_time / (tp_time + fn) if (tp_time + fn) > 0 else 0.0
    time_f1 = (2 * time_p * time_r / (time_p + time_r)
               if (time_p + time_r) > 0 else 0.0)

    # content accuracy: correct / (content scored, i.e. matched pairs)
    content_acc = (tp_content / content_scored) if content_scored > 0 else 0.0

    # joint: content-gated P/R/F1 (pred is correct only if time+content ok)
    joint_p = tp_content / (tp_time + fp) if (tp_time + fp) > 0 else 0.0
    joint_r = tp_content / (tp_time + fn) if (tp_time + fn) > 0 else 0.0
    joint_f1 = (2 * joint_p * joint_r / (joint_p + joint_r)
                if (joint_p + joint_r) > 0 else 0.0)

    return {
        "id": sample_pred["id"],
        "task": task,
        "num_emits": len(emits),
        "num_gt": len(gts),
        "tp_time": tp_time,
        "tp_content": tp_content,
        "fp": fp,
        "fn": fn,
        "content_scored": content_scored,
        "time_precision": time_p,
        "time_recall":    time_r,
        "time_f1":        time_f1,
        "content_accuracy": content_acc,
        "joint_f1":       joint_f1,
        "per_match":      per_match,
    }


def aggregate(per_sample: List[Dict]) -> Dict:
    """Task-wise + overall micro aggregation over per-sample results."""
    by_task: Dict[str, List[Dict]] = {}
    for r in per_sample:
        by_task.setdefault(r["task"], []).append(r)

    def _micro(rs):
        tp_t = sum(r["tp_time"] for r in rs)
        tp_c = sum(r["tp_content"] for r in rs)
        fp = sum(r["fp"] for r in rs)
        fn = sum(r["fn"] for r in rs)
        cs = sum(r["content_scored"] for r in rs)
        time_p = tp_t / (tp_t + fp) if (tp_t + fp) > 0 else 0.0
        time_r = tp_t / (tp_t + fn) if (tp_t + fn) > 0 else 0.0
        time_f1 = (2*time_p*time_r/(time_p+time_r)
                   if (time_p+time_r) > 0 else 0.0)
        content = tp_c / cs if cs > 0 else 0.0
        joint_p = tp_c / (tp_t + fp) if (tp_t + fp) > 0 else 0.0
        joint_r = tp_c / (tp_t + fn) if (tp_t + fn) > 0 else 0.0
        joint_f1 = (2*joint_p*joint_r/(joint_p+joint_r)
                    if (joint_p+joint_r) > 0 else 0.0)
        return {
            "n_samples": len(rs),
            "tp_time": tp_t, "tp_content": tp_c, "fp": fp, "fn": fn,
            "time_precision": time_p, "time_recall": time_r, "time_f1": time_f1,
            "content_accuracy": content,
            "joint_f1": joint_f1,
        }

    out = {task: _micro(rs) for task, rs in by_task.items()}
    out["overall"] = _micro(per_sample)
    return out


# ---------------------------------------------------------------------
# This module is now a pure library. The CLI for scoring + aggregation
# lives in scripts/compute_online_metrics.py.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("metrics.online_metrics is a library; use "
          "`python scripts/compute_online_metrics.py` instead.")
