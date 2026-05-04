"""
Probe 式评测器。

- alert 类任务 (instant_event_alert, semantic_condition_alert, explicit_target_grounding):
  GT 定点探测，每个触发点前后各问一次，固定 seed 随机选时间点。
- 其他任务: 固定间隔轮询。
"""

import os
import random
import time
from typing import List, Optional

from tqdm import tqdm

from models.base import BaseModel
from utils.video import split_video
from utils.prompts import build_probe_prompt, parse_response
from utils.io import save_prediction, get_completed_ids


# GT 定点探测任务（每个触发点前后各一次探测）
GT_PROBE_TASKS = {
    "instant_event_alert",
    "semantic_condition_alert",
    "explicit_target_grounding",
    "snapshot_counting",
    "cumulative_counting",
    "dedup_counting",
    "realtime_state_monitor",
    "event_narration",
    "sequential_step_instruction",
}

# 固定间隔轮询任务
POLLING_TASKS = set()

# Post-probe offset map:
#   grounding: offset=0  (need the exact trigger frame for position)
#   snapshot_counting: offset=1 (one second after trigger, so the scene has stabilised)
#   cumulative_counting: offset=1 (count should have incremented by now)
#   others: pick from [0, 1, 2, 3]
_POST_OFFSETS_BY_TASK = {
    "explicit_target_grounding": [0],
    "snapshot_counting": [1],
    "sequential_step_instruction": [2],  # offset+2s: probe 2s into the new step
    # cumulative_counting: random [0..3] like alert tasks
}

# Pre-probe offset map (seconds BEFORE trigger; must be negative).
#   SCA: fixed -5 (instead of the default random [-5..-2]) because with a
#        strong VLM the [-2..-4] window frequently lands inside the semantic
#        build-up that precedes the trigger, producing technically-correct
#        YES answers that are scored as false positives under the pre=NO
#        convention. Fixing at -5 keeps pre comparable to the original setup
#        while avoiding the tightest part of the build-up.
#   others: default [-5, -4, -3, -2].
_PRE_OFFSETS_BY_TASK = {
    "semantic_condition_alert": [-5],
}
_DEFAULT_PRE_OFFSETS = [-5, -4, -3, -2]


class ProbeEvaluator:
    """Probe-based evaluation with two strategies: GT-probe and fixed-interval polling."""

    def __init__(
        self,
        model: BaseModel,
        poll_interval: int = 5,
        tolerance_after: int = 10,
        clip_cache_dir: str = "/path/to/OmniProact-Bench/clip_cache",
        seed: int = 42,
    ):
        """
        Args:
            model: Model instance to evaluate.
            poll_interval: Seconds between each probe for polling tasks (default 5).
            tolerance_after: Continue polling N seconds after last GT trigger (default 10).
            clip_cache_dir: Directory to cache clipped videos.
            seed: Random seed for GT-probe time point selection.
        """
        self.model = model
        self.poll_interval = poll_interval
        self.tolerance_after = tolerance_after
        self.clip_cache_dir = clip_cache_dir
        self.seed = seed

    def evaluate(
        self,
        dataset: List[dict],
        output_dir: str,
        resume: bool = True,
    ) -> List[dict]:
        """
        Run probe evaluation on the dataset.

        Args:
            dataset: List of sample dicts from benchmark.json.
            output_dir: Directory to save per-task JSONL predictions.
            resume: If True, skip already completed samples.

        Returns:
            List of prediction dicts.
        """
        completed_ids = get_completed_ids(output_dir) if resume else set()
        if completed_ids:
            print(f"Resuming: {len(completed_ids)} samples already completed")

        predictions = []
        for sample in tqdm(dataset, desc="Probe eval"):
            if sample["id"] in completed_ids:
                continue

            try:
                if sample["task"] in GT_PROBE_TASKS:
                    pred = self._evaluate_gt_probe(sample)
                else:
                    pred = self._evaluate_polling(sample)
            except Exception as e:
                print(f"[ERROR] {sample['id']}: {e}")
                pred = self._make_error_pred(sample, str(e))

            save_prediction(pred, output_dir)
            predictions.append(pred)

        return predictions

    # ── GT 定点探测 (alert 类任务) ─────────────────────────────────────

    def _evaluate_gt_probe(self, sample: dict) -> dict:
        """
        GT-based probing for alert / grounding / counting / state_monitor.

        For each GT trigger, probe twice:
          - pre_probe:  random from [trigger-5, -4, -3, -2]
          - post_probe: task-dependent offset from the trigger

        Correctness:
          - alert / grounding / snapshot_counting: binary YES/NO (+aux check downstream).
          - cumulative / dedup counting: integer exact match. GT count before
            trigger i is `i`; after trigger i it is `i+1`.
          - realtime_state_monitor: MCQ; pre expects `state_from`, post expects
            `state_to`. Paired correctness (pre AND post correct) is the main
            metric, computed downstream.
        """
        task = sample["task"]
        question = sample["question"]
        video_path = sample["video_path"]
        duration = sample.get("duration", float("inf"))
        is_int_counting = task in ("cumulative_counting", "dedup_counting")
        is_state_monitor = (task == "realtime_state_monitor")
        is_narration_mcq = task in ("event_narration", "sequential_step_instruction")
        states = sample.get("states") if is_state_monitor else None

        # Fixed seed per sample for reproducibility
        rng = random.Random(self.seed + hash(sample["id"]))

        gt_list = sample["ground_truth"]
        probe_results = []

        # For MCQ narration: build the shuffled option list ONCE per sample.
        # Each gt[i]'s response is one of the options; the correct answer's
        # position (letter) is recorded alongside each probe.
        mcq_options = None          # list of narration strings in shuffled order
        mcq_correct_idx = None      # list: mcq_correct_idx[gt_i] = position of gt_i in mcq_options
        if is_narration_mcq:
            n = len(gt_list)
            perm = list(range(n))
            rng.shuffle(perm)
            # mcq_options[j] = gt_list[perm[j]].response
            mcq_options = [gt_list[perm[j]].get("response", "") for j in range(n)]
            # For each gt_i, find its position j in the shuffled list
            mcq_correct_idx = [perm.index(i) for i in range(n)]

        # post 位置按任务选择
        post_offsets = _POST_OFFSETS_BY_TASK.get(task, [0, 1, 2, 3])

        for gt_idx, gt in enumerate(gt_list):
            trigger_sec = gt["trigger_time_sec"]

            # ── Pre-probe (skipped for event_narration MCQ: no "before/after"
            # structure; we only do a single probe at trigger time) ──
            if not is_narration_mcq:
                pre_offsets = _PRE_OFFSETS_BY_TASK.get(task, _DEFAULT_PRE_OFFSETS)
                pre_offset = rng.choice(pre_offsets)
                pre_time = max(1, trigger_sec + pre_offset)  # at least 1s into video
                pre_time = min(pre_time, duration)

                clip_path = split_video(video_path, 0, int(pre_time), self.clip_cache_dir)
                prompt = build_probe_prompt(task, question, pre_time,
                                            occurred_count=gt_idx,
                                            event=sample.get("event"),
                                            target=sample.get("target"),
                                            states=states)

                t0 = time.time()
                raw = self.model.generate(prompt, clip_path)
                inf_time = time.time() - t0
                parsed = parse_response(task, raw, states=states) if is_state_monitor else parse_response(task, raw)

                if is_int_counting:
                    gt_count_pre = gt_idx  # count BEFORE this trigger
                    pred_count = parsed.get("count")
                    correct_pre = (pred_count is not None
                                   and int(pred_count) == gt_count_pre)
                    probe_results.append({
                        "gt_idx": gt_idx,
                        "probe_type": "pre",
                        "time_sec": pre_time,
                        "gt_expected": gt_count_pre,
                        "raw_response": raw,
                        "parsed": parsed,
                        "correct": correct_pre,
                        "inference_time": round(inf_time, 2),
                    })
                elif is_state_monitor:
                    gt_state_pre = gt["state_from"]
                    pred_state = parsed.get("state")
                    correct_pre = (pred_state is not None
                                   and pred_state.lower() == gt_state_pre.lower())
                    probe_results.append({
                        "gt_idx": gt_idx,
                        "probe_type": "pre",
                        "time_sec": pre_time,
                        "gt_expected": gt_state_pre,
                        "raw_response": raw,
                        "parsed": parsed,
                        "correct": correct_pre,
                        "inference_time": round(inf_time, 2),
                    })
                else:
                    probe_results.append({
                        "gt_idx": gt_idx,
                        "probe_type": "pre",
                        "time_sec": pre_time,
                        "gt_expected": "NO",
                        "raw_response": raw,
                        "parsed": parsed,
                        "correct": not parsed.get("triggered", False),
                        "inference_time": round(inf_time, 2),
                    })

            # ── Post-probe ──
            post_offset = rng.choice(post_offsets)
            post_time = trigger_sec + post_offset
            post_time = max(1, post_time)
            post_time = min(post_time, duration)

            clip_path = split_video(video_path, 0, int(post_time), self.clip_cache_dir)
            prompt = build_probe_prompt(task, question, post_time,
                                        occurred_count=gt_idx,
                                        event=sample.get("event"),
                                        target=sample.get("target"),
                                        states=states,
                                        options=mcq_options)

            t0 = time.time()
            raw = self.model.generate(prompt, clip_path)
            inf_time = time.time() - t0
            parsed = parse_response(task, raw, states=states) if is_state_monitor else parse_response(task, raw)

            if is_int_counting:
                gt_count_post = gt_idx + 1  # count AFTER this trigger
                pred_count = parsed.get("count")
                correct_post = (pred_count is not None
                                and int(pred_count) == gt_count_post)
                probe_results.append({
                    "gt_idx": gt_idx,
                    "probe_type": "post",
                    "time_sec": post_time,
                    "gt_expected": gt_count_post,
                    "raw_response": raw,
                    "parsed": parsed,
                    "correct": correct_post,
                    "inference_time": round(inf_time, 2),
                })
            elif is_state_monitor:
                gt_state_post = gt["state_to"]
                pred_state = parsed.get("state")
                correct_post = (pred_state is not None
                                and pred_state.lower() == gt_state_post.lower())
                probe_results.append({
                    "gt_idx": gt_idx,
                    "probe_type": "post",
                    "time_sec": post_time,
                    "gt_expected": gt_state_post,
                    "raw_response": raw,
                    "parsed": parsed,
                    "correct": correct_post,
                    "inference_time": round(inf_time, 2),
                })
            elif is_narration_mcq:
                correct_pos = mcq_correct_idx[gt_idx]        # 0-based
                correct_letter = chr(ord('A') + correct_pos)
                pred_letter = parsed.get("choice")
                correct_post = (pred_letter is not None
                                and pred_letter == correct_letter)
                probe_results.append({
                    "gt_idx": gt_idx,
                    "probe_type": "post",
                    "time_sec": post_time,
                    "gt_expected": correct_letter,
                    "raw_response": raw,
                    "parsed": parsed,
                    "correct": correct_post,
                    "inference_time": round(inf_time, 2),
                })
            else:
                probe_results.append({
                    "gt_idx": gt_idx,
                    "probe_type": "post",
                    "time_sec": post_time,
                    "gt_expected": "YES",
                    "raw_response": raw,
                    "parsed": parsed,
                    "correct": parsed.get("triggered", False),
                    "inference_time": round(inf_time, 2),
                })

        out = {
            "id": sample["id"],
            "task": task,
            "video_id": sample["video_id"],
            "question": question,
            "ground_truth": gt_list,
            "predictions": probe_results,
            "model": self.model.name(),
            "eval_mode": "gt_probe",
            "eval_config": {
                "seed": self.seed,
                "pre_offsets": _PRE_OFFSETS_BY_TASK.get(task, _DEFAULT_PRE_OFFSETS),
                "post_offsets": post_offsets,
            },
        }
        if is_narration_mcq:
            out["mcq_options"] = mcq_options
            out["mcq_correct_idx"] = mcq_correct_idx
        return out

    # ── 固定间隔轮询 (counting / monitor / narration 类任务) ──────────

    def _evaluate_polling(self, sample: dict) -> dict:
        """Evaluate one sample with fixed-interval polling."""
        task = sample["task"]
        question = sample["question"]
        video_path = sample["video_path"]
        question_time_sec = sample.get("question_time_sec", 0)

        # Determine polling range
        last_gt_time = max(gt["trigger_time_sec"] for gt in sample["ground_truth"])
        max_poll_time = last_gt_time + self.tolerance_after
        if sample.get("duration") and sample["duration"] > 0:
            max_poll_time = min(max_poll_time, sample["duration"])

        poll_results = []
        prev_response = None

        t = question_time_sec + self.poll_interval
        while t <= max_poll_time:
            clip_path = split_video(video_path, 0, int(t), self.clip_cache_dir)
            prompt = build_probe_prompt(task, question, t, prev_response,
                                        event=sample.get("event"),
                                        target=sample.get("target"))

            t0 = time.time()
            raw_response = self.model.generate(prompt, clip_path)
            inference_time = time.time() - t0

            parsed = parse_response(task, raw_response)

            poll_results.append({
                "time_sec": t,
                "raw_response": raw_response,
                "parsed": parsed,
                "inference_time": round(inference_time, 2),
            })

            if parsed.get("triggered"):
                resp_text = parsed.get("response", raw_response)
                if parsed.get("count") is not None:
                    prev_response = str(parsed["count"])
                else:
                    prev_response = resp_text

            t += self.poll_interval

        return {
            "id": sample["id"],
            "task": task,
            "video_id": sample["video_id"],
            "question": question,
            "ground_truth": sample["ground_truth"],
            "predictions": poll_results,
            "model": self.model.name(),
            "eval_mode": "probe",
            "eval_config": {
                "poll_interval": self.poll_interval,
                "tolerance_after": self.tolerance_after,
            },
        }

    # ── 公共 ──────────────────────────────────────────────────────────

    def _make_error_pred(self, sample: dict, error_msg: str) -> dict:
        return {
            "id": sample["id"],
            "task": sample["task"],
            "video_id": sample["video_id"],
            "question": sample["question"],
            "ground_truth": sample["ground_truth"],
            "predictions": [],
            "model": self.model.name(),
            "eval_mode": "gt_probe" if sample["task"] in GT_PROBE_TASKS else "probe",
            "eval_config": {},
            "error": error_msg,
        }
