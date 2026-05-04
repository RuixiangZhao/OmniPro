"""Probe-mode per-task metric aggregation.

Orchestrates temporal (matching.py) + content (content.py, llm_judge.py)
metrics for each of the 9 benchmark tasks, following the GT-probe and
polling protocols defined in ``evaluators/probe_evaluator.py``.
"""

from collections import defaultdict
from typing import List

from .matching import (
    compute_temporal_metrics,
    extract_pred_trigger_times,
    extract_gt_trigger_times,
)
from .content import evaluate_counting_sample
from ..llm_judge import LLMJudge


# --------------------------------------------------------------------- #
# Task → (temporal strategy, content strategy) config
# --------------------------------------------------------------------- #
TASK_METRIC_CONFIG = {
    "cumulative_counting":       {"temporal": "gt_probe", "content": "count_exact"},
    "dedup_counting":            {"temporal": "gt_probe", "content": "count_exact"},

    "instant_event_alert":       {"temporal": "gt_probe", "content": "gpt_judge"},
    "semantic_condition_alert":  {"temporal": "gt_probe", "content": "gpt_judge"},
    "explicit_target_grounding": {"temporal": "gt_probe", "content": "position"},
    "snapshot_counting":    {"temporal": "gt_probe", "content": "count_at_probe"},

    "realtime_state_monitor":    {"temporal": "gt_probe", "content": "state_paired"},
    "event_narration":           {"temporal": "gt_probe", "content": "mcq_single"},
    "sequential_step_instruction": {"temporal": "gt_probe", "content": "mcq_single"},
}


# --------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------- #
def compute_all_metrics(
    predictions: List[dict],
    tolerances: List[float] = (3.0, 5.0),
    use_gpt_judge: bool = False,
    gpt_judge: LLMJudge = None,
) -> dict:
    """Compute all metrics per task and an overall summary.

    Args:
        predictions:   list of probe prediction dicts
        tolerances:    time tolerances for polling-style tasks
        use_gpt_judge: enable LLM judge for alert tasks (IEA/SCA)
        gpt_judge:     LLMJudge instance (required if use_gpt_judge)
    """
    by_task = defaultdict(list)
    for p in predictions:
        by_task[p["task"]].append(p)

    results = {}
    for task, preds in sorted(by_task.items()):
        cfg = TASK_METRIC_CONFIG.get(task,
                                     {"temporal": True, "content": "gpt_judge"})
        task_results = {"num_samples": len(preds)}

        # ── Temporal ──────────────────────────────────────────────────
        temporal_mode = cfg.get("temporal", True)

        if temporal_mode == "gt_probe":
            metrics = _compute_gt_probe_metrics(
                preds,
                require_position=(task == "explicit_target_grounding"),
                require_count=(task == "snapshot_counting"),
                is_accuracy_only=(task in ("cumulative_counting",
                                           "dedup_counting")),
                is_state_paired=(task == "realtime_state_monitor"),
            )
            task_results.update(metrics)

        elif temporal_mode:
            for tol in tolerances:
                key = f"@{tol:.0f}s"
                rs, ps, f1s = [], [], []
                for pred in preds:
                    if pred.get("error"):
                        continue
                    pred_t = extract_pred_trigger_times(pred["predictions"])
                    gt_t = extract_gt_trigger_times(pred["ground_truth"])
                    m = compute_temporal_metrics(pred_t, gt_t, tol)
                    rs.append(m["recall"])
                    ps.append(m["precision"])
                    f1s.append(m["f1"])
                n = len(rs) or 1
                task_results[f"temporal_recall{key}"] = round(sum(rs) / n, 4)
                task_results[f"temporal_precision{key}"] = round(sum(ps) / n, 4)
                task_results[f"temporal_f1{key}"] = round(sum(f1s) / n, 4)

        # ── Content ───────────────────────────────────────────────────
        content_type = cfg.get("content", "gpt_judge")

        if content_type == "count":
            accs = []
            for pred in preds:
                if pred.get("error"):
                    continue
                tol = tolerances[0] if tolerances else 3.0
                m = evaluate_counting_sample(
                    pred["predictions"], pred["ground_truth"], tolerance=tol)
                accs.append(m["count_accuracy"])
            n = len(accs) or 1
            task_results["count_accuracy"] = round(sum(accs) / n, 4)

        elif content_type == "gpt_judge" and use_gpt_judge and gpt_judge:
            scores = _compute_gpt_judge_scores(
                preds, gpt_judge,
                tolerances[0] if tolerances else 3.0)
            task_results["gpt_judge_score"] = scores["avg_score"]
            task_results["gpt_judge_count"] = scores["judged_count"]

        elif content_type == "position":
            task_results["position_accuracy"] = task_results.get(
                "position_only_accuracy", 0.0)
            task_results["position_correct"] = task_results.get(
                "triggered_post_with_correct_position", 0)
            task_results["position_total"] = task_results.get(
                "triggered_post_total", 0)

        elif content_type == "count_at_probe":
            task_results["count_accuracy"] = task_results.get(
                "count_only_accuracy", 0.0)
            task_results["count_correct"] = task_results.get(
                "triggered_post_with_correct_count", 0)
            task_results["count_total"] = task_results.get(
                "triggered_post_total", 0)

        results[task] = task_results

    results["overall"] = _compute_overall(results, tolerances)
    return results


# --------------------------------------------------------------------- #
# GT-probe metric (the main gt_probe scorer)
# --------------------------------------------------------------------- #
def _compute_gt_probe_metrics(preds: List[dict],
                              require_position: bool = False,
                              require_count: bool = False,
                              is_accuracy_only: bool = False,
                              is_state_paired: bool = False) -> dict:
    """Compute paired_accuracy + diagnostic stats for GT-probe tasks."""
    total_pre = correct_pre = 0
    total_post = correct_post = 0
    total_triggered_post = correct_aux_post = 0
    paired_total = paired_correct = 0

    for pred in preds:
        if pred.get("error"):
            continue
        by_idx = {}

        for p in pred.get("predictions", []):
            gt_idx = p.get("gt_idx")
            if p["probe_type"] == "pre":
                total_pre += 1
                pre_ok = bool(p["correct"])
                if pre_ok:
                    correct_pre += 1
                by_idx.setdefault(gt_idx, {})["pre"] = pre_ok

            elif p["probe_type"] == "post":
                total_post += 1
                base_correct = bool(p["correct"])
                if require_position or require_count:
                    parsed = p.get("parsed", {})
                    gt = (pred["ground_truth"][gt_idx]
                          if gt_idx is not None else {})
                    triggered = bool(parsed.get("triggered", False))
                    if not triggered:
                        post_ok = False
                    elif require_position:
                        total_triggered_post += 1
                        gt_pos = gt.get("position")
                        pp = parsed.get("position")
                        post_ok = bool(gt_pos and pp
                                       and pp.lower() == gt_pos.lower())
                        if post_ok:
                            correct_aux_post += 1
                    else:  # require_count
                        total_triggered_post += 1
                        gc = gt.get("count")
                        pc = parsed.get("count")
                        post_ok = (gc is not None and pc is not None
                                   and int(pc) == int(gc))
                        if post_ok:
                            correct_aux_post += 1
                else:
                    post_ok = base_correct

                if post_ok:
                    correct_post += 1
                by_idx.setdefault(gt_idx, {})["post"] = post_ok

        for idx, d in by_idx.items():
            if "pre" in d and "post" in d:
                paired_total += 1
                if d["pre"] and d["post"]:
                    paired_correct += 1

    pre_acc = correct_pre / total_pre if total_pre else 0.0
    post_acc = correct_post / total_post if total_post else 0.0
    if paired_total > 0:
        paired_acc = paired_correct / paired_total
    elif total_pre == 0 and total_post > 0:
        paired_acc = post_acc
        paired_total = total_post
        paired_correct = correct_post
    else:
        paired_acc = 0.0
    total_all = total_pre + total_post
    overall_acc = ((correct_pre + correct_post) / total_all
                   if total_all else 0.0)
    f1 = (2 * pre_acc * post_acc / (pre_acc + post_acc)
          if (pre_acc + post_acc) > 0 else 0.0)

    out = {
        "paired_accuracy": round(paired_acc, 4),
        "accuracy": round(overall_acc, 4),
        "pre_accuracy": round(pre_acc, 4),
        "post_accuracy": round(post_acc, 4),
        "f1": round(f1, 4),
        "total_pairs": paired_total,
        "correct_pairs": paired_correct,
        "total_pre": total_pre,
        "correct_pre": correct_pre,
        "total_post": total_post,
        "correct_post": correct_post,
    }
    if require_position:
        pos_only = (correct_aux_post / total_triggered_post
                    if total_triggered_post else 0.0)
        out["position_only_accuracy"] = round(pos_only, 4)
        out["triggered_post_with_correct_position"] = correct_aux_post
        out["triggered_post_total"] = total_triggered_post
    elif require_count:
        cnt_only = (correct_aux_post / total_triggered_post
                    if total_triggered_post else 0.0)
        out["count_only_accuracy"] = round(cnt_only, 4)
        out["triggered_post_with_correct_count"] = correct_aux_post
        out["triggered_post_total"] = total_triggered_post
    return out


# --------------------------------------------------------------------- #
# GPT judge over matched pairs
# --------------------------------------------------------------------- #
def _compute_gpt_judge_scores(preds: List[dict], judge: LLMJudge,
                              tolerance: float) -> dict:
    """Average judge scores over all correctly time-matched post-probes."""
    all_scores = []
    for pred in preds:
        if pred.get("error"):
            continue

        # gt_probe mode
        if pred.get("eval_mode") == "gt_probe":
            for p in pred["predictions"]:
                if p.get("probe_type") != "post" or not p.get("correct"):
                    continue
                gt_idx = p["gt_idx"]
                gt_resp = pred["ground_truth"][gt_idx].get("response", "")
                pred_resp = p.get("parsed", {}).get("response", "")
                if pred_resp and gt_resp:
                    all_scores.append(
                        judge.judge(pred["question"], gt_resp,
                                    pred_resp)["score"])
            continue

        # polling mode
        pred_triggered = [p for p in pred["predictions"]
                          if p.get("parsed", {}).get("triggered")]
        gt_list = pred["ground_truth"]
        pairs = []
        for gi, gt in enumerate(gt_list):
            for pi, p in enumerate(pred_triggered):
                dt = abs(gt["trigger_time_sec"] - p["time_sec"])
                if dt <= tolerance:
                    pairs.append((dt, gi, pi))
        pairs.sort()
        matched_gt, matched_pred = set(), set()
        for dt, gi, pi in pairs:
            if gi in matched_gt or pi in matched_pred:
                continue
            matched_gt.add(gi)
            matched_pred.add(pi)
            gt_resp = gt_list[gi].get("response", "")
            pred_resp = pred_triggered[pi].get("parsed", {}).get("response", "")
            if pred_resp and gt_resp:
                all_scores.append(
                    judge.judge(pred["question"], gt_resp,
                                pred_resp)["score"])

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return {"avg_score": round(avg, 2), "judged_count": len(all_scores)}


# --------------------------------------------------------------------- #
# Overall aggregator
# --------------------------------------------------------------------- #
def _compute_overall(task_results: dict, tolerances: List[float]) -> dict:
    overall = {}
    names = [k for k in task_results if k != "overall"]
    f1_values = []
    for t in names:
        r = task_results[t]
        if "paired_accuracy" in r:
            f1_values.append(r["paired_accuracy"])
        elif f"temporal_f1@{tolerances[0]:.0f}s" in r:
            f1_values.append(r[f"temporal_f1@{tolerances[0]:.0f}s"])
    overall["avg_f1"] = (round(sum(f1_values) / len(f1_values), 4)
                         if f1_values else 0.0)
    for tol in tolerances:
        key = f"@{tol:.0f}s"
        for metric in ("temporal_recall", "temporal_precision",
                       "temporal_f1"):
            k = f"{metric}{key}"
            vals = [task_results[t][k] for t in names
                    if k in task_results[t]]
            overall[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return overall
