"""Compute per-audio-dependency-group metrics for OmniProact-Bench.

Groups samples by their `audio_dependency` label (required/helpful/none)
and computes paired_accuracy (probe) or joint_f1 (online) for each group,
averaged across all tasks.

Usage:
    python3 scripts/analyze_audio_dependency.py
"""

import json
import os
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def load_benchmark_index(data_path):
    """Build a mapping: sample_id -> audio_dependency."""
    data = json.load(open(data_path))
    index = {}
    for s in data:
        sid = s["id"]
        index[sid] = s.get("audio_dependency", "unknown")
    return index


def compute_probe_grouped_metrics(pred_dir, dep_index):
    """Compute paired_accuracy grouped by audio_dependency for probe results.

    Returns: {dep_level: {task: paired_accuracy}} and overall AVG per group.
    """
    # Collect per-sample paired correctness
    # group -> task -> list of (correct_pairs, total_pairs)
    group_task_results = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for fname in sorted(os.listdir(pred_dir)):
        if not fname.endswith(".jsonl") or fname.endswith(".scored.jsonl"):
            continue
        task = fname.replace(".jsonl", "")
        fpath = os.path.join(pred_dir, fname)

        for line in open(fpath):
            sample = json.loads(line)
            sid = sample["id"]
            dep = dep_index.get(sid, "unknown")

            # Count paired correctness for this sample
            preds = sample.get("predictions", [])
            # Group predictions by gt_idx to form pairs (pre, post)
            by_gt = defaultdict(dict)
            for p in preds:
                gt_idx = p.get("gt_idx", 0)
                probe_type = p.get("probe_type", "")
                by_gt[gt_idx][probe_type] = p.get("correct", False)

            for gt_idx, probes in by_gt.items():
                if "pre" in probes and "post" in probes:
                    group_task_results[dep][task]["total"] += 1
                    if probes["pre"] and probes["post"]:
                        group_task_results[dep][task]["correct"] += 1

    return group_task_results


def compute_online_grouped_metrics(pred_dir, dep_index):
    """Compute joint_f1 grouped by audio_dependency for online results.

    For online mode, we compute micro-averaged joint_f1 per group.
    """
    # group -> {tp_time, tp_content, fp, fn, content_scored}
    group_task_results = defaultdict(lambda: defaultdict(lambda: {
        "tp_time": 0, "tp_content": 0, "fp": 0, "fn": 0, "content_scored": 0,
        "is_time_only": False
    }))

    for fname in sorted(os.listdir(pred_dir)):
        if not fname.endswith(".scored.jsonl"):
            continue
        task = fname.replace(".scored.jsonl", "")
        fpath = os.path.join(pred_dir, fname)

        for line in open(fpath):
            sample = json.loads(line)
            sid = sample["id"]
            dep = dep_index.get(sid, "unknown")
            score = sample.get("score", {})

            r = group_task_results[dep][task]
            r["tp_time"] += score.get("tp_time", 0)
            r["tp_content"] += score.get("tp_content", 0) or 0
            r["fp"] += score.get("fp", 0)
            r["fn"] += score.get("fn", 0)
            r["content_scored"] += score.get("content_scored", 0)
            if score.get("content_kind") == "time_only":
                r["is_time_only"] = True

    return group_task_results


def main():
    data_path = os.path.join(PROJECT_ROOT, "data", "release", "benchmark.json")
    if not os.path.exists(data_path):
        data_path = os.path.join(PROJECT_ROOT, "data", "benchmark.json")

    dep_index = load_benchmark_index(data_path)

    # Full-input (A+V) models to analyze
    probe_models = {
        "Qwen2.5-Omni (A+V)": "results/probe/Qwen2.5-Omni-7B",
        "Qwen3-Omni (A+V)": "results/probe/Qwen3-Omni-30B-A3B-Instruct",
        "video-SALMONN2+ (A+V)": "results/probe/Video-SALMONN2plus-7B",
        "Gemini-3-Flash (A+V)": "results/probe/gemini-3-flash-preview",
    }

    online_models = {
        "MiniCPM-o 4.5 (A+V)": "results/online/MiniCPM-o-4.5-Duplex",
    }

    dep_levels = ["required", "helpful", "none"]

    print("=" * 80)
    print("Audio Dependency Group Analysis (Full A+V input)")
    print("=" * 80)
    print()
    print(f"{'Model':<30s}  {'required':>10s}  {'helpful':>10s}  {'none':>10s}")
    print("-" * 65)

    # Probe models
    for model_name, rel_path in probe_models.items():
        pred_dir = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.isdir(pred_dir):
            print(f"{model_name:<30s}  {'N/A':>10s}  {'N/A':>10s}  {'N/A':>10s}  (not found)")
            continue

        group_results = compute_probe_grouped_metrics(pred_dir, dep_index)

        # Compute AVG (macro avg of per-task paired_accuracy) for each group
        avgs = {}
        for dep in dep_levels:
            task_accs = []
            for task, counts in group_results[dep].items():
                if counts["total"] > 0:
                    acc = counts["correct"] / counts["total"]
                    task_accs.append(acc)
            avgs[dep] = (sum(task_accs) / len(task_accs) * 100) if task_accs else 0.0

        print(f"{model_name:<30s}  {avgs['required']:>10.1f}  {avgs['helpful']:>10.1f}  {avgs['none']:>10.1f}")

    # Online models
    for model_name, rel_path in online_models.items():
        pred_dir = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.isdir(pred_dir):
            print(f"{model_name:<30s}  {'N/A':>10s}  {'N/A':>10s}  {'N/A':>10s}  (not found)")
            continue

        group_results = compute_online_grouped_metrics(pred_dir, dep_index)

        # Compute AVG (macro avg of per-task joint_f1) for each group
        avgs = {}
        for dep in dep_levels:
            task_f1s = []
            for task, r in group_results[dep].items():
                tp_t = r["tp_time"]
                fp = r["fp"]
                fn = r["fn"]
                tp_c = r["tp_content"]

                if r["is_time_only"]:
                    # joint = time for time-only tasks
                    p = tp_t / (tp_t + fp) if (tp_t + fp) > 0 else 0
                    rec = tp_t / (tp_t + fn) if (tp_t + fn) > 0 else 0
                    f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0
                else:
                    # joint uses tp_content
                    p = tp_c / (tp_t + fp) if (tp_t + fp) > 0 else 0
                    rec = tp_c / (tp_t + fn) if (tp_t + fn) > 0 else 0
                    f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0
                task_f1s.append(f1)
            avgs[dep] = (sum(task_f1s) / len(task_f1s) * 100) if task_f1s else 0.0

        print(f"{model_name:<30s}  {avgs['required']:>10.1f}  {avgs['helpful']:>10.1f}  {avgs['none']:>10.1f}")

    print("-" * 65)
    print()
    print("Note: 'required' = audio is essential for the task")
    print("      'helpful'  = audio provides supplementary cues")
    print("      'none'     = task can be solved visually alone")


if __name__ == "__main__":
    main()
