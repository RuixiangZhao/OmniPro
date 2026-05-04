"""Score, annotate, and aggregate online-mode prediction jsonl files.

Two modes (auto-detected by --aggregate flag):

1. Single-dir mode (default):
   Reads *.jsonl from a prediction directory, runs greedy 1-to-1 temporal
   matching, writes a companion `*.scored.jsonl` for each task with per-emit
   and per-GT labels inlined, and saves metrics.json in the same directory.

   Example:
     python scripts/compute_online_metrics.py --pred_dir results/online/MiniCPM-o-4.5-Duplex/
     python scripts/compute_online_metrics.py --pred_dir <dir> --tolerance 5
     python scripts/compute_online_metrics.py --pred_dir <dir> --inplace

2. Aggregate mode (--aggregate):
   Walks a parent directory, collects metrics.json from each subdirectory
   that has one, and writes a combined overview metrics JSON.

   Example:
     python scripts/compute_online_metrics.py --aggregate \
         --pred_dir results/online

Scored jsonl fields (per emit):
    parsed            : dict extracted by online_parser
    label             : "TP" | "TP_time_only" | "FP"
    match_gt_idx      : int or null
    time_delta        : |Δt| (seconds) if matched, else null
    content_ok        : bool for structured tasks; null for free-text

Scored jsonl fields (per GT):
    matched_emit_idx  : int or null (null = FN)
    label             : "TP" | "TP_time_only" | "FN"
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from metrics.online.scorer import (
    TASK_CONTENT_KIND,
    match_emits_to_gt,
    _score_content,
)
from utils.online_parser import parse_streaming_output


# Task groups by content-scoring strategy. Also see metrics/online_metrics.py
# TASK_CONTENT_KIND for the canonical mapping.
_TIME_ONLY_TASKS = {t for t, k in TASK_CONTENT_KIND.items() if k == "time_only"}
_GPT_JUDGE_TASKS = {t for t, k in TASK_CONTENT_KIND.items() if k == "gpt_judge"}

# Canonical task display order for combined tables.
_TASK_ORDER = [
    "instant_event_alert",
    "semantic_condition_alert",
    "explicit_target_grounding",
    "snapshot_counting",
    "cumulative_counting",
    "dedup_counting",
    "realtime_state_monitor",
    "event_narration",
    "sequential_step_instruction",
]


# =============================================================================
# Core scoring (single sample)
# =============================================================================

def _label_for_emit(content_ok, matched: bool) -> str:
    if not matched:
        return "FP"
    if content_ok is True:
        return "TP"
    # content_ok False or None (free-text unscored): still time-matched.
    return "TP_time_only"


def _ensure_parsed(emit: dict, task: str) -> dict:
    """Return emit['parsed'] if present, else compute on the fly."""
    if "parsed" in emit and emit["parsed"]:
        return dict(emit["parsed"])
    p = parse_streaming_output(emit.get("raw", ""), task)
    keep = {}
    for k in ("position", "count", "state", "valid"):
        if k in p:
            keep[k] = p[k]
    return keep


def score_sample(sample: dict, tolerance: float,
                 gpt_judge=None, gpt_pass_score: int = 3) -> dict:
    """Annotate a single sample with match labels and sample-level score.

    Content-scoring policy (see TASK_CONTENT_KIND in metrics/online_metrics):
      - time_only (IEA, SCA): content ignored -> content_ok = True for any
        time-matched emit. joint_F1 equals time_F1.
      - gpt_judge (EN, SSI): content_ok filled by ``gpt_judge`` callable if
        provided (expects judge(question, gt_response, pred_response)
        -> {"score": int, ...}; passing score >= gpt_pass_score is True).
        Otherwise left as None.
      - structured (ETG, SOC, CC, DC, RSM): rule-based exact match.

    Returns a copy; does not mutate the input.
    """
    task = sample["task"]
    question = sample.get("question", "")
    emits = [dict(e) for e in sample.get("predictions", [])]
    gts = [dict(g) for g in sample.get("ground_truth", [])]

    # Inline parsed on every emit (useful for inspection of FPs too).
    for e in emits:
        parsed = _ensure_parsed(e, task)
        if parsed:
            e["parsed"] = parsed

    matches, _un_e, _un_g = match_emits_to_gt(emits, gts, tolerance=tolerance)

    for g in gts:
        g["matched_emit_idx"] = None
        g["label"] = "FN"
    for e in emits:
        e["label"] = "FP"
        e["match_gt_idx"] = None
        e["time_delta"] = None
        e["content_ok"] = None

    is_time_only = task in _TIME_ONLY_TASKS
    is_gpt_judge = task in _GPT_JUDGE_TASKS
    use_gpt = is_gpt_judge and gpt_judge is not None
    needs_gpt = is_gpt_judge and not use_gpt  # unfilled

    tp_time = tp_content = 0
    content_scored = 0
    for i, j, dt in matches:
        tp_time += 1
        ok = _score_content(task, emits[i], gts[j], gts)
        judge_info = None

        # For gpt_judge tasks with a judge available, call it now.
        if ok is None and use_gpt:
            gt_resp = gts[j].get("response", "")
            pred_resp = (emits[i].get("parsed", {}).get("payload")
                         or emits[i].get("raw", ""))
            judge_info = gpt_judge(question, gt_resp, pred_resp)
            score = int(judge_info.get("score", 0))
            ok = score >= gpt_pass_score if score > 0 else False

        if ok is None:
            # Still unscored (gpt_judge task without judge) — not counted.
            pass
        else:
            content_scored += 1
            if ok:
                tp_content += 1

        emits[i]["match_gt_idx"] = j
        emits[i]["time_delta"] = round(float(dt), 2)
        emits[i]["content_ok"] = ok
        if judge_info is not None:
            emits[i]["judge"] = judge_info
        emits[i]["label"] = _label_for_emit(ok, matched=True)

        gts[j]["matched_emit_idx"] = i
        gts[j]["label"] = "TP" if ok is True else "TP_time_only"

    fp = sum(1 for e in emits if e["label"] == "FP")
    fn = sum(1 for g in gts if g["label"] == "FN")

    time_p = tp_time / (tp_time + fp) if (tp_time + fp) > 0 else 0.0
    time_r = tp_time / (tp_time + fn) if (tp_time + fn) > 0 else 0.0
    time_f1 = (2 * time_p * time_r / (time_p + time_r)
               if (time_p + time_r) > 0 else 0.0)

    # content_accuracy: only defined when we actually scored content.
    content_acc = (tp_content / content_scored) if content_scored > 0 else None

    # joint_F1:
    #  - time_only tasks: joint == time (content trivially satisfied)
    #  - gpt_judge tasks: None until judge fills content_ok on matches
    #  - structured tasks: content-gated F1 using tp_content
    if is_time_only:
        joint_f1 = time_f1
    elif needs_gpt:
        joint_f1 = None
    else:
        joint_p = tp_content / (tp_time + fp) if (tp_time + fp) > 0 else 0.0
        joint_r = tp_content / (tp_time + fn) if (tp_time + fn) > 0 else 0.0
        joint_f1 = (2 * joint_p * joint_r / (joint_p + joint_r)
                    if (joint_p + joint_r) > 0 else 0.0)

    annotated = dict(sample)
    annotated["predictions"] = emits
    annotated["ground_truth"] = gts
    annotated["score"] = {
        "tolerance": tolerance,
        "tp_time": tp_time,
        "tp_content": tp_content if not needs_gpt else None,
        "fp": fp,
        "fn": fn,
        "content_scored": content_scored,
        "time_precision": round(time_p, 4),
        "time_recall": round(time_r, 4),
        "time_f1": round(time_f1, 4),
        "content_accuracy": (round(content_acc, 4)
                             if content_acc is not None else None),
        "joint_f1": round(joint_f1, 4) if joint_f1 is not None else None,
        "content_kind": TASK_CONTENT_KIND.get(task, "unknown"),
        "used_gpt_judge": use_gpt,
    }
    return annotated


# =============================================================================
# Aggregation helpers
# =============================================================================

def _aggregate_samples(scored: List[dict]) -> dict:
    """Micro-aggregate sample-level scores into a task-level record.

    joint_F1 policy:
      - time_only tasks: joint_F1 = time_F1 (content trivially correct).
      - gpt_judge tasks *without* a judge run: None (awaiting judge).
      - gpt_judge tasks *with* a judge run: content-gated like structured tasks.
      - structured tasks: content-gated joint F1.
    """
    tp_t = sum(s["score"]["tp_time"] for s in scored)
    fp = sum(s["score"]["fp"] for s in scored)
    fn = sum(s["score"]["fn"] for s in scored)
    tp_c = sum((s["score"]["tp_content"] or 0) for s in scored)
    cs = sum(s["score"]["content_scored"] for s in scored)
    content_kind = (scored[0]["score"].get("content_kind", "unknown")
                    if scored else "unknown")
    used_gpt = any(s["score"].get("used_gpt_judge", False) for s in scored)
    is_time_only = content_kind == "time_only"
    # gpt_judge tasks are "pending" (no joint_F1) only when no judge ran.
    is_gpt_pending = (content_kind == "gpt_judge" and not used_gpt)

    time_p = tp_t / (tp_t + fp) if (tp_t + fp) > 0 else 0.0
    time_r = tp_t / (tp_t + fn) if (tp_t + fn) > 0 else 0.0
    time_f1 = (2 * time_p * time_r / (time_p + time_r)
               if (time_p + time_r) > 0 else 0.0)
    content_acc = (tp_c / cs) if cs > 0 else None

    if is_time_only:
        joint_f1 = time_f1
    elif is_gpt_pending:
        joint_f1 = None
    else:
        joint_p = tp_c / (tp_t + fp) if (tp_t + fp) > 0 else 0.0
        joint_r = tp_c / (tp_t + fn) if (tp_t + fn) > 0 else 0.0
        joint_f1 = (2 * joint_p * joint_r / (joint_p + joint_r)
                    if (joint_p + joint_r) > 0 else 0.0)

    return {
        "n": len(scored),
        "tp_time": tp_t,
        "tp_content": tp_c if not is_gpt_pending else None,
        "fp": fp,
        "fn": fn,
        "content_scored": cs,
        "time_precision": round(time_p, 4),
        "time_recall": round(time_r, 4),
        "time_f1": round(time_f1, 4),
        "content_accuracy": (round(content_acc, 4)
                             if content_acc is not None else None),
        "joint_f1": round(joint_f1, 4) if joint_f1 is not None else None,
        "content_kind": content_kind,
        "used_gpt_judge": used_gpt,
    }


def _macro_overall(tasks: Dict[str, dict]) -> dict:
    if not tasks:
        return {}
    tf1 = [a["time_f1"] for a in tasks.values()]
    ca = [a["content_accuracy"] for a in tasks.values()
          if a.get("content_accuracy") is not None]
    jf1 = [a["joint_f1"] for a in tasks.values()
           if a.get("joint_f1") is not None]
    return {
        "num_tasks": len(tasks),
        "macro_time_f1": round(sum(tf1) / len(tf1), 4),
        "macro_content_accuracy": (round(sum(ca) / len(ca), 4)
                                   if ca else None),
        "content_scored_task_count": len(ca),
        "macro_joint_f1": (round(sum(jf1) / len(jf1), 4)
                           if jf1 else None),
        "joint_task_count": len(jf1),
    }


# =============================================================================
# Rendering
# =============================================================================

def _print_table(title: str, tasks: Dict[str, dict], overall: dict):
    print(f"\n{title}\n")
    hdr = (f"{'Task':36s} {'N':>3s} {'tp_t':>4s} {'tp_c':>4s} {'FP':>3s} "
           f"{'FN':>3s}  {'t_F1':>6s} {'content':>8s} {'joint_F1':>9s}")
    print(hdr)
    print("-" * len(hdr))

    ordered = ([t for t in _TASK_ORDER if t in tasks]
               + [t for t in tasks if t not in _TASK_ORDER])
    for task in ordered:
        a = tasks[task]
        kind = a.get("content_kind", "")
        # content column
        if a.get("content_accuracy") is not None:
            content_str = f"{a['content_accuracy']:>8.3f}"
        elif kind == "time_only":
            content_str = f"{'    n/a ':>8s}"
        elif kind == "gpt_judge":
            content_str = f"{' [gpt] ':>8s}"
        else:
            content_str = f"{'  n/a  ':>8s}"
        # joint column
        if a.get("joint_f1") is not None:
            joint_str = f"{a['joint_f1']:>9.3f}"
        elif kind == "gpt_judge":
            joint_str = f"{'  [gpt] ':>9s}"
        else:
            joint_str = f"{'    n/a ':>9s}"
        tp_c = a.get("tp_content")
        tp_c_s = str(tp_c) if tp_c is not None else "-"
        print(f"{task:36s} {a['n']:>3d} {a['tp_time']:>4d} {tp_c_s:>4s} "
              f"{a['fp']:>3d} {a['fn']:>3d}  {a['time_f1']:>6.3f} "
              f"{content_str} {joint_str}")

    print("-" * len(hdr))
    ca = f"{overall['macro_content_accuracy']:.3f}" \
         if overall.get("macro_content_accuracy") is not None else "  n/a "
    jf = f"{overall['macro_joint_f1']:.3f}" \
         if overall.get("macro_joint_f1") is not None else "   n/a "
    print(f"{'MACRO (all)':36s} {overall['num_tasks']:>3d} "
          f"{'':>4s} {'':>4s} {'':>3s} {'':>3s}  "
          f"{overall['macro_time_f1']:>6.3f} {ca:>8s} {jf:>9s}  "
          f"(content:{overall['content_scored_task_count']} tasks, "
          f"joint:{overall['joint_task_count']} tasks)")


# =============================================================================
# Mode 1: single directory (score + annotate + write metrics.json)
# =============================================================================

def run_score_mode(pred_dir: str, tolerance: float, inplace: bool,
                   write_metrics: bool,
                   gpt_judge=None, gpt_pass_score: int = 3) -> dict:
    if not os.path.isdir(pred_dir):
        print(f"[ERROR] not a directory: {pred_dir}")
        sys.exit(1)

    files = sorted(f for f in os.listdir(pred_dir)
                   if f.endswith(".jsonl") and not f.endswith(".scored.jsonl"))
    if not files:
        print(f"[WARN] no *.jsonl found in {pred_dir}")
        return {}

    tasks: Dict[str, dict] = {}
    output_files: List[str] = []
    for fname in files:
        in_path = os.path.join(pred_dir, fname)
        task = fname.replace(".jsonl", "")
        samples = [json.loads(line) for line in open(in_path) if line.strip()]

        # Only pass the judge for gpt_judge tasks (avoid wasting API calls).
        task_judge = (gpt_judge if task in _GPT_JUDGE_TASKS else None)
        if task_judge is not None:
            n_matches_est = sum(min(len(s.get("predictions", [])),
                                     len(s.get("ground_truth", [])))
                                for s in samples)
            print(f"[judge] {task}: scoring ~{n_matches_est} matches via GPT ...")

        scored = [score_sample(s, tolerance,
                               gpt_judge=task_judge,
                               gpt_pass_score=gpt_pass_score)
                  for s in samples]

        out_name = fname if inplace else fname.replace(".jsonl", ".scored.jsonl")
        out_path = os.path.join(pred_dir, out_name)
        with open(out_path, "w") as f:
            for s in scored:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        output_files.append(out_path)

        tasks[task] = _aggregate_samples(scored)

    overall = _macro_overall(tasks)
    _print_table(f"Scored online predictions in {pred_dir}  "
                 f"(tolerance ±{tolerance}s)", tasks, overall)

    print()
    for p in output_files:
        print(f"  -> {p}")

    metrics_path = None
    if write_metrics:
        doc = {
            "pred_dir": os.path.abspath(pred_dir),
            "tolerance": tolerance,
            "tasks": tasks,
            "overall": overall,
            "used_gpt_judge": gpt_judge is not None,
            "gpt_pass_score": gpt_pass_score if gpt_judge is not None else None,
        }
        metrics_path = os.path.join(pred_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"  -> {metrics_path}")

    return {"tasks": tasks, "overall": overall,
            "metrics_path": metrics_path}


# =============================================================================
# Mode 2: aggregate multiple metrics.json files
# =============================================================================

def run_aggregate_mode(parent_dir: str, pattern: str,
                       explicit_dirs: List[str], output: str) -> dict:
    if explicit_dirs:
        dirs = [d for d in explicit_dirs if os.path.isdir(d)]
    else:
        entries = sorted(os.listdir(parent_dir))
        dirs = [os.path.join(parent_dir, e) for e in entries
                if os.path.isdir(os.path.join(parent_dir, e))
                and (pattern is None or pattern in e)
                and os.path.exists(os.path.join(parent_dir, e, "metrics.json"))]

    if not dirs:
        print(f"[ERROR] no matching subdirs with metrics.json under "
              f"{parent_dir or '--dirs'}")
        sys.exit(1)

    merged: Dict[str, dict] = {}
    tol_seen = set()
    for d in dirs:
        mpath = os.path.join(d, "metrics.json")
        with open(mpath) as f:
            m = json.load(f)
        tol_seen.add(m.get("tolerance"))
        for task, stats in m.get("tasks", {}).items():
            if task in merged:
                print(f"[warn] duplicate task {task!r} from {d}, keeping first",
                      file=sys.stderr)
                continue
            merged[task] = {**stats, "source_dir": d}

    overall = _macro_overall(merged)
    doc = {
        "parent_dir": os.path.abspath(parent_dir) if parent_dir else None,
        "tolerance": (next(iter(tol_seen)) if len(tol_seen) == 1
                      else sorted(tol_seen)),
        "n_dirs": len(dirs),
        "tasks": merged,
        "overall": overall,
    }

    if not output:
        output = os.path.join(parent_dir or ".", "aggregated_metrics.json")
    with open(output, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    _print_table(f"Aggregated online metrics  "
                 f"(tolerance ±{doc['tolerance']}s, {doc['n_dirs']} dirs)",
                 merged, overall)
    print(f"\nSaved -> {output}")

    return doc


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Score online predictions and/or aggregate metrics.")
    ap.add_argument("--pred_dir", default=None,
                    help="Prediction directory (single-dir mode) OR parent "
                         "directory (aggregate mode, with --aggregate).")
    ap.add_argument("--tolerance", type=float, default=3.0,
                    help="Temporal matching tolerance seconds (single-dir "
                         "mode only).")
    ap.add_argument("--inplace", action="store_true",
                    help="Overwrite *.jsonl instead of writing *.scored.jsonl "
                         "(single-dir mode only).")
    ap.add_argument("--no_metrics", action="store_true",
                    help="Skip writing metrics.json in single-dir mode.")

    ap.add_argument("--aggregate", action="store_true",
                    help="Aggregate multiple metrics.json files under "
                         "--pred_dir into a combined overview.")
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="Explicit subdir list for aggregate mode.")
    ap.add_argument("--pattern", default=None,
                    help="Only aggregate subdirs whose name contains this "
                         "string (e.g. 'pilot8').")
    ap.add_argument("--output", default=None,
                    help="Output path for aggregate mode's merged JSON. "
                         "Defaults to <pred_dir>/aggregated_metrics.json.")

    # GPT judge options (only used in single-dir mode for EN/SSI tasks).
    ap.add_argument("--gpt_judge", action="store_true",
                    help="Enable LLM-based content judge for free-text "
                         "tasks (event_narration, sequential_step_instruction). "
                         "Requires OPENAI_API_KEY or GEMINI_API_KEY (or --gpt_api_key).")
    ap.add_argument("--gpt_api_key", default=None,
                    help="Override API key (auto-picks provider from env).")
    ap.add_argument("--gpt_api_base", default=None,
                    help="Override API base URL.")
    ap.add_argument("--gpt_model", default=None,
                    help="Override model name.")
    ap.add_argument("--gpt_pass_score", type=int, default=3,
                    help="Minimum judge score (1-5) to count as correct. "
                         "Default 3.")
    args = ap.parse_args()

    if args.aggregate:
        if not args.pred_dir and not args.dirs:
            ap.error("--aggregate requires --pred_dir or --dirs")
        run_aggregate_mode(args.pred_dir, args.pattern,
                           args.dirs or [], args.output)
    else:
        if not args.pred_dir:
            ap.error("single-dir mode requires --pred_dir")

        judge = None
        if args.gpt_judge:
            from metrics.llm_judge import LLMJudge
            gj = LLMJudge(api_key=args.gpt_api_key,
                          api_base=args.gpt_api_base,
                          model=args.gpt_model)
            judge = gj.judge  # callable(question, gt_resp, pred_resp) -> dict
            print(f"[judge] provider={gj.provider}  model={gj.model}  "
                  f"base={gj.api_base}  pass_score>={args.gpt_pass_score}")

        run_score_mode(args.pred_dir, args.tolerance, args.inplace,
                       write_metrics=not args.no_metrics,
                       gpt_judge=judge,
                       gpt_pass_score=args.gpt_pass_score)


if __name__ == "__main__":
    main()
