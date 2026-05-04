"""
独立计算指标脚本，基于已有推理结果。

Usage:
    python scripts/compute_metrics.py \
        --pred_dir results/probe/Qwen3-VL-8B-Instruct/ \
        --tolerance 3,5 \
        --use_gpt_judge \
        --output results/probe/Qwen3-VL-8B-Instruct/metrics.json
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute OmniProact-Bench metrics")

    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing prediction JSONL files")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task names (default: all found)")
    parser.add_argument("--tolerance", type=str, default="3,5",
                        help="Comma-separated time tolerance values in seconds")
    parser.add_argument("--use_gpt_judge", action="store_true",
                        help="Enable GPT-4o judge for response quality")
    parser.add_argument("--gpt_api_key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env)")
    parser.add_argument("--gpt_api_base", type=str, default=None,
                        help="OpenAI API base URL (or set OPENAI_API_BASE env)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: <pred_dir>/metrics.json)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse tolerances
    tolerances = [float(x) for x in args.tolerance.split(",")]

    # Parse tasks
    tasks = args.tasks.split(",") if args.tasks else None

    # Load predictions
    from utils.io import load_predictions
    predictions = load_predictions(args.pred_dir, tasks=tasks)
    print(f"Loaded {len(predictions)} predictions from {args.pred_dir}")

    if not predictions:
        print("No predictions found.")
        sys.exit(1)

    # Count by task
    from collections import Counter
    task_counts = Counter(p["task"] for p in predictions)
    print("Per-task counts:")
    for t, c in sorted(task_counts.items()):
        print(f"  {t:35s} {c:>6d}")

    # Setup LLM judge if needed (auto-picks OpenAI / Gemini from env vars,
    # or honours --gpt_api_key / --gpt_api_base).
    gpt_judge = None
    if args.use_gpt_judge:
        from metrics.llm_judge import LLMJudge
        gpt_judge = LLMJudge(
            api_key=args.gpt_api_key,
            api_base=args.gpt_api_base,
        )
        print(f"LLM judge enabled  provider={gpt_judge.provider}  "
              f"model={gpt_judge.model}")

    # Compute metrics
    from metrics.probe import compute_all_metrics
    print(f"\nComputing metrics (tolerances: {tolerances}s)...")
    results = compute_all_metrics(
        predictions,
        tolerances=tolerances,
        use_gpt_judge=args.use_gpt_judge,
        gpt_judge=gpt_judge,
    )

    # Output path
    if args.output is None:
        args.output = os.path.join(args.pred_dir, "metrics.json")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to: {args.output}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Task':<35} {'N':>5}", end="")
    for tol in tolerances:
        print(f"  {'F1@' + str(int(tol)) + 's':>8}", end="")
    # Content metric column
    print(f"  {'Content':>10}")
    print("-" * 90)

    for task in sorted(results.keys()):
        if task == "overall":
            continue
        r = results[task]
        row = f"{task:<35} {r['num_samples']:>5}"

        if "paired_accuracy" in r:
            # all gt_probe tasks: show paired_acc + pre/post diagnostic
            row += f"  {r['paired_accuracy']:>8.4f}  (pre={r['pre_accuracy']:.2f} post={r['post_accuracy']:.2f} pairs={r['correct_pairs']}/{r['total_pairs']})"
        else:
            for tol in tolerances:
                key = f"temporal_f1@{tol:.0f}s"
                val = r.get(key, 0)
                row += f"  {val:>8.4f}"

        # Content metric
        if "count_accuracy" in r:
            row += f"  cnt={r['count_accuracy']:.4f}"
        elif "position_accuracy" in r:
            row += f"  pos={r['position_accuracy']:.4f} ({r['position_correct']}/{r['position_total']})"
        elif "gpt_judge_score" in r:
            row += f"  gpt={r['gpt_judge_score']:.2f}/5"
        print(row)

    # Overall
    print("-" * 90)
    r = results.get("overall", {})
    row = f"{'OVERALL':<35} {'':>5}  avg_f1={r.get('avg_f1', 0):.4f}"
    print(row)
    print("=" * 90)


if __name__ == "__main__":
    main()
