"""Counting accuracy for probe-mode predictions.

For each GT trigger, finds the closest triggered prediction within tolerance,
then compares the extracted integer count.
"""

from typing import List

from .matching import extract_gt_trigger_times


def count_accuracy(pred_count, gt_count: int) -> float:
    """1.0 if pred_count equals gt_count, else 0.0."""
    if pred_count is None:
        return 0.0
    return 1.0 if pred_count == gt_count else 0.0


def evaluate_counting_sample(predictions: List[dict],
                             ground_truth: List[dict],
                             tolerance: float = 3.0) -> dict:
    """Evaluate counting accuracy over matched GT-prediction pairs."""
    _ = extract_gt_trigger_times(ground_truth)  # (kept for import symmetry)
    pred_triggered = [p for p in predictions
                      if p.get("parsed", {}).get("triggered")]

    if not pred_triggered or not ground_truth:
        return {"count_accuracy": 0.0, "matched_pairs": 0,
                "total_gt": len(ground_truth)}

    pairs = []
    for gi, gt in enumerate(ground_truth):
        gt_t = gt["trigger_time_sec"]
        for pi, pred in enumerate(pred_triggered):
            dist = abs(gt_t - pred["time_sec"])
            if dist <= tolerance:
                pairs.append((dist, gi, pi))
    pairs.sort()

    matched_gt, matched_pred = set(), set()
    count_correct = 0
    matched_pairs = 0
    for dist, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matched_pairs += 1
        gt_count = ground_truth[gi].get("count")
        pred_count = pred_triggered[pi].get("parsed", {}).get("count")
        if gt_count is not None and pred_count is not None \
                and pred_count == gt_count:
            count_correct += 1

    acc = count_correct / len(ground_truth) if ground_truth else 0.0
    return {
        "count_accuracy": round(acc, 4),
        "count_correct": count_correct,
        "matched_pairs": matched_pairs,
        "total_gt": len(ground_truth),
    }
