"""Temporal Recall / Precision / F1 for probe-mode polling evaluations.

Greedy 1-to-1 matching between predicted trigger times and GT trigger times,
constrained by a ±tolerance window.
"""

from typing import List


def compute_temporal_metrics(
    pred_times: List[float],
    gt_times: List[float],
    tolerance: float = 3.0,
) -> dict:
    """Compute Recall / Precision / F1 for a single sample."""
    if not gt_times:
        if not pred_times:
            return {"recall": 1.0, "precision": 1.0, "f1": 1.0,
                    "matched": 0, "total_gt": 0, "total_pred": 0}
        return {"recall": 1.0, "precision": 0.0, "f1": 0.0,
                "matched": 0, "total_gt": 0, "total_pred": len(pred_times)}

    if not pred_times:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0,
                "matched": 0, "total_gt": len(gt_times), "total_pred": 0}

    pairs = []
    for gi, gt in enumerate(gt_times):
        for pi, pr in enumerate(pred_times):
            dist = abs(gt - pr)
            if dist <= tolerance:
                pairs.append((dist, gi, pi))

    pairs.sort()
    matched_gt, matched_pred = set(), set()
    for dist, gi, pi in pairs:
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)

    n_matched = len(matched_gt)
    total_gt = len(gt_times)
    total_pred = len(pred_times)

    recall = n_matched / total_gt if total_gt > 0 else 0.0
    precision = n_matched / total_pred if total_pred > 0 else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "matched": n_matched,
        "total_gt": total_gt,
        "total_pred": total_pred,
    }


def extract_pred_trigger_times(predictions: List[dict]) -> List[float]:
    """Extract trigger times from probe predictions."""
    return [p["time_sec"] for p in predictions
            if p.get("parsed", {}).get("triggered")]


def extract_gt_trigger_times(ground_truth: List[dict]) -> List[float]:
    """Extract trigger times from ground truth."""
    return [gt["trigger_time_sec"] for gt in ground_truth]
