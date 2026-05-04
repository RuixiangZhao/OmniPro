"""数据加载和结果保存工具。"""

import fcntl
import json
import os
from typing import List, Optional


def load_benchmark(data_path: str, tasks: Optional[List[str]] = None) -> List[dict]:
    """
    Load benchmark data, optionally filtered by task names.

    Args:
        data_path: Path to benchmark.json
        tasks: List of task names to include. None means all.

    Returns:
        List of sample dicts.
    """
    with open(data_path) as f:
        data = json.load(f)

    if tasks:
        data = [s for s in data if s["task"] in tasks]

    return data


def save_prediction(pred: dict, output_dir: str):
    """
    Append one prediction to task-specific JSONL file.
    Uses file lock for multi-process safety.

    Args:
        pred: Prediction dict (must contain 'task' and 'id').
        output_dir: Directory to save results.
    """
    os.makedirs(output_dir, exist_ok=True)
    task = pred["task"]
    fpath = os.path.join(output_dir, f"{task}.jsonl")
    line = json.dumps(pred, ensure_ascii=False) + "\n"
    with open(fpath, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        fcntl.flock(f, fcntl.LOCK_UN)


def load_predictions(pred_dir: str, tasks: Optional[List[str]] = None) -> List[dict]:
    """
    Load all predictions from a results directory.

    Args:
        pred_dir: Directory containing per-task JSONL files.
        tasks: Task names to load. None means all found files.

    Returns:
        List of prediction dicts.
    """
    preds = []
    for fname in sorted(os.listdir(pred_dir)):
        if not fname.endswith(".jsonl"):
            continue
        task_name = fname.replace(".jsonl", "")
        if tasks and task_name not in tasks:
            continue
        fpath = os.path.join(pred_dir, fname)
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    preds.append(json.loads(line))
    return preds


def get_completed_ids(output_dir: str) -> set:
    """Get set of sample IDs that already have predictions (for resuming)."""
    completed = set()
    if not os.path.isdir(output_dir):
        return completed
    for fname in os.listdir(output_dir):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(output_dir, fname)
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        pred = json.loads(line)
                        completed.add(pred["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return completed
