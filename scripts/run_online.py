"""Online (true-streaming) evaluation entry point. Supports multi-GPU parallel.

Example:
    # Perfect-cheat dummy
    python scripts/run_online.py \
        --model dummy-perfect \
        --tasks instant_event_alert --limit 3 \
        --output_dir results/online/dummy-perfect/

    # MiniCPM-o single GPU
    python scripts/run_online.py \
        --model minicpm-o \
        --model_path /path/to/pretrained_models/MiniCPM-o-4_5 \
        --limit 3 \
        --output_dir results/online/MiniCPM-o-4.5/

    # MiniCPM-o 4-GPU parallel
    python scripts/run_online.py \
        --model minicpm-o \
        --model_path /path/to/pretrained_models/MiniCPM-o-4_5 \
        --num_gpus 4 \
        --output_dir results/online/MiniCPM-o-4.5/
"""

import argparse
import math
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="OmniProact-Bench Online Evaluation")
    p.add_argument("--model", required=True,
                   choices=["dummy-perfect", "dummy-noisy", "minicpm-o",
                            "minicpm-o-noaudio", "minicpm-o-audioonly",
                            "livestar", "mmduet2"],
                   help="Online-capable model.")
    p.add_argument("--model_path", default=None,
                   help="Model checkpoint path (required for real models)")
    p.add_argument("--data_path", default=None)
    p.add_argument("--tasks", default=None, help="Comma-separated task list.")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--max_new_speak_tokens", type=int, default=None,
                   help="Max tokens per streaming_generate / chat call. "
                        "Default: 20 (MiniCPM-o), 128 (LiveStar).")
    # LiveStar-specific knobs
    p.add_argument("--decode_factor", type=float, default=1.04,
                   help="LiveStar SVeD threshold multiplier. Higher = "
                        "more silent. Official defaults: 1.04 (eval) / "
                        "1.06 (demo).")
    p.add_argument("--max_frames_per_segment", type=int, default=60,
                   help="LiveStar hard-reset interval (frames). Caps the "
                        "O(N^2) cumulative-forward cost on long videos. "
                        "Default 60 (=1 min at 1 fps).")
    p.add_argument("--speak_threshold", type=float, default=0.725,
                   help="LiveStar SVeD threshold. Lower = more emits. "
                        "Default 0.725.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--no_resume", action="store_true")
    # Parallelism
    p.add_argument("--num_gpus", type=int, default=1,
                   help="Number of GPUs for parallel evaluation (default: 1)")
    # dummy-noisy knobs
    p.add_argument("--miss_rate", type=float, default=0.2)
    p.add_argument("--shift_max", type=float, default=2.0)
    p.add_argument("--wrong_content_rate", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_model(args, device: str):
    if args.model == "dummy-perfect":
        from models.dummy_streaming import DummyStreamingPerfect
        return DummyStreamingPerfect()

    if args.model == "dummy-noisy":
        from models.dummy_streaming import DummyStreamingNoisy
        return DummyStreamingNoisy(
            miss_rate=args.miss_rate,
            shift_max=args.shift_max,
            wrong_content_rate=args.wrong_content_rate,
            seed=args.seed,
        )

    if args.model == "minicpm-o":
        if not args.model_path:
            raise ValueError("--model_path required for minicpm-o")
        from models.minicpm_o import MiniCPMOStreaming
        return MiniCPMOStreaming(
            model_path=args.model_path,
            device=device,
            max_new_speak_tokens=args.max_new_speak_tokens if args.max_new_speak_tokens is not None else 20,
        )

    if args.model == "minicpm-o-noaudio":
        if not args.model_path:
            raise ValueError("--model_path required for minicpm-o-noaudio")
        from models.minicpm_o import MiniCPMOStreamingNoAudio
        return MiniCPMOStreamingNoAudio(
            model_path=args.model_path,
            device=device,
            max_new_speak_tokens=args.max_new_speak_tokens if args.max_new_speak_tokens is not None else 20,
        )

    if args.model == "minicpm-o-audioonly":
        if not args.model_path:
            raise ValueError("--model_path required for minicpm-o-audioonly")
        from models.minicpm_o import MiniCPMOStreamingAudioOnly
        return MiniCPMOStreamingAudioOnly(
            model_path=args.model_path,
            device=device,
            max_new_speak_tokens=args.max_new_speak_tokens if args.max_new_speak_tokens is not None else 20,
        )

    if args.model == "livestar":
        if not args.model_path:
            raise ValueError("--model_path required for livestar")
        from models.livestar import LiveStarStreaming
        kwargs = dict(
            model_path=args.model_path,
            device=device,
            decode_factor=args.decode_factor,
            max_frames_per_segment=args.max_frames_per_segment,
        )
        if args.max_new_speak_tokens is not None:
            kwargs["max_new_tokens"] = args.max_new_speak_tokens
        return LiveStarStreaming(**kwargs)




    if args.model == "mmduet2":
        if not args.model_path:
            raise ValueError("--model_path required for mmduet2")
        from models.mmduet2 import MMDuet2Streaming
        kwargs = dict(
            model_path=args.model_path,
            device=device,
        )
        if args.max_new_speak_tokens is not None:
            kwargs["max_new_tokens"] = args.max_new_speak_tokens
        return MMDuet2Streaming(**kwargs)


    raise ValueError(f"Unknown online model: {args.model}")


def split_into_chunks(lst, n):
    """Split list into *n* chunks of near-equal size.

    Uses numpy.array_split semantics: the first ``len(lst) % n`` chunks get
    one extra item. Unlike ``ceil(len/n)`` splitting, this always returns
    exactly ``n`` chunks (some possibly empty if len < n) and leaves no
    GPU idle when len > n.

    Example: 18 items into 8 chunks -> sizes [3,3,3,2,2,2,2,1]  (len=8, not 6).
    """
    n = max(1, n)
    q, r = divmod(len(lst), n)
    chunks = []
    start = 0
    for i in range(n):
        size = q + (1 if i < r else 0)
        chunks.append(lst[start:start + size])
        start += size
    return chunks


def gpu_worker(gpu_id, args, samples):
    """Worker: load model on one GPU and evaluate assigned samples."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    tag = f"[GPU {gpu_id}]"

    print(f"{tag} Loading model, {len(samples)} samples...")
    model = build_model(args, device=device)

    from evaluators.online_evaluator import OnlineEvaluator
    evaluator = OnlineEvaluator(model=model, fps=args.fps)

    preds = evaluator.evaluate(
        samples,
        output_dir=args.output_dir,
        resume=not args.no_resume,
    )
    print(f"{tag} Done! {len(preds)} predictions.")
    return preds


def load_and_filter_dataset(args):
    """Load benchmark, apply task filter and limit."""
    if args.data_path is None:
        args.data_path = os.path.join(PROJECT_ROOT, "data", "benchmark.json")
    if not os.path.exists(args.data_path):
        print(f"ERROR: benchmark not found at {args.data_path}")
        sys.exit(1)

    tasks = args.tasks.split(",") if args.tasks else None
    from utils.io import load_benchmark
    dataset = load_benchmark(args.data_path, tasks=tasks)
    print(f"Loaded {len(dataset)} samples" +
          (f" (tasks: {tasks})" if tasks else ""))

    if args.limit > 0:
        from collections import defaultdict
        by_task = defaultdict(list)
        for s in dataset:
            by_task[s["task"]].append(s)
        limited = []
        for task_samples in by_task.values():
            limited.extend(task_samples[:args.limit])
        dataset = limited
        print(f"Limited to {len(dataset)} samples ({args.limit}/task)")

    # Filter already completed (before splitting)
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.no_resume:
        from utils.io import get_completed_ids
        completed = get_completed_ids(args.output_dir)
        before = len(dataset)
        dataset = [s for s in dataset if s["id"] not in completed]
        if before > len(dataset):
            print(f"Resuming: {before - len(dataset)} already completed, "
                  f"{len(dataset)} remaining")

    return dataset


def main():
    args = parse_args()
    dataset = load_and_filter_dataset(args)

    if not dataset:
        print("No samples to evaluate.")
        sys.exit(0)

    num_gpus = min(args.num_gpus, len(dataset))

    if num_gpus <= 1:
        # Single GPU mode
        model = build_model(args, device="cuda:0")
        print(f"Model: {model.name()}")
        print(f"  accepts_audio: {model.accepts_audio}")
        print(f"  fps: {args.fps}")

        from evaluators.online_evaluator import OnlineEvaluator
        evaluator = OnlineEvaluator(model=model, fps=args.fps)

        preds = evaluator.evaluate(
            dataset,
            output_dir=args.output_dir,
            resume=False,  # already filtered above
        )
        print(f"\nDone. {len(preds)} samples -> {args.output_dir}")

    else:
        # Multi-GPU parallel
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)

        chunks = split_into_chunks(dataset, num_gpus)
        print(f"\nMulti-GPU mode: {num_gpus} GPUs, "
              f"{[len(c) for c in chunks]} samples each")
        print(f"  Output: {args.output_dir}")

        processes = []
        for worker_id, chunk in enumerate(chunks):
            if not chunk:
                continue  # skip empty chunks (dataset smaller than num_gpus)
            p = mp.Process(target=gpu_worker, args=(worker_id, args, chunk))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        total = sum(
            1 for f in os.listdir(args.output_dir) if f.endswith(".jsonl")
            for _ in open(os.path.join(args.output_dir, f))
        )
        print(f"\nAll GPUs done! Total predictions in {args.output_dir}: {total}")

    print(f"Compute metrics with:\n"
          f"  python scripts/compute_online_metrics.py "
          f"--pred_dir {args.output_dir} --tolerance 3")


if __name__ == "__main__":
    main()
