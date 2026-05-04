"""
Probe 评测入口脚本。支持多卡并行。

Usage:
    # 单卡
    python scripts/run_probe.py \
        --model qwen3-vl \
        --model_path /path/to/model \
        --tasks instant_event_alert \
        --output_dir results/probe/Qwen3-VL-8B-Instruct/

    # 多卡并行
    python scripts/run_probe.py \
        --model qwen2.5-omni \
        --model_path /path/to/model \
        --tasks instant_event_alert \
        --num_gpus 4 \
        --output_dir results/probe/Qwen2.5-Omni-7B/
"""

import argparse
import math
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="OmniProact-Bench Probe Evaluation")

    # Model
    parser.add_argument("--model", type=str, default="qwen3-vl",
                        choices=["qwen3-vl", "qwen2.5-omni", "qwen2.5-omni-noaudio",
                                 "qwen2.5-omni-audioonly",
                                 "qwen3-omni", "qwen3-omni-noaudio",
                                 "qwen3-omni-audioonly",
                                 "gemini-3-flash", "gemini-3-flash-noaudio",
                                 "gemini-3-flash-audioonly",
                                 "video-salmonn2+", "video-salmonn2+-noaudio",
                                 "video-salmonn2+-audioonly",
                                 "videollama2-av", "videollama2-av-noaudio",
                                 "phi4-multimodal", "phi4-multimodal-noaudio",
                                 "internvl3-8b",
                                 "livestar",
                                 "mmduet2",
                                 "minicpm-o", "minicpm-o-noaudio",],
                        help="Model to evaluate")
    parser.add_argument("--model_path", type=str, required=False, default=None,
                        help="Path to model checkpoint (not needed for API models)")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Video sampling FPS for model (default: 1.0)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens for model generation")

    # Data
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to benchmark.json (default: data/benchmark.json)")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task names (default: all)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit samples per task (0 = all)")

    # Evaluation config
    parser.add_argument("--poll_interval", type=int, default=5,
                        help="Polling interval in seconds (default: 5)")
    parser.add_argument("--tolerance_after", type=int, default=10,
                        help="Continue polling N seconds after last GT (default: 10)")
    parser.add_argument("--clip_cache_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "clip_cache"),
                        help="Directory to cache clipped videos")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for GT-probe time selection (default: 42)")

    # Parallelism
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs for parallel evaluation (default: 1)")

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save results")
    parser.add_argument("--no_resume", action="store_true",
                        help="Do not resume from existing results")

    return parser.parse_args()


def build_model(args, device: str):
    """Build model instance on a specific device."""
    if args.model == "qwen3-vl":
        from models.qwen3_vl import Qwen3VL
        return Qwen3VL(
            model_path=args.model_path,
            device=device,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.model == "qwen2.5-omni":
        from models.qwen2_5_omni import Qwen2_5Omni
        return Qwen2_5Omni(
            model_path=args.model_path,
            device=device,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
            use_audio=True,
        )
    elif args.model == "qwen2.5-omni-noaudio":
        from models.qwen2_5_omni import Qwen2_5Omni_NoAudio
        return Qwen2_5Omni_NoAudio(
            model_path=args.model_path,
            device=device,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.model == "qwen2.5-omni-audioonly":
        from models.qwen2_5_omni import Qwen2_5Omni_AudioOnly
        return Qwen2_5Omni_AudioOnly(
            model_path=args.model_path,
            device=device,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.model == "qwen3-omni":
        from models.qwen3_omni import Qwen3Omni
        return Qwen3Omni(
            model_path=args.model_path,
            device=device,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
            use_audio=True,
        )
    elif args.model == "qwen3-omni-noaudio":
        from models.qwen3_omni import Qwen3Omni_NoAudio
        return Qwen3Omni_NoAudio(
            model_path=args.model_path,
            device=device,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.model == "qwen3-omni-audioonly":
        from models.qwen3_omni import Qwen3Omni_AudioOnly
        return Qwen3Omni_AudioOnly(
            model_path=args.model_path,
            device=device,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.model == "gemini-3-flash":
        from models.gemini import Gemini3Flash
        return Gemini3Flash(
            max_new_tokens=args.max_new_tokens,
        )
    elif args.model == "gemini-3-flash-noaudio":
        from models.gemini import Gemini3Flash_NoAudio
        return Gemini3Flash_NoAudio(
            max_new_tokens=args.max_new_tokens,
        )
    elif args.model == "gemini-3-flash-audioonly":
        from models.gemini import Gemini3Flash_AudioOnly
        return Gemini3Flash_AudioOnly(
            max_new_tokens=args.max_new_tokens,
        )
    elif args.model == "video-salmonn2+":
        from models.video_salmonn2 import VideoSALMONN2Plus
        return VideoSALMONN2Plus(
            model_path=args.model_path,
            device=device,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
            use_audio=True,
        )
    elif args.model == "video-salmonn2+-noaudio":
        from models.video_salmonn2 import VideoSALMONN2Plus_NoAudio
        return VideoSALMONN2Plus_NoAudio(
            model_path=args.model_path,
            device=device,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.model == "video-salmonn2+-audioonly":
        from models.video_salmonn2 import VideoSALMONN2Plus_AudioOnly
        return VideoSALMONN2Plus_AudioOnly(
            model_path=args.model_path,
            device=device,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.model == "videollama2-av":
        from models.videollama2_av import VideoLLaMA2AV
        return VideoLLaMA2AV(
            model_path=args.model_path,
            device=device,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
            use_audio=True,
        )
    elif args.model == "videollama2-av-noaudio":
        from models.videollama2_av import VideoLLaMA2AV_NoAudio
        return VideoLLaMA2AV_NoAudio(
            model_path=args.model_path,
            device=device,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.model == "phi4-multimodal":
        from models.phi4_multimodal import Phi4Multimodal
        return Phi4Multimodal(
            model_path=args.model_path,
            device=device,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
            use_audio=True,
        )
    elif args.model == "phi4-multimodal-noaudio":
        from models.phi4_multimodal import Phi4Multimodal_NoAudio
        return Phi4Multimodal_NoAudio(
            model_path=args.model_path,
            device=device,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.model == "internvl3-8b":
        from models.internvl3 import InternVL3
        return InternVL3(
            model_path=args.model_path,
            device=device,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.model == "livestar":
        from models.livestar_probe import LiveStarProbe
        return LiveStarProbe(
            model_path=args.model_path,
            device=device,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.model == "mmduet2":
        from models.mmduet2_probe import MMDuet2Probe
        return MMDuet2Probe(
            model_path=args.model_path,
            device=device,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.model == "minicpm-o":
        from models.minicpm_o_probe import MiniCPMOProbe
        return MiniCPMOProbe(
            model_path=args.model_path,
            device=device,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
            use_audio=True,
        )
    elif args.model == "minicpm-o-noaudio":
        from models.minicpm_o_probe import MiniCPMOProbe_NoAudio
        return MiniCPMOProbe_NoAudio(
            model_path=args.model_path,
            device=device,
            fps=args.fps,
            max_new_tokens=args.max_new_tokens,
        )

    else:
        raise ValueError(f"Unknown model: {args.model}")


def gpu_worker(gpu_id, args, samples):
    """Worker function: load model on one GPU (or call API) and evaluate assigned samples."""
    is_api = args.model.startswith("gemini")
    if not is_api:
        import torch
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = "cuda:0"
        tag = f"[GPU {gpu_id}]"
    else:
        device = None
        tag = f"[Worker {gpu_id}]"

    print(f"{tag} Loading model, {len(samples)} samples...")
    model = build_model(args, device=device)

    from evaluators.probe_evaluator import ProbeEvaluator
    evaluator = ProbeEvaluator(
        model=model,
        poll_interval=args.poll_interval,
        tolerance_after=args.tolerance_after,
        clip_cache_dir=args.clip_cache_dir,
        seed=args.seed,
    )

    preds = evaluator.evaluate(
        samples,
        output_dir=args.output_dir,
        resume=not args.no_resume,
    )
    print(f"{tag} Done! {len(preds)} predictions.")
    return preds


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


def main():
    args = parse_args()

    # Default data path
    if args.data_path is None:
        args.data_path = os.path.join(PROJECT_ROOT, "data", "benchmark.json")

    if not os.path.exists(args.data_path):
        print(f"ERROR: Data file not found: {args.data_path}")
        print("Clone the data repo into data/ first. See README.md for instructions.")
        sys.exit(1)

    tasks = args.tasks.split(",") if args.tasks else None

    from utils.io import load_benchmark, get_completed_ids
    dataset = load_benchmark(args.data_path, tasks=tasks)
    print(f"Loaded {len(dataset)} samples" + (f" (tasks: {tasks})" if tasks else ""))

    # Apply limit
    if args.limit > 0:
        from collections import defaultdict
        by_task = defaultdict(list)
        for s in dataset:
            by_task[s["task"]].append(s)
        limited = []
        for task_samples in by_task.values():
            limited.extend(task_samples[:args.limit])
        dataset = limited
        print(f"Limited to {len(dataset)} samples ({args.limit} per task)")

    if not dataset:
        print("No samples to evaluate.")
        sys.exit(1)

    # Filter already completed (before splitting)
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.no_resume:
        completed = get_completed_ids(args.output_dir)
        before = len(dataset)
        dataset = [s for s in dataset if s["id"] not in completed]
        if before > len(dataset):
            print(f"Resuming: {before - len(dataset)} already completed, {len(dataset)} remaining")

    if not dataset:
        print("All samples already completed.")
        sys.exit(0)

    num_gpus = min(args.num_gpus, len(dataset))

    if num_gpus <= 1:
        # Single GPU: run directly
        print(f"Single GPU mode")
        model = build_model(args, device="auto")
        print(f"Model loaded: {model.name()}")

        from evaluators.probe_evaluator import ProbeEvaluator
        evaluator = ProbeEvaluator(
            model=model,
            poll_interval=args.poll_interval,
            tolerance_after=args.tolerance_after,
            clip_cache_dir=args.clip_cache_dir,
            seed=args.seed,
        )

        print(f"\nStarting probe evaluation ({len(dataset)} samples):")
        print(f"  Output: {args.output_dir}")
        predictions = evaluator.evaluate(dataset, output_dir=args.output_dir, resume=False)
        print(f"\nDone! {len(predictions)} predictions saved to {args.output_dir}")

    else:
        # Multi-worker: spawn workers (torch.multiprocessing for CUDA,
        # standard multiprocessing for API models)
        is_api = args.model.startswith("gemini")
        if is_api:
            import multiprocessing as mp
        else:
            import torch.multiprocessing as mp
            mp.set_start_method("spawn", force=True)

        chunks = split_into_chunks(dataset, num_gpus)
        mode = "workers" if is_api else "GPUs"
        print(f"\nMulti-{mode[:-1]} mode: {num_gpus} {mode}, {[len(c) for c in chunks]} samples each")
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

    print("Run `python scripts/compute_metrics.py` to calculate metrics.")


if __name__ == "__main__":
    main()
