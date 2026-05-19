# OmniPro

**A Comprehensive Benchmark for Omni-Proactive Streaming Video Understanding**

<p align="center">
  <a href="https://arxiv.org/abs/2605.18577"><img src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=red&logo=arxiv"></a>
  <a href="https://ruixiangzhao.github.io/OmniPro/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/omniproact-bench/OmniPro"><img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-yellow" alt="Dataset"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

OmniPro evaluates multimodal models on their ability to proactively interact with streaming video — detecting events, monitoring states, counting objects, and providing timely narrations without explicit user queries at each moment.

## 📋 Overview

OmniPro consists of **9 evaluation tasks** across two modes:

### Tasks

| Task                        | Abbr.         | Type        | Description                       |
| --------------------------- | ------------- | ----------- | --------------------------------- |
| Instant Event Alert         | Event-Alert   | Alert       | Detect and report specific events |
| Semantic Condition Alert    | Cond.-Alert   | Alert       | Monitor for semantic conditions   |
| Explicit Target Grounding   | Target-Ground | Grounding   | Locate targets when events occur  |
| Snapshot Counting           | Snap.-Count   | Counting    | Count objects at trigger moments  |
| Cumulative Counting         | Cum.-Count    | Counting    | Track cumulative event counts     |
| Dedup Counting              | Dedup.-Count  | Counting    | Count unique instances            |
| Realtime State Monitor      | State-Monitor | Monitor     | Track state changes               |
| Event Narration             | Event-Narr.   | Narration   | Narrate events as they happen     |
| Sequential Step Instruction | Step-Inst.    | Instruction | Guide through procedures          |

### Evaluation Modes

- **Probe Mode**: Model receives a video clip up to time *t* and answers whether an event has occurred. Tests temporal awareness and content understanding.
- **Online Mode**: Model processes video frame-by-frame in real-time and autonomously decides *when* to speak and *what* to say.

## 🏗️ Project Structure

```
OmniPro/
├── models/                 # Model adapters (probe + streaming)
│   ├── base.py            # BaseModel abstract class
│   ├── streaming_base.py  # StreamingModel abstract class
│   ├── qwen3_vl.py        # Qwen3-VL probe adapter
│   ├── qwen2_5_omni.py    # Qwen2.5-Omni (audio+visual)
│   ├── qwen3_omni.py      # Qwen3-Omni
│   ├── internvl3.py       # InternVL3.5-8B
│   ├── phi4_multimodal.py # Phi-4-multimodal
│   ├── video_salmonn2.py  # Video-SALMONN2+
│   ├── videollama2_av.py  # VideoLLaMA2.1-7B-AV
│   ├── livestar.py        # LiveStar-8B (online)
│   ├── livestar_probe.py  # LiveStar-8B (probe)
│   ├── mmduet2.py         # MMDuet2 (online)
│   ├── mmduet2_probe.py   # MMDuet2 (probe)
│   ├── minicpm_o.py       # MiniCPM-o 4.5 (online)
│   ├── minicpm_o_probe.py # MiniCPM-o 4.5 (probe)
│   ├── gemini.py          # Gemini-3-Flash (API)
│   └── ...
├── evaluators/             # Evaluation engines
│   ├── probe_evaluator.py # GT-probe evaluation logic
│   └── online_evaluator.py# Frame-by-frame streaming evaluation
├── metrics/                # Scoring and metrics
│   ├── probe/             # Probe metrics (paired accuracy, F1)
│   └── online/            # Online metrics (time F1, content accuracy)
├── utils/                  # Utilities
│   ├── prompts.py         # Task-specific prompt templates
│   ├── video.py           # Video splitting/processing
│   ├── io.py              # Data I/O
│   └── online_parser.py   # Response parsing
├── scripts/                # Run scripts (one-click evaluation)
│   ├── run_probe.py       # Probe evaluation entry point
│   ├── run_online.py      # Online evaluation entry point
│   ├── compute_metrics.py # Compute probe metrics
│   ├── compute_online_metrics.py # Compute online metrics
│   ├── run_probe_*.sh     # Per-model probe scripts
│   └── run_online_*.sh    # Per-model online scripts
├── data/
│   └── benchmark.json     # Benchmark annotations (2700 samples)
├── third_party/            # Third-party model code (see README inside)
├── visualization/          # Demo and visualization
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repo_url>
cd OmniPro

# Install base dependencies
pip install -r requirements.txt

# Clone third-party model repos (see third_party/README.md)
```

### 2. Data Preparation

Clone the data repository (videos + metadata) into the `data/` directory:

```bash
# Clone the benchmark data (videos, annotations)
cd data
git clone https://huggingface.co/datasets/omniproact-bench/OmniPro .
cd ..
```

The `data/` directory should contain:

- `benchmark.json` — Benchmark annotations (2700 samples, 9 tasks × 300)
- `raw_videos/` — Source video files referenced by `benchmark.json`

### 3. Run Probe Evaluation

```bash
# Example: Qwen3-VL-8B
bash scripts/run_probe_qwen3_vl.sh

# Quick smoke test (2 samples/task)
LIMIT=2 bash scripts/run_probe_qwen3_vl.sh

# Custom: specific tasks, limited samples
python scripts/run_probe.py \
    --model qwen3-vl \
    --model_path /path/to/model \
    --tasks instant_event_alert,event_narration \
    --limit 50 \
    --num_gpus 8 \
    --output_dir results/probe/Qwen3-VL-8B/
```

### 4. Run Online Evaluation

```bash
# Example: MiniCPM-o 4.5
bash scripts/run_online_minicpmo.sh

# Quick test
LIMIT=4 bash scripts/run_online_minicpmo.sh
```

### 5. Compute Metrics

```bash
# Probe metrics
python scripts/compute_metrics.py --pred_dir results/probe/Qwen3-VL-8B/ --tolerance 3,5

# Online metrics
python scripts/compute_online_metrics.py --pred_dir results/online/MiniCPM-o-4.5-Duplex/ --tolerance 3
```

## 📊 Supported Models

### Probe Mode

| Model               | Size | Audio | Script                           |
| ------------------- | ---- | ----- | -------------------------------- |
| Qwen3-VL-8B         | 8B   | ❌    | `run_probe_qwen3_vl.sh`        |
| Qwen2.5-Omni-7B     | 7B   | ✅    | `run_probe_qwen2_5_omni.sh`    |
| Qwen3-Omni-30B-A3B  | 30B  | ✅    | `run_probe_qwen3_omni.sh`      |
| InternVL3.5-8B      | 8B   | ❌    | `run_probe_internvl3_5.sh`     |
| Phi-4-multimodal    | 5.6B | ✅    | `run_probe_phi4_multimodal.sh` |
| Video-SALMONN2+     | 7B   | ✅    | `run_probe_video_salmonn2.sh`  |
| VideoLLaMA2.1-7B-AV | 7B   | ✅    | `run_probe_videollama2_av.sh`  |
| LiveStar-8B         | 8B   | ❌    | `run_probe_livestar.sh`        |
| MMDuet2             | 3B   | ❌    | `run_probe_mmduet2.sh`         |
| MiniCPM-o 4.5       | 9B   | ✅    | `run_probe_minicpmo.sh`        |
| Gemini-3-Flash      | —   | ✅    | `run_probe_gemini.sh`          |

### Online Mode

| Model         | Size | Audio | Script                     |
| ------------- | ---- | ----- | -------------------------- |
| MiniCPM-o 4.5 | 9B   | ✅    | `run_online_minicpmo.sh` |
| MMDuet2       | 3B   | ❌    | `run_online_mmduet2.sh`  |
| LiveStar-8B   | 8B   | ❌    | `run_online_livestar.sh` |

## 📐 Metrics

### Probe Mode

- **Paired Accuracy**: Both pre-probe and post-probe must be correct
- **Content F1**: F1 score across all probe points
- **Pre/Post Accuracy**: Separate accuracy for before/after trigger

### Online Mode

- **Time F1**: Precision × Recall of emit timestamps within tolerance window
- **Content Accuracy**: Correctness of emitted content (parsed or GPT-judged)
- **Joint F1**: Combined timing + content score

## 🔧 Adding a New Model

### Probe Mode

1. Create `models/your_model.py` inheriting from `BaseModel`
2. Implement `name()` and `generate(instruction, video_path)`
3. Register in `scripts/run_probe.py` (choices + build_model)
4. Create `scripts/run_probe_your_model.sh`

### Online Mode

1. Create `models/your_model.py` inheriting from `StreamingModel`
2. Implement `begin()`, `observe()`, `end()`
3. Register in `scripts/run_online.py`
4. Create `scripts/run_online_your_model.sh`

## 📝 Citation

```bibtex
@article{omnipro2026,
  title={OmniPro: A Comprehensive Benchmark for Omni-Proactive Streaming Video Understanding},
  author={Zhao, Ruixiang and Yang, Jie and Xin, Zijie and Wang, Tianyi and Rao, Fengyun and LYU, Jing and Li, Xirong},
  journal={arXiv preprint arXiv:2605.18577},
  year={2026}
}
```

## 📄 License

This project is released under the [MIT License](LICENSE).
