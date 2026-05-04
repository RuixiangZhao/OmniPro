"""
OmniProact-Bench — Visualization Demo

Loads the unified benchmark.json (single file, grouped by `task` field) and
provides an interactive Gradio app to browse samples, search by video_id,
and optionally label them.

Usage:
    python benchmark_demo.py [--port 7861] [--bench_path /path/to/benchmark.json]

Differences from proactive_qa_demo.py:
  * Single JSON source (benchmark.json) rather than per-task results.jsonl.
  * Uses `ground_truth` instead of `responses`.
  * Shows top-level `event` / `target` fields (populated for alert / grounding
    / static-counting tasks).
"""

import argparse
import json
import os
import random
import subprocess
import threading
from collections import defaultdict

import gradio as gr

# ──────────────────────────────────────────────
# Task display metadata
# ──────────────────────────────────────────────
TASK_META = {
    "instant_event_alert": {
        "label": "Instant Event Alert",
        "level": "Perception",
        "icon": "⚡",
        "color": "#e74c3c",
        "desc": "Detect a specific instantaneous event and alert the user.",
    },
    "realtime_state_monitor": {
        "label": "Realtime State Monitor",
        "level": "Perception",
        "icon": "📡",
        "color": "#3498db",
        "desc": "Monitor a discrete state and report when it changes.",
    },
    "snapshot_counting": {
        "label": "Static Object Counting",
        "level": "Perception",
        "icon": "🔢",
        "color": "#2ecc71",
        "desc": "Count objects in the scene when a trigger fires.",
    },
    "explicit_target_grounding": {
        "label": "Explicit Target Grounding",
        "level": "Perception",
        "icon": "🎯",
        "color": "#9b59b6",
        "desc": "Locate a user-specified target (9-region position).",
    },
    "event_narration": {
        "label": "Event Narration",
        "level": "Comprehension",
        "icon": "📝",
        "color": "#e67e22",
        "desc": "Periodically summarize or narrate ongoing events.",
    },
    "cumulative_counting": {
        "label": "Cumulative Counting",
        "level": "Comprehension",
        "icon": "📊",
        "color": "#1abc9c",
        "desc": "Count cumulative occurrences of an event over time.",
    },
    "semantic_condition_alert": {
        "label": "Semantic Condition Alert",
        "level": "Comprehension",
        "icon": "🧠",
        "color": "#8e44ad",
        "desc": "Alert when a fuzzy semantic condition is met.",
    },
    "dedup_counting": {
        "label": "De-duplicated Counting",
        "level": "Reasoning",
        "icon": "🔍",
        "color": "#c0392b",
        "desc": "Count unique targets with identity matching across time.",
    },
    "sequential_step_instruction": {
        "label": "Sequential Step Instruction",
        "level": "Reasoning",
        "icon": "🎓",
        "color": "#2c3e50",
        "desc": "Infer the next instructional step based on current progress.",
    },
}

DEFAULT_META = {
    "label": None,
    "level": "Unknown",
    "icon": "📌",
    "color": "#7f8c8d",
    "desc": "No description available.",
}

LEVEL_COLORS = {
    "Perception": "#e74c3c",
    "Comprehension": "#e67e22",
    "Reasoning": "#2c3e50",
    "Unknown": "#7f8c8d",
}

LABEL_OPTIONS = ["All", "Unlabeled", "Good", "Needs Edit", "Bad"]
LABEL_COLORS = {
    "Good": "#27ae60",
    "Needs Edit": "#f39c12",
    "Bad": "#e74c3c",
}


# ──────────────────────────────────────────────
# Annotations persistence
# ──────────────────────────────────────────────
class AnnotationStore:
    """Thread-safe annotation store: { "task::id": {"label": "...", "comment": "..."} }."""

    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.data = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.data = {}

    def _key(self, sample_id: str) -> str:
        return sample_id

    def get(self, sample_id: str) -> dict:
        with self.lock:
            return self.data.get(self._key(sample_id), {})

    def save(self, sample_id: str, label: str, comment: str):
        with self.lock:
            if not label and not comment:
                self.data.pop(sample_id, None)
            else:
                self.data[sample_id] = {"label": label, "comment": comment}
            with open(self.path, "w") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)

    def get_stats(self) -> dict:
        with self.lock:
            stats = {}
            for v in self.data.values():
                lbl = v.get("label", "")
                if lbl:
                    stats[lbl] = stats.get(lbl, 0) + 1
            return stats


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
def load_benchmark(bench_path: str) -> dict:
    """Group samples by task."""
    with open(bench_path) as f:
        data = json.load(f)
    tasks = defaultdict(list)
    for s in data:
        tasks[s["task"]].append(s)
    return dict(tasks)


def get_meta(task_name: str) -> dict:
    meta = TASK_META.get(task_name, DEFAULT_META).copy()
    if meta["label"] is None:
        meta["label"] = task_name.replace("_", " ").title()
    return meta


def get_video_aspect_ratio(video_path: str) -> str:
    if not video_path or not os.path.exists(video_path):
        return "16/9"
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
             video_path],
            capture_output=True, text=True, timeout=5,
        )
        parts = result.stdout.strip().split("x")
        if len(parts) == 2:
            w, h = int(parts[0]), int(parts[1])
            if w > 0 and h > 0:
                return f"{w}/{h}"
    except Exception:
        pass
    return "16/9"


def make_grid_overlay_html(aspect_ratio: str = "16/9") -> str:
    cell = ('<div style="border:1px solid rgba(155,89,182,0.4); display:flex; '
            'align-items:center; justify-content:center;">'
            '<span style="color:rgba(155,89,182,0.7); font-size:11px; '
            'font-weight:600; text-shadow:0 0 3px rgba(0,0,0,0.5);">{}</span></div>')
    labels = ["top-left", "top-center", "top-right",
              "center-left", "center", "center-right",
              "bottom-left", "bottom-center", "bottom-right"]
    cells = "\n".join(cell.format(l) for l in labels)
    return (f'<div style="position:absolute; top:0; left:50%; transform:translateX(-50%); '
            f'height:100%; aspect-ratio:{aspect_ratio}; max-width:100%; display:grid; '
            f'grid-template-columns:1fr 1fr 1fr; grid-template-rows:1fr 1fr 1fr; '
            f'pointer-events:none; z-index:10;">\n{cells}\n</div>')


def fmt_duration(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}:{s:02d}"


# ──────────────────────────────────────────────
# HTML rendering
# ──────────────────────────────────────────────
def render_sample_html(sample: dict, annotation: dict = None) -> str:
    task_name = sample.get("task", "unknown")
    meta = get_meta(task_name)
    duration = sample.get("duration", 0) or 0
    gts = sample.get("ground_truth", [])
    audio_dep = sample.get("audio_dependency", "none")
    question = sample.get("question", "N/A")
    question_time = sample.get("question_time", "00:00")
    event = sample.get("event")
    target = sample.get("target")

    # Annotation badge
    ann_html = ""
    if annotation and annotation.get("label"):
        lbl = annotation["label"]
        lbl_color = LABEL_COLORS.get(lbl, "#999")
        ann_html = (f'<span style="background:{lbl_color};color:#fff;padding:2px 8px;'
                    f'border-radius:10px;font-size:12px;">🏷️ {lbl}</span>')

    audio_badge_color = {
        "required": "#e74c3c",
        "helpful": "#e67e22",
        "none": "#95a5a6",
    }.get(audio_dep, "#95a5a6")
    audio_badge = (f'<span style="background:{audio_badge_color};color:#fff;'
                   f'padding:2px 8px;border-radius:10px;font-size:12px;">'
                   f'🔊 Audio: {audio_dep}</span>')

    level_color = LEVEL_COLORS.get(meta["level"], "#7f8c8d")
    level_badge = (f'<span style="background:{level_color};color:#fff;'
                   f'padding:2px 8px;border-radius:10px;font-size:12px;">{meta["level"]}</span>')

    # Event / target callout (only for tasks that have them)
    event_target_html = ""
    if event or target:
        chips = []
        if event:
            chips.append(f'<div style="font-size:12px;color:#666;margin-bottom:2px;">🔎 EVENT</div>'
                         f'<div style="font-size:14px;color:#1a1a2e;">{event}</div>')
        if target:
            chips.append(f'<div style="font-size:12px;color:#666;margin-top:8px;margin-bottom:2px;">🎯 TARGET</div>'
                         f'<div style="font-size:14px;color:#1a1a2e;">{target}</div>')
        event_target_html = (
            f'<div style="background:#fffaf0;border-left:4px solid #f39c12;'
            f'padding:10px 14px;border-radius:0 6px 6px 0;margin-bottom:12px;">'
            + "".join(chips) + '</div>'
        )

    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 820px;">
      <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px; flex-wrap:wrap;">
        <span style="font-size:24px;">{meta['icon']}</span>
        <span style="font-size:18px; font-weight:600; color:{meta['color']};">{meta['label']}</span>
        {level_badge}
        {audio_badge}
        {ann_html}
        <span style="color:#888; font-size:13px; margin-left:auto;">Video: {sample.get('video_id', 'N/A')} &middot; Duration: {fmt_duration(duration)}</span>
      </div>
      <div style="background:#f0f4ff; border-left:4px solid {meta['color']}; padding:12px 16px; border-radius:0 8px 8px 0; margin-bottom:12px;">
        <div style="font-size:11px; color:#666; margin-bottom:4px;">💬 QUESTION  <span style="color:#999;">@ {question_time}</span></div>
        <div style="font-size:16px; font-weight:500; color:#1a1a2e;">{question}</div>
      </div>
      {event_target_html}
      <div style="position:relative; margin-bottom:8px;">
        <div style="font-size:13px; font-weight:600; color:#444; margin-bottom:10px;">📍 Ground-Truth Triggers ({len(gts)})</div>
    """

    # Timeline
    if duration > 0 and gts:
        html += (
            f'<div style="position:relative; height:32px; background:linear-gradient(90deg, #e8e8e8, #d0d0d0); '
            f'border-radius:16px; margin:8px 0 20px 0; overflow:visible;">'
            f'<div style="position:absolute; left:4px; top:8px; font-size:10px; color:#666;">0:00</div>'
            f'<div style="position:absolute; right:4px; top:8px; font-size:10px; color:#666;">{fmt_duration(duration)}</div>'
        )
        for gt in gts:
            t_sec = gt.get("trigger_time_sec", 0)
            pct = min(max(t_sec / duration * 100, 2), 98)
            html += (
                f'<div style="position:absolute; left:{pct}%; top:-2px; transform:translateX(-50%); z-index:10;" '
                f'title="{gt.get("trigger_time", "")}">'
                f'<div style="width:18px; height:18px; background:{meta["color"]}; border:3px solid #fff; '
                f'border-radius:50%; box-shadow:0 1px 4px rgba(0,0,0,0.3); margin:0 auto;"></div>'
                f'<div style="font-size:10px; color:{meta["color"]}; font-weight:600; text-align:center; '
                f'margin-top:2px; white-space:nowrap;">{gt.get("trigger_time", "")}</div>'
                f'</div>'
            )
        html += "</div>"

    # Response cards
    for i, gt in enumerate(gts):
        trigger_type = gt.get("trigger_type", "unknown")
        tt_icon = {"sound": "🔊", "speech": "🗣️", "visual": "👁️",
                   "sound+speech": "🔊🗣️", "visual+speech": "👁️🗣️",
                   "visual+sound": "👁️🔊"}.get(trigger_type, "📌")

        count_html = ""
        if "count" in gt and gt["count"] is not None:
            count_html = (f'<span style="background:#27ae60;color:#fff;padding:1px 8px;'
                          f'border-radius:10px;font-size:12px;margin-left:6px;">'
                          f'Count: {gt["count"]}</span>')

        position_html = ""
        if "position" in gt and gt["position"]:
            position_html = (f'<span style="background:#9b59b6;color:#fff;padding:1px 8px;'
                             f'border-radius:10px;font-size:12px;margin-left:6px;">'
                             f'📍 {gt["position"]}</span>')

        html += f"""
      <div style="background:#fff; border:1px solid #e0e0e0; border-radius:8px; padding:12px 16px; margin-bottom:8px; box-shadow:0 1px 3px rgba(0,0,0,0.05);">
        <div style="display:flex; align-items:center; gap:6px; margin-bottom:6px; flex-wrap:wrap;">
          <span style="background:{meta['color']}; color:#fff; width:22px; height:22px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-size:12px; font-weight:700;">{'#' + str(i+1) if len(gts) > 1 else '●'}</span>
          <span style="font-weight:600; color:#333;">@ {gt.get('trigger_time', '?')}</span>
          <span style="font-size:13px;">{tt_icon} {trigger_type}</span>
          {count_html}
          {position_html}
        </div>
        <div style="font-size:15px; color:#1a1a2e; margin-bottom:6px; line-height:1.4;">💬 {gt.get('response', '')}</div>
        <div style="font-size:12px; color:#888; line-height:1.3;">📎 {gt.get('event_description', '')}</div>
      </div>
        """

    html += "</div></div>"
    return html


# ──────────────────────────────────────────────
# Gradio app
# ──────────────────────────────────────────────
def build_app(bench_path: str):
    tasks_data = load_benchmark(bench_path)
    if not tasks_data:
        raise ValueError(f"No tasks found in {bench_path}.")

    ann_path = os.path.join(os.path.dirname(bench_path), "benchmark_annotations.json")
    ann_store = AnnotationStore(ann_path)

    # Allowed paths for video playback
    video_dirs = set()
    for samples in tasks_data.values():
        for s in samples:
            vp = s.get("video_path", "")
            if vp:
                video_dirs.add(os.path.dirname(vp))

    # Task dropdown choices (ordered by level then name for nicer UX)
    level_order = {"Perception": 0, "Comprehension": 1, "Reasoning": 2, "Unknown": 3}
    task_keys = sorted(
        tasks_data.keys(),
        key=lambda k: (level_order.get(get_meta(k)["level"], 99), k),
    )
    task_choices = []
    for k in task_keys:
        m = get_meta(k)
        task_choices.append(f"{m['icon']} {m['label']}  ({len(tasks_data[k])} samples)")

    # video_id → {task: sample}
    video_id_index = defaultdict(dict)
    for k, samples in tasks_data.items():
        for s in samples:
            vid = s.get("video_id", "")
            if vid:
                video_id_index[vid][k] = s

    current_sample = {"id": None}

    def _get_task_key(task_display: str) -> str:
        idx = task_choices.index(task_display)
        return task_keys[idx]

    def _show_sample(sample: dict):
        sid = sample.get("id", "")
        current_sample["id"] = sid
        ann = ann_store.get(sid)
        video_path = sample.get("video_path", "")
        if not os.path.exists(video_path):
            video_path = None
        html = render_sample_html(sample, ann)

        is_grounding = "grounding" in sample.get("task", "").lower()
        if is_grounding:
            ar = get_video_aspect_ratio(video_path)
            grid = gr.update(value=make_grid_overlay_html(ar), visible=True)
        else:
            grid = gr.update(value="", visible=False)

        return video_path, html, ann.get("label", ""), ann.get("comment", ""), grid

    def pick_random(task_display: str, filter_label: str):
        task_key = _get_task_key(task_display)
        samples = tasks_data[task_key]
        if filter_label and filter_label != "All":
            filtered = []
            for s in samples:
                ann = ann_store.get(s.get("id", ""))
                lbl = ann.get("label", "")
                if filter_label == "Unlabeled" and not lbl:
                    filtered.append(s)
                elif lbl == filter_label:
                    filtered.append(s)
            if not filtered:
                return (None,
                        f'<div style="color:#888; padding:20px;">No samples with label "{filter_label}" in this task.</div>',
                        "", "", gr.update(value="", visible=False))
            sample = random.choice(filtered)
        else:
            sample = random.choice(samples)
        return _show_sample(sample)

    def on_task_change(task_display: str, filter_label: str):
        return pick_random(task_display, filter_label)

    def search_by_video_id(video_id: str, task_display: str):
        video_id = video_id.strip()
        if not video_id:
            return (None, '<div style="color:#888; padding:20px;">Please enter a video ID.</div>',
                    "", "", gr.update(value="", visible=False))
        task_key = _get_task_key(task_display)
        matches = video_id_index.get(video_id, {})
        sample = matches.get(task_key)
        if sample is None:
            for _, other_sample in matches.items():
                sample = other_sample
                break
        if sample is None:
            return (None,
                    f'<div style="color:#e74c3c; padding:20px;">Video ID "<b>{video_id}</b>" not found.</div>',
                    "", "", gr.update(value="", visible=False))
        return _show_sample(sample)

    def save_annotation(label: str, comment: str):
        sid = current_sample.get("id")
        if not sid:
            return '<div style="color:#e74c3c;">No sample loaded yet.</div>'
        ann_store.save(sid, label, comment)
        stats = ann_store.get_stats()
        parts = [f"{k}: {v}" for k, v in sorted(stats.items())]
        summary = ", ".join(parts) if parts else "0"
        return f'<div style="color:#27ae60; padding:4px;">✅ Saved for <b>{sid}</b>. Total annotations: {summary}</div>'

    # Dataset stats table
    stats_rows = ""
    for k in task_keys:
        m = get_meta(k)
        n = len(tasks_data[k])
        total_triggers = sum(len(s.get("ground_truth", [])) for s in tasks_data[k])
        avg_trig = total_triggers / n if n else 0
        # audio_dependency breakdown
        dep_counter = defaultdict(int)
        for s in tasks_data[k]:
            dep_counter[s.get("audio_dependency", "none")] += 1
        dep_str = ", ".join(f"{d}={c}" for d, c in sorted(dep_counter.items()))
        stats_rows += f"""
        <tr>
            <td style="padding:6px 12px;">{m['icon']} {m['label']}</td>
            <td style="padding:6px 12px;"><span style="background:{LEVEL_COLORS.get(m['level'],'#999')};color:#fff;padding:1px 8px;border-radius:10px;font-size:12px;">{m['level']}</span></td>
            <td style="padding:6px 12px; text-align:right;">{n}</td>
            <td style="padding:6px 12px; text-align:right;">{total_triggers}</td>
            <td style="padding:6px 12px; text-align:right;">{avg_trig:.1f}</td>
            <td style="padding:6px 12px; font-size:12px; color:#555;">{dep_str}</td>
        </tr>
        """
    stats_html = f"""
    <div style="font-family: -apple-system, sans-serif;">
      <table style="border-collapse:collapse; width:100%; font-size:14px;">
        <tr style="background:#f5f5f5; font-weight:600;">
          <td style="padding:8px 12px;">Task</td>
          <td style="padding:8px 12px;">Level</td>
          <td style="padding:8px 12px; text-align:right;">Samples</td>
          <td style="padding:8px 12px; text-align:right;">Total Triggers</td>
          <td style="padding:8px 12px; text-align:right;">Avg Trig/Sample</td>
          <td style="padding:8px 12px;">Audio Dep</td>
        </tr>
        {stats_rows}
      </table>
    </div>
    """

    # ── Layout ──
    with gr.Blocks(title="OmniProact-Bench — Benchmark Viewer") as app:
        gr.Markdown("# 🎬 OmniProact-Bench — Benchmark Visualization")
        gr.Markdown(
            "Interactive viewer of `benchmark.json`. "
            "Select a subtask, browse samples randomly or by video_id, and label for QA review."
        )

        with gr.Row():
            task_dropdown = gr.Dropdown(choices=task_choices, value=task_choices[0], label="Subtask", scale=2)
            filter_dropdown = gr.Dropdown(choices=LABEL_OPTIONS, value="All", label="🏷️ Filter by Label", scale=1)
            random_btn = gr.Button("🎲 Random Sample", variant="primary", scale=1)

        with gr.Row():
            video_id_input = gr.Textbox(label="🔍 Search by Video ID",
                                        placeholder="e.g. -l9fxSCGJ4M", scale=3)
            search_btn = gr.Button("🔍 Search", variant="secondary", scale=1)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, elem_classes=["video-col"]):
                video_player = gr.Video(label="Video", height=400)
                grid_overlay = gr.HTML(visible=False, elem_classes=["grid-overlay"])
            with gr.Column(scale=1):
                qa_display = gr.HTML(elem_classes=["qa-html"])

        # Annotation section
        with gr.Row():
            with gr.Column(scale=1):
                label_radio = gr.Radio(choices=["Good", "Needs Edit", "Bad"],
                                       label="🏷️ Label this sample", value="")
            with gr.Column(scale=2):
                comment_box = gr.Textbox(label="💬 Comment",
                                         placeholder="Optional notes...", lines=2)
            with gr.Column(scale=1):
                save_btn = gr.Button("💾 Save Annotation", variant="primary")
                save_status = gr.HTML()

        with gr.Accordion("📊 Benchmark Statistics", open=False):
            gr.HTML(stats_html)

        # Events
        _outputs = [video_player, qa_display, label_radio, comment_box, grid_overlay]
        random_btn.click(fn=pick_random, inputs=[task_dropdown, filter_dropdown], outputs=_outputs)
        task_dropdown.change(fn=on_task_change, inputs=[task_dropdown, filter_dropdown], outputs=_outputs)
        search_btn.click(fn=search_by_video_id, inputs=[video_id_input, task_dropdown], outputs=_outputs)
        video_id_input.submit(fn=search_by_video_id, inputs=[video_id_input, task_dropdown], outputs=_outputs)
        save_btn.click(fn=save_annotation, inputs=[label_radio, comment_box], outputs=[save_status])

        app.load(fn=lambda: pick_random(task_choices[0], "All"), outputs=_outputs)

    return app, sorted(video_dirs)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniProact-Bench Visualization Demo")
    parser.add_argument(
        "--bench_path",
        type=str,
        default="/path/to/OmniProact-Bench/data/benchmark.json",
        help="Path to benchmark.json",
    )
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    app, allowed_paths = build_app(args.bench_path)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        allowed_paths=allowed_paths,
        theme=gr.themes.Soft(),
        css="""
.qa-html { min-height: 200px; }
.video-col { position: relative !important; }
.video-col .grid-overlay {
    position: absolute !important;
    top: 0; left: 50%;
    transform: translateX(-50%);
    height: 100%;
    aspect-ratio: 16/9;
    max-width: 100%;
    pointer-events: none;
    z-index: 10;
}
""",
    )
