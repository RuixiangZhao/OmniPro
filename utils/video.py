"""视频裁剪工具。"""

import os
import subprocess


def _is_valid_clip(clip_path: str) -> bool:
    """Check if a cached clip file is valid (has moov atom / is readable)."""
    if not os.path.exists(clip_path):
        return False
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of", "csv=p=0", clip_path],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def split_video(video_path: str, start_sec: int, end_sec: int,
                cache_dir: str = "/tmp/omniproact_clips") -> str:
    """
    Clip video [start_sec, end_sec] using ffmpeg. Results are cached.

    Validates cached files with ffprobe to detect incomplete writes
    (e.g. missing moov atom from interrupted ffmpeg processes).

    Returns:
        Path to the clipped video file.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(cache_dir, exist_ok=True)
    clip_path = os.path.join(cache_dir, f"{video_name}_{start_sec}_{end_sec}.mp4")

    # Use cached clip only if it passes integrity check
    if os.path.exists(clip_path) and _is_valid_clip(clip_path):
        return clip_path

    # Remove potentially corrupted cached file
    if os.path.exists(clip_path):
        os.unlink(clip_path)

    duration = end_sec - start_sec
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(int(start_sec)),
        "-i", video_path,
        "-t", str(int(duration)),
        "-vcodec", "libx264",
        "-acodec", "aac",
        "-movflags", "+faststart",  # put moov atom at the beginning
        "-loglevel", "error",
        clip_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg failed for {video_path}: {e.stderr.decode()[:200]}")
        raise
    except subprocess.TimeoutExpired:
        print(f"[ERROR] ffmpeg timeout for {video_path} [{start_sec}-{end_sec}s]")
        raise

    return clip_path
