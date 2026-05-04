"""Gemini-3-Flash API model (non-GPU)."""

import base64
import json
import os
import subprocess
import time
import traceback

import requests

from models.base import BaseModel


GEMINI_API_URL = os.environ.get(
    "GEMINI_API_BASE",
    "YOUR_API_BASE_URL/models/{model}:generateContent",
)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")

# Default cache dir for 480p re-encoded clips sent to Gemini
_COMPRESS_CACHE_DIR = os.environ.get(
    "OMNIPROACT_GEMINI_CACHE",
    "/path/to/OmniProact-Bench/gemini_clip_cache",
)


def _encode_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _compress_video(src_path: str, cache_dir: str = _COMPRESS_CACHE_DIR,
                    max_height: int = 360, fps: int = 8, crf: int = 30,
                    strip_audio: bool = False) -> str:
    """Compress video to a smaller size for API upload.

    Scales to max_height (keep aspect), drops fps to `fps`, encodes with
    libx264 + CRF. Audio kept as AAC 64k mono by default; pass
    strip_audio=True to drop the audio track entirely (for no-audio Gemini
    evaluation). Results are cached by the original path's inode + size +
    params so cache never collides.
    """
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(src_path))[0]
    # hash of src mtime + size to invalidate if the source clip changes
    try:
        st = os.stat(src_path)
        sig = f"{st.st_size}_{int(st.st_mtime)}"
    except OSError:
        sig = "nosig"
    noaudio_tag = "_noaudio" if strip_audio else ""
    out_path = os.path.join(
        cache_dir,
        f"{base}_h{max_height}_fps{fps}_crf{crf}{noaudio_tag}_{sig}.mp4",
    )
    if os.path.exists(out_path):
        return out_path

    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-vf", f"scale=-2:'min({max_height},ih)'",
        "-r", str(fps),
        "-vcodec", "libx264",
        "-preset", "veryfast",
        "-crf", str(crf),
    ]
    if strip_audio:
        cmd += ["-an"]
    else:
        cmd += ["-acodec", "aac", "-b:a", "64k", "-ac", "1"]
    cmd += ["-loglevel", "error", out_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except subprocess.CalledProcessError as e:
        print(f"[Gemini] compress failed for {src_path}: {e.stderr.decode()[:200]}")
        return src_path  # fall back to original
    except subprocess.TimeoutExpired:
        print(f"[Gemini] compress timed out for {src_path}")
        return src_path
    return out_path


class Gemini3Flash(BaseModel):
    """HTTP client for gemini-3-flash-preview.

    Not a GPU model; `device` argument is ignored. Each worker process makes
    independent HTTP requests — set run_probe.py's --num_gpus to the desired
    concurrency. Videos are re-encoded to 360p / 8 fps / CRF 30 before upload
    to keep payload small.
    """

    MODEL_NAME = "gemini-3-flash-preview"

    def __init__(self, model_path: str = None, device: str = None,
                 fps: float = 1.0, max_new_tokens: int = 256,
                 max_retries: int = 3, retry_backoff: float = 2.0,
                 request_timeout: int = 180,
                 compress: bool = True,
                 compress_height: int = 360,
                 compress_fps: int = 2,
                 compress_crf: int = 33,
                 use_audio: bool = True,
                 proxy: str = None):
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.request_timeout = request_timeout
        self.compress = compress
        self.compress_height = compress_height
        self.compress_fps = compress_fps
        self.compress_crf = compress_crf
        self.use_audio = use_audio
        self._url = GEMINI_API_URL.format(model=self.MODEL_NAME)
        self._headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {GEMINI_API_KEY}",
            "Content-Type": "application/json",
        }
        # Proxy: first CLI arg, else HTTP(S)_PROXY env var, else None
        proxy = proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        self._proxies = {"http": proxy, "https": proxy} if proxy else None

    def name(self) -> str:
        return self.MODEL_NAME + ("" if self.use_audio else "-NoAudio")

    # ──────────────────────────────────────────────────────────
    def _build_payload(self, instruction: str, video_b64: str) -> dict:
        return {
            "model": self.MODEL_NAME,
            "contents": [
                {
                    "parts": [
                        {"text": instruction},
                        {"inline_data": {"mime_type": "video/mp4", "data": video_b64}},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 1.0,
                "maxOutputTokens": self.max_new_tokens,
                "thinkingConfig": {"thinkingLevel": "LOW"},
            },
        }

    def _parse_response(self, resp_json: dict) -> str:
        try:
            cands = resp_json["candidates"]
            parts = cands[0]["content"]["parts"]
            for p in parts:
                if "text" in p and p["text"]:
                    return p["text"].strip()
            return ""
        except (KeyError, IndexError, TypeError):
            return ""

    # ──────────────────────────────────────────────────────────
    def generate(self, instruction: str, video_path: str, **kwargs) -> str:
        """Send (instruction + video) to Gemini and return text response."""
        send_path = video_path
        if self.compress:
            send_path = _compress_video(
                video_path,
                max_height=self.compress_height,
                fps=self.compress_fps,
                crf=self.compress_crf,
                strip_audio=not self.use_audio,
            )
        video_b64 = _encode_file(send_path)
        payload = json.dumps(self._build_payload(instruction, video_b64))

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self._url,
                    headers=self._headers,
                    data=payload,
                    timeout=self.request_timeout,
                    proxies=self._proxies,
                )
                if resp.status_code == 200:
                    text = self._parse_response(resp.json())
                    if text:
                        return text
                    last_err = f"empty text in response: {resp.text[:200]}"
                else:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except requests.RequestException as e:
                last_err = f"{type(e).__name__}: {e}"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_backoff * (2 ** attempt))

        print(f"[Gemini] generate failed after {self.max_retries} tries: {last_err}")
        return ""


class Gemini3Flash_NoAudio(Gemini3Flash):
    """Gemini-3-Flash with the audio track stripped before upload.

    Identical to Gemini3Flash except that `_compress_video` is invoked with
    `strip_audio=True`, so the video bytes sent to the API have no audio.
    Cached compressed clips are keyed separately (filename contains
    `_noaudio`), so this does not clobber the audio-version cache.
    """

    def __init__(self, **kwargs):
        kwargs["use_audio"] = False
        super().__init__(**kwargs)


def _extract_audio(src_path: str, cache_dir: str = _COMPRESS_CACHE_DIR,
                   fmt: str = "mp3") -> str:
    """Extract audio track from a video file as mp3.

    Returns path to the cached mp3 file. If the video has no audio track,
    returns None.
    """
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(src_path))[0]
    try:
        st = os.stat(src_path)
        sig = f"{st.st_size}_{int(st.st_mtime)}"
    except OSError:
        sig = "nosig"
    out_path = os.path.join(cache_dir, f"{base}_audioonly_{sig}.{fmt}")
    if os.path.exists(out_path):
        return out_path

    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-vn",  # no video
        "-acodec", "libmp3lame",
        "-b:a", "128k",
        "-ac", "1",
        "-loglevel", "error",
        out_path,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"[Gemini] audio extraction failed for {src_path}: {e}")
        return None
    # Check output file has non-zero size
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return None
    return out_path


class Gemini3Flash_AudioOnly(Gemini3Flash):
    """Gemini-3-Flash with only audio input (no video frames).

    Extracts the audio track from the video clip as mp3 and sends it
    to the Gemini API as an audio/mpeg inline_data part.
    """

    def __init__(self, **kwargs):
        kwargs["use_audio"] = True
        super().__init__(**kwargs)

    def name(self) -> str:
        return self.MODEL_NAME + "-AudioOnly"

    def _build_payload_audio(self, instruction: str, audio_b64: str) -> dict:
        return {
            "model": self.MODEL_NAME,
            "contents": [
                {
                    "parts": [
                        {"text": instruction},
                        {"inline_data": {"mime_type": "audio/mpeg", "data": audio_b64}},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 1.0,
                "maxOutputTokens": self.max_new_tokens,
                "thinkingConfig": {"thinkingLevel": "LOW"},
            },
        }

    def generate(self, instruction: str, video_path: str, **kwargs) -> str:
        """Extract audio from video and send audio-only to Gemini."""
        audio_path = _extract_audio(video_path)
        if audio_path is None:
            print(f"[Gemini-AudioOnly] No audio track in {video_path}")
            return ""

        audio_b64 = _encode_file(audio_path)
        payload = json.dumps(self._build_payload_audio(instruction, audio_b64))

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self._url,
                    headers=self._headers,
                    data=payload,
                    timeout=self.request_timeout,
                    proxies=self._proxies,
                )
                if resp.status_code == 200:
                    text = self._parse_response(resp.json())
                    if text:
                        return text
                    last_err = f"empty text in response: {resp.text[:200]}"
                else:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except requests.RequestException as e:
                last_err = f"{type(e).__name__}: {e}"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_backoff * (2 ** attempt))

        print(f"[Gemini-AudioOnly] generate failed after {self.max_retries} tries: {last_err}")
        return ""
