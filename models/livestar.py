"""LiveStar streaming adapter for online evaluation.

Paper: "LiveStar: Live Streaming Assistant for Real-World Online Video
Understanding" (NeurIPS 2025). Repo: https://github.com/sotayang/LiveStar

Key behaviour preserved from the official `inference/demo.py` and
`evaluate/eval_benchmark.py`:

  * Vision-only model (InternViT + InternLM2), no audio input.
  * Pre-extracted 1-fps frames are fed one at a time through
    ``model.chat(...)``. There is no duplex prefill; each tick re-runs a
    forward pass over ``pixel_values[:tick+1]``. Token-merging (ToMe) +
    peak-end memory compression inside the model keep compute bounded.
  * Response-Silence (SVeD) decision: compare the perplexity of reusing
    the previous answer under the new frame vs. an (self-checked)
    baseline threshold. Speak iff  ``ppl > decode_threshold * decode_factor``.

Contract matches ``StreamingModel``:
  begin(instruction, task_meta)  — store the benchmark's per-sample prompt.
  observe(frame, t_sec, history, audio_chunk)
      — first tick  : run decode to get baseline answer + self_check ppl.
      — later ticks : ppl check → speak (new decode + reset threshold) or
                      stay silent (only absorb the frame token into history).
  end()  — clear per-sample state.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from .streaming_base import StreamingModel

logger = logging.getLogger(__name__)


# ─── Image preprocessing (verbatim from LiveStar repo) ──────────────────

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        tar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - tar)
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio
        elif diff == best_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 1,
                        image_size: int = 448, use_thumbnail: bool = True):
    """Split one frame into tiles according to its aspect ratio.

    For benchmark inference we match the official LiveStar settings:
    ``max_num=1`` (no tiling, just the thumbnail), ``image_size=448``.
    """
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    tar = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_w,
                                     orig_h, image_size)
    target_w = image_size * tar[0]
    target_h = image_size * tar[1]
    blocks = tar[0] * tar[1]
    resized = image.resize((target_w, target_h))
    tiles = []
    for i in range(blocks):
        box = (
            (i % (target_w // image_size)) * image_size,
            (i // (target_w // image_size)) * image_size,
            ((i % (target_w // image_size)) + 1) * image_size,
            ((i // (target_w // image_size)) + 1) * image_size,
        )
        tiles.append(resized.crop(box))
    assert len(tiles) == blocks
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


# ─── The streaming adapter ─────────────────────────────────────────────

class LiveStarStreaming(StreamingModel):
    """LiveStar-8B as an OmniProact-Bench streaming model.

    Args:
        model_path: Local path containing the HF snapshot (safetensors
            + ``modeling_livestar_chat.py`` etc.).
        device: Single device string, e.g. ``"cuda:0"``.
        decode_factor: Multiplier applied to the self-check baseline ppl to
            decide whether to emit. Official defaults: 1.06 in demo, 1.04 in
            eval_benchmark. Higher = more silent, lower = more talkative.
        max_new_tokens: Cap on generated tokens per emit. Benchmark answers
            are short (one sentence / single integer / state name) so we
            default to 128 — far below the demo's 1024.
        input_size: ViT input resolution. Official value 448.
        max_num: Max number of dynamic tiles per frame. ``1`` keeps just
            the thumbnail as in LiveStar's official evaluation.
        check_len: Prefix of the last answer used as ``check_answer`` when
            computing ppl (mirrors the ``check_len`` constant in demo.py).
        max_frames_per_segment: When the cumulative frame count reaches
            this value, the streaming session is reset (pixel_values /
            chat_history cleared) and a fresh one is started from the
            next tick, using the previous ``output_last`` as seed context
            in the instruction. This caps the O(N^2)-ish cost of
            LiveStar's cumulative forward passes so long benchmark videos
            (~10 min) remain tractable. Defaults to 60 (1 min @ 1 fps).
    """

    accepts_audio: bool = False
    native_fps: float = 1.0

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        decode_factor: float = 1.04,
        max_new_tokens: int = 128,
        input_size: int = 448,
        max_num: int = 1,
        check_len: int = 1000,
        max_frames_per_segment: int = 60,
        torch_dtype=None,
    ):
        from transformers import AutoModel, AutoTokenizer

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        logger.info(f"Loading LiveStar from {model_path} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        # Follow the official recipe: .half() then .to(bfloat16) keeps the
        # weights in bf16 while also clearing any fp32 buffers.
        self._model = (
            AutoModel.from_pretrained(model_path, trust_remote_code=True)
            .half()
            .to(device)
            .to(torch_dtype)
            .eval()
        )
        self._device = device
        self._decode_factor = decode_factor
        self._max_new_tokens = max_new_tokens
        self._input_size = input_size
        self._max_num = max_num
        self._check_len = check_len
        self._max_frames_per_segment = max_frames_per_segment
        self._transform = _build_transform(input_size)

        # LiveStar's decode settings (from demo.py / eval_benchmark.py).
        self._generation_config = dict(
            temperature=0.0,
            max_new_tokens=self._max_new_tokens,
            top_p=0.1,
            num_beams=1,
            repetition_penalty=1.05,
        )

        # Per-sample state (reset by begin/end).
        self._instruction: str = ""
        self._tick: int = 0
        self._pixel_values: Optional[torch.Tensor] = None  # (N_tiles, 3, H, W)
        self._num_patches_list: List[int] = []
        self._chat_history: Optional[List] = None
        self._output_last: str = ""
        self._decode_threshold: Optional[float] = None

        logger.info("LiveStar loaded.")

    def name(self) -> str:
        return "LiveStar-8B"

    # ── StreamingModel interface ───────────────────────────────────────

    def begin(self, instruction: str, task_meta: Dict[str, Any]) -> None:
        self._instruction = instruction or ""
        self._tick = 0
        self._pixel_values = None
        self._num_patches_list = []
        self._chat_history = None
        self._output_last = ""
        self._decode_threshold = None
        self._segment_idx = 0  # which segment we are in (0 = first)

    def end(self) -> None:
        # Drop large tensors so GC can reclaim GPU memory between samples.
        self._pixel_values = None
        self._num_patches_list = []
        self._chat_history = None

    def _reset_segment(self) -> None:
        """Hard-reset the streaming session while keeping the global tick
        counter. Used when a segment exceeds ``max_frames_per_segment`` so
        that LiveStar's O(N^2)-ish cumulative inference stays tractable on
        long benchmark videos.

        The previous answer (``output_last``) is carried into the new
        instruction as a seed so the model has some context continuity.
        """
        self._segment_idx += 1
        self._pixel_values = None
        self._num_patches_list = []
        self._chat_history = None
        self._decode_threshold = None
        # output_last is kept so the next first-tick can use it as seed.

    @torch.no_grad()
    def observe(
        self,
        frame,
        t_sec: float,
        history: List[Dict[str, Any]],
        audio_chunk: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        # Check whether we should reset *before* appending this frame.
        if (self._pixel_values is not None
                and len(self._num_patches_list) >= self._max_frames_per_segment):
            self._reset_segment()

        # 1. Preprocess the new frame → tiles → tensor.
        pil = _to_pil(frame)
        if pil is None:
            # No frame this tick (shouldn't happen in normal runs). Skip.
            return None
        tiles = _dynamic_preprocess(
            pil, image_size=self._input_size, use_thumbnail=True,
            max_num=self._max_num,
        )
        frame_tensor = torch.stack([self._transform(t) for t in tiles])
        frame_tensor = frame_tensor.to(self._device).to(torch.bfloat16)
        n_tiles = frame_tensor.shape[0]

        # 2. Append to the running pixel_values + num_patches_list.
        if self._pixel_values is None:
            self._pixel_values = frame_tensor
        else:
            self._pixel_values = torch.cat(
                [self._pixel_values, frame_tensor], dim=0,
            )
        self._num_patches_list.append(n_tiles)

        # Frame index within the current segment (starts at 1 each reset,
        # matching LiveStar's "Frame-1:, Frame-2: ..." convention).
        seg_frame_idx = len(self._num_patches_list)
        frame_tag = f"Frame-{seg_frame_idx}: <image>\n"
        is_first_in_segment = (seg_frame_idx == 1)
        emit: Optional[str] = None

        if is_first_in_segment:
            # First tick of a fresh segment: run decode to establish
            # baseline answer, then self_check ppl → decode_threshold.
            if self._segment_idx == 0:
                # Very first segment: plain task instruction.
                question = self._instruction + "\n" + frame_tag
            else:
                # Mid-sample reset: carry forward the previous answer as
                # seed so narration/alerts keep some continuity.
                seed = (self._output_last or "").strip()
                if seed:
                    question = (
                        self._instruction
                        + f"\n(Continuing. Your previous note was: "
                          f"\"{seed[:self._check_len]}\")\n"
                        + frame_tag
                    )
                else:
                    question = self._instruction + "\n" + frame_tag
            self._output_last, self._chat_history, _ = self._chat(
                question=question,
                history=None,
                return_history=True,
            )
            ppl = self._chat(
                question=frame_tag,
                history=self._chat_history,
                return_history=False,
                check_answer=self._output_last[:self._check_len],
                self_check=True,
            )
            self._decode_threshold = ppl
            emit = self._output_last
        else:
            # Later ticks within a segment: probe ppl of reusing the
            # previous answer.
            ppl = self._chat(
                question=frame_tag,
                history=self._chat_history,
                return_history=False,
                check_answer=self._output_last[:self._check_len],
                self_check=False,
            )
            if ppl > self._decode_threshold * self._decode_factor:
                # Emit: re-decode a fresh answer, reset the threshold.
                self._output_last, self._chat_history, _ = self._chat(
                    question=frame_tag,
                    history=self._chat_history,
                    return_history=True,
                )
                new_thresh = self._chat(
                    question=frame_tag,
                    history=self._chat_history,
                    return_history=False,
                    check_answer=self._output_last[:self._check_len],
                    self_check=True,
                )
                self._decode_threshold = new_thresh
                emit = self._output_last
            else:
                # Silent: absorb the frame tag into the last user turn so
                # the model still "sees" this frame in subsequent context.
                if self._chat_history:
                    last_q, last_a = self._chat_history[-1]
                    self._chat_history[-1] = (last_q + frame_tag, last_a)
                emit = None

        self._tick += 1
        # Normalise empty strings to STANDBY.
        if isinstance(emit, str):
            emit = emit.strip()
            if not emit:
                emit = None
        return emit

    # ── helpers ───────────────────────────────────────────────────────

    def _chat(self, *, question: str, history, return_history: bool,
              check_answer: Optional[str] = None, self_check: bool = False):
        """Single wrapper around ``model.chat`` that protects our cumulative
        buffers from LiveStar's in-place side-effects.

        ``model.chat(..., self_check=True)`` **appends** the last frame a
        second time to both ``pixel_values`` and ``num_patches_list``
        (see ``modeling_livestar_chat.py`` lines 561-565). If we passed our
        own buffers in, they would grow by one every call. We therefore
        always clone them before the call.
        """
        pv = self._pixel_values
        if pv is not None:
            pv_in = pv.clone()
        else:
            pv_in = None
        npl_in = list(self._num_patches_list)

        # A fresh generation_config per call: the model mutates it to set
        # ``eos_token_id`` inside ``chat``.
        gen_cfg = dict(self._generation_config)

        result = self._model.chat(
            self._tokenizer,
            pv_in,
            question,
            gen_cfg,
            num_patches_list=npl_in,
            history=history,
            return_history=return_history,
            check_answer=check_answer,
            self_check=self_check,
        )
        if check_answer is not None:
            # returns (ppl_float, None)
            ppl, _ = result
            return float(ppl)
        if return_history:
            # returns (response, history, past_key_values)
            return result
        # return_history=False and no check_answer: plain response string
        return result


# ─── utils ─────────────────────────────────────────────────────────────

def _to_pil(frame) -> Optional[Image.Image]:
    if frame is None:
        return None
    if isinstance(frame, Image.Image):
        return frame
    if isinstance(frame, np.ndarray):
        return Image.fromarray(frame)
    # Torch tensor (HWC or CHW, uint8 or float)
    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu().numpy()
        if arr.dtype != np.uint8:
            arr = arr.clip(0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.transpose(1, 2, 0)
        return Image.fromarray(arr)
    return None
