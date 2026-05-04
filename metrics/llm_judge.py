"""Unified LLM judge for scoring free-text predictions against ground truth.

Two providers supported:
  - "openai"  : chat/completions-style endpoint (default env: OPENAI_API_KEY)
  - "gemini"  : Google :generateContent endpoint (env: GEMINI_API_KEY,
                GEMINI_API_BASE)

Select a provider explicitly, or auto-detect from whichever env var is set.
The ``judge(question, gt_response, pred_response) -> {"score": int,
"explanation": str}`` interface is identical across providers, so downstream
code can treat them interchangeably.
"""

import json
import os
import re
import time
from typing import Optional

import requests


# --------------------------------------------------------------------------- #
# Shared prompt
# --------------------------------------------------------------------------- #
_JUDGE_PROMPT = """You are evaluating the quality of a streaming video \
assistant's response.

Task context: The user is watching a video stream and asked the assistant \
to monitor for specific events/conditions. When triggered, the assistant \
should provide an accurate and relevant response.

User's original question: {question}

Ground truth (what actually happened): {gt_response}

Model's prediction: {pred_response}

Rate the model's response on a scale of 1-5:
- 5: Perfect match — captures the same event/information accurately
- 4: Good — mostly correct, minor differences in detail
- 3: Acceptable — captures the right event but with notable inaccuracies
- 2: Poor — partially relevant but significantly wrong or vague
- 1: Wrong — completely irrelevant or describes wrong event

Respond with ONLY a JSON object: \
{{"score": <int>, "explanation": "<brief reason>"}}"""


# --------------------------------------------------------------------------- #
# Provider defaults
# --------------------------------------------------------------------------- #
_PROVIDER_DEFAULTS = {
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "model": "gpt-4o-2024-08-06",
        "key_env": "OPENAI_API_KEY",
        "base_env": "OPENAI_API_BASE",
        "model_env": "OPENAI_MODEL",
    },
    "gemini": {
        "api_base": "YOUR_API_BASE_URL",
        "model": "gemini-3-flash-preview",
        "key_env": "GEMINI_API_KEY",
        "base_env": "GEMINI_API_BASE",
        "model_env": "GEMINI_MODEL",
    },
}


def _auto_provider() -> Optional[str]:
    """Pick a provider based on which credential env var is set."""
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _extract_text(body: dict) -> str:
    """Best-effort text extraction from either API shape."""
    try:
        cand = body.get("candidates") or []
        if cand:
            parts = (cand[0].get("content") or {}).get("parts") or []
            if parts:
                return parts[0].get("text", "")
    except Exception:
        pass
    try:
        ch = body.get("choices") or []
        if ch:
            return (ch[0].get("message") or {}).get("content", "")
    except Exception:
        pass
    return ""


def _parse_score_json(text: str) -> Optional[dict]:
    """Extract ``{score:int, explanation:str}`` from judge output."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
        score = int(obj.get("score", 0))
        return {"score": max(0, min(5, score)),
                "explanation": str(obj.get("explanation", ""))[:300]}
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# Unified judge
# --------------------------------------------------------------------------- #
class LLMJudge:
    """LLM-as-a-judge for free-text content correctness.

    Usage:
        judge = LLMJudge(provider="gemini")             # or "openai"
        result = judge.judge(question, gt_resp, pred_resp)
        # -> {"score": 4, "explanation": "..."}
    """

    def __init__(self,
                 provider: Optional[str] = None,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 model: Optional[str] = None,
                 max_retries: int = 3,
                 timeout: int = 30):
        provider = provider or _auto_provider()
        if provider not in _PROVIDER_DEFAULTS:
            raise ValueError(
                f"Provider must be one of {list(_PROVIDER_DEFAULTS)}; "
                f"got {provider!r}. Set OPENAI_API_KEY or GEMINI_API_KEY, "
                f"or pass provider= explicitly.")
        self.provider = provider
        d = _PROVIDER_DEFAULTS[provider]
        self.api_key = api_key or os.environ.get(d["key_env"], "")
        if not self.api_key:
            raise ValueError(
                f"{d['key_env']} not set. Export it or pass api_key=...")
        self.api_base = api_base or os.environ.get(d["base_env"],
                                                    d["api_base"])
        self.model = model or os.environ.get(d["model_env"], d["model"])
        self.max_retries = max_retries
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    def _build_request(self, prompt: str):
        """Return ``(url, headers, payload)`` for the selected provider."""
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "gemini":
            url = (f"{self.api_base.rstrip('/')}/models/"
                   f"{self.model}:generateContent")
            payload = {
                "model": self.model,
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    # Room for thinking tokens + final JSON output.
                    # Thinking alone can consume ~250 tokens on LOW; keep
                    # the cap well above so the JSON isn't truncated.
                    "maxOutputTokens": 1024,
                    "thinkingConfig": {"thinkingLevel": "LOW"},
                },
            }
        else:  # openai
            url = f"{self.api_base.rstrip('/')}/chat/completions"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
                "temperature": 0.0,
            }
        return url, headers, payload

    # ------------------------------------------------------------------ #
    def judge(self, question: str, gt_response: str,
              pred_response: str) -> dict:
        prompt = _JUDGE_PROMPT.format(
            question=question or "",
            gt_response=gt_response or "",
            pred_response=pred_response or "",
        )
        url, headers, payload = self._build_request(prompt)

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url,
                                     headers=headers,
                                     data=json.dumps(payload),
                                     timeout=self.timeout)
                if resp.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                    last_err = "429 rate limit"
                    continue
                resp.raise_for_status()
                body = resp.json()
                text = _extract_text(body)
                if not text:
                    last_err = f"empty response: {str(body)[:200]}"
                    time.sleep(1)
                    continue
                result = _parse_score_json(text)
                if result is not None:
                    return result
                last_err = f"unparseable response: {text[:200]}"
            except Exception as e:  # network, HTTP, JSON parse...
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(2 ** attempt)
        return {"score": 0, "explanation": f"judge failed: {last_err}"}
