"""各子任务的 Prompt 模板和响应解析逻辑。"""

import re
from typing import Optional


# ─── Prompt 模板 ───────────────────────────────────────────────────────

def build_probe_prompt(task: str, question: str, current_time_sec: float,
                       prev_response: Optional[str] = None,
                       occurred_count: int = 0,
                       event: Optional[str] = None,
                       target: Optional[str] = None,
                       states: Optional[list] = None,
                       options: Optional[list] = None) -> str:
    """
    Build the probe prompt for a given task and polling time.

    Args:
        task: Task name.
        question: Original user question.
        current_time_sec: Current video time in seconds.
        prev_response: Previous model response (for multi-turn tasks).
        occurred_count: Number of times the event has already occurred (for alert tasks).
        event: Pre-extracted trigger event string (for alert tasks).
        target: Pre-extracted target object noun phrase (for grounding task).
        states: Closed state vocabulary (for realtime_state_monitor task).
        options: MCQ options list (for event_narration task). Each element
            is a shuffled narration string; one of them is the correct answer
            and its index is tracked separately by the evaluator.

    Returns:
        Formatted prompt string.
    """
    builder = PROMPT_BUILDERS.get(task, _default_prompt)
    # Alert-type prompts accept occurred_count + event + task (for noun choice)
    if task == "explicit_target_grounding":
        return builder(question, current_time_sec, prev_response, occurred_count, event, task, target)
    if task in ("instant_event_alert", "semantic_condition_alert"):
        return builder(question, current_time_sec, prev_response, occurred_count, event, task)
    if task == "snapshot_counting":
        return builder(question, current_time_sec, prev_response, occurred_count, event, target)
    if task == "realtime_state_monitor":
        return builder(question, current_time_sec, prev_response, states)
    if task == "event_narration":
        return builder(question, current_time_sec, options)
    if task == "sequential_step_instruction":
        return builder(question, current_time_sec, options)
    return builder(question, current_time_sec, prev_response)


# Noun used in "Has the following <noun> just happened/held?" and follow-up "Note: this ..." reference.
_TASK_NOUN = {
    "instant_event_alert": "event",
    "semantic_condition_alert": "condition",
    "explicit_target_grounding": "event",
}


def _alert_prompt(question: str, current_time_sec: float,
                  prev_response: Optional[str] = None,
                  occurred_count: int = 0,
                  event: Optional[str] = None,
                  task: Optional[str] = None) -> str:
    """For instant_event_alert, semantic_condition_alert.

    Uses the pre-extracted `event` field when available; falls back to
    `_convert_to_question(question)` only if event is missing.
    """
    noun = _TASK_NOUN.get(task, "event")

    if event:
        core_q = f"Did {event} in this video? YES or NO."
    else:
        core_q = _convert_to_question(question)

    if occurred_count == 0:
        occurrence_ctx = ""
    else:
        occurrence_ctx = (
            f"Note: this {noun} has already happened {occurred_count} "
            f"time{'s' if occurred_count > 1 else ''} earlier in the video. "
            f"Focus ONLY on the most recent occurrence; do NOT answer YES if you are only "
            f"relying on those earlier occurrences.\n"
        )
    # Evidence requirement used to be injected for semantic_condition_alert,
    # but in practice it made the problem worse (pre false-positive rate went
    # up under "strict word" variant; even the paraphrase-allowed variant only
    # matched baseline paired acc). Fix moved to probe_evaluator.py where
    # SCA's pre offset is pinned to -5 instead of random [-5..-2]. Keep the
    # lean alert prompt here for all tasks.
    evidence_block = ""

    return (
        f"Listen carefully to both the visual and audio content of this video stream "
        f"(up to {_fmt_time(current_time_sec)}).\n"
        f"{core_q}\n"
        f"{occurrence_ctx}"
        f"{evidence_block}"
        f"FORMAT REQUIREMENT (strict):\n"
        f"- Start your response with the single token YES or NO.\n"
        f"- If YES, follow with a colon and a brief description: `YES: <description>`.\n"
        f"- If NO, reply with just `NO`.\n"
        f"- Do NOT add any preamble like \"The answer is\" or \"Therefore\"."
    )


def _target_grounding_prompt(question: str, current_time_sec: float,
                             prev_response: Optional[str] = None,
                             occurred_count: int = 0,
                             event: Optional[str] = None,
                             task: Optional[str] = None,
                             target: Optional[str] = None) -> str:
    """For explicit_target_grounding: alert trigger + 9-region position output.

    Uses the pre-extracted `event` and `target` fields when available.
    Falls back to `_convert_to_question` and a generic "target object" phrase otherwise.
    """
    noun = _TASK_NOUN.get(task, "event")

    if event:
        core_q = f"Did {event} in this video? YES or NO."
    else:
        core_q = _convert_to_question(question)

    if occurred_count == 0:
        occurrence_ctx = ""
    else:
        occurrence_ctx = (
            f"Note: this {noun} has already happened {occurred_count} "
            f"time{'s' if occurred_count > 1 else ''} earlier in the video. "
            f"Focus ONLY on the most recent occurrence; do NOT answer YES if you are only "
            f"relying on those earlier occurrences.\n"
        )

    target_phrase = target if target else "the target object"

    return (
        f"Listen carefully to both the visual and audio content of this video stream "
        f"(up to {_fmt_time(current_time_sec)}).\n"
        f"{core_q}\n"
        f"{occurrence_ctx}"
        f"Evidence requirement:\n"
        f"- Answer YES only if you can identify the specific audible words, visible action, "
        f"or on-screen text in the clip that constitutes the event.\n"
        f"- Do not infer, predict, or imagine content that is not directly present in the "
        f"clip's audio or video.\n"
        f"If YES, also tell me the position of {target_phrase} in the frame where the event occurred, "
        f"choosing ONE of the following 9 regions:\n"
        f"  top-left | top-center | top-right\n"
        f"  center-left | center | center-right\n"
        f"  bottom-left | bottom-center | bottom-right\n\n"
        f"FORMAT REQUIREMENT (strict):\n"
        f"- Start your response with the single token YES or NO.\n"
        f"- If YES, use format: `YES: <brief description>. Position: <one of the 9 regions>`.\n"
        f"- If NO, reply with just `NO`.\n"
        f"- Do NOT add any preamble like \"The answer is\" or \"Therefore\"."
    )


def _convert_to_question(instruction: str) -> str:
    """
    Convert user-style instructions like 'Let me know when X'
    into direct questions like 'Did X happen in this video?'.
    """
    lower = instruction.lower().strip()
    for prefix in [
        "let me know when ", "let me know whenever ",
        "let me know if ", "let me know once ",
        "alert me when ", "alert me whenever ",
        "alert me if ", "alert me once ",
        "notify me when ", "notify me whenever ",
        "notify me if ", "notify me once ",
        "tell me when ", "tell me whenever ",
        "tell me if ", "tell me once ",
        "please let me know when ", "please let me know whenever ",
        "please let me know if ", "please let me know once ",
        "please alert me when ", "please notify me when ",
        "please tell me when ",
    ]:
        if lower.startswith(prefix):
            rest = instruction[len(prefix):].rstrip(".")
            return f"Did {rest} in this video? YES or NO."
    if instruction.rstrip().endswith("?"):
        return instruction
    return f"Did the following happen in this video: {instruction.rstrip('.')}? YES or NO."


def _counting_prompt(question: str, current_time_sec: float,
                     prev_response: Optional[str] = None) -> str:
    """For dedup_counting (polling)."""
    prev_ctx = ""
    if prev_response is not None:
        prev_ctx = f"\nYour previous count was: {prev_response}\n"
    return (
        f"You are watching a video stream. The user asked: \"{question}\"\n\n"
        f"The video currently shows content up to {_fmt_time(current_time_sec)}.\n"
        f"{prev_ctx}\n"
        f"Based on the video content so far, provide the current count.\n"
        f"Answer with ONLY a single integer number, nothing else."
    )


def _cumulative_counting_prompt(question: str, current_time_sec: float,
                                prev_response: Optional[str] = None) -> str:
    """For cumulative_counting: GT-probe style. Uses the raw question directly.

    Ask for an integer count of how many times the event has occurred so far.
    0 is a valid answer (i.e. it has not happened yet).
    """
    return (
        f"Listen carefully to both the visual and audio content of this video stream "
        f"(up to {_fmt_time(current_time_sec)}).\n"
        f"The user asked: \"{question}\"\n"
        f"Based on everything in the video so far, how many times has this happened?\n\n"
        f"FORMAT REQUIREMENT (strict):\n"
        f"- Reply with a SINGLE INTEGER only (e.g. 0, 1, 2, 3).\n"
        f"- Do NOT include any words, explanations, punctuation, timestamps, "
        f"units, quotes, or line breaks.\n"
        f"- 0 is the correct answer if it has not happened yet."
    )


def _dedup_counting_prompt(question: str, current_time_sec: float,
                           prev_response: Optional[str] = None) -> str:
    """For dedup_counting: GT-probe style. Uses the raw question directly.

    Ask for an integer count of DISTINCT items satisfying the question so far
    (deduplicated — don't double-count the same item across appearances).
    0 is a valid answer.
    """
    return (
        f"Listen carefully to both the visual and audio content of this video stream "
        f"(up to {_fmt_time(current_time_sec)}).\n"
        f"The user asked: \"{question}\"\n"
        f"Based on everything in the video so far, how many DISTINCT such items "
        f"have appeared? Do not double-count the same item when it appears multiple times.\n\n"
        f"FORMAT REQUIREMENT (strict):\n"
        f"- Reply with a SINGLE INTEGER only (e.g. 0, 1, 2, 3).\n"
        f"- Do NOT include any words, explanations, punctuation, timestamps, "
        f"units, quotes, or line breaks.\n"
        f"- 0 is the correct answer if none have appeared yet."
    )


def _snapshot_counting_prompt(question: str, current_time_sec: float,
                            prev_response: Optional[str] = None,
                            occurred_count: int = 0,
                            event: Optional[str] = None,
                            target: Optional[str] = None) -> str:
    """For snapshot_counting: alert-style trigger + count.

    Uses the pre-extracted `event` (trigger) and `target` (object to count)
    when available; falls back to the raw question otherwise.
    Evaluated via GT-probe (pre + post @ trigger_time+1), same protocol as grounding.
    """
    if event:
        core_q = f"Did {event} in this video? YES or NO."
    else:
        core_q = _convert_to_question(question)

    if occurred_count == 0:
        occurrence_ctx = ""
    else:
        occurrence_ctx = (
            f"Note: this event has already happened {occurred_count} "
            f"time{'s' if occurred_count > 1 else ''} earlier in the video. "
            f"Focus ONLY on the most recent occurrence; do NOT answer YES if you are only "
            f"relying on those earlier occurrences.\n"
        )

    target_phrase = target if target else "the requested objects"

    return (
        f"Listen carefully to both the visual and audio content of this video stream "
        f"(up to {_fmt_time(current_time_sec)}).\n"
        f"{core_q}\n"
        f"{occurrence_ctx}"
        f"If YES, count {target_phrase} in the frame where the event occurred.\n\n"
        f"FORMAT REQUIREMENT (strict):\n"
        f"- Start your response with YES or NO.\n"
        f"- If YES, use exactly this format: `YES: <integer>`. e.g. `YES: 5`.\n"
        f"- If NO, reply with just `NO`.\n"
        f"- Do NOT add any preamble like \"The answer is\" or \"Therefore\"."
    )


def _state_monitor_prompt(question: str, current_time_sec: float,
                          prev_response: Optional[str] = None,
                          states: Optional[list] = None) -> str:
    """For realtime_state_monitor: GT-probe style MCQ.

    Strongly emphasises that we want the state at the END of the clip
    (not a majority vote over the whole video), to counter the
    "model reports the dominant earlier state" failure mode.
    """
    states = states or []
    state_lines = "\n".join(f"  - {s}" for s in states)
    return (
        f"Listen carefully to both the visual and audio content of this video stream.\n"
        f"The video clip shown ends at {_fmt_time(current_time_sec)}. "
        f"You are being asked about the state at the VERY END of the clip, "
        f"NOT the overall video.\n\n"
        f"The user asked: \"{question}\"\n\n"
        f"At the final moment of the clip (the last 1-2 seconds), which of the "
        f"following states is currently active?\n"
        f"Possible states:\n{state_lines}\n\n"
        f"Even if earlier parts of the video showed a different state, answer "
        f"based on the CURRENT state at the end of the clip.\n"
        f"Answer with ONLY one state from the list above, using the exact same "
        f"wording. Do not include any other text."
    )


def _narration_prompt(question: str, current_time_sec: float,
                      options: Optional[list] = None) -> str:
    """For event_narration: multiple-choice question.

    Given the current video clip and a set of candidate narration summaries
    (shuffled by the evaluator, one of them is the correct answer for the
    current trigger segment), ask the model to pick the best-matching one.
    """
    options = options or []
    # Label options as (A), (B), ...
    letters = [chr(ord('A') + i) for i in range(len(options))]
    opt_lines = "\n".join(f"  ({l}) {txt}" for l, txt in zip(letters, options))
    letter_list = ", ".join(f"({l})" for l in letters)
    return (
        f"You are watching a video stream. The clip shown so far runs up to "
        f"{_fmt_time(current_time_sec)}.\n"
        f"The user asked: \"{question}\"\n\n"
        f"Several candidate narration summaries are listed below. Which ONE "
        f"best describes what has just happened in the video at the current "
        f"moment (around {_fmt_time(current_time_sec)})?\n\n"
        f"{opt_lines}\n\n"
        f"Answer with ONLY the single letter of your choice ({letter_list}). "
        f"Do not include any other text."
    )


def _default_prompt(question: str, current_time_sec: float,
                    prev_response: Optional[str] = None) -> str:
    return _alert_prompt(question, current_time_sec, prev_response)


def _seq_step_prompt(question: str, current_time_sec: float,
                     options: Optional[list] = None) -> str:
    """For sequential_step_instruction: MCQ for NEXT-step prediction.

    Probe at trigger_time (offset 0): the previous step has just been
    completed; the next step has NOT yet started in the clip shown. The
    model must predict what comes next from context.
    """
    options = options or []
    letters = [chr(ord('A') + i) for i in range(len(options))]
    opt_lines = "\n".join(f"  ({l}) {txt}" for l, txt in zip(letters, options))
    letter_list = ", ".join(f"({l})" for l in letters)
    return (
        f"You are watching a tutorial video stream. The clip shown so far "
        f"runs up to {_fmt_time(current_time_sec)}.\n"
        f"The user asked: \"{question}\"\n\n"
        f"Based on the progress in the video so far, which ONE of the "
        f"following steps should the user perform NEXT?\n\n"
        f"Important:\n"
        f"- You are being asked to PREDICT the upcoming step BEFORE it "
        f"starts. Do NOT describe what is currently on screen — the next "
        f"step has not happened yet.\n"
        f"- Some listed options describe steps that have ALREADY been "
        f"completed earlier in the video. Do NOT pick those.\n"
        f"- Some listed options describe steps that come much later. Pick "
        f"the ONE that should logically come IMMEDIATELY NEXT given what "
        f"has just finished.\n\n"
        f"{opt_lines}\n\n"
        f"Answer with ONLY the single letter of your choice ({letter_list}). "
        f"Do not include any other text."
    )


PROMPT_BUILDERS = {
    "instant_event_alert": _alert_prompt,
    "semantic_condition_alert": _alert_prompt,
    "explicit_target_grounding": _target_grounding_prompt,
    "cumulative_counting": _cumulative_counting_prompt,
    "dedup_counting": _dedup_counting_prompt,
    "snapshot_counting": _snapshot_counting_prompt,
    "realtime_state_monitor": _state_monitor_prompt,
    "event_narration": _narration_prompt,
    "sequential_step_instruction": _seq_step_prompt,
}


# ─── 响应解析 ──────────────────────────────────────────────────────────

def parse_response(task: str, raw_response: str, states: Optional[list] = None) -> dict:
    """
    Parse model response according to task type.

    Args:
        task: Task name.
        raw_response: Raw model output text.
        states: Closed state vocabulary (for realtime_state_monitor).

    Returns:
        dict with at least 'triggered' (bool) and task-specific fields.
    """
    if task == "realtime_state_monitor":
        return _parse_state_monitor(raw_response, states or [])
    parser = RESPONSE_PARSERS.get(task, _parse_alert)
    return parser(raw_response)


def _parse_alert(raw: str) -> dict:
    """Parse alert-type response.

    Robust to models that emit natural-language sentences (e.g. "The answer is
    YES, the event occurred.") by looking for standalone YES/NO tokens anywhere
    in the response. Prefers the FIRST occurrence.
    """
    raw_stripped = raw.strip()
    if not raw_stripped:
        return {"triggered": False}

    # Find the first standalone YES / NO (case-insensitive, word-boundary).
    yes_m = re.search(r"\byes\b", raw_stripped, re.IGNORECASE)
    no_m = re.search(r"\bno\b", raw_stripped, re.IGNORECASE)
    yes_pos = yes_m.start() if yes_m else -1
    no_pos = no_m.start() if no_m else -1

    if yes_pos < 0 and no_pos < 0:
        return {"triggered": False}
    if yes_pos < 0:
        return {"triggered": False}
    if no_pos < 0 or yes_pos < no_pos:
        # YES wins: extract description after "YES:" or "YES," or after yes itself
        desc = re.sub(r"^.*?\byes\b[:\s,]*", "", raw_stripped, count=1,
                      flags=re.IGNORECASE).strip()
        return {"triggered": True, "response": desc if desc else raw_stripped}
    # NO comes first → treat as NO
    return {"triggered": False}


def _strip_timestamps(text: str) -> str:
    """Remove common timestamp patterns that could be mistaken for integer
    counts by number extractors.

    Covers:
      - HH:MM:SS, MM:SS, M:SS (with optional leading zero in minutes)
      - Bracketed/parenthesized variants: [01:23], (1:23)
      - Ranges split by `-`, `–`, `to`: 0:21 - 0:34
      - Dangling partial timestamps from truncation: trailing "1:" at end
      - Prefix-like "00:" with no seconds (model fragments)
    """
    # 1. Full timestamps HH:MM:SS / MM:SS / M:SS (word boundary on both ends).
    #    Also consume a trailing "." or "," right after (e.g. "01:06.").
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b[.,]?", " ", text)
    # 2. Truncated / partial timestamps like a dangling "1:" at end of string
    #    or followed by non-digit (common in truncated model outputs).
    text = re.sub(r"\b\d{1,2}:(?=\D|$)", " ", text)
    return text


def _parse_counting(raw: str) -> dict:
    """Parse counting response: extract integer. count=0 means not triggered.

    Strategy (in order), timestamps are stripped up-front so keyword patterns
    can't mistakenly grab MM or SS digits:
      1. Pure integer string → return it.
      2. Keyword-anchored patterns on timestamp-stripped text:
         "a total of N", "N times", "answer/count/total/number is N",
         trailing ": N".
      3. Last integer in the timestamp-stripped text.
    """
    raw_stripped = raw.strip()
    if not raw_stripped:
        return {"triggered": False}

    # 1. Pure integer
    m = re.match(r"^\s*(\d+)\s*[\.\)]?\s*$", raw_stripped)
    if m:
        count = int(m.group(1))
        return {"triggered": count > 0, "count": count, "response": raw_stripped}

    # Strip timestamps first so MM/SS don't leak into keyword matches.
    cleaned = _strip_timestamps(raw_stripped)

    # 2. Keyword-anchored patterns (check on cleaned text)
    for pat in [
        r"(?:a\s+)?total\s+of\s+(\d+)",                        # "a total of 4"
        r"\b(\d+)\s+times?\b",                                  # "5 times"
        r"(?:answer|count|total|number)\s*(?:is|:|=)\s*(\d+)", # "answer is N"
        r"(?:reply|response)\s*(?:is|:)\s*['\"]?(\d+)",
        r":\s*(\d+)\s*[\.\)]?\s*$",                             # trailing ": 5"
    ]:
        m = re.search(pat, cleaned, re.IGNORECASE)
        if m:
            count = int(m.group(1))
            return {"triggered": count > 0, "count": count, "response": raw_stripped}

    # 3. Last integer in cleaned text
    numbers = re.findall(r"\d+", cleaned)
    if numbers:
        count = int(numbers[-1])
        return {"triggered": count > 0, "count": count, "response": raw_stripped}

    return {"triggered": False}


def _parse_state_monitor(raw: str, states: list) -> dict:
    """Parse state_monitor MCQ response: match one of the provided states.

    Matching strategy:
      1. Whole-string lowercased exact match.
      2. Otherwise: find the FIRST state (by order in `states`) that appears
         as a substring of the response (case-insensitive).
      3. If none match → state=None.
    """
    raw_stripped = raw.strip()
    raw_lower = raw_stripped.lower().strip(' \n\t".,;:!?')
    # Exact match after light strip
    for s in states:
        if raw_lower == s.lower():
            return {"triggered": True, "state": s, "response": raw_stripped}
    # Substring match: pick the first state (by list order) found
    # in the response. Prefer longer states first when they share a prefix
    # (e.g. "not playing" before "playing" to avoid misclassifying).
    for s in sorted(states, key=lambda x: -len(x)):
        if s.lower() in raw_lower:
            return {"triggered": True, "state": s, "response": raw_stripped}
    return {"triggered": False, "state": None, "response": raw_stripped}


def _parse_narration_mcq(raw: str) -> dict:
    """Parse event_narration / sequential_step_instruction MCQ response:
    extract the choice letter (A-Z).

    Robust to natural-language answers. Priority:
      1. Bare single letter as full (or near-full) response.
      2. Explicit "(X)" pattern.
      3. "Answer: X" / "answer is X" / "option X" / "choice X".
      4. First standalone uppercase letter EXCLUDING common English words
         like "I" and "A" (when "A" is article rather than option A).
    """
    raw_stripped = raw.strip()
    if not raw_stripped:
        return {"triggered": False, "choice": None, "response": raw_stripped}

    # 1. Bare letter (allow trailing "." ")" ":" etc.)
    m = re.match(r"^\s*\(?([A-Z])\)?[\s\.\):,;!?]*$", raw_stripped)
    if m:
        return {"triggered": True, "choice": m.group(1), "response": raw_stripped}

    # 2. "(X)" pattern anywhere
    m = re.search(r"\(([A-Z])\)", raw_stripped)
    if m:
        return {"triggered": True, "choice": m.group(1), "response": raw_stripped}

    # 3. Keyword-anchored extraction
    #    e.g. "Answer: B", "The answer is B.", "option B", "choice B", "response is B"
    for pat in [
        r"answer\s*(?:is|:)\s*([A-Z])\b",
        r"correct\s*answer\s*(?:is|:)\s*([A-Z])\b",
        r"\boption\s+([A-Z])\b",
        r"\bchoice\s+([A-Z])\b",
        r"response\s*(?:is|:)\s*([A-Z])\b",
        r"\bselect\s+([A-Z])\b",
        r"^\s*([A-Z])\s*[:\.\)]",  # "B." / "B)" / "B:" at very start
    ]:
        m = re.search(pat, raw_stripped, re.IGNORECASE)
        if m:
            return {"triggered": True, "choice": m.group(1).upper(),
                    "response": raw_stripped}

    # 4. Fallback: first standalone uppercase letter that is NOT a common
    #    English pronoun/article. Scan word by word.
    skip_letters = {"I", "A"}  # will handle "A" below if followed by MCQ marker
    for m in re.finditer(r"\b([A-Z])\b", raw_stripped):
        letter = m.group(1)
        # Context: if letter is surrounded by mcq-like markers, accept even if "A"
        start = m.start()
        window = raw_stripped[max(0, start - 6):min(len(raw_stripped), start + 6)]
        # If letter is right before/after an option indicator, accept
        if re.search(r"(option|choice|answer|\(|\)|\.)\s*" + letter + r"\b",
                     window, re.IGNORECASE):
            return {"triggered": True, "choice": letter, "response": raw_stripped}
        if letter not in skip_letters:
            return {"triggered": True, "choice": letter, "response": raw_stripped}

    return {"triggered": False, "choice": None, "response": raw_stripped}


def _parse_narration(raw: str) -> dict:
    """Parse narration response: UPDATE:/NO UPDATE."""
    raw_lower = raw.strip().lower()
    if "no update" in raw_lower:
        return {"triggered": False}
    if raw_lower.startswith("update"):
        desc = re.sub(r"^update[:\s,]*", "", raw.strip(), flags=re.IGNORECASE).strip()
        return {"triggered": True, "response": desc if desc else raw.strip()}
    if len(raw.strip()) > 15 and "no update" not in raw_lower:
        return {"triggered": True, "response": raw.strip()}
    return {"triggered": False}


def _parse_snapshot_counting(raw: str) -> dict:
    """Parse static counting response: YES: [number] / NO.

    Robust to natural-language outputs. Uses the same YES/NO detection as
    `_parse_alert`; if YES wins, extracts the first integer from the cleaned
    text (timestamps stripped).
    """
    alert = _parse_alert(raw)
    if not alert.get("triggered"):
        return {"triggered": False}
    cleaned = _strip_timestamps(raw.strip())
    numbers = re.findall(r"\d+", cleaned)
    if numbers:
        count = int(numbers[0])
        return {"triggered": True, "count": count, "response": raw.strip()}
    return {"triggered": True, "response": raw.strip()}


_POSITION_REGIONS = [
    "top-left", "top-center", "top-right",
    "center-left", "center", "center-right",
    "bottom-left", "bottom-center", "bottom-right",
]


def _parse_target_grounding(raw: str) -> dict:
    """Parse target grounding response: YES: <desc>. Position: <region> / NO.

    Robust to natural-language outputs. If YES wins (anywhere in the
    response), look for 'Position: <region>' or any region keyword.
    """
    alert = _parse_alert(raw)
    if not alert.get("triggered"):
        return {"triggered": False}

    raw_stripped = raw.strip()
    raw_lower = raw_stripped.lower()
    result = {"triggered": True, "response": raw_stripped}

    # Extract position via regex
    m = re.search(r"position\s*[:\-]?\s*([a-z\-]+)", raw_lower)
    if m:
        candidate = m.group(1).strip().rstrip(".,;")
        if candidate in _POSITION_REGIONS:
            result["position"] = candidate
            return result

    # Fallback: scan for any region keyword (prefer longer first to avoid
    # picking "center" when "center-left" is meant)
    for region in sorted(_POSITION_REGIONS, key=lambda x: -len(x)):
        if region in raw_lower:
            result["position"] = region
            break

    return result


RESPONSE_PARSERS = {
    "instant_event_alert": _parse_alert,
    "semantic_condition_alert": _parse_alert,
    "explicit_target_grounding": _parse_target_grounding,
    "cumulative_counting": _parse_counting,
    "dedup_counting": _parse_counting,
    "snapshot_counting": _parse_snapshot_counting,
    "realtime_state_monitor": _parse_state_monitor,
    "event_narration": _parse_narration_mcq,
    "sequential_step_instruction": _parse_narration_mcq,
}


# ─── 辅助 ─────────────────────────────────────────────────────────────

def _fmt_time(sec: float) -> str:
    """Format seconds to MM:SS."""
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


# =============================================================================
# Online (true-streaming) prompts
# =============================================================================
# Shared contract:
#   * Evaluator feeds one frame per second and calls model.observe(frame, t, history).
#   * Model output grammar depends on the task (TRIGGER/UPDATE/STATE/NARRATION/STEP).
#   * Empty output = STANDBY (no emit this tick).
#   * History is a list of dicts [{"t_sec": float, "raw": str}, ...] and is
#     rendered verbatim so the model knows what it has already said.
#
# Two helper functions:
#   build_online_session_prompt(task, sample) -> str    # passed to model.begin()
#   render_observe_prompt(task, sample, current_t, history) -> str
# The first is a one-time instruction; the second is the per-tick prompt fed
# alongside the current frame. Concrete StreamingModel implementations may use
# either or both (some models keep the session prompt in KV cache).


def _render_history(history) -> str:
    """Render [{'t_sec':float, 'raw':str}, ...] as bracketed MM:SS lines."""
    if not history:
        return "(none)"
    lines = []
    for h in history:
        ts = _fmt_time(h.get("t_sec", 0))
        raw = str(h.get("raw", "")).strip()
        lines.append(f"[{ts}] {raw}")
    return "\n".join(lines)


# Per-task online instructions. Each string describes:
#   (a) what the user wants monitored,
#   (b) output rules (silent when nothing to report; concise when triggered),
#   (c) don't-repeat constraint worded for the task's semantics.
# Single unified template set used for ALL online streaming models.
_ONLINE_TASK_INSTRUCTIONS = {
    # ── Text类：输出非空=检测到事件，全文即描述 ──
    "instant_event_alert": (
        "Watch this live video stream and monitor for the event described "
        "below. The user asked:\n"
        "  \"{question}\"\n\n"
        "Rules:\n"
        "- If the event is NOT happening right now, stay completely silent "
        "(output nothing).\n"
        "- If the event has just occurred, briefly describe what happened "
        "in one sentence.\n"
        "- Do NOT repeat yourself: once you have alerted for an occurrence, "
        "do NOT alert again for the SAME occurrence. Only alert again if a "
        "NEW, separate occurrence happens later."
    ),
    "semantic_condition_alert": (
        "Watch this live video stream and monitor for the semantic condition "
        "described below. The user asked:\n"
        "  \"{question}\"\n\n"
        "Rules:\n"
        "- If the condition is not currently satisfied, stay silent.\n"
        "- When the condition becomes satisfied, briefly describe what made "
        "the condition hold in one sentence.\n"
        "- Do NOT repeat yourself: only alert again if the condition becomes "
        "satisfied again later in a distinct moment."
    ),
    "event_narration": (
        "Watch this live video stream. The user asked:\n"
        "  \"{question}\"\n\n"
        "Rules:\n"
        "- When a noteworthy event occurs, briefly describe what just "
        "happened in one sentence.\n"
        "- Otherwise, stay silent.\n"
        "- Do NOT repeat yourself: only narrate when something substantially "
        "new happens."
    ),
    "sequential_step_instruction": (
        "Watch this live video stream. The user asked:\n"
        "  \"{question}\"\n\n"
        "Rules:\n"
        "- When a new step begins or is about to begin, describe the step "
        "in one sentence.\n"
        "- Otherwise, stay silent.\n"
        "- Do NOT repeat yourself: announce a step only once when it begins."
    ),
    # ── 结构化类：约束输出格式便于提取 ──
    "explicit_target_grounding": (
        "Watch this live video stream. The user asked:\n"
        "  \"{question}\"\n\n"
        "Rules:\n"
        "- If the target event has NOT just occurred, stay silent.\n"
        "- When the target event occurs, output in this format:\n"
        "    <brief description>. Position: <region>\n"
        "  where <region> is EXACTLY one of:\n"
        "    top-left | top-center | top-right\n"
        "    center-left | center | center-right\n"
        "    bottom-left | bottom-center | bottom-right\n"
        "- Do NOT repeat yourself."
    ),
    "snapshot_counting": (
        "Watch this live video stream. The user asked:\n"
        "  \"{question}\"\n\n"
        "Rules:\n"
        "- If the trigger condition has NOT yet occurred, stay silent.\n"
        "- When the trigger condition occurs, output ONLY the count as a "
        "single integer (e.g. 5). No other text.\n"
        "- Do NOT repeat yourself."
    ),
    "cumulative_counting": (
        "Watch this live video stream. The user asked:\n"
        "  \"{question}\"\n\n"
        "Rules:\n"
        "- When you see a NEW occurrence that increments the cumulative "
        "count, output ONLY the updated total as a single integer "
        "(e.g. 3). No other text.\n"
        "- Otherwise, stay silent.\n"
        "- Do NOT repeat yourself: emit only once per new occurrence."
    ),
    "dedup_counting": (
        "Watch this live video stream. The user asked:\n"
        "  \"{question}\"\n\n"
        "Rules:\n"
        "- When a NEW, previously-unseen instance appears, output ONLY "
        "the updated distinct count as a single integer (e.g. 2). "
        "No other text.\n"
        "- Otherwise, stay silent.\n"
        "- Do NOT repeat yourself: re-appearances do NOT count."
    ),
    "realtime_state_monitor": (
        "Watch this live video stream. The user asked:\n"
        "  \"{question}\"\n\n"
        "Possible states:\n{states}\n\n"
        "Rules:\n"
        "- When the monitored state CHANGES, output ONLY the new state "
        "name from the list above. No other text.\n"
        "- Otherwise, stay silent.\n"
        "- Do NOT repeat yourself: only emit when the state actually changes."
    ),
}


def build_online_session_prompt(task: str, sample: dict, model_name: str = "") -> str:
    """One-time instruction passed to model.begin() for a sample.

    Does NOT include history or current timestamp — those change per tick
    and are rendered by `render_observe_prompt`.

    Args:
        task: Task name
        sample: Sample dictionary
        model_name: Kept for backward compatibility (unused — the same
            unified template set is used for every online model).
    """
    # Extract states vocabulary for RSM
    extra_kwargs = {}
    if task == "realtime_state_monitor":
        states = _extract_states(sample)
        if states:
            extra_kwargs["states"] = "\n".join(f"  - {s}" for s in states)

    tmpl = _ONLINE_TASK_INSTRUCTIONS.get(task)
    if tmpl is None:
        raise ValueError(f"No online prompt template for task {task!r}")
    return tmpl.format(question=sample.get("question", ""), **extra_kwargs)


def _extract_states(sample: dict) -> list:
    """Extract unique state names from RSM ground truth."""
    states = set()
    for gt in sample.get("ground_truth", []):
        if gt.get("state_to"):
            states.add(gt["state_to"])
        if gt.get("state_from"):
            states.add(gt["state_from"])
    return sorted(states)


def render_observe_prompt(task: str, sample: dict, current_time_sec: float,
                          history) -> str:
    """Per-tick prompt fed alongside the current frame.

    Includes:
      - session instruction (so models without persistent state still know
        the task),
      - the clock,
      - rendered history so the model sees what it already said.

    Models that keep the session prompt in persistent context (KV cache) may
    ignore the leading instruction and use only the tail portion.
    """
    session = build_online_session_prompt(task, sample)
    return (
        f"{session}\n\n"
        f"Your previous responses so far:\n{_render_history(history)}\n\n"
        f"Current clock: {_fmt_time(current_time_sec)}. "
        f"What is your output at this moment? (Remember: empty = stay silent.)"
    )
