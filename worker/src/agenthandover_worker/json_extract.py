"""Robust JSON extraction from model output (the single JSON extractor).

Models wrap JSON in ``<think>`` blocks, markdown fences, preamble text or
chat-template end tokens, and sometimes stop mid-object when they hit
their token budget.  :func:`extract_json` is the one place that copes
with all of it; callers validate the fields they need on the dict it
returns.
"""

from __future__ import annotations

import json
import re
from typing import Any

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
# Chat-template tokens some models (Qwen) leak at the end of the output;
# everything from the first one on is noise.
_END_TOKENS = ("<|endoftext|>", "<|im_end|>", "<|end|>")
# How many cut points the truncation repair tries, from the end backwards.
_MAX_REPAIR_CANDIDATES = 200


def clean_model_text(text: str) -> str:
    """Model output without ``<think>`` blocks and trailing end tokens."""
    text = _THINK_RE.sub("", text)
    for token in _END_TOKENS:
        text = text.split(token)[0]
    return text.strip()


def extract_json(text: str, *, repair_truncated: bool = False) -> dict[str, Any]:
    """Extract a JSON object from model output text.

    Strategy, after removing ``<think>`` blocks and end tokens:

    1. Direct ``json.loads`` (fast path for clean output).
    2. The content of the first markdown code block (```json ... ```).
    3. Each ``{`` position up to the last ``}`` (text with unrelated
       braces or prose around the JSON).
    4. With *repair_truncated*: an object cut off by the token limit is
       closed at its last complete value (see :func:`_repair_truncated`).

    Raises:
        ValueError: If no JSON object can be extracted (including when
            the output is valid JSON but not an object).
    """
    stripped = clean_model_text(text)
    if not stripped:
        raise ValueError("Empty input text")

    # Tier 1: direct parse
    try:
        result = json.loads(stripped)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(result, dict):
            return result
        raise ValueError(
            f"Expected JSON object, got {type(result).__name__}: {stripped[:200]}"
        )

    # Tier 2: markdown code block (```json ... ``` or ``` ... ```)
    fenced = _FENCE_RE.search(stripped)
    if fenced:
        block = fenced.group(1).strip()
        try:
            result = json.loads(block)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Tier 3: try each { position (handles text with unrelated braces).
    # A { inside an object that never closes is part of a truncated
    # object, not an answer of its own: a nested step must not pass for
    # the whole response.
    last_brace = stripped.rfind("}")
    open_at_end = _first_unclosed(stripped)
    start = 0
    while last_brace != -1:
        pos = stripped.find("{", start)
        if pos == -1 or pos >= last_brace:
            break
        if open_at_end is not None and pos > open_at_end:
            break
        try:
            result = json.loads(stripped[pos : last_brace + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        start = pos + 1

    # Tier 4: close an object cut off mid-way
    if repair_truncated:
        # A fenced block that did not parse holds the cut-off object.
        candidate = fenced.group(1) if fenced else stripped
        # The cut-off object is the one still open at the end, not a
        # closed brace in the prose before it.
        first = _first_unclosed(candidate)
        if first is None or candidate[first] != "{":
            first = candidate.find("{")
        if first != -1:
            repaired = _repair_truncated(candidate[first:])
            if repaired is not None:
                return repaired

    raise ValueError(f"Could not extract JSON object from text: {stripped[:200]}")


def _first_unclosed(text: str) -> int | None:
    """Index of the outermost ``{`` or ``[`` still open at the end of
    *text* (string-aware), or None when every bracket closes."""
    openers: list[int] = []
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            openers.append(i)
        elif ch in "}]" and openers:
            openers.pop()
    return openers[0] if openers else None


def _repair_truncated(text: str) -> dict[str, Any] | None:
    """Close a JSON object that stops mid-way (the model hit its token limit).

    Scans *text* (which starts at ``{``) once, string-aware, noting every
    point where a complete value ends together with the brackets still
    open there.  From the last such point backwards, it cuts the text,
    appends the closing brackets and returns the first result that
    parses as an object.  A partial trailing value (an unterminated
    string, a key without a value) is dropped rather than guessed.
    """
    stack: list[str] = []
    cuts: list[tuple[int, str]] = []
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                cuts.append((i + 1, "".join(reversed(stack))))
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if not stack:
                break
            stack.pop()
            cuts.append((i + 1, "".join(reversed(stack))))
            if not stack:
                break  # the object closed; nothing to repair after it
        elif ch == ",":
            cuts.append((i, "".join(reversed(stack))))

    for end, closers in reversed(cuts[-_MAX_REPAIR_CANDIDATES:]):
        try:
            result = json.loads(text[:end] + closers)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result
    return None
