"""Robust JSON extraction from VLM model output.

VLM models often wrap JSON in markdown code blocks, add preamble text,
or produce slightly malformed output. This module provides a three-tier
parsing strategy to extract valid JSON from such output.
"""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from model output text.

    Three-tier parsing strategy:
    1. Direct ``json.loads`` (fast path for clean output).
    2. Extract from markdown code block (```json ... ``` or ``` ... ```).
    3. Try each ``{`` position paired with matching ``}`` positions
       (handles text with unrelated braces before the JSON).

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty input text")

    # Tier 1: direct parse
    try:
        result = json.loads(stripped)
        if isinstance(result, dict):
            return result
        raise ValueError(
            f"Expected JSON object, got {type(result).__name__}: "
            f"{stripped[:200]}"
        )
    except json.JSONDecodeError:
        pass

    # Tier 2: markdown code block (```json ... ``` or ``` ... ```)
    code_block_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```",
        stripped,
        re.DOTALL,
    )
    if code_block_match:
        block_content = code_block_match.group(1).strip()
        try:
            result = json.loads(block_content)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Tier 3: try each { position (handles text with unrelated braces)
    last_brace = stripped.rfind("}")
    if last_brace != -1:
        start = 0
        while True:
            pos = stripped.find("{", start)
            if pos == -1 or pos >= last_brace:
                break
            candidate = stripped[pos : last_brace + 1]
            try:
                result = json.loads(candidate)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass
            start = pos + 1

    raise ValueError(
        f"Could not extract JSON object from text: {stripped[:200]}"
    )
