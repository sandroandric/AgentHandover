"""Frame-to-frame diff engine for consecutive scene annotations.

Compares pairs of consecutive annotations to produce action descriptions:
what the user DID between two frames (typed text, clicked buttons,
navigated pages, etc.).

Produces four types of markers:
- "action": real diff via text-only LLM call (~3.6s)
- "app_switch": app changed (code-only, free)
- "session_gap": time gap > threshold (code-only, free)
- "no_change": identical content (code-only, free)
- "stale_skip": previous frame was stale-skipped (code-only, free)
- "diff_failed": LLM diff failed (fallback marker)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from agenthandover_worker.json_extract import extract_json
from agenthandover_worker.scene_annotator import as_compact

logger = logging.getLogger(__name__)

# Time gap threshold for session_gap marker (seconds)
SESSION_GAP_THRESHOLD = 600  # 10 minutes


@dataclass
class DiffConfig:
    """Configuration for the frame differ."""

    model: str = "qwen3.5:4b"
    ollama_host: str = "http://localhost:11434"
    num_predict: int = 400
    session_gap_seconds: int = SESSION_GAP_THRESHOLD


@dataclass
class DiffResult:
    """Result of comparing two consecutive annotations."""

    event_id: str
    diff: dict
    inference_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Code-only markers (no LLM call needed)
# ---------------------------------------------------------------------------

def _make_app_switch_marker(from_app: str, to_app: str) -> dict:
    return {
        "diff_type": "app_switch",
        "from_app": from_app,
        "to_app": to_app,
    }


def _make_session_gap_marker(gap_seconds: float, reason: str = "time_gap") -> dict:
    return {
        "diff_type": "session_gap",
        "gap_seconds": int(gap_seconds),
        "reason": reason,
    }


def _make_no_change_marker(what_doing: str, duration_seconds: float) -> dict:
    return {
        "diff_type": "no_change",
        "what_doing": what_doing,
        "duration_seconds": int(duration_seconds),
    }


def _make_stale_skip_marker(reason: str = "stale_frame") -> dict:
    return {
        "diff_type": "stale_skip",
        "reason": reason,
    }


def _make_diff_failed_marker(error: str) -> dict:
    return {
        "diff_type": "diff_failed",
        "error": error,
    }


# ---------------------------------------------------------------------------
# Edge case detection
# ---------------------------------------------------------------------------

def _parse_annotation(event: dict) -> dict | None:
    """Extract the parsed annotation from an event row, in the compact
    schema (:func:`scene_annotator.as_compact`)."""
    raw = event.get("scene_annotation_json")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return as_compact(parsed) if isinstance(parsed, dict) else None


def _parse_timestamp(ts_str: str) -> float:
    """Parse ISO timestamp to Unix epoch seconds."""
    from datetime import datetime, timezone

    # Handle various ISO formats
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f+00:00",
        "%Y-%m-%dT%H:%M:%S+00:00",
    ):
        try:
            dt = datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue

    # Fallback: try fromisoformat
    try:
        from datetime import datetime as _dt
        dt = _dt.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, AttributeError):
        return 0.0


def _detect_edge_case(
    prev_event: dict,
    curr_event: dict,
    prev_annotation: dict,
    curr_annotation: dict,
    config: DiffConfig,
) -> dict | None:
    """Check for edge cases that produce code-only markers (no LLM needed).

    Returns a marker dict if an edge case is detected, None otherwise.
    """
    prev_ts = prev_event.get("timestamp", "")
    curr_ts = curr_event.get("timestamp", "")

    # Time gap check
    if prev_ts and curr_ts:
        gap = _parse_timestamp(curr_ts) - _parse_timestamp(prev_ts)
        if gap > config.session_gap_seconds:
            return _make_session_gap_marker(gap, "time_gap")

    # App switch: don't short-circuit — fall through to LLM diff.
    # App switches are often the most meaningful transitions (e.g.,
    # "copied product info from PH, switched to Gmail, pasted into email").
    # The LLM diff will capture what happened during the switch.
    prev_app = prev_annotation.get("app", "")
    curr_app = curr_annotation.get("app", "")

    # No-change check: same app + location + same screen text and typing
    prev_loc = prev_annotation.get("location", "")
    curr_loc = curr_annotation.get("location", "")
    prev_what = prev_annotation.get("task_context", {}).get("what_doing", "")
    curr_what = curr_annotation.get("task_context", {}).get("what_doing", "")

    def _content(annotation: dict) -> tuple:
        kt = annotation.get("key_text") or {}
        return (annotation.get("_screen_text", ""), kt.get("typed_text", ""))

    if (
        prev_app == curr_app
        and prev_loc == curr_loc
        and _content(prev_annotation) == _content(curr_annotation)
        and prev_what == curr_what
    ):
        gap = 0.0
        if prev_ts and curr_ts:
            gap = _parse_timestamp(curr_ts) - _parse_timestamp(prev_ts)
        return _make_no_change_marker(curr_what, gap)

    # No edge case — needs LLM diff
    return None


# ---------------------------------------------------------------------------
# LLM diff prompt
# ---------------------------------------------------------------------------

DIFF_SYSTEM_PROMPT = (
    "You compare two consecutive screen annotations and identify what "
    "actions the user took between them. "
    "Respond with ONLY valid JSON, no markdown fences, no commentary."
)

DIFF_PROMPT_TEMPLATE = """\
Compare these two consecutive screen annotations and identify what the user did between Frame A and Frame B.

FRAME A (before):
{frame_a}

FRAME B (after):
{frame_b}

Respond with this JSON structure:
{{
  "diff_type": "action",
  "actions": ["<list each distinct user action as a sentence>"],
  "inputs": [
    {{"field": "<field name>", "value": "<what was typed/selected>"}}
  ],
  "navigation": "<describe any page/URL/screen changes, or 'none'>",
  "step_description": "<one sentence summarizing what the user did>"
}}

Respond with ONLY the JSON object."""


# The screen text per frame in the diff prompt (two frames per call).
_DIFF_SCREEN_TEXT_MAX_CHARS = 600


def _format_annotation_for_diff(annotation: dict) -> str:
    """Format an annotation dict into a compact text representation for the diff prompt."""
    lines = []
    app = annotation.get("app", "?")
    location = annotation.get("location", "?")
    lines.append(f"App: {app}")
    lines.append(f"Location: {location}")

    screen = " | ".join(
        line.strip() for line in str(annotation.get("_screen_text") or "").splitlines()
        if line.strip()
    )
    if screen:
        lines.append(f"Screen text: {screen[:_DIFF_SCREEN_TEXT_MAX_CHARS]}")
    kt = annotation.get("key_text") or {}
    if kt.get("typed_text"):
        lines.append(f"Typed: {kt['typed_text']}")
    items = [
        f"{item.get('label')}{' (chosen)' if item.get('chosen') else ''}"
        for item in annotation.get("visible_items") or []
        if isinstance(item, dict) and item.get("label")
    ]
    if items:
        lines.append(f"Items: {'; '.join(items)}")

    ui = annotation.get("ui_state", {})
    if ui.get("active_element"):
        lines.append(f"Active element: {ui['active_element']}")
    if ui.get("modal"):
        lines.append(f"Modal/popup: {ui['modal']}")

    tc = annotation.get("task_context", {})
    if tc.get("what_doing"):
        lines.append(f"Doing: {tc['what_doing']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON validation for diff output
# ---------------------------------------------------------------------------

def _validate_diff(raw: str) -> dict | None:
    """Parse and validate a diff JSON response."""
    try:
        data = extract_json(raw)
    except ValueError:
        return None

    # Must have diff_type and at least step_description or actions
    if "diff_type" not in data:
        data["diff_type"] = "action"

    if "actions" not in data and "step_description" not in data:
        return None

    return data


# ---------------------------------------------------------------------------
# FrameDiffer
# ---------------------------------------------------------------------------

class FrameDiffer:
    """Compares consecutive annotation pairs to produce action diffs.

    Designed to run in a dedicated thread within the worker process.
    """

    def __init__(self, config: DiffConfig | None = None) -> None:
        self.config = config or DiffConfig()
        self._stats = {
            "diffs_computed": 0,
            "edge_cases": 0,
            "failed": 0,
        }

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def diff_pair(
        self,
        prev_event: dict,
        curr_event: dict,
    ) -> DiffResult:
        """Compute the diff between two consecutive annotated events.

        Returns a DiffResult with the diff dict (may be an edge-case
        marker or an LLM-generated action diff).
        """
        curr_id = curr_event.get("id", "unknown")

        # Parse annotations
        prev_ann = _parse_annotation(prev_event)
        curr_ann = _parse_annotation(curr_event)

        if prev_ann is None or curr_ann is None:
            self._stats["failed"] += 1
            return DiffResult(
                event_id=curr_id,
                diff=_make_diff_failed_marker("missing_annotation"),
            )

        # Check for stale-skipped previous frame
        prev_status = prev_event.get("annotation_status", "")
        if prev_status == "skipped":
            self._stats["edge_cases"] += 1
            return DiffResult(
                event_id=curr_id,
                diff=_make_stale_skip_marker("previous_frame_skipped"),
            )

        # Check edge cases (app switch, time gap, no change)
        edge_case = _detect_edge_case(
            prev_event, curr_event, prev_ann, curr_ann, self.config
        )
        if edge_case is not None:
            self._stats["edge_cases"] += 1
            return DiffResult(event_id=curr_id, diff=edge_case)

        # --- LLM diff ---
        frame_a = _format_annotation_for_diff(prev_ann)
        frame_b = _format_annotation_for_diff(curr_ann)

        prompt = DIFF_PROMPT_TEMPLATE.format(frame_a=frame_a, frame_b=frame_b)

        try:
            from agenthandover_worker.scene_annotator import (
                ANNOTATION_KEEP_ALIVE, _call_ollama_vlm,
            )
            from agenthandover_worker.model_profiles import get_profile
            profile = get_profile(self.config.model)

            # The differ runs on the annotation model: the same num_ctx and
            # keep_alive keep Ollama from reloading or unloading it early.
            raw_response, inference_time = _call_ollama_vlm(
                model=self.config.model,
                prompt=prompt,
                host=self.config.ollama_host,
                num_predict=profile.diff_num_predict,
                num_ctx=profile.ann_num_ctx,
                keep_alive=ANNOTATION_KEEP_ALIVE,
                system=profile.diff_system or DIFF_SYSTEM_PROMPT,
                think=profile.diff_think,
                extra_options={
                    k: v for k, v in profile.diff_options().items()
                    if k not in ("num_predict",)
                },
            )
        except ConnectionError as exc:
            self._stats["failed"] += 1
            return DiffResult(
                event_id=curr_id,
                diff=_make_diff_failed_marker(f"ollama_connection: {exc}"),
            )
        except Exception as exc:
            self._stats["failed"] += 1
            return DiffResult(
                event_id=curr_id,
                diff=_make_diff_failed_marker(f"llm_error: {exc}"),
            )

        # Validate
        diff = _validate_diff(raw_response)
        if diff is None:
            self._stats["failed"] += 1
            return DiffResult(
                event_id=curr_id,
                diff=_make_diff_failed_marker("invalid_json"),
                inference_time_seconds=inference_time,
            )

        self._stats["diffs_computed"] += 1
        return DiffResult(
            event_id=curr_id,
            diff=diff,
            inference_time_seconds=inference_time,
        )
