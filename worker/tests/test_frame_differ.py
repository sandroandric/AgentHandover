"""Tests for the frame diff engine.

All tests run without a real LLM — edge cases are tested with code-only
markers and LLM diffs use mocked responses.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agenthandover_worker.frame_differ import (
    DiffConfig,
    DiffResult,
    FrameDiffer,
    _detect_edge_case,
    _format_annotation_for_diff,
    _make_app_switch_marker,
    _make_no_change_marker,
    _make_session_gap_marker,
    _make_stale_skip_marker,
    _parse_timestamp,
    _validate_diff,
)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

ANN_CHROME_GITHUB = {
    "app": "Google Chrome",
    "location": "https://github.com/issues/new",
    "visible_content": {
        "headings": ["New Issue"],
        "labels": ["Title", "Description"],
        "values": ["Title: Bug report"],
    },
    "ui_state": {
        "active_element": "Title field",
        "modals_or_popups": "none",
        "scroll_position": "top",
    },
    "task_context": {
        "what_doing": "Creating a new bug report",
        "likely_next": "Fill description",
        "is_workflow": True,
    },
}

ANN_CHROME_GITHUB_FILLED = {
    "app": "Google Chrome",
    "location": "https://github.com/issues/new",
    "visible_content": {
        "headings": ["New Issue"],
        "labels": ["Title", "Description", "Labels"],
        "values": ["Title: Bug report", "Description: Steps to reproduce...", "Label: bug"],
    },
    "ui_state": {
        "active_element": "Submit button",
        "modals_or_popups": "none",
        "scroll_position": "bottom",
    },
    "task_context": {
        "what_doing": "Submitting bug report on GitHub",
        "likely_next": "Click submit",
        "is_workflow": True,
    },
}

ANN_FINDER = {
    "app": "Finder",
    "location": "/Users/sandro/Documents",
    "visible_content": {
        "headings": ["Documents"],
        "labels": [],
        "values": [],
    },
    "ui_state": {
        "active_element": "file list",
        "modals_or_popups": "none",
        "scroll_position": "top",
    },
    "task_context": {
        "what_doing": "Browsing documents folder",
        "likely_next": "Open a file",
        "is_workflow": False,
    },
}

ANN_REDDIT = {
    "app": "Google Chrome",
    "location": "https://reddit.com",
    "visible_content": {
        "headings": ["Popular"],
        "labels": [],
        "values": [],
    },
    "ui_state": {
        "active_element": "none",
        "modals_or_popups": "none",
        "scroll_position": "top",
    },
    "task_context": {
        "what_doing": "Browsing Reddit front page",
        "likely_next": "Click on a post",
        "is_workflow": False,
    },
}

VALID_DIFF_RESPONSE = {
    "diff_type": "action",
    "actions": [
        "Typed 'Steps to reproduce...' in Description",
        "Selected 'bug' label from dropdown",
    ],
    "inputs": [
        {"field": "Description", "value": "Steps to reproduce..."},
        {"field": "Labels", "value": "bug"},
    ],
    "navigation": "none (same page)",
    "step_description": "User filled description and selected bug label",
}


def _make_event(
    event_id: str,
    timestamp: str,
    annotation: dict,
    annotation_status: str = "completed",
) -> dict:
    return {
        "id": event_id,
        "timestamp": timestamp,
        "scene_annotation_json": json.dumps(annotation),
        "annotation_status": annotation_status,
        "window_json": "{}",
    }


# ---------------------------------------------------------------------------
# Marker construction tests
# ---------------------------------------------------------------------------

class TestMarkers:

    def test_app_switch_marker(self):
        m = _make_app_switch_marker("Chrome", "Finder")
        assert m["diff_type"] == "app_switch"
        assert m["from_app"] == "Chrome"
        assert m["to_app"] == "Finder"

    def test_session_gap_marker(self):
        m = _make_session_gap_marker(43200, "laptop_sleep")
        assert m["diff_type"] == "session_gap"
        assert m["gap_seconds"] == 43200
        assert m["reason"] == "laptop_sleep"

    def test_no_change_marker(self):
        m = _make_no_change_marker("reading docs", 180)
        assert m["diff_type"] == "no_change"
        assert m["what_doing"] == "reading docs"
        assert m["duration_seconds"] == 180

    def test_stale_skip_marker(self):
        m = _make_stale_skip_marker()
        assert m["diff_type"] == "stale_skip"


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

class TestParseTimestamp:

    def test_iso_with_millis(self):
        ts = _parse_timestamp("2026-03-03T09:14:50.123Z")
        assert ts > 0

    def test_iso_no_millis(self):
        ts = _parse_timestamp("2026-03-03T09:14:50Z")
        assert ts > 0

    def test_iso_offset(self):
        ts = _parse_timestamp("2026-03-03T09:14:50.123+00:00")
        assert ts > 0

    def test_invalid(self):
        assert _parse_timestamp("not-a-date") == 0.0

    def test_ordering(self):
        t1 = _parse_timestamp("2026-03-03T09:14:50.000Z")
        t2 = _parse_timestamp("2026-03-03T09:15:00.000Z")
        assert t2 > t1


# ---------------------------------------------------------------------------
# Edge case detection
# ---------------------------------------------------------------------------

class TestDetectEdgeCase:

    def test_app_switch_falls_through_to_llm(self):
        """App switches no longer short-circuit — they fall through to LLM diff
        for richer action descriptions (changed 2026-03-28)."""
        prev = _make_event("e1", "2026-03-03T09:14:50.000Z", ANN_CHROME_GITHUB)
        curr = _make_event("e2", "2026-03-03T09:14:55.000Z", ANN_FINDER)
        config = DiffConfig()

        result = _detect_edge_case(
            prev, curr,
            ANN_CHROME_GITHUB, ANN_FINDER,
            config,
        )
        # App switches now return None so they get full LLM analysis
        assert result is None

    def test_session_gap(self):
        prev = _make_event("e1", "2026-03-03T09:00:00.000Z", ANN_CHROME_GITHUB)
        curr = _make_event("e2", "2026-03-03T10:00:00.000Z", ANN_CHROME_GITHUB_FILLED)
        config = DiffConfig(session_gap_seconds=600)

        result = _detect_edge_case(
            prev, curr,
            ANN_CHROME_GITHUB, ANN_CHROME_GITHUB_FILLED,
            config,
        )
        assert result is not None
        assert result["diff_type"] == "session_gap"
        assert result["gap_seconds"] == 3600

    def test_no_change(self):
        """Same app + location + values + what_doing → no_change."""
        prev = _make_event("e1", "2026-03-03T09:14:50.000Z", ANN_REDDIT)
        curr = _make_event("e2", "2026-03-03T09:15:05.000Z", ANN_REDDIT)
        config = DiffConfig()

        result = _detect_edge_case(
            prev, curr,
            ANN_REDDIT, ANN_REDDIT,
            config,
        )
        assert result is not None
        assert result["diff_type"] == "no_change"
        assert result["what_doing"] == "Browsing Reddit front page"

    def test_real_change_no_edge_case(self):
        """Different values on same page → None (needs LLM diff)."""
        prev = _make_event("e1", "2026-03-03T09:14:50.000Z", ANN_CHROME_GITHUB)
        curr = _make_event("e2", "2026-03-03T09:15:05.000Z", ANN_CHROME_GITHUB_FILLED)
        config = DiffConfig()

        result = _detect_edge_case(
            prev, curr,
            ANN_CHROME_GITHUB, ANN_CHROME_GITHUB_FILLED,
            config,
        )
        assert result is None, "Real change should require LLM diff"


# ---------------------------------------------------------------------------
# Diff validation
# ---------------------------------------------------------------------------

class TestValidateDiff:

    def test_valid_diff(self):
        raw = json.dumps(VALID_DIFF_RESPONSE)
        result = _validate_diff(raw)
        assert result is not None
        assert result["diff_type"] == "action"
        assert len(result["actions"]) == 2

    def test_markdown_fenced(self):
        raw = f"```json\n{json.dumps(VALID_DIFF_RESPONSE)}\n```"
        result = _validate_diff(raw)
        assert result is not None

    def test_thinking_tags(self):
        raw = f"<think>analyzing...</think>\n{json.dumps(VALID_DIFF_RESPONSE)}"
        result = _validate_diff(raw)
        assert result is not None

    def test_missing_actions_and_description(self):
        raw = json.dumps({"diff_type": "action"})
        result = _validate_diff(raw)
        assert result is None

    def test_minimal_valid(self):
        raw = json.dumps({
            "actions": ["Clicked submit button"],
            "step_description": "User submitted the form",
        })
        result = _validate_diff(raw)
        assert result is not None
        assert result["diff_type"] == "action"  # auto-added

    def test_invalid_json(self):
        assert _validate_diff("not json") is None

    def test_not_a_dict(self):
        assert _validate_diff("[1,2,3]") is None


# ---------------------------------------------------------------------------
# Format annotation for diff
# ---------------------------------------------------------------------------

class TestFormatAnnotation:

    def test_full_annotation(self):
        text = _format_annotation_for_diff(ANN_CHROME_GITHUB)
        assert "Google Chrome" in text
        assert "github.com" in text
        assert "Title" in text
        assert "Creating a new bug report" in text

    def test_minimal_annotation(self):
        ann = {"app": "Finder", "location": "/tmp"}
        text = _format_annotation_for_diff(ann)
        assert "Finder" in text
        assert "/tmp" in text


# ---------------------------------------------------------------------------
# FrameDiffer integration (with mocked LLM)
# ---------------------------------------------------------------------------

class TestFrameDiffer:

    def test_app_switch_goes_to_llm(self):
        """App switches now fall through to LLM diff for richer analysis.
        Without Ollama, falls back to a failed marker."""
        differ = FrameDiffer()
        prev = _make_event("e1", "2026-03-03T09:14:50.000Z", ANN_CHROME_GITHUB)
        curr = _make_event("e2", "2026-03-03T09:14:55.000Z", ANN_FINDER)

        result = differ.diff_pair(prev, curr)
        # Without LLM available, should get a failed marker (not an edge case)
        assert result.diff["diff_type"] in ("action", "diff_failed")
        assert differ.stats["edge_cases"] == 0

    def test_edge_case_session_gap(self):
        differ = FrameDiffer(DiffConfig(session_gap_seconds=600))
        prev = _make_event("e1", "2026-03-03T09:00:00.000Z", ANN_CHROME_GITHUB)
        curr = _make_event("e2", "2026-03-03T10:00:00.000Z", ANN_CHROME_GITHUB_FILLED)

        result = differ.diff_pair(prev, curr)
        assert result.diff["diff_type"] == "session_gap"

    def test_edge_case_no_change(self):
        differ = FrameDiffer()
        prev = _make_event("e1", "2026-03-03T09:14:50.000Z", ANN_REDDIT)
        curr = _make_event("e2", "2026-03-03T09:15:00.000Z", ANN_REDDIT)

        result = differ.diff_pair(prev, curr)
        assert result.diff["diff_type"] == "no_change"

    def test_stale_skipped_prev(self):
        differ = FrameDiffer()
        prev = _make_event("e1", "2026-03-03T09:14:50.000Z", ANN_REDDIT, "skipped")
        curr = _make_event("e2", "2026-03-03T09:15:00.000Z", ANN_CHROME_GITHUB)

        result = differ.diff_pair(prev, curr)
        assert result.diff["diff_type"] == "stale_skip"

    def test_llm_diff_success(self):
        """Real content change → calls LLM, gets valid diff."""
        differ = FrameDiffer()
        prev = _make_event("e1", "2026-03-03T09:14:50.000Z", ANN_CHROME_GITHUB)
        curr = _make_event("e2", "2026-03-03T09:15:05.000Z", ANN_CHROME_GITHUB_FILLED)

        def _mock_llm(*args, **kwargs):
            return json.dumps(VALID_DIFF_RESPONSE), 3.6

        with patch(
            "agenthandover_worker.scene_annotator._call_ollama_vlm",
            side_effect=_mock_llm,
        ):
            result = differ.diff_pair(prev, curr)

        assert result.diff["diff_type"] == "action"
        assert len(result.diff["actions"]) == 2
        assert result.inference_time_seconds == pytest.approx(3.6)
        assert differ.stats["diffs_computed"] == 1

    def test_llm_diff_failure(self):
        """LLM returns garbage → diff_failed marker."""
        differ = FrameDiffer()
        prev = _make_event("e1", "2026-03-03T09:14:50.000Z", ANN_CHROME_GITHUB)
        curr = _make_event("e2", "2026-03-03T09:15:05.000Z", ANN_CHROME_GITHUB_FILLED)

        def _mock_llm(*args, **kwargs):
            return "This is not JSON", 1.0

        with patch(
            "agenthandover_worker.scene_annotator._call_ollama_vlm",
            side_effect=_mock_llm,
        ):
            result = differ.diff_pair(prev, curr)

        assert result.diff["diff_type"] == "diff_failed"
        assert differ.stats["failed"] == 1

    def test_llm_connection_error(self):
        """Ollama not running → diff_failed marker."""
        differ = FrameDiffer()
        prev = _make_event("e1", "2026-03-03T09:14:50.000Z", ANN_CHROME_GITHUB)
        curr = _make_event("e2", "2026-03-03T09:15:05.000Z", ANN_CHROME_GITHUB_FILLED)

        def _mock_llm(*args, **kwargs):
            raise ConnectionError("Ollama not reachable")

        with patch(
            "agenthandover_worker.scene_annotator._call_ollama_vlm",
            side_effect=_mock_llm,
        ):
            result = differ.diff_pair(prev, curr)

        assert result.diff["diff_type"] == "diff_failed"
        assert "ollama_connection" in result.diff["error"]

    def test_missing_annotation_in_prev(self):
        differ = FrameDiffer()
        prev = {"id": "e1", "timestamp": "2026-03-03T09:14:50.000Z",
                "scene_annotation_json": None, "annotation_status": "completed"}
        curr = _make_event("e2", "2026-03-03T09:15:05.000Z", ANN_CHROME_GITHUB)

        result = differ.diff_pair(prev, curr)
        assert result.diff["diff_type"] == "diff_failed"
        assert "missing_annotation" in result.diff["error"]
