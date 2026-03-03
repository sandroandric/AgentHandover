"""Tests for the VLM scene annotation module.

All tests run without a real VLM — they use canned responses and mock
the Ollama call. Tests marked ``@pytest.mark.vlm`` require a running
Ollama instance and are excluded from CI.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from oc_apprentice_worker.scene_annotator import (
    AnnotationConfig,
    AnnotationResult,
    SceneAnnotator,
    _StaleTracker,
    _strip_markdown_fences,
    _validate_annotation,
    build_annotation_prompt,
    _build_context_section,
)

from conftest import insert_event, DAEMON_SCHEMA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_ANNOTATION = {
    "app": "Google Chrome",
    "location": "https://github.com/issues/new",
    "visible_content": {
        "headings": ["New Issue"],
        "labels": ["Title", "Description", "Labels", "Submit"],
        "values": ["Title: Bug report"],
    },
    "ui_state": {
        "active_element": "Title field (just filled)",
        "modals_or_popups": "none",
        "scroll_position": "top",
    },
    "task_context": {
        "what_doing": "Creating a new bug report on GitHub",
        "likely_next": "Fill in the description field",
        "is_workflow": True,
    },
}

NON_WORKFLOW_ANNOTATION = {
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
        "scroll_position": "middle",
    },
    "task_context": {
        "what_doing": "Browsing Reddit front page",
        "likely_next": "Click on a post",
        "is_workflow": False,
    },
}


def _canned_vlm_response(annotation: dict = VALID_ANNOTATION) -> str:
    """Return a canned VLM JSON response."""
    return json.dumps(annotation)


def _canned_vlm_call(*args, **kwargs) -> tuple[str, float]:
    """Mock Ollama VLM call returning a valid annotation."""
    return _canned_vlm_response(), 1.5


def _canned_vlm_call_non_workflow(*args, **kwargs) -> tuple[str, float]:
    return json.dumps(NON_WORKFLOW_ANNOTATION), 1.2


def _failing_vlm_call(*args, **kwargs) -> tuple[str, float]:
    """Mock that returns invalid JSON."""
    return "This is not JSON at all, sorry!", 0.5


def _fenced_vlm_call(*args, **kwargs) -> tuple[str, float]:
    """Mock that returns JSON inside markdown fences."""
    return f"```json\n{_canned_vlm_response()}\n```", 1.5


def _thinking_vlm_call(*args, **kwargs) -> tuple[str, float]:
    """Mock that returns JSON with <think> tags."""
    return f"<think>Analyzing the screenshot...</think>\n{_canned_vlm_response()}", 1.5


# ---------------------------------------------------------------------------
# JSON validation tests
# ---------------------------------------------------------------------------

class TestStripMarkdownFences:

    def test_plain_json(self):
        raw = '{"key": "value"}'
        assert _strip_markdown_fences(raw) == raw

    def test_json_code_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        assert _strip_markdown_fences(raw) == '{"key": "value"}'

    def test_plain_code_fence(self):
        raw = '```\n{"key": "value"}\n```'
        assert _strip_markdown_fences(raw) == '{"key": "value"}'

    def test_thinking_tags(self):
        raw = '<think>reasoning here</think>\n{"key": "value"}'
        result = _strip_markdown_fences(raw)
        assert result == '{"key": "value"}'

    def test_combined_thinking_and_fences(self):
        raw = '<think>hmm</think>\n```json\n{"k": 1}\n```'
        result = _strip_markdown_fences(raw)
        assert result == '{"k": 1}'

    def test_empty_string(self):
        assert _strip_markdown_fences("") == ""

    def test_whitespace(self):
        assert _strip_markdown_fences("   ") == ""


class TestValidateAnnotation:

    def test_valid_full_annotation(self):
        raw = json.dumps(VALID_ANNOTATION)
        result = _validate_annotation(raw)
        assert result is not None
        assert result["app"] == "Google Chrome"
        assert result["task_context"]["is_workflow"] is True

    def test_minimal_valid(self):
        """Only task_context.what_doing is mandatory."""
        raw = json.dumps({"task_context": {"what_doing": "reading docs"}})
        result = _validate_annotation(raw)
        assert result is not None

    def test_missing_what_doing(self):
        raw = json.dumps({"task_context": {"likely_next": "something"}})
        assert _validate_annotation(raw) is None

    def test_missing_task_context(self):
        raw = json.dumps({"app": "Chrome"})
        assert _validate_annotation(raw) is None

    def test_is_workflow_string_true(self):
        raw = json.dumps({
            "task_context": {"what_doing": "test", "is_workflow": "true"}
        })
        result = _validate_annotation(raw)
        assert result["task_context"]["is_workflow"] is True

    def test_is_workflow_string_false(self):
        raw = json.dumps({
            "task_context": {"what_doing": "test", "is_workflow": "false"}
        })
        result = _validate_annotation(raw)
        assert result["task_context"]["is_workflow"] is False

    def test_is_workflow_int(self):
        raw = json.dumps({
            "task_context": {"what_doing": "test", "is_workflow": 1}
        })
        result = _validate_annotation(raw)
        assert result["task_context"]["is_workflow"] is False

    def test_not_a_dict(self):
        assert _validate_annotation("[1,2,3]") is None

    def test_empty_string(self):
        assert _validate_annotation("") is None

    def test_markdown_fenced(self):
        raw = f"```json\n{json.dumps(VALID_ANNOTATION)}\n```"
        result = _validate_annotation(raw)
        assert result is not None
        assert result["app"] == "Google Chrome"

    def test_garbage(self):
        assert _validate_annotation("not json at all") is None


# ---------------------------------------------------------------------------
# Prompt construction tests
# ---------------------------------------------------------------------------

class TestBuildAnnotationPrompt:

    def test_no_context(self):
        prompt = build_annotation_prompt()
        assert "task_context" in prompt
        assert "PREVIOUS FRAMES" not in prompt

    def test_empty_context(self):
        prompt = build_annotation_prompt([])
        assert "PREVIOUS FRAMES" not in prompt

    def test_with_context(self):
        recent = [
            {
                "timestamp": "2026-03-03T09:14:50.000Z",
                "scene_annotation_json": json.dumps({
                    "app": "Chrome",
                    "location": "https://example.com",
                    "task_context": {"what_doing": "Filling form"},
                }),
            },
        ]
        prompt = build_annotation_prompt(recent)
        assert "PREVIOUS FRAMES" in prompt
        assert "Chrome" in prompt
        assert "Filling form" in prompt

    def test_context_with_bad_annotation(self):
        recent = [
            {"timestamp": "2026-03-03T09:14:50.000Z", "scene_annotation_json": "not json"},
        ]
        prompt = build_annotation_prompt(recent)
        # Should gracefully handle bad JSON — no crash, no context section
        assert "task_context" in prompt


class TestBuildContextSection:

    def test_no_annotations(self):
        assert _build_context_section([]) == ""

    def test_multiple_annotations(self):
        recent = [
            {
                "timestamp": "2026-03-03T09:15:02.000Z",
                "scene_annotation_json": json.dumps({
                    "app": "Chrome",
                    "location": "github.com",
                    "task_context": {"what_doing": "Reviewing PR"},
                }),
            },
            {
                "timestamp": "2026-03-03T09:14:50.000Z",
                "scene_annotation_json": json.dumps({
                    "app": "Chrome",
                    "location": "github.com",
                    "task_context": {"what_doing": "Opening PR page"},
                }),
            },
        ]
        section = _build_context_section(recent)
        assert "Opening PR page" in section
        assert "Reviewing PR" in section


# ---------------------------------------------------------------------------
# Stale tracker tests
# ---------------------------------------------------------------------------

class TestStaleTracker:

    def test_workflow_resets(self):
        tracker = _StaleTracker()
        ann = {
            "app": "Chrome",
            "location": "github.com",
            "task_context": {"is_workflow": True, "what_doing": "coding"},
        }
        tracker.update(ann)
        assert not tracker.should_skip(3)

    def test_non_workflow_accumulates(self):
        tracker = _StaleTracker()
        ann = {
            "app": "Chrome",
            "location": "reddit.com",
            "task_context": {"is_workflow": False, "what_doing": "browsing"},
        }
        for _ in range(3):
            tracker.update(ann)

        assert tracker.should_skip(3)

    def test_app_change_resets(self):
        tracker = _StaleTracker()
        ann1 = {
            "app": "Chrome",
            "location": "reddit.com",
            "task_context": {"is_workflow": False, "what_doing": "browsing"},
        }
        ann2 = {
            "app": "Finder",
            "location": "/Users",
            "task_context": {"is_workflow": False, "what_doing": "file browsing"},
        }
        for _ in range(3):
            tracker.update(ann1)
        assert tracker.should_skip(3)

        tracker.update(ann2)
        assert not tracker.should_skip(3)

    def test_location_change_resets(self):
        tracker = _StaleTracker()
        ann1 = {
            "app": "Chrome",
            "location": "reddit.com",
            "task_context": {"is_workflow": False, "what_doing": "browsing reddit"},
        }
        ann2 = {
            "app": "Chrome",
            "location": "github.com",
            "task_context": {"is_workflow": False, "what_doing": "browsing github"},
        }
        for _ in range(3):
            tracker.update(ann1)
        assert tracker.should_skip(3)

        tracker.update(ann2)
        assert not tracker.should_skip(3)

    def test_reset_method(self):
        tracker = _StaleTracker()
        ann = {
            "app": "Chrome",
            "location": "reddit.com",
            "task_context": {"is_workflow": False, "what_doing": "browsing"},
        }
        for _ in range(5):
            tracker.update(ann)
        assert tracker.should_skip(3)

        tracker.reset()
        assert not tracker.should_skip(3)


# ---------------------------------------------------------------------------
# SceneAnnotator integration (with mocked VLM)
# ---------------------------------------------------------------------------

class TestSceneAnnotator:

    def test_annotate_valid_response(self, tmp_path):
        """Mock VLM returns valid JSON → annotation succeeds."""
        # Create a fake screenshot
        img = tmp_path / "screenshot.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        config = AnnotationConfig(delete_screenshot_on_success=False)
        annotator = SceneAnnotator(config)

        event = {
            "id": "evt-001",
            "timestamp": "2026-03-03T09:14:50.000Z",
            "artifact_ids_json": '["screenshot"]',
            "metadata_json": json.dumps({"screenshot_path": str(img)}),
            "window_json": "{}",
            "annotation_status": "pending",
        }

        with patch(
            "oc_apprentice_worker.scene_annotator._call_ollama_vlm",
            side_effect=_canned_vlm_call,
        ):
            result = annotator.annotate_event(event)

        assert result.status == "completed"
        assert result.annotation is not None
        assert result.annotation["app"] == "Google Chrome"
        assert result.annotation["task_context"]["is_workflow"] is True

    def test_annotate_missing_screenshot(self):
        config = AnnotationConfig()
        annotator = SceneAnnotator(config)

        event = {
            "id": "evt-002",
            "timestamp": "2026-03-03T09:14:50.000Z",
            "artifact_ids_json": "[]",
            "metadata_json": "{}",
            "window_json": "{}",
            "annotation_status": "pending",
        }

        result = annotator.annotate_event(event)
        assert result.status == "missing_screenshot"

    def test_annotate_invalid_json_retries(self, tmp_path):
        """Invalid JSON triggers one retry."""
        img = tmp_path / "screenshot.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        call_count = [0]

        def _mock_call(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "not json", 0.5
            return _canned_vlm_response(), 1.0

        config = AnnotationConfig(delete_screenshot_on_success=False)
        annotator = SceneAnnotator(config)

        event = {
            "id": "evt-003",
            "timestamp": "2026-03-03T09:14:50.000Z",
            "artifact_ids_json": '["screenshot"]',
            "metadata_json": json.dumps({"screenshot_path": str(img)}),
            "window_json": "{}",
            "annotation_status": "pending",
        }

        with patch(
            "oc_apprentice_worker.scene_annotator._call_ollama_vlm",
            side_effect=_mock_call,
        ):
            result = annotator.annotate_event(event)

        assert result.status == "completed"
        assert call_count[0] == 2  # original + retry

    def test_annotate_both_attempts_fail(self, tmp_path):
        """Both attempts fail → status=failed."""
        img = tmp_path / "screenshot.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        config = AnnotationConfig(delete_screenshot_on_success=False)
        annotator = SceneAnnotator(config)

        event = {
            "id": "evt-004",
            "timestamp": "2026-03-03T09:14:50.000Z",
            "artifact_ids_json": '["screenshot"]',
            "metadata_json": json.dumps({"screenshot_path": str(img)}),
            "window_json": "{}",
            "annotation_status": "pending",
        }

        with patch(
            "oc_apprentice_worker.scene_annotator._call_ollama_vlm",
            side_effect=_failing_vlm_call,
        ):
            result = annotator.annotate_event(event)

        assert result.status == "failed"
        assert "invalid_json" in result.error

    def test_markdown_fenced_response(self, tmp_path):
        """VLM wraps response in markdown fences → still succeeds."""
        img = tmp_path / "screenshot.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        config = AnnotationConfig(delete_screenshot_on_success=False)
        annotator = SceneAnnotator(config)

        event = {
            "id": "evt-005",
            "timestamp": "2026-03-03T09:14:50.000Z",
            "artifact_ids_json": '["screenshot"]',
            "metadata_json": json.dumps({"screenshot_path": str(img)}),
            "window_json": "{}",
            "annotation_status": "pending",
        }

        with patch(
            "oc_apprentice_worker.scene_annotator._call_ollama_vlm",
            side_effect=_fenced_vlm_call,
        ):
            result = annotator.annotate_event(event)

        assert result.status == "completed"

    def test_thinking_tags_stripped(self, tmp_path):
        """<think> tags are stripped before JSON parsing."""
        img = tmp_path / "screenshot.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        config = AnnotationConfig(delete_screenshot_on_success=False)
        annotator = SceneAnnotator(config)

        event = {
            "id": "evt-006",
            "timestamp": "2026-03-03T09:14:50.000Z",
            "artifact_ids_json": '["screenshot"]',
            "metadata_json": json.dumps({"screenshot_path": str(img)}),
            "window_json": "{}",
            "annotation_status": "pending",
        }

        with patch(
            "oc_apprentice_worker.scene_annotator._call_ollama_vlm",
            side_effect=_thinking_vlm_call,
        ):
            result = annotator.annotate_event(event)

        assert result.status == "completed"

    def test_connection_error(self, tmp_path):
        """Ollama not running → graceful failure."""
        img = tmp_path / "screenshot.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        config = AnnotationConfig()
        annotator = SceneAnnotator(config)

        event = {
            "id": "evt-007",
            "timestamp": "2026-03-03T09:14:50.000Z",
            "artifact_ids_json": '["screenshot"]',
            "metadata_json": json.dumps({"screenshot_path": str(img)}),
            "window_json": "{}",
            "annotation_status": "pending",
        }

        def _raise_conn(*args, **kwargs):
            raise ConnectionError("Ollama not reachable")

        with patch(
            "oc_apprentice_worker.scene_annotator._call_ollama_vlm",
            side_effect=_raise_conn,
        ):
            result = annotator.annotate_event(event)

        assert result.status == "failed"
        assert "ollama_connection" in result.error

    def test_screenshot_deleted_on_success(self, tmp_path):
        """Screenshot file is deleted after successful annotation."""
        img = tmp_path / "screenshot.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        config = AnnotationConfig(delete_screenshot_on_success=True)
        annotator = SceneAnnotator(config)

        event = {
            "id": "evt-008",
            "timestamp": "2026-03-03T09:14:50.000Z",
            "artifact_ids_json": '["screenshot"]',
            "metadata_json": json.dumps({"screenshot_path": str(img)}),
            "window_json": "{}",
            "annotation_status": "pending",
        }

        with patch(
            "oc_apprentice_worker.scene_annotator._call_ollama_vlm",
            side_effect=_canned_vlm_call,
        ):
            result = annotator.annotate_event(event)

        assert result.status == "completed"
        assert not img.exists(), "Screenshot should be deleted after success"

    def test_screenshot_kept_on_failure(self, tmp_path):
        """Screenshot is NOT deleted when annotation fails."""
        img = tmp_path / "screenshot.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        config = AnnotationConfig(delete_screenshot_on_success=True)
        annotator = SceneAnnotator(config)

        event = {
            "id": "evt-009",
            "timestamp": "2026-03-03T09:14:50.000Z",
            "artifact_ids_json": '["screenshot"]',
            "metadata_json": json.dumps({"screenshot_path": str(img)}),
            "window_json": "{}",
            "annotation_status": "pending",
        }

        with patch(
            "oc_apprentice_worker.scene_annotator._call_ollama_vlm",
            side_effect=_failing_vlm_call,
        ):
            result = annotator.annotate_event(event)

        assert result.status == "failed"
        assert img.exists(), "Screenshot should be kept on failure"

    def test_stats_tracking(self, tmp_path):
        """Stats counters increment correctly."""
        img = tmp_path / "screenshot.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        config = AnnotationConfig(delete_screenshot_on_success=False)
        annotator = SceneAnnotator(config)

        event = {
            "id": "evt-010",
            "timestamp": "2026-03-03T09:14:50.000Z",
            "artifact_ids_json": '["screenshot"]',
            "metadata_json": json.dumps({"screenshot_path": str(img)}),
            "window_json": "{}",
            "annotation_status": "pending",
        }

        with patch(
            "oc_apprentice_worker.scene_annotator._call_ollama_vlm",
            side_effect=_canned_vlm_call,
        ):
            annotator.annotate_event(event)

        assert annotator.stats["annotated"] == 1
        assert annotator.stats["failed"] == 0


# ---------------------------------------------------------------------------
# DB annotation methods
# ---------------------------------------------------------------------------

class TestDBAnnotationMethods:

    def test_save_and_retrieve_annotation(self, tmp_db_path, write_conn):
        """save_annotation + get_recent_annotations round-trip."""
        from oc_apprentice_worker.db import WorkerDB

        eid = insert_event(
            write_conn,
            timestamp="2026-03-03T09:14:50.000Z",
        )

        with WorkerDB(tmp_db_path) as db:
            ok = db.save_annotation(
                eid,
                json.dumps(VALID_ANNOTATION),
                status="completed",
            )
            assert ok

        # Re-open to read back
        with WorkerDB(tmp_db_path) as db:
            recent = db.get_recent_annotations(
                before_timestamp="2026-03-03T09:15:00.000Z",
                limit=5,
            )
            assert len(recent) == 1
            assert recent[0]["id"] == eid

    def test_get_unannotated_events(self, tmp_db_path, write_conn):
        from oc_apprentice_worker.db import WorkerDB

        eid = insert_event(write_conn, timestamp="2026-03-03T09:14:50.000Z")

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unannotated_events(limit=10)
            assert len(events) == 1
            assert events[0]["id"] == eid

    def test_save_frame_diff(self, tmp_db_path, write_conn):
        from oc_apprentice_worker.db import WorkerDB

        eid = insert_event(write_conn, timestamp="2026-03-03T09:14:50.000Z")
        diff = {"diff_type": "no_change", "what_doing": "reading", "duration_seconds": 30}

        with WorkerDB(tmp_db_path) as db:
            ok = db.save_frame_diff(eid, json.dumps(diff))
            assert ok

        # Verify via raw query
        row = write_conn.execute(
            "SELECT frame_diff_json FROM events WHERE id = ?",
            (eid,),
        ).fetchone()
        assert row is not None
        parsed = json.loads(row[0])
        assert parsed["diff_type"] == "no_change"

    def test_get_events_needing_diff(self, tmp_db_path, write_conn):
        from oc_apprentice_worker.db import WorkerDB

        eid = insert_event(write_conn, timestamp="2026-03-03T09:14:50.000Z")
        # Manually mark as completed with annotation but no diff
        write_conn.execute(
            "UPDATE events SET annotation_status = 'completed', "
            "scene_annotation_json = ? WHERE id = ?",
            (json.dumps(VALID_ANNOTATION), eid),
        )
        write_conn.commit()

        with WorkerDB(tmp_db_path) as db:
            events = db.get_events_needing_diff(limit=10)
            assert len(events) == 1
            assert events[0]["id"] == eid

    def test_get_annotation_before(self, tmp_db_path, write_conn):
        from oc_apprentice_worker.db import WorkerDB

        eid1 = insert_event(write_conn, timestamp="2026-03-03T09:14:50.000Z")
        eid2 = insert_event(write_conn, timestamp="2026-03-03T09:15:00.000Z")

        write_conn.execute(
            "UPDATE events SET annotation_status = 'completed', "
            "scene_annotation_json = ? WHERE id = ?",
            (json.dumps(VALID_ANNOTATION), eid1),
        )
        write_conn.commit()

        with WorkerDB(tmp_db_path) as db:
            prev = db.get_annotation_before("2026-03-03T09:15:00.000Z")
            assert prev is not None
            assert prev["id"] == eid1

    def test_focus_first_ordering(self, tmp_db_path, write_conn):
        """Focus session events should come before non-focus events."""
        from oc_apprentice_worker.db import WorkerDB

        # Normal event (earlier timestamp)
        eid1 = insert_event(
            write_conn,
            timestamp="2026-03-03T09:14:00.000Z",
        )
        # Focus event (later timestamp, but should come first)
        eid2 = insert_event(
            write_conn,
            timestamp="2026-03-03T09:15:00.000Z",
        )
        write_conn.execute(
            "UPDATE events SET metadata_json = ? WHERE id = ?",
            (json.dumps({"focus_session_id": "focus-123"}), eid2),
        )
        write_conn.commit()

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unannotated_events(limit=10, focus_first=True)
            assert len(events) == 2
            # Focus event should be first despite later timestamp
            assert events[0]["id"] == eid2
