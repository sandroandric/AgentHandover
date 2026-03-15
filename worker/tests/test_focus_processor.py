"""Tests for the v2 focus session processor."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from oc_apprentice_worker.focus_processor import FocusProcessor
from oc_apprentice_worker.scene_annotator import (
    AnnotationConfig,
    AnnotationResult,
    SceneAnnotator,
)
from oc_apprentice_worker.frame_differ import DiffConfig, DiffResult, FrameDiffer
from oc_apprentice_worker.sop_generator import (
    SOPGenerator,
    SOPGeneratorConfig,
    GeneratedSOP,
)
from oc_apprentice_worker.db import WorkerDB


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

DAEMON_SCHEMA = """
CREATE TABLE events (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    kind_json TEXT,
    window_json TEXT,
    metadata_json TEXT DEFAULT '{}',
    artifact_ids_json TEXT DEFAULT '[]',
    processed INTEGER DEFAULT 0,
    display_ids_spanned_json TEXT,
    scene_annotation_json TEXT DEFAULT NULL,
    annotation_status TEXT DEFAULT 'pending',
    frame_diff_json TEXT DEFAULT NULL
);
CREATE INDEX idx_events_annotation_status ON events(annotation_status);
CREATE TABLE vlm_queue (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    priority REAL DEFAULT 0.0,
    status TEXT DEFAULT 'pending',
    ttl_expires_at TEXT,
    result_json TEXT,
    processed_at TEXT
);
CREATE TABLE episodes (
    id TEXT PRIMARY KEY,
    segment_id TEXT,
    status TEXT DEFAULT 'active',
    event_count INTEGER DEFAULT 0,
    created_at TEXT
);
CREATE TABLE artifacts (
    id TEXT PRIMARY KEY,
    event_id TEXT,
    artifact_type TEXT,
    file_path TEXT
);
"""


def _create_test_db(tmp_path: Path) -> Path:
    """Create a fresh test database with v2 schema."""
    db_path = tmp_path / "events.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(DAEMON_SCHEMA)
    conn.close()
    return db_path


def _insert_focus_events(
    db_path: Path,
    session_id: str,
    count: int = 5,
    *,
    annotated: bool = False,
):
    """Insert focus session events into the test DB."""
    conn = sqlite3.connect(str(db_path))
    for i in range(count):
        event_id = f"evt-{session_id[:8]}-{i}"
        metadata = json.dumps({"focus_session_id": session_id})
        ann_status = "completed" if annotated else "pending"
        ann_json = json.dumps({
            "app": "Google Chrome",
            "location": "https://example.com",
            "visible_content": {"headings": [f"Step {i+1}"], "labels": [], "values": []},
            "ui_state": {"active_element": "input", "modals_or_popups": "none", "scroll_position": "top"},
            "task_context": {
                "what_doing": f"Performing step {i+1}",
                "likely_next": "Next action",
                "is_workflow": True,
            },
        }) if annotated else None

        diff_json = json.dumps({
            "diff_type": "action",
            "actions": [f"Action for step {i+1}"],
            "inputs": [{"field": "input", "value": f"val_{i}"}],
            "step_description": f"User did step {i+1}",
        }) if annotated and i > 0 else (
            json.dumps({"diff_type": "first_frame"}) if annotated and i == 0 else None
        )

        conn.execute(
            "INSERT INTO events "
            "(id, timestamp, kind_json, window_json, metadata_json, "
            " annotation_status, scene_annotation_json, frame_diff_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                f"2026-03-03T10:00:{i:02d}Z",
                '{"DwellSnapshot":{}}',
                '{"app_bundle_id":"com.google.Chrome","title":"Example"}',
                metadata,
                ann_status,
                ann_json,
                diff_json,
            ),
        )
    conn.commit()
    conn.close()


def _make_mock_annotator():
    """Create a mock SceneAnnotator that returns success for all events."""
    annotator = MagicMock(spec=SceneAnnotator)
    annotator.config = AnnotationConfig()

    def mock_annotate(event, *, recent_annotations=None, artifact_dir=None, skip_stale_check=False):
        return AnnotationResult(
            event_id=event.get("id", "unknown"),
            status="completed",
            annotation={
                "app": "Google Chrome",
                "location": "https://example.com",
                "visible_content": {"headings": ["Test"], "labels": [], "values": []},
                "ui_state": {"active_element": "", "modals_or_popups": "none", "scroll_position": "top"},
                "task_context": {
                    "what_doing": "Testing the task",
                    "likely_next": "Next step",
                    "is_workflow": True,
                },
            },
            inference_time_seconds=1.0,
        )

    annotator.annotate_event.side_effect = mock_annotate
    return annotator


def _make_mock_differ():
    """Create a mock FrameDiffer that returns action diffs."""
    differ = MagicMock(spec=FrameDiffer)

    def mock_diff(prev, current):
        return DiffResult(
            event_id=current.get("id", "unknown"),
            diff={
                "diff_type": "action",
                "actions": ["Mock action"],
                "inputs": [{"field": "test", "value": "mock"}],
                "step_description": "Mock diff",
            },
            inference_time_seconds=0.5,
        )

    differ.diff_pair.side_effect = mock_diff
    return differ


def _make_mock_sop_generator(success=True, title="Test Task"):
    """Create a mock SOPGenerator."""
    generator = MagicMock(spec=SOPGenerator)

    def mock_generate(timeline, title_arg):
        if not success:
            return GeneratedSOP(
                sop={}, success=False, error="Mock failure"
            )
        return GeneratedSOP(
            sop={
                "slug": "test-task",
                "title": title_arg,
                "steps": [{"step": "action", "target": "test", "selector": None, "parameters": {}, "confidence": 0.7, "pre_state": {}}],
                "variables": [],
                "confidence_avg": 0.70,
                "episode_count": 1,
                "abs_support": 1,
                "apps_involved": ["Google Chrome"],
                "preconditions": [],
                "task_description": "A test task",
                "execution_overview": {},
                "source": "v2_focus_recording",
            },
            inference_time_seconds=72.0,
        )

    generator.generate_from_focus.side_effect = mock_generate
    return generator


# ---------------------------------------------------------------------------
# TestFocusProcessor
# ---------------------------------------------------------------------------

class TestFocusProcessorBasic:
    def test_empty_events(self, tmp_path):
        """Empty events list returns error."""
        db_path = _create_test_db(tmp_path)
        with WorkerDB(db_path) as db:
            processor = FocusProcessor(
                _make_mock_annotator(),
                _make_mock_differ(),
                _make_mock_sop_generator(),
            )
            result = processor.process_session(
                db, "session-1", "Test", []
            )
        assert not result.success
        assert "No events" in result.error

    def test_success_with_unannotated_events(self, tmp_path):
        """Events get annotated, diffed, and SOP generated."""
        db_path = _create_test_db(tmp_path)
        session_id = "focus-sess-001"
        _insert_focus_events(db_path, session_id, count=3, annotated=False)

        with WorkerDB(db_path) as db:
            annotator = _make_mock_annotator()
            differ = _make_mock_differ()
            sop_gen = _make_mock_sop_generator()

            processor = FocusProcessor(annotator, differ, sop_gen)
            result = processor.process_session(
                db, session_id, "Expense Report",
                db.get_focus_session_events(session_id),
                screenshots_dir=str(tmp_path),
            )

        assert result.success
        assert result.sop["title"] == "Expense Report"
        assert annotator.annotate_event.call_count == 3
        assert sop_gen.generate_from_focus.call_count == 1

    def test_success_with_already_annotated_events(self, tmp_path):
        """Already-annotated events skip annotation step."""
        db_path = _create_test_db(tmp_path)
        session_id = "focus-sess-002"
        _insert_focus_events(db_path, session_id, count=4, annotated=True)

        with WorkerDB(db_path) as db:
            annotator = _make_mock_annotator()
            differ = _make_mock_differ()
            sop_gen = _make_mock_sop_generator()

            processor = FocusProcessor(annotator, differ, sop_gen)
            result = processor.process_session(
                db, session_id, "Already Done",
                db.get_focus_session_events(session_id),
                screenshots_dir=str(tmp_path),
            )

        assert result.success
        # Annotator should NOT be called since all events already annotated
        assert annotator.annotate_event.call_count == 0
        # SOP generator should still be called
        assert sop_gen.generate_from_focus.call_count == 1

    def test_sop_generation_failure(self, tmp_path):
        """SOP generation failure propagates error."""
        db_path = _create_test_db(tmp_path)
        session_id = "focus-sess-003"
        _insert_focus_events(db_path, session_id, count=2, annotated=True)

        with WorkerDB(db_path) as db:
            processor = FocusProcessor(
                _make_mock_annotator(),
                _make_mock_differ(),
                _make_mock_sop_generator(success=False),
            )
            result = processor.process_session(
                db, session_id, "Failing Task",
                db.get_focus_session_events(session_id),
                screenshots_dir=str(tmp_path),
            )

        assert not result.success
        assert "Mock failure" in result.error


class TestFocusProcessorAnnotation:
    def test_skip_stale_check_passed(self, tmp_path):
        """Focus processor passes skip_stale_check=True to annotator."""
        db_path = _create_test_db(tmp_path)
        session_id = "focus-sess-010"
        _insert_focus_events(db_path, session_id, count=2, annotated=False)

        with WorkerDB(db_path) as db:
            annotator = _make_mock_annotator()
            processor = FocusProcessor(
                annotator,
                _make_mock_differ(),
                _make_mock_sop_generator(),
            )
            processor.process_session(
                db, session_id, "Task",
                db.get_focus_session_events(session_id),
                screenshots_dir=str(tmp_path),
            )

        # Verify skip_stale_check=True was passed
        for call in annotator.annotate_event.call_args_list:
            assert call.kwargs.get("skip_stale_check") is True

    def test_mixed_annotated_unannotated(self, tmp_path):
        """Some events annotated, some not — only unannotated get processed."""
        db_path = _create_test_db(tmp_path)
        session_id = "focus-sess-011"

        # Insert 2 annotated + 1 unannotated
        conn = sqlite3.connect(str(db_path))
        metadata = json.dumps({"focus_session_id": session_id})

        # Annotated event
        conn.execute(
            "INSERT INTO events (id, timestamp, kind_json, metadata_json, "
            "annotation_status, scene_annotation_json) VALUES (?, ?, ?, ?, ?, ?)",
            ("evt-ann-0", "2026-03-03T10:00:00Z", '{}', metadata,
             "completed", json.dumps({
                 "app": "Chrome", "location": "https://x.com",
                 "visible_content": {}, "ui_state": {},
                 "task_context": {"what_doing": "Test", "likely_next": "", "is_workflow": True},
             })),
        )
        # Unannotated event
        conn.execute(
            "INSERT INTO events (id, timestamp, kind_json, metadata_json, "
            "annotation_status) VALUES (?, ?, ?, ?, ?)",
            ("evt-unann-1", "2026-03-03T10:00:01Z", '{}', metadata, "pending"),
        )
        conn.commit()
        conn.close()

        with WorkerDB(db_path) as db:
            annotator = _make_mock_annotator()
            processor = FocusProcessor(
                annotator,
                _make_mock_differ(),
                _make_mock_sop_generator(),
            )
            processor.process_session(
                db, session_id, "Mixed",
                db.get_focus_session_events(session_id),
                screenshots_dir=str(tmp_path),
            )

        # Only the unannotated event should be processed
        assert annotator.annotate_event.call_count == 1

    def test_annotation_failure_continues(self, tmp_path):
        """Failed annotation doesn't stop processing other events."""
        db_path = _create_test_db(tmp_path)
        session_id = "focus-sess-012"
        _insert_focus_events(db_path, session_id, count=3, annotated=False)

        call_count = {"n": 0}

        def mock_annotate(event, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                return AnnotationResult(
                    event_id=event.get("id"), status="failed", error="VLM error"
                )
            return AnnotationResult(
                event_id=event.get("id"),
                status="completed",
                annotation={
                    "app": "Chrome", "location": "https://x.com",
                    "visible_content": {}, "ui_state": {},
                    "task_context": {"what_doing": "Test", "likely_next": "", "is_workflow": True},
                },
                inference_time_seconds=1.0,
            )

        annotator = MagicMock(spec=SceneAnnotator)
        annotator.config = AnnotationConfig()
        annotator.annotate_event.side_effect = mock_annotate

        with WorkerDB(db_path) as db:
            processor = FocusProcessor(
                annotator,
                _make_mock_differ(),
                _make_mock_sop_generator(),
            )
            result = processor.process_session(
                db, session_id, "Partial Fail",
                db.get_focus_session_events(session_id),
                screenshots_dir=str(tmp_path),
            )

        # All 3 events should be attempted
        assert annotator.annotate_event.call_count == 3


class TestFocusProcessorDiffs:
    def test_first_frame_gets_first_frame_marker(self, tmp_path):
        """First event in session gets first_frame diff marker."""
        db_path = _create_test_db(tmp_path)
        session_id = "focus-sess-020"
        # Insert 2 annotated events (no diffs yet)
        conn = sqlite3.connect(str(db_path))
        metadata = json.dumps({"focus_session_id": session_id})
        ann = json.dumps({
            "app": "Chrome", "location": "https://x.com",
            "visible_content": {}, "ui_state": {},
            "task_context": {"what_doing": "Test", "likely_next": "", "is_workflow": True},
        })
        conn.execute(
            "INSERT INTO events (id, timestamp, kind_json, metadata_json, "
            "annotation_status, scene_annotation_json) VALUES (?, ?, ?, ?, ?, ?)",
            ("evt-d0", "2026-03-03T09:00:00Z", '{}', metadata, "completed", ann),
        )
        conn.execute(
            "INSERT INTO events (id, timestamp, kind_json, metadata_json, "
            "annotation_status, scene_annotation_json) VALUES (?, ?, ?, ?, ?, ?)",
            ("evt-d1", "2026-03-03T09:00:01Z", '{}', metadata, "completed", ann),
        )
        conn.commit()
        conn.close()

        with WorkerDB(db_path) as db:
            differ = _make_mock_differ()
            processor = FocusProcessor(
                _make_mock_annotator(),
                differ,
                _make_mock_sop_generator(),
            )
            processor.process_session(
                db, session_id, "Diff Test",
                db.get_focus_session_events(session_id),
                screenshots_dir=str(tmp_path),
            )

        # First event: no predecessor (in session context, DB might have one
        # from outside the session). The differ should be called for the 2nd.
        # The exact behavior depends on whether get_annotation_before finds
        # a previous event — in an empty DB, first frame gets first_frame marker.


class TestFocusProcessorFirstFrameNoDiff:
    """The first frame in a focus session must have diff=None."""

    def test_first_frame_diff_is_none_in_timeline(self, tmp_path):
        """First focus frame's diff is None, subsequent frames keep theirs."""
        db_path = _create_test_db(tmp_path)
        session_id = "focus-sess-first-diff"
        _insert_focus_events(db_path, session_id, count=3, annotated=True)

        with WorkerDB(db_path) as db:
            sop_gen = _make_mock_sop_generator()
            processor = FocusProcessor(
                _make_mock_annotator(),
                _make_mock_differ(),
                sop_gen,
            )
            processor.process_session(
                db, session_id, "First Frame Test",
                db.get_focus_session_events(session_id),
                screenshots_dir=str(tmp_path),
            )

        assert sop_gen.generate_from_focus.call_count == 1
        timeline = sop_gen.generate_from_focus.call_args[0][0]
        assert len(timeline) == 3

        # Frame 0: diff MUST be None (clean start)
        assert timeline[0]["diff"] is None, (
            "First focus frame must have diff=None to avoid pre-session pollution"
        )

        # Frames 1+: diff should be present (within-session diffs)
        for i, frame in enumerate(timeline[1:], start=1):
            assert frame["diff"] is not None, (
                f"Frame {i} should have a diff (within-session)"
            )

    def test_first_frame_diff_none_even_with_presession_data(self, tmp_path):
        """Even if DB contains a pre-session annotation, first frame diff is None."""
        db_path = _create_test_db(tmp_path)
        session_id = "focus-sess-presession"

        # Insert a pre-session event (different session / no session)
        conn = sqlite3.connect(str(db_path))
        pre_ann = json.dumps({
            "app": "Mail",
            "location": "inbox",
            "visible_content": {"headings": ["Inbox"], "labels": [], "values": []},
            "ui_state": {"active_element": "", "modals_or_popups": "none", "scroll_position": "top"},
            "task_context": {"what_doing": "Reading email", "likely_next": "", "is_workflow": False},
        })
        conn.execute(
            "INSERT INTO events (id, timestamp, kind_json, metadata_json, "
            "annotation_status, scene_annotation_json) VALUES (?, ?, ?, ?, ?, ?)",
            ("evt-presession", "2026-03-03T09:59:50Z", '{}', '{}',
             "completed", pre_ann),
        )
        conn.commit()
        conn.close()

        # Now insert focus session events
        _insert_focus_events(db_path, session_id, count=2, annotated=True)

        with WorkerDB(db_path) as db:
            sop_gen = _make_mock_sop_generator()
            processor = FocusProcessor(
                _make_mock_annotator(),
                _make_mock_differ(),
                sop_gen,
            )
            processor.process_session(
                db, session_id, "Pre-session Test",
                db.get_focus_session_events(session_id),
                screenshots_dir=str(tmp_path),
            )

        assert sop_gen.generate_from_focus.call_count == 1
        timeline = sop_gen.generate_from_focus.call_args[0][0]
        assert len(timeline) == 2

        # First frame must NOT carry a diff from the pre-session Mail event
        assert timeline[0]["diff"] is None, (
            "First focus frame must not carry a diff from pre-session activity"
        )


class TestFocusProcessorTimeline:
    def test_timeline_includes_annotations_and_diffs(self, tmp_path):
        """Collected timeline has both annotation and diff data."""
        db_path = _create_test_db(tmp_path)
        session_id = "focus-sess-030"
        _insert_focus_events(db_path, session_id, count=3, annotated=True)

        with WorkerDB(db_path) as db:
            sop_gen = _make_mock_sop_generator()
            processor = FocusProcessor(
                _make_mock_annotator(),
                _make_mock_differ(),
                sop_gen,
            )
            processor.process_session(
                db, session_id, "Timeline Test",
                db.get_focus_session_events(session_id),
                screenshots_dir=str(tmp_path),
            )

        # The SOP generator should have been called with a timeline
        assert sop_gen.generate_from_focus.call_count == 1
        call_args = sop_gen.generate_from_focus.call_args
        timeline = call_args[0][0]  # first positional arg
        assert len(timeline) == 3
        for frame in timeline:
            assert "annotation" in frame
            assert "timestamp" in frame
            assert isinstance(frame["annotation"], dict)


# ---------------------------------------------------------------------------
# TestDBFocusMethods
# ---------------------------------------------------------------------------

class TestDBFocusMethods:
    def test_get_focus_session_annotated_events(self, tmp_path):
        db_path = _create_test_db(tmp_path)
        session_id = "focus-db-001"
        _insert_focus_events(db_path, session_id, count=5, annotated=True)
        # Also insert 2 unannotated
        conn = sqlite3.connect(str(db_path))
        metadata = json.dumps({"focus_session_id": session_id})
        conn.execute(
            "INSERT INTO events (id, timestamp, kind_json, metadata_json, "
            "annotation_status) VALUES (?, ?, ?, ?, ?)",
            ("evt-unann", "2026-03-03T10:00:05Z", '{}', metadata, "pending"),
        )
        conn.commit()
        conn.close()

        with WorkerDB(db_path) as db:
            annotated = db.get_focus_session_annotated_events(session_id)
            assert len(annotated) == 5  # only the annotated ones

    def test_count_focus_unannotated(self, tmp_path):
        db_path = _create_test_db(tmp_path)
        session_id = "focus-db-002"
        _insert_focus_events(db_path, session_id, count=3, annotated=False)

        with WorkerDB(db_path) as db:
            count = db.count_focus_unannotated(session_id)
            assert count == 3

    def test_count_focus_unannotated_mixed(self, tmp_path):
        db_path = _create_test_db(tmp_path)
        session_id = "focus-db-003"
        _insert_focus_events(db_path, session_id, count=3, annotated=True)
        # Add 2 pending
        conn = sqlite3.connect(str(db_path))
        metadata = json.dumps({"focus_session_id": session_id})
        for i in range(2):
            conn.execute(
                "INSERT INTO events (id, timestamp, kind_json, metadata_json, "
                "annotation_status) VALUES (?, ?, ?, ?, ?)",
                (f"evt-p-{i}", f"2026-03-03T10:01:{i:02d}Z", '{}', metadata, "pending"),
            )
        conn.commit()
        conn.close()

        with WorkerDB(db_path) as db:
            count = db.count_focus_unannotated(session_id)
            assert count == 2

    def test_nonexistent_session(self, tmp_path):
        db_path = _create_test_db(tmp_path)
        with WorkerDB(db_path) as db:
            annotated = db.get_focus_session_annotated_events("no-such-session")
            assert annotated == []
            count = db.count_focus_unannotated("no-such-session")
            assert count == 0


# ---------------------------------------------------------------------------
# TestProcessFocusSessionsV2 (integration with main.py)
# ---------------------------------------------------------------------------

class TestProcessFocusSessionsV2:
    def test_no_signal_file_returns_zero(self, tmp_path):
        """No focus-session.json → returns 0."""
        from oc_apprentice_worker.main import _process_focus_sessions_v2

        db_path = _create_test_db(tmp_path)
        with WorkerDB(db_path) as db:
            mock_writer = MagicMock()
            mock_writer.write_all_sops.return_value = []
            mock_writer.get_sops_dir.return_value = tmp_path / "sops"

            with patch(
                "oc_apprentice_worker.main._status_dir",
                return_value=tmp_path,
            ):
                result = _process_focus_sessions_v2(
                    db,
                    focus_processor=MagicMock(),
                    openclaw_writer=mock_writer,
                    index_generator=MagicMock(),
                )
            assert result == 0

    def test_signal_still_recording_returns_zero(self, tmp_path):
        """Signal file with status=recording → returns 0."""
        from oc_apprentice_worker.main import _process_focus_sessions_v2

        signal_path = tmp_path / "focus-session.json"
        signal_path.write_text(json.dumps({
            "session_id": "sess-1",
            "status": "recording",
            "title": "In Progress",
        }))

        db_path = _create_test_db(tmp_path)
        with WorkerDB(db_path) as db:
            with patch(
                "oc_apprentice_worker.main._status_dir",
                return_value=tmp_path,
            ):
                result = _process_focus_sessions_v2(
                    db,
                    focus_processor=MagicMock(),
                    openclaw_writer=MagicMock(),
                    index_generator=MagicMock(),
                )
            assert result == 0

    def test_stopped_session_processes_and_clears_signal(self, tmp_path):
        """Stopped session → processes → clears signal file."""
        from oc_apprentice_worker.main import _process_focus_sessions_v2

        session_id = "focus-v2-test"
        signal_path = tmp_path / "focus-session.json"
        signal_path.write_text(json.dumps({
            "session_id": session_id,
            "status": "stopped",
            "title": "Test Task v2",
        }))

        db_path = _create_test_db(tmp_path)
        _insert_focus_events(db_path, session_id, count=3, annotated=True)

        mock_processor = MagicMock()
        mock_processor.process_session.return_value = GeneratedSOP(
            sop={
                "slug": "test-task",
                "title": "Test Task v2",
                "steps": [{"step": "action"}],
            },
            inference_time_seconds=50.0,
        )

        mock_writer = MagicMock()
        mock_writer.write_all_sops.return_value = [Path("/fake/SKILL.test.md")]
        mock_writer.get_sops_dir.return_value = tmp_path / "sops"

        mock_index = MagicMock()

        with WorkerDB(db_path) as db:
            with patch(
                "oc_apprentice_worker.main._status_dir",
                return_value=tmp_path,
            ):
                result = _process_focus_sessions_v2(
                    db,
                    focus_processor=mock_processor,
                    openclaw_writer=mock_writer,
                    skill_md_writer=None,
                    index_generator=mock_index,
                    screenshots_dir=str(tmp_path),
                )

        assert result == 1
        assert mock_processor.process_session.call_count == 1
        assert mock_writer.write_sop.call_count == 1
        assert mock_index.update_index.call_count == 1
        # Signal file should be removed
        assert not signal_path.exists()

    def test_failed_generation_clears_signal(self, tmp_path):
        """Failed SOP generation still clears the signal file."""
        from oc_apprentice_worker.main import _process_focus_sessions_v2

        session_id = "focus-v2-fail"
        signal_path = tmp_path / "focus-session.json"
        signal_path.write_text(json.dumps({
            "session_id": session_id,
            "status": "stopped",
            "title": "Failing Task",
        }))

        db_path = _create_test_db(tmp_path)
        _insert_focus_events(db_path, session_id, count=2, annotated=True)

        mock_processor = MagicMock()
        mock_processor.process_session.return_value = GeneratedSOP(
            sop={}, success=False, error="VLM unavailable"
        )

        with WorkerDB(db_path) as db:
            with patch(
                "oc_apprentice_worker.main._status_dir",
                return_value=tmp_path,
            ):
                result = _process_focus_sessions_v2(
                    db,
                    focus_processor=mock_processor,
                    openclaw_writer=MagicMock(),
                    index_generator=MagicMock(),
                    screenshots_dir=str(tmp_path),
                )

        assert result == 0
        # Signal should still be cleared to prevent infinite retry
        assert not signal_path.exists()


# ---------------------------------------------------------------------------
# TestFocusProcessorDomNodes
# ---------------------------------------------------------------------------

class TestFocusProcessorDomNodes:
    """Test that _collect_timeline includes dom_nodes when available."""

    def test_timeline_includes_dom_nodes_for_browser_events(self) -> None:
        """When DB has DOM snapshots matching a browser annotation, dom_nodes is populated."""

        # Create mock DB
        class MockDB:
            def get_event_by_id(self, eid):
                if eid == "e1":
                    return {
                        "id": "e1",
                        "timestamp": "2026-03-04T10:00:00Z",
                        "annotation_status": "completed",
                        "scene_annotation_json": json.dumps({
                            "app": "Google Chrome",
                            "location": "https://example.com/search",
                        }),
                        "frame_diff_json": json.dumps({"diff_type": "action"}),
                    }
                return None

            def get_dom_snapshots_near_timestamp(self, ts, url, tolerance_sec=5.0):
                if "example.com" in url:
                    return [{
                        "event_id": "dom-1",
                        "timestamp": ts,
                        "url": url,
                        "nodes": [
                            {"tag": "button", "text": "Search", "id": "search-btn"},
                        ],
                    }]
                return []

        processor = FocusProcessor(
            annotator=None,  # type: ignore
            differ=None,  # type: ignore
            sop_generator=None,  # type: ignore
        )

        events = [{"id": "e1", "timestamp": "2026-03-04T10:00:00Z"}]
        timeline = processor._collect_timeline(MockDB(), events)

        assert len(timeline) == 1
        assert timeline[0]["dom_nodes"] is not None
        assert len(timeline[0]["dom_nodes"]) == 1
        assert timeline[0]["dom_nodes"][0]["tag"] == "button"

    def test_timeline_no_dom_for_non_browser_events(self) -> None:
        """Non-browser events (Finder, VS Code) have dom_nodes = None."""

        class MockDB:
            def get_event_by_id(self, eid):
                return {
                    "id": eid,
                    "timestamp": "2026-03-04T10:00:00Z",
                    "annotation_status": "completed",
                    "scene_annotation_json": json.dumps({
                        "app": "Finder",
                        "location": "/Users/test/Documents",
                    }),
                    "frame_diff_json": json.dumps({"diff_type": "action"}),
                }

            def get_dom_snapshots_near_timestamp(self, ts, url, tolerance_sec=5.0):
                return []  # Should never be called for non-http locations

        processor = FocusProcessor(
            annotator=None,  # type: ignore
            differ=None,  # type: ignore
            sop_generator=None,  # type: ignore
        )

        events = [{"id": "e1", "timestamp": "2026-03-04T10:00:00Z"}]
        timeline = processor._collect_timeline(MockDB(), events)

        assert len(timeline) == 1
        assert timeline[0]["dom_nodes"] is None
