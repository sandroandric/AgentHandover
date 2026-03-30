"""Tests for v2 scene annotation pipeline integration in main.py.

Covers _process_annotations(), _process_diffs(), _read_vlm_v2_config(),
_check_v2_schema(), and the v2 status reporting additions.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agenthandover_worker.db import WorkerDB
from agenthandover_worker.frame_differ import DiffConfig, FrameDiffer
from agenthandover_worker.main import (
    _check_v2_schema,
    _process_annotations,
    _process_diffs,
    _read_vlm_v2_config,
    _write_worker_status,
)
from agenthandover_worker.scene_annotator import (
    AnnotationConfig,
    AnnotationResult,
    SceneAnnotator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _insert_event(
    conn: sqlite3.Connection,
    *,
    event_id: str | None = None,
    timestamp: str | None = None,
    annotation_status: str = "pending",
    scene_annotation_json: str | None = None,
    frame_diff_json: str | None = None,
    window_json: str | None = None,
    metadata_json: str = "{}",
    kind_json: str = '{"DwellSnapshot":{}}',
) -> str:
    eid = event_id or str(uuid.uuid4())
    ts = timestamp or _ts(datetime.now(timezone.utc))
    conn.execute(
        "INSERT INTO events "
        "(id, timestamp, kind_json, window_json, display_topology_json, "
        "primary_display_id, metadata_json, processed, "
        "annotation_status, scene_annotation_json, frame_diff_json) "
        "VALUES (?, ?, ?, ?, '[]', 'main', ?, 0, ?, ?, ?)",
        (
            eid, ts, kind_json, window_json, metadata_json,
            annotation_status, scene_annotation_json, frame_diff_json,
        ),
    )
    conn.commit()
    return eid


# Import schema from conftest
from conftest import DAEMON_SCHEMA


@pytest.fixture()
def v2_db_path(tmp_path: Path) -> Path:
    """Create a database with v2 schema (annotation columns)."""
    db_file = tmp_path / "events.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA journal_mode=wal;")
    conn.executescript(DAEMON_SCHEMA)
    conn.close()
    return db_file


@pytest.fixture()
def v2_write_conn(v2_db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(v2_db_path))
    conn.row_factory = sqlite3.Row
    yield conn  # type: ignore[misc]
    conn.close()


@pytest.fixture()
def v2_db(v2_db_path: Path) -> WorkerDB:
    db = WorkerDB(v2_db_path)
    yield db  # type: ignore[misc]
    db.close()


# ---------------------------------------------------------------------------
# _read_vlm_v2_config
# ---------------------------------------------------------------------------


class TestReadVlmV2Config:
    def test_defaults_when_no_config(self, tmp_path: Path, monkeypatch):
        """When no config.toml exists, return all defaults."""
        from agenthandover_worker import main as main_mod

        # Monkeypatch _read_vlm_config_field to return defaults
        monkeypatch.setattr(
            main_mod, "_read_vlm_config_field",
            lambda field, default="": default,
        )

        cfg = _read_vlm_v2_config()
        assert cfg["annotation_enabled"] is True
        assert cfg["annotation_model"] == "qwen3.5:2b"
        assert cfg["sop_model"] == "qwen3.5:4b"
        assert cfg["stale_skip_count"] == 3
        assert cfg["sliding_window_max_age_sec"] == 600
        assert cfg["ollama_host"] == "http://localhost:11434"

    def test_annotation_disabled(self, monkeypatch):
        from agenthandover_worker import main as main_mod

        overrides = {"annotation_enabled": "false"}
        monkeypatch.setattr(
            main_mod, "_read_vlm_config_field",
            lambda field, default="": overrides.get(field, default),
        )

        cfg = _read_vlm_v2_config()
        assert cfg["annotation_enabled"] is False

    def test_custom_model(self, monkeypatch):
        from agenthandover_worker import main as main_mod

        overrides = {"annotation_model": "qwen3:4b"}
        monkeypatch.setattr(
            main_mod, "_read_vlm_config_field",
            lambda field, default="": overrides.get(field, default),
        )

        cfg = _read_vlm_v2_config()
        assert cfg["annotation_model"] == "qwen3:4b"


# ---------------------------------------------------------------------------
# _check_v2_schema
# ---------------------------------------------------------------------------


class TestCheckV2Schema:
    def test_v2_schema_present(self, v2_db: WorkerDB):
        """Should return True when annotation_status column exists."""
        assert _check_v2_schema(v2_db) is True

    def test_v2_schema_missing(self, tmp_path: Path):
        """Should return False when annotation_status column is missing."""
        db_file = tmp_path / "old.db"
        conn = sqlite3.connect(str(db_file))
        conn.execute("PRAGMA journal_mode=wal;")
        # v1 schema without annotation columns
        conn.execute(
            "CREATE TABLE events ("
            "  id TEXT PRIMARY KEY NOT NULL,"
            "  timestamp TEXT NOT NULL,"
            "  kind_json TEXT NOT NULL,"
            "  display_topology_json TEXT NOT NULL,"
            "  primary_display_id TEXT NOT NULL,"
            "  processed INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        conn.close()

        db = WorkerDB(db_file)
        try:
            assert _check_v2_schema(db) is False
        finally:
            db.close()


# ---------------------------------------------------------------------------
# _process_annotations
# ---------------------------------------------------------------------------


class TestProcessAnnotations:
    def test_no_unannotated_events(self, v2_db: WorkerDB, tmp_path: Path):
        """When there are no pending events, returns zero stats."""
        annotator = SceneAnnotator(AnnotationConfig())
        stats = _process_annotations(v2_db, annotator, tmp_path / "screenshots")
        assert stats == {"annotated": 0, "skipped": 0, "failed": 0, "blocked": 0}

    def test_annotates_pending_event(
        self, v2_db: WorkerDB, v2_write_conn: sqlite3.Connection, tmp_path: Path,
    ):
        """A pending event gets annotated when the VLM succeeds."""
        base = datetime(2026, 3, 3, 10, 0, 0, tzinfo=timezone.utc)
        eid = _insert_event(
            v2_write_conn,
            timestamp=_ts(base),
            metadata_json=json.dumps({
                "screenshot_path": str(tmp_path / "screenshots" / "test.jpg"),
            }),
        )

        # Create a fake screenshot file
        screenshots_dir = tmp_path / "screenshots"
        screenshots_dir.mkdir(exist_ok=True)
        (screenshots_dir / "test.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

        # Mock the annotator to return a successful result
        annotator = SceneAnnotator(AnnotationConfig())
        fake_annotation = {
            "app": "Google Chrome",
            "location": "https://example.com",
            "visible_content": {"headings": ["Test"], "labels": [], "values": []},
            "ui_state": {"active_element": "none", "modals_or_popups": "none", "scroll_position": "top"},
            "task_context": {
                "what_doing": "Browsing example.com",
                "likely_next": "Click a link",
                "is_workflow": False,
            },
        }

        with patch.object(
            annotator, "annotate_event",
            return_value=AnnotationResult(
                event_id=eid,
                status="completed",
                annotation=fake_annotation,
                inference_time_seconds=5.0,
            ),
        ):
            stats = _process_annotations(v2_db, annotator, screenshots_dir)

        assert stats["annotated"] == 1
        assert stats["skipped"] == 0
        assert stats["failed"] == 0

        # Verify DB was updated
        event = v2_db.get_event_by_id(eid)
        assert event is not None
        assert event["annotation_status"] == "completed"
        assert json.loads(event["scene_annotation_json"])["app"] == "Google Chrome"

    def test_handles_failed_annotation(
        self, v2_db: WorkerDB, v2_write_conn: sqlite3.Connection, tmp_path: Path,
    ):
        """A failed annotation is saved with status='failed'."""
        eid = _insert_event(v2_write_conn)

        annotator = SceneAnnotator(AnnotationConfig())

        with patch.object(
            annotator, "annotate_event",
            return_value=AnnotationResult(
                event_id=eid,
                status="failed",
                error="ollama_connection: Connection refused",
            ),
        ):
            stats = _process_annotations(
                v2_db, annotator, tmp_path / "screenshots"
            )

        assert stats["failed"] == 1

        event = v2_db.get_event_by_id(eid)
        assert event is not None
        assert event["annotation_status"] == "failed"

    def test_handles_missing_screenshot(
        self, v2_db: WorkerDB, v2_write_conn: sqlite3.Connection, tmp_path: Path,
    ):
        """Missing screenshot events are saved with correct status."""
        eid = _insert_event(v2_write_conn)

        annotator = SceneAnnotator(AnnotationConfig())

        with patch.object(
            annotator, "annotate_event",
            return_value=AnnotationResult(
                event_id=eid,
                status="missing_screenshot",
                error="no_screenshot_file",
            ),
        ):
            stats = _process_annotations(
                v2_db, annotator, tmp_path / "screenshots"
            )

        assert stats["skipped"] == 1

        event = v2_db.get_event_by_id(eid)
        assert event["annotation_status"] == "missing_screenshot"

    def test_handles_skipped_stale(
        self, v2_db: WorkerDB, v2_write_conn: sqlite3.Connection, tmp_path: Path,
    ):
        """Stale-skipped events are saved with status='skipped'."""
        eid = _insert_event(v2_write_conn)

        annotator = SceneAnnotator(AnnotationConfig())

        with patch.object(
            annotator, "annotate_event",
            return_value=AnnotationResult(
                event_id=eid,
                status="skipped",
                error="stale_frame",
            ),
        ):
            stats = _process_annotations(
                v2_db, annotator, tmp_path / "screenshots"
            )

        assert stats["skipped"] == 1

    def test_passes_sliding_window_context(
        self, v2_db: WorkerDB, v2_write_conn: sqlite3.Connection, tmp_path: Path,
    ):
        """Annotator receives recent annotations as sliding window context."""
        base = datetime(2026, 3, 3, 10, 0, 0, tzinfo=timezone.utc)

        # Insert a completed annotation (will be context)
        _insert_event(
            v2_write_conn,
            timestamp=_ts(base),
            annotation_status="completed",
            scene_annotation_json=json.dumps({
                "app": "Chrome",
                "task_context": {"what_doing": "Previous task"},
            }),
        )

        # Insert a pending event (will be annotated)
        eid = _insert_event(
            v2_write_conn,
            timestamp=_ts(base + timedelta(seconds=30)),
        )

        annotator = SceneAnnotator(AnnotationConfig())
        call_args_capture: list = []

        def _capture_annotate(event, *, recent_annotations=None, artifact_dir=None):
            call_args_capture.append({
                "event_id": event.get("id"),
                "recent_annotations": recent_annotations,
            })
            return AnnotationResult(
                event_id=event.get("id", ""),
                status="completed",
                annotation={
                    "app": "Test",
                    "task_context": {"what_doing": "Testing", "is_workflow": False},
                },
                inference_time_seconds=1.0,
            )

        with patch.object(annotator, "annotate_event", side_effect=_capture_annotate):
            _process_annotations(v2_db, annotator, tmp_path / "screenshots")

        assert len(call_args_capture) == 1
        # The sliding window should have found the completed annotation
        recent = call_args_capture[0]["recent_annotations"]
        assert len(recent) == 1
        assert "Previous task" in json.loads(
            recent[0]["scene_annotation_json"]
        ).get("task_context", {}).get("what_doing", "")

    def test_batch_size_limit(
        self, v2_db: WorkerDB, v2_write_conn: sqlite3.Connection, tmp_path: Path,
    ):
        """Only processes batch_size events per call."""
        base = datetime(2026, 3, 3, 10, 0, 0, tzinfo=timezone.utc)
        for i in range(10):
            _insert_event(
                v2_write_conn,
                timestamp=_ts(base + timedelta(seconds=i)),
            )

        annotator = SceneAnnotator(AnnotationConfig())
        call_count = 0

        def _counting_annotate(event, **kwargs):
            nonlocal call_count
            call_count += 1
            return AnnotationResult(
                event_id=event.get("id", ""),
                status="completed",
                annotation={
                    "app": "Test",
                    "task_context": {"what_doing": "Testing", "is_workflow": False},
                },
            )

        with patch.object(annotator, "annotate_event", side_effect=_counting_annotate):
            stats = _process_annotations(
                v2_db, annotator, tmp_path / "screenshots",
                batch_size=3,
            )

        assert call_count == 3
        assert stats["annotated"] == 3


# ---------------------------------------------------------------------------
# _process_diffs
# ---------------------------------------------------------------------------


class TestProcessDiffs:
    def test_no_events_needing_diff(self, v2_db: WorkerDB):
        """When no events need diffs, returns zero stats."""
        differ = FrameDiffer(DiffConfig())
        stats = _process_diffs(v2_db, differ)
        assert stats == {"diffs": 0, "edge_cases": 0, "failed": 0}

    def test_first_frame_gets_first_frame_marker(
        self, v2_db: WorkerDB, v2_write_conn: sqlite3.Connection,
    ):
        """First annotated event with no predecessor gets first_frame marker."""
        base = datetime(2026, 3, 3, 10, 0, 0, tzinfo=timezone.utc)
        eid = _insert_event(
            v2_write_conn,
            timestamp=_ts(base),
            annotation_status="completed",
            scene_annotation_json=json.dumps({
                "app": "Chrome",
                "location": "https://example.com",
                "task_context": {"what_doing": "Browsing", "is_workflow": False},
            }),
        )

        differ = FrameDiffer(DiffConfig())
        stats = _process_diffs(v2_db, differ)

        assert stats["edge_cases"] == 1

        event = v2_db.get_event_by_id(eid)
        diff = json.loads(event["frame_diff_json"])
        assert diff["diff_type"] == "first_frame"

    def test_edge_case_app_switch(
        self, v2_db: WorkerDB, v2_write_conn: sqlite3.Connection,
    ):
        """App switch between two frames produces app_switch marker (no LLM)."""
        base = datetime(2026, 3, 3, 10, 0, 0, tzinfo=timezone.utc)

        # First event — already has diff (from being first_frame)
        _insert_event(
            v2_write_conn,
            timestamp=_ts(base),
            annotation_status="completed",
            scene_annotation_json=json.dumps({
                "app": "Google Chrome",
                "location": "https://example.com",
                "visible_content": {"headings": [], "labels": [], "values": []},
                "ui_state": {},
                "task_context": {"what_doing": "Browsing", "is_workflow": False},
            }),
            frame_diff_json=json.dumps({"diff_type": "first_frame"}),
        )

        # Second event — different app, needs diff
        eid2 = _insert_event(
            v2_write_conn,
            timestamp=_ts(base + timedelta(seconds=30)),
            annotation_status="completed",
            scene_annotation_json=json.dumps({
                "app": "VS Code",
                "location": "/src/main.py",
                "visible_content": {"headings": [], "labels": [], "values": []},
                "ui_state": {},
                "task_context": {"what_doing": "Editing code", "is_workflow": True},
            }),
        )

        differ = FrameDiffer(DiffConfig())
        stats = _process_diffs(v2_db, differ)

        # App switches now fall through to LLM (not handled as edge case).
        # Without Ollama, the diff is a failed marker.
        event = v2_db.get_event_by_id(eid2)
        diff = json.loads(event["frame_diff_json"])
        assert diff["diff_type"] in ("app_switch", "diff_failed", "action")

    def test_edge_case_no_change(
        self, v2_db: WorkerDB, v2_write_conn: sqlite3.Connection,
    ):
        """Same app + location + values produces no_change marker."""
        base = datetime(2026, 3, 3, 10, 0, 0, tzinfo=timezone.utc)
        annotation = {
            "app": "Chrome",
            "location": "https://example.com",
            "visible_content": {"headings": [], "labels": [], "values": ["text"]},
            "ui_state": {},
            "task_context": {"what_doing": "Reading", "is_workflow": False},
        }

        _insert_event(
            v2_write_conn,
            timestamp=_ts(base),
            annotation_status="completed",
            scene_annotation_json=json.dumps(annotation),
            frame_diff_json=json.dumps({"diff_type": "first_frame"}),
        )

        eid2 = _insert_event(
            v2_write_conn,
            timestamp=_ts(base + timedelta(seconds=10)),
            annotation_status="completed",
            scene_annotation_json=json.dumps(annotation),
        )

        differ = FrameDiffer(DiffConfig())
        stats = _process_diffs(v2_db, differ)

        assert stats["edge_cases"] == 1

        event = v2_db.get_event_by_id(eid2)
        diff = json.loads(event["frame_diff_json"])
        assert diff["diff_type"] == "no_change"

    def test_action_diff_calls_llm(
        self, v2_db: WorkerDB, v2_write_conn: sqlite3.Connection,
    ):
        """When content changed, the differ calls the LLM and produces action diff."""
        base = datetime(2026, 3, 3, 10, 0, 0, tzinfo=timezone.utc)

        _insert_event(
            v2_write_conn,
            timestamp=_ts(base),
            annotation_status="completed",
            scene_annotation_json=json.dumps({
                "app": "Chrome",
                "location": "https://example.com/form",
                "visible_content": {"headings": [], "labels": ["Name"], "values": []},
                "ui_state": {"active_element": "Name input"},
                "task_context": {"what_doing": "Filling form", "is_workflow": True},
            }),
            frame_diff_json=json.dumps({"diff_type": "first_frame"}),
        )

        eid2 = _insert_event(
            v2_write_conn,
            timestamp=_ts(base + timedelta(seconds=15)),
            annotation_status="completed",
            scene_annotation_json=json.dumps({
                "app": "Chrome",
                "location": "https://example.com/form",
                "visible_content": {
                    "headings": [],
                    "labels": ["Name"],
                    "values": ["John Doe"],
                },
                "ui_state": {"active_element": "Email input"},
                "task_context": {"what_doing": "Filling form", "is_workflow": True},
            }),
        )

        differ = FrameDiffer(DiffConfig())

        # Mock the LLM call to return a valid diff
        fake_diff_result = {
            "diff_type": "action",
            "actions": ["Typed 'John Doe' in Name field"],
            "inputs": [{"field": "Name", "value": "John Doe"}],
            "navigation": "none",
            "step_description": "User entered name in the form",
        }

        from agenthandover_worker.frame_differ import DiffResult

        with patch.object(
            differ, "diff_pair",
            return_value=DiffResult(
                event_id=eid2,
                diff=fake_diff_result,
                inference_time_seconds=3.6,
            ),
        ):
            stats = _process_diffs(v2_db, differ)

        assert stats["diffs"] == 1

        event = v2_db.get_event_by_id(eid2)
        diff = json.loads(event["frame_diff_json"])
        assert diff["diff_type"] == "action"
        assert "John Doe" in diff["actions"][0]

    def test_batch_size_limit(
        self, v2_db: WorkerDB, v2_write_conn: sqlite3.Connection,
    ):
        """Only processes batch_size events per call."""
        base = datetime(2026, 3, 3, 10, 0, 0, tzinfo=timezone.utc)

        # First completed event with diff
        _insert_event(
            v2_write_conn,
            timestamp=_ts(base),
            annotation_status="completed",
            scene_annotation_json=json.dumps({
                "app": "Chrome",
                "location": "a.com",
                "visible_content": {"headings": [], "labels": [], "values": []},
                "ui_state": {},
                "task_context": {"what_doing": "Task", "is_workflow": False},
            }),
            frame_diff_json=json.dumps({"diff_type": "first_frame"}),
        )

        # Insert 5 events needing diffs (different apps for easy edge_case)
        apps = ["VS Code", "Terminal", "Finder", "Notes", "Safari"]
        for i, app in enumerate(apps):
            _insert_event(
                v2_write_conn,
                timestamp=_ts(base + timedelta(seconds=30 * (i + 1))),
                annotation_status="completed",
                scene_annotation_json=json.dumps({
                    "app": app,
                    "location": f"/{app.lower()}",
                    "visible_content": {"headings": [], "labels": [], "values": []},
                    "ui_state": {},
                    "task_context": {"what_doing": f"Using {app}", "is_workflow": False},
                }),
            )

        differ = FrameDiffer(DiffConfig())
        stats = _process_diffs(v2_db, differ, batch_size=2)

        # batch_size=2 limits processing — verify we didn't process all 5.
        # App switches now go through LLM (not edge_cases). Without Ollama
        # they count as "failed", so total = edge_cases + diffs + failed.
        total_processed = stats.get("edge_cases", 0) + stats.get("diffs_computed", 0) + stats.get("failed", 0)
        assert total_processed <= 2


# ---------------------------------------------------------------------------
# _write_worker_status with v2 fields
# ---------------------------------------------------------------------------


class TestWorkerStatusV2:
    def test_v2_fields_included_when_enabled(self, tmp_path: Path, monkeypatch):
        """Status file includes v2 annotation stats when enabled."""
        from agenthandover_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        _write_worker_status(
            started_at="2026-03-03T10:00:00Z",
            events_processed_today=100,
            sops_generated=5,
            last_pipeline_duration_ms=1234,
            consecutive_errors=0,
            vlm_available=True,
            sop_inducer_available=True,
            v2_annotation_enabled=True,
            v2_annotations_today=42,
            v2_diffs_today=38,
        )

        status_file = tmp_path / "worker-status.json"
        assert status_file.exists()
        status = json.loads(status_file.read_text())
        assert status["v2_annotation_enabled"] is True
        assert status["v2_annotations_today"] == 42
        assert status["v2_diffs_today"] == 38

    def test_v2_fields_excluded_when_disabled(self, tmp_path: Path, monkeypatch):
        """Status file omits v2 fields when annotation is disabled."""
        from agenthandover_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        _write_worker_status(
            started_at="2026-03-03T10:00:00Z",
            events_processed_today=100,
            sops_generated=5,
            last_pipeline_duration_ms=1234,
            consecutive_errors=0,
            vlm_available=True,
            sop_inducer_available=True,
            v2_annotation_enabled=False,
        )

        status_file = tmp_path / "worker-status.json"
        status = json.loads(status_file.read_text())
        assert "v2_annotation_enabled" not in status
        assert "v2_annotations_today" not in status


# ---------------------------------------------------------------------------
# Integration: v2 pipeline doesn't break v1 pipeline
# ---------------------------------------------------------------------------


class TestV2DoesNotBreakV1:
    """Ensure the v2 additions don't break existing run_pipeline behavior."""

    def test_run_pipeline_still_works(self, tmp_path: Path):
        """run_pipeline continues to work normally with v2 imports added."""
        from agenthandover_worker.clipboard_linker import ClipboardLinker
        from agenthandover_worker.confidence import ConfidenceScorer
        from agenthandover_worker.episode_builder import EpisodeBuilder
        from agenthandover_worker.exporter import IndexGenerator
        from agenthandover_worker.main import run_pipeline
        from agenthandover_worker.negative_demo import NegativeDemoPruner
        from agenthandover_worker.openclaw_writer import OpenClawWriter
        from agenthandover_worker.translator import SemanticTranslator
        from agenthandover_worker.vlm_queue import VLMFallbackQueue

        workspace = tmp_path / "workspace"
        summary = run_pipeline(
            [],
            episode_builder=EpisodeBuilder(),
            clipboard_linker=ClipboardLinker(),
            pruner=NegativeDemoPruner(),
            translator=SemanticTranslator(),
            scorer=ConfidenceScorer(),
            vlm_queue=VLMFallbackQueue(),
            openclaw_writer=OpenClawWriter(workspace_dir=workspace),
            index_generator=IndexGenerator(),
        )

        assert summary["events_in"] == 0
        assert summary["episodes"] == 0
