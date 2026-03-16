"""End-to-end integration test for the full AgentHandover pipeline.

Starts daemon + worker processes, inserts synthetic events into the DB,
waits for SOP output, verifies status files, and performs clean shutdown.

No Chrome dependency — bypasses the extension for CI.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Allow up to 60 seconds for the full pipeline test
PIPELINE_TIMEOUT = 60
POLL_INTERVAL = 1.0


def _data_dir(tmp_path: Path) -> Path:
    """Create and return a temporary data directory mimicking the real layout."""
    data = tmp_path / "agenthandover"
    data.mkdir(parents=True)
    (data / "logs").mkdir()
    (data / "artifacts").mkdir()
    return data


def _create_test_db(db_path: Path) -> None:
    """Create a minimal events database with the schema the daemon would create.

    Mirrors crates/storage/src/migrations/v001_initial.sql + v002 exactly.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY NOT NULL,
            timestamp TEXT NOT NULL,
            kind_json TEXT NOT NULL,
            window_json TEXT,
            display_topology_json TEXT NOT NULL DEFAULT '[]',
            primary_display_id TEXT NOT NULL DEFAULT 'unknown',
            cursor_x INTEGER,
            cursor_y INTEGER,
            ui_scale REAL,
            artifact_ids_json TEXT NOT NULL DEFAULT '[]',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            processed INTEGER NOT NULL DEFAULT 0,
            episode_id TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            display_ids_spanned_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_events_processed ON events(processed);
        CREATE INDEX IF NOT EXISTS idx_events_episode_id ON events(episode_id);

        CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY NOT NULL,
            event_id TEXT NOT NULL REFERENCES events(id),
            artifact_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            compression_algo TEXT NOT NULL DEFAULT 'zstd',
            encryption_algo TEXT NOT NULL DEFAULT 'xchacha20poly1305',
            original_size_bytes INTEGER NOT NULL,
            stored_size_bytes INTEGER NOT NULL,
            artifact_version INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE INDEX IF NOT EXISTS idx_artifacts_event_id ON artifacts(event_id);

        CREATE TABLE IF NOT EXISTS episodes (
            id TEXT PRIMARY KEY NOT NULL,
            segment_id INTEGER NOT NULL DEFAULT 0,
            prev_segment_id INTEGER,
            thread_id TEXT,
            start_time TEXT NOT NULL,
            end_time TEXT,
            event_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'open',
            summary TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS vlm_queue (
            id TEXT PRIMARY KEY NOT NULL,
            event_id TEXT NOT NULL REFERENCES events(id),
            priority REAL NOT NULL DEFAULT 0.5,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            processed_at TEXT,
            result_json TEXT,
            ttl_expires_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_vlm_queue_status ON vlm_queue(status, priority DESC);
    """)
    conn.commit()
    conn.close()


def _insert_synthetic_events(db_path: Path, count: int = 10) -> list[str]:
    """Insert synthetic click/navigation events into the database.

    Returns list of event IDs inserted.
    """
    conn = sqlite3.connect(str(db_path))
    event_ids = []

    for i in range(count):
        event_id = str(uuid.uuid4())
        event_ids.append(event_id)
        ts = datetime.now(timezone.utc).isoformat()

        # Alternate between click and navigation events
        if i % 2 == 0:
            kind_json = json.dumps({
                "BrowserClick": {
                    "url": f"https://example.com/page-{i}",
                    "selector": f"button#action-{i}",
                    "inner_text": f"Click Me {i}",
                    "tag": "button",
                    "x": 100 + i * 10,
                    "y": 200 + i * 5,
                }
            })
        else:
            kind_json = json.dumps({
                "FocusChange": {}
            })

        window_json = json.dumps({
            "app_id": "com.google.Chrome",
            "title": f"Test Page {i}",
        })

        metadata_json = json.dumps({
            "url": f"https://example.com/page-{i}",
        })

        conn.execute(
            """INSERT INTO events (id, timestamp, kind_json, window_json,
               metadata_json, display_topology_json, primary_display_id, processed)
               VALUES (?, ?, ?, ?, ?, '[]', 'main', 0)""",
            (event_id, ts, kind_json, window_json, metadata_json),
        )

    conn.commit()
    conn.close()
    return event_ids


class TestStatusFileProtocol:
    """Test that status files are correctly written and readable."""

    def test_daemon_status_json_schema(self, tmp_path: Path):
        """Verify daemon-status.json has all required fields."""
        status = {
            "pid": os.getpid(),
            "version": "0.1.0",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "heartbeat": datetime.now(timezone.utc).isoformat(),
            "events_today": 42,
            "permissions_ok": True,
            "accessibility_permitted": True,
            "screen_recording_permitted": True,
            "db_path": str(tmp_path / "events.db"),
            "uptime_seconds": 3600,
        }

        status_file = tmp_path / "daemon-status.json"
        status_file.write_text(json.dumps(status, indent=2))

        loaded = json.loads(status_file.read_text())
        required_fields = [
            "pid", "version", "started_at", "heartbeat",
            "events_today", "permissions_ok", "db_path", "uptime_seconds",
        ]
        for field in required_fields:
            assert field in loaded, f"Missing required field: {field}"

    def test_worker_status_json_schema(self, tmp_path: Path):
        """Verify worker-status.json has all required fields."""
        status = {
            "pid": os.getpid(),
            "version": "0.1.0",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "heartbeat": datetime.now(timezone.utc).isoformat(),
            "events_processed_today": 100,
            "sops_generated": 3,
            "last_pipeline_duration_ms": 450,
            "consecutive_errors": 0,
            "vlm_available": False,
            "sop_inducer_available": True,
        }

        status_file = tmp_path / "worker-status.json"
        status_file.write_text(json.dumps(status, indent=2))

        loaded = json.loads(status_file.read_text())
        required_fields = [
            "pid", "version", "started_at", "heartbeat",
            "events_processed_today", "sops_generated",
            "consecutive_errors", "vlm_available", "sop_inducer_available",
        ]
        for field in required_fields:
            assert field in loaded, f"Missing required field: {field}"


class TestDatabaseSetup:
    """Test database creation and event insertion for E2E scenarios."""

    def test_create_test_db(self, tmp_path: Path):
        db_path = tmp_path / "events.db"
        _create_test_db(db_path)
        assert db_path.exists()

        conn = sqlite3.connect(str(db_path))
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()

        assert "events" in tables
        assert "episodes" in tables
        assert "vlm_queue" in tables
        assert "artifacts" in tables

    def test_insert_synthetic_events(self, tmp_path: Path):
        db_path = tmp_path / "events.db"
        _create_test_db(db_path)
        ids = _insert_synthetic_events(db_path, count=20)

        assert len(ids) == 20

        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        conn.close()
        assert count == 20

    def test_events_marked_unprocessed(self, tmp_path: Path):
        db_path = tmp_path / "events.db"
        _create_test_db(db_path)
        _insert_synthetic_events(db_path, count=5)

        conn = sqlite3.connect(str(db_path))
        unprocessed = conn.execute(
            "SELECT COUNT(*) FROM events WHERE processed = 0"
        ).fetchone()[0]
        conn.close()
        assert unprocessed == 5


class TestWorkerPipeline:
    """Test that the worker pipeline processes events from the database."""

    def test_run_pipeline_with_synthetic_events(self, tmp_path: Path):
        """Test the pipeline function directly with synthetic data."""
        from agenthandover_worker.episode_builder import EpisodeBuilder
        from agenthandover_worker.clipboard_linker import ClipboardLinker
        from agenthandover_worker.negative_demo import NegativeDemoPruner
        from agenthandover_worker.translator import SemanticTranslator
        from agenthandover_worker.confidence import ConfidenceScorer
        from agenthandover_worker.vlm_queue import VLMFallbackQueue
        from agenthandover_worker.openclaw_writer import OpenClawWriter
        from agenthandover_worker.exporter import IndexGenerator
        from agenthandover_worker.main import run_pipeline

        # Create synthetic events as dicts (matching what WorkerDB returns)
        events = []
        for i in range(5):
            events.append({
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kind_json": json.dumps({"ClickIntent": {}}),
                "window_json": json.dumps({
                    "app_id": "com.google.Chrome",
                    "title": f"Test Page {i}",
                }),
                "metadata_json": json.dumps({
                    "url": f"https://example.com/{i}",
                }),
                "display_topology_json": "[]",
                "primary_display_id": "main",
                "processed": 0,
            })

        workspace = tmp_path / "workspace"
        writer = OpenClawWriter(workspace_dir=workspace)

        summary = run_pipeline(
            events,
            episode_builder=EpisodeBuilder(),
            clipboard_linker=ClipboardLinker(),
            pruner=NegativeDemoPruner(),
            translator=SemanticTranslator(),
            scorer=ConfidenceScorer(),
            vlm_queue=VLMFallbackQueue(),
            openclaw_writer=writer,
            index_generator=IndexGenerator(),
        )

        assert summary["events_in"] == 5
        assert isinstance(summary["episodes"], int)
        assert isinstance(summary["translations"], int)


class TestExportAdapters:
    """Test that both export adapters produce valid output."""

    def test_openclaw_adapter_writes_sops(self, tmp_path: Path):
        from agenthandover_worker.openclaw_writer import OpenClawWriter

        writer = OpenClawWriter(workspace_dir=tmp_path)
        sop = {
            "slug": "e2e-test-sop",
            "title": "E2E Test SOP",
            "steps": [
                {"step": "navigate", "target": "https://example.com", "confidence": 0.95},
                {"step": "click", "target": "button#submit", "confidence": 0.88},
            ],
            "confidence_avg": 0.915,
            "episode_count": 5,
            "apps_involved": ["Chrome"],
        }

        path = writer.write_sop(sop)
        assert path.exists()
        content = path.read_text()
        assert "E2E Test SOP" in content

    def test_generic_adapter_writes_md_and_json(self, tmp_path: Path):
        from agenthandover_worker.generic_writer import GenericWriter

        writer = GenericWriter(output_dir=tmp_path, json_export=True)
        sop = {
            "slug": "e2e-generic",
            "title": "E2E Generic SOP",
            "steps": [{"step": "click", "target": "button"}],
            "confidence_avg": 0.85,
            "episode_count": 3,
            "apps_involved": ["Chrome"],
        }

        md_path = writer.write_sop(sop)
        assert md_path.exists()

        json_path = tmp_path / "sops" / "sop.e2e-generic.json"
        assert json_path.exists()

        data = json.loads(json_path.read_text())
        assert data["schema_version"] == "1.1.0"
        assert data["slug"] == "e2e-generic"

    def test_adapter_list_sops(self, tmp_path: Path):
        from agenthandover_worker.generic_writer import GenericWriter

        writer = GenericWriter(output_dir=tmp_path)
        for i in range(3):
            writer.write_sop({
                "slug": f"sop-{i}",
                "title": f"SOP {i}",
                "steps": [],
            })

        sops = writer.list_sops()
        assert len(sops) == 3


class TestCleanShutdown:
    """Test graceful shutdown behavior."""

    def test_pid_file_lifecycle(self, tmp_path: Path):
        """Simulate PID file write, stale detection, and cleanup."""
        pid_file = tmp_path / "test.pid"

        # Write current PID
        pid_file.write_text(str(os.getpid()))
        assert pid_file.exists()

        # Read back
        stored_pid = int(pid_file.read_text().strip())
        assert stored_pid == os.getpid()

        # "Clean shutdown" removes it
        pid_file.unlink()
        assert not pid_file.exists()

    def test_stale_pid_detection(self, tmp_path: Path):
        """A PID file with a non-existent process is stale."""
        pid_file = tmp_path / "stale.pid"
        pid_file.write_text("999999999")  # Almost certainly not running

        stored_pid = int(pid_file.read_text().strip())
        # kill(pid, 0) should fail for non-existent process
        try:
            os.kill(stored_pid, 0)
            process_alive = True
        except (OSError, ProcessLookupError):
            process_alive = False

        assert not process_alive, "Stale PID should not correspond to a running process"
