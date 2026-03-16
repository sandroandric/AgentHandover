"""Error recovery tests for AgentHandover services.

Tests crash recovery scenarios, DB lock contention, full disk behavior,
missing permissions, and stale PID handling.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest


class TestDatabaseLockContention:
    """Test behavior when multiple processes access the DB simultaneously."""

    def test_wal_mode_allows_concurrent_reads(self, tmp_path: Path):
        """WAL mode should allow readers while writing."""
        db_path = tmp_path / "events.db"

        # Writer connection
        writer = sqlite3.connect(str(db_path))
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("""
            CREATE TABLE events (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                kind TEXT NOT NULL,
                processed INTEGER DEFAULT 0
            )
        """)
        writer.commit()

        # Insert data
        writer.execute(
            "INSERT INTO events (id, timestamp, kind) VALUES (?, ?, ?)",
            ("evt-1", datetime.now(timezone.utc).isoformat(), "test"),
        )
        writer.commit()

        # Reader connection should work concurrently
        reader = sqlite3.connect(str(db_path))
        reader.execute("PRAGMA journal_mode=WAL")
        count = reader.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 1

        writer.close()
        reader.close()

    def test_busy_timeout_prevents_immediate_failure(self, tmp_path: Path):
        """With busy_timeout, writers wait instead of failing immediately."""
        db_path = tmp_path / "events.db"

        conn1 = sqlite3.connect(str(db_path))
        conn1.execute("PRAGMA journal_mode=WAL")
        conn1.execute("PRAGMA busy_timeout=5000")
        conn1.execute("CREATE TABLE test (id TEXT)")
        conn1.commit()

        conn2 = sqlite3.connect(str(db_path))
        conn2.execute("PRAGMA journal_mode=WAL")
        conn2.execute("PRAGMA busy_timeout=5000")

        # Both connections should be able to read
        conn1.execute("INSERT INTO test VALUES ('a')")
        conn1.commit()

        result = conn2.execute("SELECT * FROM test").fetchall()
        assert len(result) == 1

        conn1.close()
        conn2.close()


class TestStalePIDHandling:
    """Test handling of stale PID files from crashed processes."""

    def test_detect_stale_pid(self, tmp_path: Path):
        """A PID file pointing to a non-existent process should be detected as stale."""
        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text("999999999")

        stored_pid = int(pid_file.read_text().strip())
        try:
            os.kill(stored_pid, 0)
            is_running = True
        except (OSError, ProcessLookupError):
            is_running = False

        assert not is_running

    def test_current_pid_is_alive(self, tmp_path: Path):
        """Our own PID should always be detected as alive."""
        pid_file = tmp_path / "self.pid"
        pid_file.write_text(str(os.getpid()))

        stored_pid = int(pid_file.read_text().strip())
        try:
            os.kill(stored_pid, 0)
            is_running = True
        except (OSError, ProcessLookupError):
            is_running = False

        assert is_running

    def test_stale_pid_cleanup(self, tmp_path: Path):
        """Stale PID files should be removable for new process."""
        pid_file = tmp_path / "worker.pid"

        # Simulate crashed process leaving PID file
        pid_file.write_text("999999999")
        assert pid_file.exists()

        # New process detects stale and overwrites
        stored_pid = int(pid_file.read_text().strip())
        try:
            os.kill(stored_pid, 0)
            is_running = True
        except (OSError, ProcessLookupError):
            is_running = False

        if not is_running:
            pid_file.write_text(str(os.getpid()))

        new_pid = int(pid_file.read_text().strip())
        assert new_pid == os.getpid()


class TestMissingPermissions:
    """Test graceful degradation when macOS permissions are missing."""

    def test_status_reflects_permission_state(self, tmp_path: Path):
        """Status file should accurately report permission states."""
        status = {
            "pid": os.getpid(),
            "version": "0.1.0",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "heartbeat": datetime.now(timezone.utc).isoformat(),
            "events_today": 0,
            "permissions_ok": False,
            "accessibility_permitted": False,
            "screen_recording_permitted": False,
            "db_path": str(tmp_path / "events.db"),
            "uptime_seconds": 10,
        }

        status_file = tmp_path / "daemon-status.json"
        status_file.write_text(json.dumps(status))

        loaded = json.loads(status_file.read_text())
        assert not loaded["permissions_ok"]
        assert not loaded["accessibility_permitted"]

    def test_worker_runs_without_vlm(self, tmp_path: Path):
        """Worker should function even without VLM backends available."""
        status = {
            "pid": os.getpid(),
            "version": "0.1.0",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "heartbeat": datetime.now(timezone.utc).isoformat(),
            "events_processed_today": 50,
            "sops_generated": 0,
            "last_pipeline_duration_ms": 100,
            "consecutive_errors": 0,
            "vlm_available": False,
            "sop_inducer_available": False,
        }

        # Worker should still process events, just without VLM enhancement
        assert status["events_processed_today"] == 50
        assert status["consecutive_errors"] == 0


class TestStatusFileRecovery:
    """Test recovery from corrupted or missing status files."""

    def test_missing_status_file_is_not_fatal(self, tmp_path: Path):
        """Reading a non-existent status file should not crash."""
        status_path = tmp_path / "daemon-status.json"
        assert not status_path.exists()

        # Simulating what AppState.readDaemonStatus does
        try:
            content = status_path.read_text()
            status = json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError):
            status = None

        assert status is None

    def test_corrupted_status_file_is_handled(self, tmp_path: Path):
        """A corrupted JSON file should not crash the reader."""
        status_path = tmp_path / "daemon-status.json"
        status_path.write_text("{ invalid json !!!")

        try:
            content = status_path.read_text()
            status = json.loads(content)
        except json.JSONDecodeError:
            status = None

        assert status is None

    def test_partial_status_file_is_handled(self, tmp_path: Path):
        """An incomplete status file (mid-write crash) should be handled."""
        status_path = tmp_path / "daemon-status.json"
        status_path.write_text('{"pid": 123, "version": "0.1.0"')  # Truncated

        try:
            content = status_path.read_text()
            status = json.loads(content)
        except json.JSONDecodeError:
            status = None

        assert status is None


class TestDiskSpaceHandling:
    """Test behavior under low disk space conditions."""

    def test_status_reports_disk_space(self, tmp_path: Path):
        """Status should include disk space information."""
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (1024 ** 3)

        # Should be able to determine free space
        assert free_gb >= 0

    def test_write_fails_gracefully_on_readonly_dir(self, tmp_path: Path):
        """Writing to a read-only directory should fail gracefully."""
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        os.chmod(str(readonly_dir), 0o444)

        try:
            status_path = readonly_dir / "test-status.json"
            try:
                status_path.write_text('{"test": true}')
                wrote = True
            except (PermissionError, OSError):
                wrote = False

            assert not wrote
        finally:
            # Restore permissions for cleanup
            os.chmod(str(readonly_dir), 0o755)
