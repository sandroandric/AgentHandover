"""Tests for usability improvements in main.py.

Covers:
- DB retry loop (_wait_for_db)
- Worker status file writing (_write_worker_status / _remove_worker_status)
- Pipeline progress logging
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agenthandover_worker.main import (
    _wait_for_db,
    _write_worker_status,
    _remove_worker_status,
    _status_dir,
    _WORKER_VERSION,
)


# ------------------------------------------------------------------
# 1. _wait_for_db: retry loop
# ------------------------------------------------------------------


class TestWaitForDB:
    def test_returns_true_if_db_exists(self, tmp_path: Path) -> None:
        db = tmp_path / "events.db"
        db.write_text("")
        assert _wait_for_db(db, [False]) is True

    def test_returns_false_on_timeout(self, tmp_path: Path) -> None:
        db = tmp_path / "events.db"
        with patch("agenthandover_worker.main._DB_RETRY_MAX_SECONDS", 1), \
             patch("agenthandover_worker.main._DB_RETRY_POLL_SECONDS", 0.2):
            result = _wait_for_db(db, [False])
        assert result is False

    def test_returns_false_on_shutdown(self, tmp_path: Path) -> None:
        db = tmp_path / "events.db"
        shutdown_flag = [True]
        with patch("agenthandover_worker.main._DB_RETRY_MAX_SECONDS", 10), \
             patch("agenthandover_worker.main._DB_RETRY_POLL_SECONDS", 0.1):
            result = _wait_for_db(db, shutdown_flag)
        assert result is False

    def test_detects_db_appearing_during_wait(self, tmp_path: Path) -> None:
        db = tmp_path / "events.db"
        call_count = [0]
        original_is_file = Path.is_file

        def patched_is_file(p: Path) -> bool:
            if p == db:
                call_count[0] += 1
                if call_count[0] >= 3:
                    db.write_text("")
                    return True
                return False
            return original_is_file(p)

        with patch("agenthandover_worker.main._DB_RETRY_MAX_SECONDS", 10), \
             patch("agenthandover_worker.main._DB_RETRY_POLL_SECONDS", 0.05), \
             patch.object(Path, "is_file", patched_is_file):
            result = _wait_for_db(db, [False])
        assert result is True


# ------------------------------------------------------------------
# 2. _write_worker_status: atomic status file
# ------------------------------------------------------------------


class TestWriteWorkerStatus:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        with patch("agenthandover_worker.main._status_dir", return_value=tmp_path):
            _write_worker_status(
                started_at="2026-02-18T10:00:00Z",
                events_processed_today=42,
                sops_generated=3,
                last_pipeline_duration_ms=450,
                consecutive_errors=0,
                vlm_available=True,
                sop_inducer_available=True,
            )
        status_file = tmp_path / "worker-status.json"
        assert status_file.exists()
        data = json.loads(status_file.read_text())
        assert data["pid"] == os.getpid()
        assert data["version"] == _WORKER_VERSION
        assert data["started_at"] == "2026-02-18T10:00:00Z"
        assert data["events_processed_today"] == 42
        assert data["sops_generated"] == 3
        assert data["last_pipeline_duration_ms"] == 450
        assert data["consecutive_errors"] == 0
        assert data["vlm_available"] is True
        assert data["sop_inducer_available"] is True
        assert "heartbeat" in data

    def test_overwrites_previous_status(self, tmp_path: Path) -> None:
        with patch("agenthandover_worker.main._status_dir", return_value=tmp_path):
            _write_worker_status(
                started_at="2026-02-18T10:00:00Z",
                events_processed_today=10,
                sops_generated=0,
                last_pipeline_duration_ms=None,
                consecutive_errors=0,
                vlm_available=False,
                sop_inducer_available=False,
            )
            _write_worker_status(
                started_at="2026-02-18T10:00:00Z",
                events_processed_today=50,
                sops_generated=2,
                last_pipeline_duration_ms=300,
                consecutive_errors=1,
                vlm_available=True,
                sop_inducer_available=True,
            )
        data = json.loads((tmp_path / "worker-status.json").read_text())
        assert data["events_processed_today"] == 50
        assert data["sops_generated"] == 2
        assert data["last_pipeline_duration_ms"] == 300
        assert data["consecutive_errors"] == 1

    def test_null_pipeline_duration(self, tmp_path: Path) -> None:
        with patch("agenthandover_worker.main._status_dir", return_value=tmp_path):
            _write_worker_status(
                started_at="2026-02-18T10:00:00Z",
                events_processed_today=0,
                sops_generated=0,
                last_pipeline_duration_ms=None,
                consecutive_errors=0,
                vlm_available=False,
                sop_inducer_available=False,
            )
        data = json.loads((tmp_path / "worker-status.json").read_text())
        assert data["last_pipeline_duration_ms"] is None

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "sub" / "dir"
        with patch("agenthandover_worker.main._status_dir", return_value=nested):
            _write_worker_status(
                started_at="2026-02-18T10:00:00Z",
                events_processed_today=0,
                sops_generated=0,
                last_pipeline_duration_ms=None,
                consecutive_errors=0,
                vlm_available=False,
                sop_inducer_available=False,
            )
        assert (nested / "worker-status.json").exists()


# ------------------------------------------------------------------
# 3. _remove_worker_status
# ------------------------------------------------------------------


class TestRemoveWorkerStatus:
    def test_removes_existing_file(self, tmp_path: Path) -> None:
        status_file = tmp_path / "worker-status.json"
        status_file.write_text("{}")
        with patch("agenthandover_worker.main._status_dir", return_value=tmp_path):
            _remove_worker_status()
        assert not status_file.exists()

    def test_no_error_if_missing(self, tmp_path: Path) -> None:
        with patch("agenthandover_worker.main._status_dir", return_value=tmp_path):
            _remove_worker_status()  # Should not raise


# ------------------------------------------------------------------
# 4. _status_dir returns correct platform paths
# ------------------------------------------------------------------


class TestStatusDir:
    def test_returns_path_object(self) -> None:
        result = _status_dir()
        assert isinstance(result, Path)

    def test_darwin_path(self) -> None:
        with patch("agenthandover_worker.main._platform") as mock_platform:
            mock_platform.system.return_value = "Darwin"
            result = _status_dir()
        assert "Library/Application Support/agenthandover" in str(result)

    def test_linux_path(self) -> None:
        with patch("agenthandover_worker.main._platform") as mock_platform:
            mock_platform.system.return_value = "Linux"
            result = _status_dir()
        assert ".local/share/agenthandover" in str(result)


# ------------------------------------------------------------------
# 5. Worker version constant
# ------------------------------------------------------------------


class TestWorkerVersion:
    def test_version_format(self) -> None:
        parts = _WORKER_VERSION.split(".")
        assert len(parts) == 3
        for p in parts:
            assert p.isdigit()
