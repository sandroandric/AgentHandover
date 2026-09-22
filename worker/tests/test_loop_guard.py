"""The worker loop survives repeated errors and quarantines poison events.

Unit pins for :mod:`agenthandover_worker.loop_guard` and
``WorkerDB.quarantine_events``, plus a run of the real ``main()`` loop
against a database holding one event the pipeline always fails on.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import time
from pathlib import Path

import pytest

from agenthandover_worker.db import WorkerDB
from agenthandover_worker.loop_guard import (
    ERROR_THRESHOLD,
    MAX_SOLO_FAILURES,
    EventRetryGuard,
    error_backoff_seconds,
    interruptible_sleep,
)

from conftest import insert_event


class TestErrorBackoff:
    def test_short_wait_below_threshold(self) -> None:
        for n in range(1, ERROR_THRESHOLD):
            assert error_backoff_seconds(n) == 5.0

    def test_doubles_from_threshold_and_caps_at_five_minutes(self) -> None:
        waits = [error_backoff_seconds(ERROR_THRESHOLD + i) for i in range(10)]
        assert waits[:7] == [5, 10, 20, 40, 80, 160, 300]
        assert all(w == 300 for w in waits[6:])


class TestInterruptibleSleep:
    def test_sleeps_full_duration_in_steps(self) -> None:
        slept: list[float] = []
        interruptible_sleep(2.5, lambda: False, sleep=slept.append)
        assert slept == [1.0, 1.0, 0.5]

    def test_returns_as_soon_as_stop_requested(self) -> None:
        slept: list[float] = []
        interruptible_sleep(300, lambda: len(slept) >= 2, sleep=slept.append)
        assert slept == [1.0, 1.0]


class TestEventRetryGuard:
    def test_failure_outside_event_processing_changes_nothing(self) -> None:
        guard = EventRetryGuard(max_batch=100)
        assert guard.failed([]) == []
        assert guard.batch_limit == 100

    def test_failed_batch_is_halved_until_single(self) -> None:
        guard = EventRetryGuard(max_batch=100)
        ids = [f"e{i}" for i in range(100)]
        limits = []
        while guard.batch_limit > 1:
            assert guard.failed(ids[: guard.batch_limit]) == []
            limits.append(guard.batch_limit)
        assert limits == [50, 25, 12, 6, 3, 1]

    def test_single_event_quarantined_after_solo_failures(self) -> None:
        guard = EventRetryGuard(max_batch=100)
        for _ in range(MAX_SOLO_FAILURES - 1):
            assert guard.failed(["poison"]) == []
            assert guard.batch_limit == 1
        assert guard.failed(["poison"]) == ["poison"]
        # Back to full batches once the poison event is out of the queue.
        assert guard.batch_limit == 100

    def test_success_restores_batch_and_forgets_failures(self) -> None:
        guard = EventRetryGuard(max_batch=100)
        guard.failed(["a"])
        guard.failed(["a"])
        guard.succeeded(["a"])
        assert guard.batch_limit == 100
        # The count restarted: two more solo failures do not quarantine.
        assert guard.failed(["a"]) == []
        assert guard.failed(["a"]) == []


class TestQuarantineEvents:
    def test_quarantined_event_leaves_queue(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        bad = insert_event(write_conn)
        good = insert_event(write_conn)
        with WorkerDB(tmp_db_path) as db:
            assert db.quarantine_events([bad]) == 1
            remaining = [e["id"] for e in db.get_unprocessed_events()]
        assert remaining == [good]
        row = write_conn.execute(
            "SELECT processed FROM events WHERE id = ?", (bad,)
        ).fetchone()
        assert row["processed"] == WorkerDB.QUARANTINED


class TestMainLoopSurvivesPoisonEvent:
    """Drive the real ``main()`` loop: one event makes the pipeline raise
    every time. The worker must not exit, must keep writing
    worker-status.json, must quarantine the poison event and must still
    process the healthy event behind it."""

    TICKS = 15

    def test_worker_stays_up_and_quarantines(
        self,
        tmp_db_path: Path,
        write_conn: sqlite3.Connection,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import agenthandover_worker.main as main_mod

        status_dir = tmp_path / "status"
        monkeypatch.setattr(main_mod, "_status_dir", lambda: status_dir)

        poison = insert_event(write_conn, timestamp="2026-09-14T10:00:00Z")
        healthy = insert_event(write_conn, timestamp="2026-09-14T10:00:01Z")

        real_pipeline = main_mod.run_pipeline
        pipeline_batches: list[list[str]] = []

        def failing_pipeline(events, **kwargs):
            ids = [e["id"] for e in events]
            pipeline_batches.append(ids)
            if poison in ids:
                raise RuntimeError("poison event")
            return real_pipeline(events, **kwargs)

        monkeypatch.setattr(main_mod, "run_pipeline", failing_pipeline)

        ticks: list[str] = []
        heartbeats: list[int] = []

        def tick(kind: str) -> None:
            ticks.append(kind)
            status = status_dir / "worker-status.json"
            if status.exists():
                heartbeats.append(json.loads(status.read_text())["consecutive_errors"])
            if len(ticks) >= self.TICKS:
                os.kill(os.getpid(), signal.SIGTERM)

        def fake_backoff(seconds, should_stop, **_kw):
            tick("error")

        real_sleep = time.sleep

        def fake_sleep(seconds: float) -> None:
            # Only the main loop's poll sleep is counted; anything shorter
            # than the poll interval is library-internal.
            if seconds >= 0.05:
                tick("poll")
            else:
                real_sleep(seconds)

        monkeypatch.setattr(main_mod, "interruptible_sleep", fake_backoff)
        monkeypatch.setattr(main_mod.time, "sleep", fake_sleep)

        old_term = signal.getsignal(signal.SIGTERM)
        old_int = signal.getsignal(signal.SIGINT)
        try:
            main_mod.main(
                [
                    "--db-path",
                    str(tmp_db_path),
                    "--poll-interval",
                    "0.05",
                    "--sops-dir",
                    str(tmp_path / "sops"),
                    "--knowledge-dir",
                    str(tmp_path / "knowledge"),
                ]
            )
        except SystemExit as exc:  # pragma: no cover - failure path
            pytest.fail(f"worker exited with {exc.code} instead of staying up")
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)

        # It ran all ticks (exited only on our SIGTERM) and the poison
        # event failed alone MAX_SOLO_FAILURES times before quarantine.
        assert len(ticks) >= self.TICKS
        assert pipeline_batches.count([poison]) == MAX_SOLO_FAILURES
        assert "error" in ticks
        # The heartbeat was written while failing, with the error count.
        assert max(heartbeats) >= 1

        rows = {
            r["id"]: r["processed"]
            for r in write_conn.execute("SELECT id, processed FROM events")
        }
        assert rows[poison] == WorkerDB.QUARANTINED
        assert rows[healthy] == WorkerDB.PROCESSED
