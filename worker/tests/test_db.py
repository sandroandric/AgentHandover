"""Tests for oc_apprentice_worker.db.WorkerDB.

Each test gets a temporary SQLite database (created by conftest fixtures)
pre-initialised with the daemon's schema.  We insert test data via a
read-write connection, then verify that WorkerDB (read-only) reads it
correctly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from oc_apprentice_worker.db import WorkerDB

from conftest import insert_episode, insert_event, insert_vlm_job


# ------------------------------------------------------------------
# 1. Basic open / read-only
# ------------------------------------------------------------------


class TestOpenReadonlyDatabase:
    """WorkerDB opens and reads from a pre-populated database."""

    def test_opens_existing_database(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        event_id = insert_event(write_conn, kind_json='{"AppSwitch":{}}')

        with WorkerDB(tmp_db_path) as db:
            row = db.get_event_by_id(event_id)

        assert row is not None
        assert row["id"] == event_id
        assert row["kind_json"] == '{"AppSwitch":{}}'

    def test_rejects_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            WorkerDB(tmp_path / "does_not_exist.db")


# ------------------------------------------------------------------
# 2. Unprocessed events
# ------------------------------------------------------------------


class TestGetUnprocessedEvents:
    """Insert events with mixed processed flags; only unprocessed returned."""

    def test_returns_only_unprocessed(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        # 2 processed, 3 unprocessed
        insert_event(write_conn, processed=1)
        insert_event(write_conn, processed=1)
        insert_event(write_conn, processed=0)
        insert_event(write_conn, processed=0)
        insert_event(write_conn, processed=0)

        with WorkerDB(tmp_db_path) as db:
            results = db.get_unprocessed_events()

        assert len(results) == 3
        for row in results:
            assert row["processed"] == 0

    def test_respects_limit(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        for _ in range(10):
            insert_event(write_conn, processed=0)

        with WorkerDB(tmp_db_path) as db:
            results = db.get_unprocessed_events(limit=3)

        assert len(results) == 3

    def test_ordered_by_timestamp_ascending(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        insert_event(
            write_conn,
            event_id="e-late",
            timestamp="2026-02-16T12:00:00.000Z",
        )
        insert_event(
            write_conn,
            event_id="e-early",
            timestamp="2026-02-16T08:00:00.000Z",
        )

        with WorkerDB(tmp_db_path) as db:
            results = db.get_unprocessed_events()

        assert results[0]["id"] == "e-early"
        assert results[1]["id"] == "e-late"


# ------------------------------------------------------------------
# 3. Get event by ID
# ------------------------------------------------------------------


class TestGetEventById:
    """Fetch a single event by UUID and verify fields."""

    def test_returns_matching_event(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        eid = insert_event(
            write_conn,
            event_id="abc-123",
            timestamp="2026-02-16T10:00:00.000Z",
            kind_json='{"DwellSnapshot":{}}',
        )

        with WorkerDB(tmp_db_path) as db:
            row = db.get_event_by_id(eid)

        assert row is not None
        assert row["id"] == "abc-123"
        assert row["timestamp"] == "2026-02-16T10:00:00.000Z"
        assert row["kind_json"] == '{"DwellSnapshot":{}}'
        assert row["processed"] == 0
        assert row["display_topology_json"] == "[]"
        assert row["primary_display_id"] == "main"

    def test_returns_none_for_missing_id(self, tmp_db_path: Path) -> None:
        with WorkerDB(tmp_db_path) as db:
            assert db.get_event_by_id("nonexistent-uuid") is None


# ------------------------------------------------------------------
# 4. Episodes
# ------------------------------------------------------------------


class TestGetEpisodes:
    """Insert episodes and verify retrieval (newest first)."""

    def test_returns_episodes(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        id1 = insert_episode(
            write_conn,
            episode_id="ep-1",
            start_time="2026-02-16T08:00:00.000Z",
        )
        id2 = insert_episode(
            write_conn,
            episode_id="ep-2",
            start_time="2026-02-16T12:00:00.000Z",
        )

        with WorkerDB(tmp_db_path) as db:
            episodes = db.get_episodes()

        assert len(episodes) == 2
        ids = {e["id"] for e in episodes}
        assert id1 in ids
        assert id2 in ids

    def test_respects_limit(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        for i in range(5):
            insert_episode(write_conn, episode_id=f"ep-{i}")

        with WorkerDB(tmp_db_path) as db:
            episodes = db.get_episodes(limit=2)

        assert len(episodes) == 2


# ------------------------------------------------------------------
# 5. Pending VLM jobs
# ------------------------------------------------------------------


class TestGetPendingVlmJobs:
    """Insert VLM jobs with mixed statuses; only pending returned."""

    def test_returns_only_pending(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        ev1 = insert_event(write_conn)
        ev2 = insert_event(write_conn)
        ev3 = insert_event(write_conn)

        insert_vlm_job(write_conn, event_id=ev1, status="pending", priority=0.8)
        insert_vlm_job(write_conn, event_id=ev2, status="completed", priority=0.9)
        insert_vlm_job(write_conn, event_id=ev3, status="pending", priority=0.3)

        with WorkerDB(tmp_db_path) as db:
            jobs = db.get_pending_vlm_jobs()

        assert len(jobs) == 2
        for job in jobs:
            assert job["status"] == "pending"

    def test_ordered_by_priority_descending(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        ev1 = insert_event(write_conn)
        ev2 = insert_event(write_conn)
        ev3 = insert_event(write_conn)

        insert_vlm_job(write_conn, event_id=ev1, priority=0.2, vlm_id="low")
        insert_vlm_job(write_conn, event_id=ev2, priority=0.9, vlm_id="high")
        insert_vlm_job(write_conn, event_id=ev3, priority=0.5, vlm_id="mid")

        with WorkerDB(tmp_db_path) as db:
            jobs = db.get_pending_vlm_jobs()

        assert jobs[0]["id"] == "high"
        assert jobs[1]["id"] == "mid"
        assert jobs[2]["id"] == "low"

    def test_respects_limit(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        for _ in range(5):
            ev = insert_event(write_conn)
            insert_vlm_job(write_conn, event_id=ev)

        with WorkerDB(tmp_db_path) as db:
            jobs = db.get_pending_vlm_jobs(limit=2)

        assert len(jobs) == 2


# ------------------------------------------------------------------
# 6. Context manager
# ------------------------------------------------------------------


class TestContextManager:
    """WorkerDB works as a context manager and closes cleanly."""

    def test_enter_returns_self(self, tmp_db_path: Path) -> None:
        db = WorkerDB(tmp_db_path)
        with db as ctx:
            assert ctx is db

    def test_connection_closed_after_exit(self, tmp_db_path: Path) -> None:
        with WorkerDB(tmp_db_path) as db:
            # Connection is usable inside the block
            db.get_unprocessed_events()

        # After exiting the block, the connection should be closed.
        # Attempting a query should raise.
        with pytest.raises(Exception):
            db.get_unprocessed_events()


# ------------------------------------------------------------------
# 7. Empty database
# ------------------------------------------------------------------


class TestEmptyDatabase:
    """All queries return empty results on a fresh (schema-only) database."""

    def test_no_events(self, tmp_db_path: Path) -> None:
        with WorkerDB(tmp_db_path) as db:
            assert db.get_unprocessed_events() == []

    def test_no_event_by_id(self, tmp_db_path: Path) -> None:
        with WorkerDB(tmp_db_path) as db:
            assert db.get_event_by_id("anything") is None

    def test_no_episodes(self, tmp_db_path: Path) -> None:
        with WorkerDB(tmp_db_path) as db:
            assert db.get_episodes() == []

    def test_no_vlm_jobs(self, tmp_db_path: Path) -> None:
        with WorkerDB(tmp_db_path) as db:
            assert db.get_pending_vlm_jobs() == []


# ------------------------------------------------------------------
# 8. Episode store — persist translated episodes across pipeline cycles
# ------------------------------------------------------------------


class TestEpisodeStore:
    """Episode store persists and retrieves translated episode steps."""

    def test_save_and_retrieve_episode(self, tmp_db_path: Path) -> None:
        steps = [
            {"step": "click", "target": "button", "confidence": 0.8},
            {"step": "type", "target": "input", "confidence": 0.7},
        ]

        with WorkerDB(tmp_db_path) as db:
            assert db.save_episode_steps("ep-1", "app:safari", steps)
            stored = db.get_all_episode_steps()

        assert len(stored) == 1
        assert len(stored[0]) == 2
        assert stored[0][0]["step"] == "click"
        assert stored[0][1]["step"] == "type"

    def test_multiple_episodes_accumulated(self, tmp_db_path: Path) -> None:
        steps_a = [{"step": "click", "target": "a"}]
        steps_b = [{"step": "navigate", "target": "b"}, {"step": "read", "target": "c"}]

        with WorkerDB(tmp_db_path) as db:
            db.save_episode_steps("ep-1", "app:a", steps_a)
            db.save_episode_steps("ep-2", "app:b", steps_b)
            stored = db.get_all_episode_steps()

        assert len(stored) == 2

    def test_replace_on_duplicate_episode_id(self, tmp_db_path: Path) -> None:
        steps_v1 = [{"step": "click", "target": "old"}]
        steps_v2 = [{"step": "click", "target": "new"}, {"step": "type", "target": "x"}]

        with WorkerDB(tmp_db_path) as db:
            db.save_episode_steps("ep-1", "app:a", steps_v1)
            db.save_episode_steps("ep-1", "app:a", steps_v2)
            stored = db.get_all_episode_steps()

        assert len(stored) == 1
        assert len(stored[0]) == 2
        assert stored[0][0]["target"] == "new"

    def test_empty_steps_skipped(self, tmp_db_path: Path) -> None:
        with WorkerDB(tmp_db_path) as db:
            assert db.save_episode_steps("ep-1", "app:a", [])
            stored = db.get_all_episode_steps()

        assert len(stored) == 0

    def test_count_stored_episodes(self, tmp_db_path: Path) -> None:
        with WorkerDB(tmp_db_path) as db:
            assert db.count_stored_episodes() == 0
            db.save_episode_steps("ep-1", "a", [{"step": "click"}])
            db.save_episode_steps("ep-2", "b", [{"step": "type"}])
            assert db.count_stored_episodes() == 2

    def test_cleanup_old_episodes(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        with WorkerDB(tmp_db_path) as db:
            db.save_episode_steps("ep-1", "a", [{"step": "click"}])
            # Backdate the record to 30 days ago
            write_conn.execute(
                "UPDATE translated_episodes SET created_at = "
                "datetime('now', '-30 days') WHERE episode_id = 'ep-1'"
            )
            write_conn.commit()
            # Cleanup with 14-day retention should remove it
            deleted = db.cleanup_old_episodes(max_age_days=14)
            assert deleted == 1
            assert db.count_stored_episodes() == 0

    def test_get_episode_steps_on_fresh_db(self, tmp_db_path: Path) -> None:
        """Episode store works on a DB that has never had the table created."""
        with WorkerDB(tmp_db_path) as db:
            stored = db.get_all_episode_steps()
        assert stored == []
