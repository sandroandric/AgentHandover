"""2.0 Memory thumbnails: the worker deletes the daemon's encrypted
thumbnails past ``[memory] thumbnail_days`` (by file age and by event age),
those whose event is gone, and, for forget, those of given events."""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agenthandover_worker import main as wm
from agenthandover_worker import thumbnails
from agenthandover_worker.db import WorkerDB
from daemon_db import create_daemon_db, insert_event

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


@pytest.fixture()
def events_db(tmp_path: Path) -> Path:
    return create_daemon_db(tmp_path / "events.db")


@pytest.fixture()
def thumbs(tmp_path: Path) -> Path:
    path = tmp_path / "thumbs"
    path.mkdir()
    return path


def _event(db_path: Path, when: datetime) -> str:
    conn = sqlite3.connect(str(db_path))
    try:
        return insert_event(conn, event_id=str(uuid.uuid4()), timestamp=_iso(when))
    finally:
        conn.close()


def _thumb(directory: Path, name: str, written: datetime) -> Path:
    path = directory / name
    path.write_bytes(os.urandom(64))
    os.utime(path, (written.timestamp(), written.timestamp()))
    return path


def _names(directory: Path) -> set[str]:
    return {p.name for p in directory.iterdir()}


class TestExpiry:
    def test_deletes_by_file_age_and_by_event_age(self, events_db, thumbs):
        fresh = _event(events_db, NOW - timedelta(days=1))
        old_event = _event(events_db, NOW - timedelta(days=8))
        aged_file = _event(events_db, NOW - timedelta(days=1))
        _thumb(thumbs, f"{fresh}.ahthumb", NOW - timedelta(days=1))
        # The file was touched recently, but its frame is 8 days old.
        _thumb(thumbs, f"{old_event}.ahthumb", NOW - timedelta(hours=1))
        # The file itself is past the limit.
        _thumb(thumbs, f"{aged_file}.ahthumb", NOW - timedelta(days=7, minutes=1))

        with WorkerDB(events_db) as db:
            removed = thumbnails.expire_thumbnails(db, days=7, directory=thumbs, now=NOW)

        assert removed == 2
        assert _names(thumbs) == {f"{fresh}.ahthumb"}

    def test_thumbnails_of_vanished_events_go_once_settled(self, events_db, thumbs):
        # A forgotten or purged event: its thumbnail is an orphan.
        _thumb(thumbs, f"{uuid.uuid4()}.ahthumb", NOW - timedelta(hours=2))
        # Just written: its event may not be in the database yet.
        young = f"{uuid.uuid4()}.ahthumb"
        _thumb(thumbs, young, NOW - timedelta(seconds=30))
        # A temp file left by a crash mid-write, and anything not ours.
        _thumb(thumbs, f".{uuid.uuid4()}.ahthumb.tmp", NOW - timedelta(hours=1))
        _thumb(thumbs, "notes.txt", NOW - timedelta(days=30))

        with WorkerDB(events_db) as db:
            removed = thumbnails.expire_thumbnails(db, days=7, directory=thumbs, now=NOW)

        assert removed == 2
        assert _names(thumbs) == {young, "notes.txt"}

    def test_a_memory_row_keeps_its_thumbnail_after_the_raw_event_is_purged(
        self, events_db, thumbs,
    ):
        # Raw events go after 14 days; a Memory moment (and, with
        # thumbnail_days > 14, its thumbnail) outlives them.
        shown = str(uuid.uuid4())
        _thumb(thumbs, f"{shown}.ahthumb", NOW - timedelta(days=20))
        conn = sqlite3.connect(str(events_db))
        conn.execute(
            "INSERT INTO memory_frames (id, first_ts, last_ts, frames, app, window_title, "
            "embedding, model, dim, thumb_id, created_at) "
            "VALUES (?, ?, ?, 1, 'Safari', 't', x'00', 'm', 256, ?, ?)",
            (shown, _iso(NOW), _iso(NOW), shown, _iso(NOW)),
        )
        conn.commit()
        conn.close()

        with WorkerDB(events_db) as db:
            assert thumbnails.expire_thumbnails(db, days=30, directory=thumbs, now=NOW) == 0
            assert _names(thumbs) == {f"{shown}.ahthumb"}
            assert thumbnails.expire_thumbnails(db, days=7, directory=thumbs, now=NOW) == 1

    def test_zero_days_keeps_none_and_missing_dir_is_fine(self, events_db, thumbs, tmp_path):
        event = _event(events_db, NOW - timedelta(minutes=10))
        _thumb(thumbs, f"{event}.ahthumb", NOW - timedelta(minutes=10))
        with WorkerDB(events_db) as db:
            assert thumbnails.expire_thumbnails(db, days=0, directory=thumbs, now=NOW) == 1
            assert thumbnails.expire_thumbnails(
                db, days=7, directory=tmp_path / "absent", now=NOW) == 0
        assert _names(thumbs) == set()

    def test_daily_maintenance_expires_thumbnails_in_the_data_dir(
        self, events_db, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(thumbnails.paths, "data_dir", lambda: tmp_path / "data")
        directory = thumbnails.thumbs_dir()
        assert directory == tmp_path / "data" / "thumbs" == thumbnails.thumbs_dir(tmp_path / "data")
        directory.mkdir(parents=True)
        old = _event(events_db, datetime.now(timezone.utc) - timedelta(days=10))
        _thumb(directory, f"{old}.ahthumb", datetime.now(timezone.utc) - timedelta(days=10))
        with WorkerDB(events_db) as db:
            wm._run_daily_maintenance(db=db, thumbnail_days=7)
        assert _names(directory) == set()

    def test_daily_maintenance_survives_a_failing_expiry(self, tmp_path, monkeypatch):
        monkeypatch.setattr(thumbnails.paths, "data_dir", lambda: tmp_path / "data")
        db = MagicMock()
        db.get_events_by_ids.side_effect = RuntimeError("locked")
        directory = thumbnails.thumbs_dir()
        directory.mkdir(parents=True)
        _thumb(directory, f"{uuid.uuid4()}.ahthumb", datetime.now(timezone.utc))
        wm._run_daily_maintenance(db=db, thumbnail_days=7)  # logged, not raised


class TestForget:
    def test_deletes_only_the_named_events(self, thumbs):
        keep, gone = str(uuid.uuid4()), str(uuid.uuid4())
        for event_id in (keep, gone):
            _thumb(thumbs, f"{event_id}.ahthumb", NOW)
        assert thumbnails.delete_thumbnails([gone, str(uuid.uuid4())], thumbs) == 1
        assert _names(thumbs) == {f"{keep}.ahthumb"}

    def test_never_leaves_the_directory(self, tmp_path, thumbs):
        outside = tmp_path / "victim.ahthumb"
        outside.write_bytes(b"x")
        assert thumbnails.delete_thumbnails(["../victim", "", ".hidden"], thumbs) == 0
        assert outside.exists()
