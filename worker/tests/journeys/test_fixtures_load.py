"""Session fixtures load through the daemon's migrations and match their manifests."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agenthandover_worker.db import WorkerDB
from daemon_db import STORAGE_SRC, migration_files, schema_version
from journeys.fixture_loader import (
    DAEMON_COLUMNS,
    FIXTURES_DIR,
    read_manifest,
    shift_timestamp,
)

FIXTURE_NAMES = sorted(
    p.name for p in FIXTURES_DIR.iterdir() if (p / "manifest.json").is_file()
)
DAEMON_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3}|\.\d{6}|\.\d{9})?\+00:00$")


def _counts(db_path: Path) -> dict:
    conn = sqlite3.connect(str(db_path))
    try:
        one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
        by_kind = dict(conn.execute(
            "SELECT json_extract(kind_json, '$.type'), COUNT(*) FROM events GROUP BY 1"
        ).fetchall())
        return {
            "events": one("SELECT COUNT(*) FROM events"),
            "by_kind": by_kind,
            "chrome_extension": one(
                "SELECT COUNT(*) FROM events "
                "WHERE json_extract(window_json, '$.app_id') = 'com.google.Chrome'"
            ),
            "clipboard_with_preview": one(
                "SELECT COUNT(*) FROM events "
                "WHERE json_extract(kind_json, '$.content_preview') <> ''"
            ),
            "dom_snapshots": one(
                "SELECT COUNT(*) FROM events "
                "WHERE json_type(metadata_json, '$.anchors') = 'array' "
                "   OR json_type(metadata_json, '$.nodes') = 'array'"
            ),
            "focus_tagged": one(
                "SELECT COUNT(*) FROM events "
                "WHERE json_extract(metadata_json, '$.focus_session_id') IS NOT NULL"
            ),
            "with_ocr_text": one(
                "SELECT COUNT(*) FROM events "
                "WHERE json_extract(metadata_json, '$.ocr.full_text') <> ''"
            ),
            "with_screenshot": one(
                "SELECT COUNT(*) FROM events "
                "WHERE json_extract(metadata_json, '$.screenshot_path') IS NOT NULL"
            ),
        }
    finally:
        conn.close()


def test_migrations_are_the_daemons() -> None:
    """Every .sql file is applied, in version order, and the count is the schema version."""
    on_disk = sorted((STORAGE_SRC / "migrations").glob("*.sql"))
    assert [p.resolve() for p in migration_files()] == [p.resolve() for p in on_disk]
    assert len(on_disk) == schema_version()


def test_every_fixture_is_listed() -> None:
    assert "focus_dailynews" in FIXTURE_NAMES


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_loads_through_real_migrations(name: str, events_db, temp_home) -> None:
    fixture = events_db(name)
    manifest = read_manifest(name)

    conn = sqlite3.connect(str(fixture.db_path))
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == schema_version()
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        # The daemon's defaults apply to every loaded row, except the
        # annotator's columns for the rows the fixture annotates.
        assert conn.execute(
            "SELECT COUNT(*) FROM events WHERE processed <> 0"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM events WHERE annotation_status <> 'pending'"
        ).fetchone()[0] == manifest.get("annotated_rows", 0)
    finally:
        conn.close()

    assert _counts(fixture.db_path) == manifest["row_counts"]
    assert fixture.db_path.parent == temp_home.data_dir


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_rows_keep_the_daemons_text_formats(name: str, events_db) -> None:
    fixture = events_db(name)
    conn = sqlite3.connect(str(fixture.db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(f"SELECT {', '.join(DAEMON_COLUMNS)} FROM events").fetchall()
    finally:
        conn.close()
    for row in rows:
        assert DAEMON_TIMESTAMP.match(row["timestamp"]), row["timestamp"]
        kind = json.loads(row["kind_json"])
        assert isinstance(kind, dict) and "type" in kind  # internally tagged
        if row["window_json"] is not None:
            assert set(json.loads(row["window_json"])) == {
                "window_id", "app_id", "title", "bounds_global_px", "z_order", "is_fullscreen",
            }
        metadata = json.loads(row["metadata_json"])
        assert row["metadata_json"] == json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        shot = metadata.get("screenshot_path")
        if shot is not None:
            path = Path(shot)
            assert path.parent == fixture.screenshots_dir
            assert path.is_file() and path.read_bytes()[:2] == b"\xff\xd8"


def test_focus_dailynews_is_read_by_the_worker(events_db) -> None:
    fixture = events_db("focus_dailynews")
    manifest = fixture.manifest
    session_id = manifest["focus_session"]["session_id"]
    with WorkerDB(fixture.db_path) as db:
        events = db.get_focus_session_events(session_id)
    kinds = [json.loads(e["kind_json"])["type"] for e in events]
    # Tagged native events plus the untagged clipboard rows inside the session.
    assert len(events) == manifest["row_counts"]["focus_tagged"] + kinds.count("ClipboardChange")
    assert kinds.count("ClipboardChange") == manifest["row_counts"]["by_kind"]["ClipboardChange"]


def test_rebase_moves_the_recording_and_keeps_formats(events_db, temp_home) -> None:
    target = datetime(2027, 1, 4, 12, 0, 0, tzinfo=timezone.utc)
    fixture = events_db("focus_dailynews", rebase_to=target)
    recorded_end = fixture.manifest["recorded_end"]

    conn = sqlite3.connect(str(fixture.db_path))
    try:
        last = conn.execute("SELECT MAX(timestamp) FROM events").fetchone()[0]
        first = conn.execute("SELECT MIN(timestamp) FROM events").fetchone()[0]
    finally:
        conn.close()
    assert last == shift_timestamp(recorded_end, fixture.shift)
    last_dt = datetime.fromisoformat(last)
    assert abs(last_dt - target) < timedelta(seconds=1)
    assert first == shift_timestamp(fixture.manifest["recorded_start"], fixture.shift)
    assert DAEMON_TIMESTAMP.match(last)

    signal_path = fixture.write_focus_signal(temp_home.data_dir)
    signal = json.loads(signal_path.read_text(encoding="utf-8"))
    assert signal["status"] == "stopped"
    assert signal["session_id"] == fixture.manifest["focus_session"]["session_id"]
    assert signal["started_at"].startswith("2027-01-04T11:50")


def test_shift_timestamp_keeps_text_format() -> None:
    assert shift_timestamp("2026-09-15T08:30:00.900802+00:00", timedelta(days=1)) == (
        "2026-09-16T08:30:00.900802+00:00"
    )
    assert shift_timestamp("2026-09-15T23:59:59.123Z", timedelta(seconds=2)) == (
        "2026-09-16T00:00:01.123Z"
    )
    assert shift_timestamp("2026-09-15T08:29:59Z", timedelta(0)) == "2026-09-15T08:29:59Z"
