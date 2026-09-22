"""``record/record_focus_session.py`` against a live-shaped daemon database:
it selects the session's rows, anonymises them, and keeps the daemon's text."""

from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

from journeys.fixture_loader import (
    DAEMON_COLUMNS,
    EVENTS_FILE,
    SCREENSHOTS_TOKEN,
    load_fixture,
    row_counts,
)
from journeys.record import record_focus_session as recorder


def _rows(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(
            f"SELECT {', '.join(DAEMON_COLUMNS)} FROM events ORDER BY timestamp, rowid"
        )]
    finally:
        conn.close()


def test_records_and_anonymises_a_session(tmp_path: Path) -> None:
    live = load_fixture("focus_dailynews", tmp_path / "live")
    session_id = live.manifest["focus_session"]["session_id"]
    original = _rows(live.db_path)

    conn = sqlite3.connect(str(live.db_path))
    try:
        # Personal text on one OCR frame.
        frame_id, meta = conn.execute(
            "SELECT id, metadata_json FROM events "
            "WHERE json_extract(metadata_json, '$.ocr.full_text') IS NOT NULL LIMIT 1"
        ).fetchone()
        meta = json.loads(meta)
        meta["ocr"]["full_text"] += "\nJane Doe <jane.doe@corp.io> /Users/jane/Desktop"
        conn.execute(
            "UPDATE events SET metadata_json = ? WHERE id = ?",
            (json.dumps(meta, sort_keys=True, separators=(",", ":")), frame_id),
        )
        # Rows that are not part of the session.
        conn.execute(
            "INSERT INTO events (id, timestamp, kind_json, display_topology_json, "
            "primary_display_id, metadata_json) VALUES "
            "('other-session', ?, '{\"type\":\"FocusChange\"}', '[]', '1', "
            "'{\"focus_session_id\":\"another\"}')",
            (original[10]["timestamp"],),
        )
        conn.execute(
            "INSERT INTO events (id, timestamp, kind_json, display_topology_json, "
            "primary_display_id) VALUES ('later', '2026-09-15T10:00:00.5+00:00', "
            "'{\"type\":\"FocusChange\"}', '[]', '1')"
        )
        conn.commit()
    finally:
        conn.close()

    out_root = tmp_path / "fixtures"
    recorder.main([
        "--db", str(live.db_path),
        "--session-id", session_id,
        "--title", "Daily news digest for Jane Doe",
        "--name", "recorded_session",
        "--replace", "Jane Doe=Alex Example",
        "--until", live.manifest["focus_session"]["stopped_at"],
        "--fixtures-dir", str(out_root),
    ])

    fixture_dir = out_root / "recorded_session"
    with gzip.open(fixture_dir / EVENTS_FILE, "rt", encoding="utf-8") as fh:
        recorded = [json.loads(line) for line in fh]
    manifest = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))

    # The session's rows, nothing else, in order.
    assert [r["id"] for r in recorded] == [r["id"] for r in original]
    text = (fixture_dir / EVENTS_FILE).read_bytes()
    plain = gzip.decompress(text).decode("utf-8")
    for leaked in ("Jane Doe", "jane.doe@corp.io", "/Users/jane", str(live.screenshots_dir)):
        assert leaked not in plain
    assert "Alex Example <person1@example.com> /Users/user/Desktop" in plain

    by_id = {r["id"]: r for r in original}
    for row in recorded:
        before = by_id[row["id"]]
        meta = json.loads(row["metadata_json"])
        if "screenshot_path" in meta:
            assert meta["screenshot_path"].startswith(SCREENSHOTS_TOKEN + "/")
            assert meta["screenshot_path"].endswith(Path(json.loads(
                before["metadata_json"])["screenshot_path"]).name)
        else:
            # Rows with nothing to rewrite are the daemon's text, unchanged.
            assert row == before

    assert manifest["source"]["kind"] == "recorded"
    assert manifest["focus_session"]["title"] == "Daily news digest for Alex Example"
    assert manifest["focus_session"]["session_id"] == session_id
    assert manifest["row_counts"] == row_counts(recorded) == live.manifest["row_counts"]
    assert manifest["recorded_start"] == original[0]["timestamp"]
