"""Shared fixtures for worker tests.

Creates temporary SQLite databases with the same schema that the
Rust daemon writes (``crates/storage/src/migrations/v001_initial.sql``
plus ``v002_add_display_ids_spanned.sql``).
"""

from __future__ import annotations

import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Schema — mirrors crates/storage/src/migrations/v001 + v002 exactly
# ---------------------------------------------------------------------------

DAEMON_SCHEMA = """\
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY NOT NULL,
    timestamp TEXT NOT NULL,
    kind_json TEXT NOT NULL,
    window_json TEXT,
    display_topology_json TEXT NOT NULL,
    primary_display_id TEXT NOT NULL,
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
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db_path(tmp_path: Path) -> Path:
    """Create an empty database with the daemon schema, return its path."""
    db_file = tmp_path / "events.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA journal_mode=wal;")
    conn.executescript(DAEMON_SCHEMA)
    conn.close()
    return db_file


@pytest.fixture()
def write_conn(tmp_db_path: Path) -> sqlite3.Connection:
    """Return a read-write connection to the temp database (for inserting
    test data).  Callers must NOT close this; the fixture handles cleanup.
    """
    conn = sqlite3.connect(str(tmp_db_path))
    conn.row_factory = sqlite3.Row
    yield conn  # type: ignore[misc]
    conn.close()


# ---------------------------------------------------------------------------
# Helpers available to all tests
# ---------------------------------------------------------------------------


def insert_event(
    conn: sqlite3.Connection,
    *,
    event_id: str | None = None,
    timestamp: str | None = None,
    kind_json: str = '{"FocusChange":{}}',
    processed: int = 0,
) -> str:
    """Insert a minimal event row and return its id."""
    eid = event_id or _new_uuid()
    ts = timestamp or _now_iso()
    conn.execute(
        "INSERT INTO events "
        "(id, timestamp, kind_json, display_topology_json, primary_display_id, processed) "
        "VALUES (?, ?, ?, '[]', 'main', ?)",
        (eid, ts, kind_json, processed),
    )
    conn.commit()
    return eid


def insert_episode(
    conn: sqlite3.Connection,
    *,
    episode_id: str | None = None,
    start_time: str | None = None,
    status: str = "open",
) -> str:
    """Insert a minimal episode row and return its id."""
    eid = episode_id or _new_uuid()
    st = start_time or _now_iso()
    conn.execute(
        "INSERT INTO episodes (id, start_time, status) VALUES (?, ?, ?)",
        (eid, st, status),
    )
    conn.commit()
    return eid


def insert_vlm_job(
    conn: sqlite3.Connection,
    *,
    event_id: str,
    priority: float = 0.5,
    status: str = "pending",
    vlm_id: str | None = None,
) -> str:
    """Insert a VLM queue job linked to an existing event. Returns the job id."""
    jid = vlm_id or _new_uuid()
    ttl = _now_iso()
    conn.execute(
        "INSERT INTO vlm_queue (id, event_id, priority, status, ttl_expires_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (jid, event_id, priority, status, ttl),
    )
    conn.commit()
    return jid
