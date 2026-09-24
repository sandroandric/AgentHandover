"""Daemon database schema and row helpers shared by worker tests.

Databases are built by applying the daemon's own migrations
(``crates/storage/src/migrations``, in the order ``crates/storage/src/lib.rs``
applies them), so the tests cannot drift from the schema the daemon writes.
"""

from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Schema: the daemon's own migrations, applied in the order the daemon applies them
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
STORAGE_SRC = REPO_ROOT / "crates" / "storage" / "src"


def migration_files() -> list[Path]:
    """The ``.sql`` files ``EventStore::run_migrations`` applies, in its order.

    Read from the ``include_str!`` calls in ``crates/storage/src/lib.rs``,
    so a migration the daemon runs is a migration the tests run.
    """
    lib_rs = (STORAGE_SRC / "lib.rs").read_text(encoding="utf-8")
    names = re.findall(r'include_str!\("(migrations/[^"]+\.sql)"\)', lib_rs)
    if not names:
        raise RuntimeError(f"no migrations found in {STORAGE_SRC / 'lib.rs'}")
    return [STORAGE_SRC / name for name in names]


def schema_version() -> int:
    """``CURRENT_SCHEMA_VERSION`` from ``crates/storage/src/schema.rs``."""
    schema_rs = (STORAGE_SRC / "schema.rs").read_text(encoding="utf-8")
    match = re.search(r"CURRENT_SCHEMA_VERSION\s*:\s*u32\s*=\s*(\d+)", schema_rs)
    if match is None:
        raise RuntimeError(f"CURRENT_SCHEMA_VERSION not found in {STORAGE_SRC / 'schema.rs'}")
    return int(match.group(1))


def apply_daemon_migrations(conn: sqlite3.Connection) -> None:
    """Bring an empty database to the daemon's current schema.

    Same statements and ``user_version`` as ``EventStore::open`` on a new
    file (WAL journal, every migration in order).
    """
    conn.execute("PRAGMA journal_mode=WAL")
    for sql_file in migration_files():
        conn.executescript(sql_file.read_text(encoding="utf-8"))
    conn.execute(f"PRAGMA user_version = {schema_version()}")
    conn.commit()


def create_daemon_db(path: Path) -> Path:
    """Create ``path`` as a new daemon database and return it."""
    conn = sqlite3.connect(str(path))
    try:
        apply_daemon_migrations(conn)
    finally:
        conn.close()
    return path


def kb_db(root: Path) -> Path:
    """The daemon database a test's ``KnowledgeBase(root=root)`` keeps its
    procedures in: ``.<name>.events.db`` beside the knowledge directory,
    created on first use, so every ``KnowledgeBase`` a test builds on the
    same root shares one store and different roots do not."""
    root = Path(root)
    path = root.parent / f".{root.name}.events.db"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        create_daemon_db(path)
    return path


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Row helpers
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
