"""Read-only SQLite interface to the daemon's event database.

The daemon owns all writes; this module opens the DB in read-only mode
and uses WAL journal so reads never block the daemon's writes.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Self

logger = logging.getLogger(__name__)


class WorkerDB:
    """Read-only connection to the daemon's SQLite database.

    Opens the database via the SQLite URI interface with ``mode=ro``
    so no accidental writes can occur.  WAL journal mode is set to
    allow concurrent reads while the daemon continues to write.

    Implements the context-manager protocol so it can be used as::

        with WorkerDB(path) as db:
            events = db.get_unprocessed_events()
    """

    def __init__(self, db_path: str | Path) -> None:
        resolved = Path(db_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Database file does not exist: {resolved}")

        uri = f"file:{resolved}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True)
        self._conn.row_factory = sqlite3.Row

        # Enable WAL reads — even though we are read-only, setting the
        # journal_mode pragma to wal tells our connection to use the WAL
        # reader path.  If the daemon already created the DB in WAL mode
        # this is a no-op; if not, the pragma is silently ignored on a
        # read-only connection.
        try:
            self._conn.execute("PRAGMA journal_mode=wal;")
        except sqlite3.OperationalError:
            # Read-only connections on some SQLite builds cannot change
            # journal mode — that is fine, the daemon sets it on create.
            pass

        logger.info("WorkerDB opened (read-only): %s", resolved)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def _rows_to_dicts(self, rows: list[sqlite3.Row]) -> list[dict]:
        """Convert a list of sqlite3.Row objects to plain dicts."""
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def get_unprocessed_events(self, limit: int = 100) -> list[dict]:
        """Return up to *limit* events where ``processed = 0``,
        ordered by timestamp ascending (oldest first).
        """
        cur = self._conn.execute(
            "SELECT * FROM events WHERE processed = 0 "
            "ORDER BY timestamp ASC LIMIT ?",
            (limit,),
        )
        return self._rows_to_dicts(cur.fetchall())

    def get_event_by_id(self, event_id: str) -> dict | None:
        """Return a single event by its UUID, or ``None``."""
        cur = self._conn.execute(
            "SELECT * FROM events WHERE id = ?",
            (event_id,),
        )
        row = cur.fetchone()
        return dict(row) if row is not None else None

    # ------------------------------------------------------------------
    # Episodes
    # ------------------------------------------------------------------

    def get_episodes(self, limit: int = 50) -> list[dict]:
        """Return up to *limit* episodes, newest first."""
        cur = self._conn.execute(
            "SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return self._rows_to_dicts(cur.fetchall())

    # ------------------------------------------------------------------
    # VLM queue
    # ------------------------------------------------------------------

    def get_pending_vlm_jobs(self, limit: int = 10) -> list[dict]:
        """Return up to *limit* VLM queue jobs with ``status = 'pending'``,
        ordered by priority descending (highest priority first).
        """
        cur = self._conn.execute(
            "SELECT * FROM vlm_queue WHERE status = 'pending' "
            "ORDER BY priority DESC LIMIT ?",
            (limit,),
        )
        return self._rows_to_dicts(cur.fetchall())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
        logger.info("WorkerDB connection closed")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()
