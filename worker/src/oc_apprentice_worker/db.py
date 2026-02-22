"""SQLite interface to the daemon's event database.

The primary connection is opened read-only (``mode=ro``) so queries never
block the daemon's writes.  A small number of write operations (e.g.
marking events as processed) open a **separate, short-lived** writable
connection to minimise lock contention.  See ``mark_events_processed``
for details.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Self

logger = logging.getLogger(__name__)


class WorkerDB:
    """Hybrid read/write connection to the daemon's SQLite database.

    The primary connection is opened via the SQLite URI interface with
    ``mode=ro`` so routine queries never contend with daemon writes.
    WAL journal mode enables concurrent reads.

    Write operations (``mark_events_processed``) open a **separate**
    short-lived writable connection that is closed immediately after
    each call to minimise lock contention.

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

    def count_pending_vlm_jobs(self) -> int:
        """Return the count of pending VLM jobs in the database.

        This is authoritative (unlike in-memory queue stats) and should
        be used for status reporting after DB-side processing.
        """
        cur = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM vlm_queue WHERE status = 'pending'"
        )
        row = cur.fetchone()
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Write operations (via separate writable connection)
    # ------------------------------------------------------------------

    def _get_writable_path(self) -> str | None:
        """Extract the file path from the read-only connection."""
        resolved = self._conn.execute("PRAGMA database_list").fetchone()
        return resolved["file"] if resolved else None

    def enqueue_vlm_job(
        self,
        job_id: str,
        event_id: str,
        priority: float,
        ttl_expires_at: str,
    ) -> bool:
        """Insert a VLM job into the persistent queue.

        Returns True on success, False on failure.
        """
        db_path = self._get_writable_path()
        if not db_path:
            logger.error("Cannot determine DB path for VLM enqueue")
            return False

        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            write_conn.execute(
                "INSERT OR IGNORE INTO vlm_queue (id, event_id, priority, status, ttl_expires_at) "
                "VALUES (?, ?, ?, 'pending', ?)",
                (job_id, event_id, priority, ttl_expires_at),
            )
            write_conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.error("Failed to enqueue VLM job: %s", exc)
            return False
        finally:
            write_conn.close()

    def mark_vlm_job_completed(
        self,
        job_id: str,
        result_json: str | None = None,
    ) -> bool:
        """Mark a VLM job as completed (or failed)."""
        db_path = self._get_writable_path()
        if not db_path:
            return False

        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            write_conn.execute(
                "UPDATE vlm_queue SET status = 'completed', "
                "processed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), "
                "result_json = ? WHERE id = ?",
                (result_json, job_id),
            )
            write_conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.error("Failed to mark VLM job completed: %s", exc)
            return False
        finally:
            write_conn.close()

    def mark_vlm_job_failed(self, job_id: str) -> bool:
        """Mark a VLM job as failed."""
        db_path = self._get_writable_path()
        if not db_path:
            return False

        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            write_conn.execute(
                "UPDATE vlm_queue SET status = 'failed', "
                "processed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                "WHERE id = ?",
                (job_id,),
            )
            write_conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.error("Failed to mark VLM job failed: %s", exc)
            return False
        finally:
            write_conn.close()

    def mark_events_processed(self, event_ids: list[str]) -> int:
        """Mark the given events as processed (``processed = 1``).

        Opens a **separate** writable connection for this operation since
        the main connection is read-only.  The writable connection is
        opened and closed within this call to minimise lock contention
        with the daemon.

        Returns the number of rows updated.
        """
        if not event_ids:
            return 0

        # Validate event IDs before building the query
        if not all(isinstance(eid, str) and len(eid) <= 36 for eid in event_ids):
            raise ValueError("Invalid event IDs")

        db_path = self._get_writable_path()
        if not db_path:
            logger.error("Cannot determine database path for writable connection")
            return 0

        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            placeholders = ",".join("?" for _ in event_ids)
            write_conn.execute("BEGIN IMMEDIATE")
            cursor = write_conn.execute(
                f"UPDATE events SET processed = 1 WHERE id IN ({placeholders})",
                event_ids,
            )
            write_conn.commit()
            updated = cursor.rowcount
            logger.info("Marked %d events as processed", updated)
            return updated
        except sqlite3.Error as exc:
            logger.error("Failed to mark events as processed: %s", exc)
            return 0
        finally:
            write_conn.close()

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
