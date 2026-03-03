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

    def get_focus_session_events(self, session_id: str) -> list[dict]:
        """Return all events tagged with the given focus session ID.

        Events are identified by a ``focus_session_id`` key in their
        ``metadata_json`` column.  Ordered by timestamp ascending.
        """
        cur = self._conn.execute(
            "SELECT * FROM events "
            "WHERE json_extract(metadata_json, '$.focus_session_id') = ? "
            "ORDER BY timestamp ASC",
            (session_id,),
        )
        return self._rows_to_dicts(cur.fetchall())

    # ------------------------------------------------------------------
    # Scene Annotations (v2 pipeline)
    # ------------------------------------------------------------------

    def get_unannotated_events(
        self,
        limit: int = 20,
        *,
        focus_first: bool = True,
    ) -> list[dict]:
        """Return events pending scene annotation, ordered for processing.

        When *focus_first* is True, events belonging to a focus session
        (identified by a non-null ``focus_session_id`` in metadata_json)
        are sorted before non-focus events so the annotation loop
        prioritises explicit recording sessions.
        """
        order = (
            "ORDER BY "
            "(CASE WHEN json_extract(metadata_json, '$.focus_session_id') IS NOT NULL "
            "THEN 0 ELSE 1 END) ASC, "
            "timestamp ASC"
        ) if focus_first else "ORDER BY timestamp ASC"

        cur = self._conn.execute(
            f"SELECT * FROM events WHERE annotation_status = 'pending' "
            f"{order} LIMIT ?",
            (limit,),
        )
        return self._rows_to_dicts(cur.fetchall())

    def get_recent_annotations(
        self,
        before_timestamp: str,
        limit: int = 3,
        max_age_seconds: int = 600,
    ) -> list[dict]:
        """Return the most recent completed annotations before a timestamp.

        Used to build the sliding-window context for the annotation prompt.
        Only returns annotations within *max_age_seconds* of *before_timestamp*
        to avoid stale context bleeding across sessions.
        """
        cur = self._conn.execute(
            "SELECT id, timestamp, scene_annotation_json, "
            "  json_extract(window_json, '$.app_bundle_id') AS app_bundle, "
            "  json_extract(window_json, '$.title') AS window_title, "
            "  json_extract(metadata_json, '$.focus_session_id') AS focus_session_id "
            "FROM events "
            "WHERE annotation_status = 'completed' "
            "  AND scene_annotation_json IS NOT NULL "
            "  AND timestamp < ? "
            "  AND timestamp >= datetime(?, '-' || ? || ' seconds') "
            "ORDER BY timestamp DESC LIMIT ?",
            (before_timestamp, before_timestamp, str(max_age_seconds), limit),
        )
        return self._rows_to_dicts(cur.fetchall())

    def save_annotation(
        self,
        event_id: str,
        annotation_json: str,
        status: str = "completed",
    ) -> bool:
        """Write the scene annotation result for an event.

        *status* should be one of: completed, failed, skipped,
        missing_screenshot.
        """
        db_path = self._get_writable_path()
        if not db_path:
            logger.error("Cannot determine DB path for annotation save")
            return False

        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            write_conn.execute(
                "UPDATE events SET scene_annotation_json = ?, "
                "annotation_status = ? WHERE id = ?",
                (annotation_json, status, event_id),
            )
            write_conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.error("Failed to save annotation for %s: %s", event_id, exc)
            return False
        finally:
            write_conn.close()

    def save_frame_diff(self, event_id: str, diff_json: str) -> bool:
        """Write the frame diff result for an event."""
        db_path = self._get_writable_path()
        if not db_path:
            logger.error("Cannot determine DB path for frame diff save")
            return False

        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            write_conn.execute(
                "UPDATE events SET frame_diff_json = ? WHERE id = ?",
                (diff_json, event_id),
            )
            write_conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.error("Failed to save frame diff for %s: %s", event_id, exc)
            return False
        finally:
            write_conn.close()

    def get_events_needing_diff(self, limit: int = 20) -> list[dict]:
        """Return annotated events that don't yet have a frame diff.

        Returns pairs of consecutive completed annotations where the later
        event has no frame_diff_json yet.
        """
        cur = self._conn.execute(
            "SELECT * FROM events "
            "WHERE annotation_status = 'completed' "
            "  AND scene_annotation_json IS NOT NULL "
            "  AND frame_diff_json IS NULL "
            "ORDER BY timestamp ASC LIMIT ?",
            (limit,),
        )
        return self._rows_to_dicts(cur.fetchall())

    def get_annotation_before(self, timestamp: str) -> dict | None:
        """Return the most recent completed annotation before *timestamp*."""
        cur = self._conn.execute(
            "SELECT * FROM events "
            "WHERE annotation_status = 'completed' "
            "  AND scene_annotation_json IS NOT NULL "
            "  AND timestamp < ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (timestamp,),
        )
        row = cur.fetchone()
        return dict(row) if row is not None else None

    def get_focus_session_annotated_events(
        self,
        session_id: str,
    ) -> list[dict]:
        """Return all annotated events for a focus session, ordered by timestamp.

        Only returns events with ``annotation_status = 'completed'`` and
        non-null ``scene_annotation_json``.  Used by the v2 focus processor
        to collect the timeline for SOP generation.
        """
        cur = self._conn.execute(
            "SELECT * FROM events "
            "WHERE json_extract(metadata_json, '$.focus_session_id') = ? "
            "  AND annotation_status = 'completed' "
            "  AND scene_annotation_json IS NOT NULL "
            "ORDER BY timestamp ASC",
            (session_id,),
        )
        return self._rows_to_dicts(cur.fetchall())

    def count_focus_unannotated(self, session_id: str) -> int:
        """Count focus session events that are still pending annotation."""
        cur = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM events "
            "WHERE json_extract(metadata_json, '$.focus_session_id') = ? "
            "  AND annotation_status = 'pending'",
            (session_id,),
        )
        row = cur.fetchone()
        return row["cnt"] if row else 0

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

    def mark_events_unprocessed(self, event_ids: list[str]) -> int:
        """Reset events to unprocessed so the pipeline re-evaluates them.

        Used after VLM job completion to re-score translations with the
        VLM confidence boost applied.
        """
        if not event_ids:
            return 0

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
                f"UPDATE events SET processed = 0 WHERE id IN ({placeholders})",
                event_ids,
            )
            write_conn.commit()
            updated = cursor.rowcount
            logger.info("Reset %d events to unprocessed (VLM reconciliation)", updated)
            return updated
        except sqlite3.Error as exc:
            logger.error("Failed to reset events to unprocessed: %s", exc)
            return 0
        finally:
            write_conn.close()

    def get_completed_vlm_boost(self, event_id: str) -> float:
        """Return the VLM confidence boost for a completed VLM job on this event.

        Returns 0.0 if no completed VLM job exists for the event.
        """
        try:
            cursor = self._conn.execute(
                "SELECT result_json FROM vlm_queue "
                "WHERE event_id = ? AND status = 'completed' "
                "LIMIT 1",
                (event_id,),
            )
            row = cursor.fetchone()
            if row and row[0]:
                import json as _json
                result = _json.loads(row[0])
                return float(result.get("confidence_boost", 0.0))
        except Exception:
            pass
        return 0.0

    def has_completed_vlm_job(self, event_id: str) -> bool:
        """Check if there is a completed VLM job for this event."""
        cursor = self._conn.execute(
            "SELECT 1 FROM vlm_queue "
            "WHERE event_id = ? AND status = 'completed' "
            "LIMIT 1",
            (event_id,),
        )
        return cursor.fetchone() is not None

    # ------------------------------------------------------------------
    # Episode store — persist translated episodes for cross-cycle SOP mining
    # ------------------------------------------------------------------

    def _ensure_episode_store_table(self, write_conn: sqlite3.Connection) -> None:
        """Create the translated_episodes table if it doesn't exist."""
        write_conn.execute(
            "CREATE TABLE IF NOT EXISTS translated_episodes ("
            "  episode_id TEXT PRIMARY KEY,"
            "  thread_id TEXT NOT NULL,"
            "  steps_json TEXT NOT NULL,"
            "  step_count INTEGER NOT NULL,"
            "  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
            ")"
        )

    def save_episode_steps(
        self,
        episode_id: str,
        thread_id: str,
        steps: list[dict],
    ) -> bool:
        """Persist translated episode steps for future SOP induction.

        Uses INSERT OR REPLACE so re-processed episodes (e.g. after VLM
        reconciliation) update rather than duplicate.
        """
        if not steps:
            return True

        db_path = self._get_writable_path()
        if not db_path:
            logger.error("Cannot determine DB path for episode store")
            return False

        import json as _json

        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            self._ensure_episode_store_table(write_conn)
            write_conn.execute(
                "INSERT OR REPLACE INTO translated_episodes "
                "(episode_id, thread_id, steps_json, step_count) "
                "VALUES (?, ?, ?, ?)",
                (episode_id, thread_id, _json.dumps(steps), len(steps)),
            )
            write_conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.error("Failed to save episode steps: %s", exc)
            return False
        finally:
            write_conn.close()

    def get_all_episode_steps(self, max_age_days: int = 14) -> list[list[dict]]:
        """Load all stored episode step sequences for SOP induction.

        Returns a list of episode step lists, each being a list of step
        dicts suitable for ``SOPInducer.induce()``.  Only episodes within
        *max_age_days* are returned.
        """
        import json as _json

        # Ensure table exists before querying (read-only conn can't create)
        # We need to use a writable connection to ensure the table exists.
        db_path = self._get_writable_path()
        if db_path:
            init_conn = sqlite3.connect(db_path)
            try:
                init_conn.execute("PRAGMA busy_timeout = 5000;")
                self._ensure_episode_store_table(init_conn)
                init_conn.commit()
            except sqlite3.Error:
                pass
            finally:
                init_conn.close()

        try:
            cursor = self._conn.execute(
                "SELECT steps_json FROM translated_episodes "
                "WHERE created_at >= datetime('now', ?)",
                (f"-{max_age_days} days",),
            )
            result: list[list[dict]] = []
            for row in cursor.fetchall():
                try:
                    steps = _json.loads(row[0])
                    if steps:
                        result.append(steps)
                except (ValueError, TypeError):
                    continue
            return result
        except sqlite3.OperationalError:
            # Table doesn't exist yet (first run)
            return []

    def cleanup_old_episodes(self, max_age_days: int = 14) -> int:
        """Remove episodes older than *max_age_days*.

        Returns the number of rows deleted.
        """
        db_path = self._get_writable_path()
        if not db_path:
            return 0

        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            self._ensure_episode_store_table(write_conn)
            cursor = write_conn.execute(
                "DELETE FROM translated_episodes "
                "WHERE created_at < datetime('now', ?)",
                (f"-{max_age_days} days",),
            )
            write_conn.commit()
            deleted = cursor.rowcount
            if deleted:
                logger.info("Cleaned up %d old episode(s) from store", deleted)
            return deleted
        except sqlite3.Error as exc:
            logger.error("Failed to cleanup old episodes: %s", exc)
            return 0
        finally:
            write_conn.close()

    def count_stored_episodes(self) -> int:
        """Return the number of episodes in the store."""
        db_path = self._get_writable_path()
        if db_path:
            init_conn = sqlite3.connect(db_path)
            try:
                init_conn.execute("PRAGMA busy_timeout = 5000;")
                self._ensure_episode_store_table(init_conn)
                init_conn.commit()
            except sqlite3.Error:
                pass
            finally:
                init_conn.close()

        try:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM translated_episodes"
            )
            row = cursor.fetchone()
            return row[0] if row else 0
        except sqlite3.OperationalError:
            return 0

    # ------------------------------------------------------------------
    # Passive discovery (v2 task segmentation)
    # ------------------------------------------------------------------

    def get_annotated_events_in_window(
        self,
        *,
        hours: int = 4,
        min_timestamp: str | None = None,
    ) -> list[dict]:
        """Return all completed annotations within a time window.

        When *min_timestamp* is provided, uses it as the window start.
        Otherwise, returns events from the last *hours* hours.

        Only returns events with completed annotations and non-null
        scene_annotation_json.  Used by the task segmenter.
        """
        if min_timestamp:
            cur = self._conn.execute(
                "SELECT * FROM events "
                "WHERE annotation_status = 'completed' "
                "  AND scene_annotation_json IS NOT NULL "
                "  AND timestamp >= ? "
                "ORDER BY timestamp ASC",
                (min_timestamp,),
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM events "
                "WHERE annotation_status = 'completed' "
                "  AND scene_annotation_json IS NOT NULL "
                "  AND timestamp >= datetime('now', ?) "
                "ORDER BY timestamp ASC",
                (f"-{hours} hours",),
            )
        return self._rows_to_dicts(cur.fetchall())

    def get_workflow_annotated_events(
        self,
        *,
        hours: int = 4,
    ) -> list[dict]:
        """Return annotated events with is_workflow=true within time window.

        Filters at the SQL level for efficiency — only returns events
        where the annotation's task_context.is_workflow is true.
        """
        cur = self._conn.execute(
            "SELECT * FROM events "
            "WHERE annotation_status = 'completed' "
            "  AND scene_annotation_json IS NOT NULL "
            "  AND timestamp >= datetime('now', ?) "
            "  AND json_extract(scene_annotation_json, "
            "    '$.task_context.is_workflow') = 1 "
            "ORDER BY timestamp ASC",
            (f"-{hours} hours",),
        )
        return self._rows_to_dicts(cur.fetchall())

    def _ensure_task_segments_table(
        self, write_conn: "sqlite3.Connection",
    ) -> None:
        """Create the task_segments table if it doesn't exist."""
        write_conn.execute(
            "CREATE TABLE IF NOT EXISTS task_segments ("
            "  segment_id TEXT PRIMARY KEY,"
            "  cluster_id INTEGER NOT NULL,"
            "  task_label TEXT NOT NULL DEFAULT '',"
            "  event_ids_json TEXT NOT NULL,"
            "  frame_count INTEGER NOT NULL,"
            "  apps_json TEXT NOT NULL DEFAULT '[]',"
            "  start_time TEXT,"
            "  end_time TEXT,"
            "  sop_generated INTEGER NOT NULL DEFAULT 0,"
            "  created_at TEXT NOT NULL DEFAULT "
            "    (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
            ")"
        )

    def save_task_segment(
        self,
        segment_id: str,
        cluster_id: int,
        task_label: str,
        event_ids: list[str],
        apps: list[str],
        start_time: str,
        end_time: str,
    ) -> bool:
        """Persist a task segment from the segmenter.

        Uses INSERT OR REPLACE so re-segmentation updates existing entries.
        """
        import json as _json

        db_path = self._get_writable_path()
        if not db_path:
            return False

        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            self._ensure_task_segments_table(write_conn)
            write_conn.execute(
                "INSERT OR REPLACE INTO task_segments "
                "(segment_id, cluster_id, task_label, event_ids_json, "
                " frame_count, apps_json, start_time, end_time) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    segment_id,
                    cluster_id,
                    task_label,
                    _json.dumps(event_ids),
                    len(event_ids),
                    _json.dumps(apps),
                    start_time,
                    end_time,
                ),
            )
            write_conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.error("Failed to save task segment %s: %s", segment_id, exc)
            return False
        finally:
            write_conn.close()

    def get_cluster_segments(
        self, cluster_id: int,
    ) -> list[dict]:
        """Return all segments for a given cluster, ordered by start time."""
        import json as _json

        db_path = self._get_writable_path()
        if db_path:
            init_conn = sqlite3.connect(db_path)
            try:
                init_conn.execute("PRAGMA busy_timeout = 5000;")
                self._ensure_task_segments_table(init_conn)
                init_conn.commit()
            except sqlite3.Error:
                pass
            finally:
                init_conn.close()

        try:
            cur = self._conn.execute(
                "SELECT * FROM task_segments "
                "WHERE cluster_id = ? AND sop_generated = 0 "
                "ORDER BY start_time ASC",
                (cluster_id,),
            )
            return self._rows_to_dicts(cur.fetchall())
        except sqlite3.OperationalError:
            return []

    def mark_segment_sop_generated(self, segment_id: str) -> bool:
        """Mark a segment as having had its SOP generated."""
        db_path = self._get_writable_path()
        if not db_path:
            return False

        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            write_conn.execute(
                "UPDATE task_segments SET sop_generated = 1 "
                "WHERE segment_id = ?",
                (segment_id,),
            )
            write_conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.error("Failed to mark segment SOP generated: %s", exc)
            return False
        finally:
            write_conn.close()

    def get_sop_pending_clusters(self) -> list[dict]:
        """Return cluster IDs that have >= 2 segments without SOPs generated.

        Used by the passive discovery pipeline to find clusters ready for
        SOP generation.
        """
        db_path = self._get_writable_path()
        if db_path:
            init_conn = sqlite3.connect(db_path)
            try:
                init_conn.execute("PRAGMA busy_timeout = 5000;")
                self._ensure_task_segments_table(init_conn)
                init_conn.commit()
            except sqlite3.Error:
                pass
            finally:
                init_conn.close()

        try:
            cur = self._conn.execute(
                "SELECT cluster_id, task_label, COUNT(*) AS seg_count "
                "FROM task_segments "
                "WHERE sop_generated = 0 "
                "GROUP BY cluster_id "
                "HAVING seg_count >= 2 "
                "ORDER BY seg_count DESC",
            )
            return self._rows_to_dicts(cur.fetchall())
        except sqlite3.OperationalError:
            return []

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
