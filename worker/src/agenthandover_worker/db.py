"""SQLite interface to the daemon's event database.

The primary connection is opened read-only (``mode=ro``) so queries never
block the daemon's writes.  A small number of write operations (e.g.
marking events as processed) open a **separate, short-lived** writable
connection to minimise lock contention.  See ``mark_events_processed``
for details.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Self

from agenthandover_worker.event_helpers import parse_json_field
from agenthandover_worker.redaction import redact

logger = logging.getLogger(__name__)


# The daemon's schema this worker reads and writes: ``CURRENT_SCHEMA_VERSION``
# in ``crates/storage/src/schema.rs``.  The daemon creates and migrates
# every table (``crates/storage/src/migrations``); the worker never does.
SCHEMA_VERSION = 9


class SchemaVersionError(RuntimeError):
    """The database's ``user_version`` is not :data:`SCHEMA_VERSION`."""


def database_schema_version(db_path: str | Path) -> int:
    """The ``user_version`` the daemon recorded in *db_path*."""
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])
    finally:
        conn.close()


def check_schema_version(db_path: str | Path) -> None:
    """Raise :class:`SchemaVersionError`, saying what to do, unless the
    daemon has brought *db_path* to exactly :data:`SCHEMA_VERSION`."""
    found = database_schema_version(db_path)
    if found == SCHEMA_VERSION:
        return
    if found < SCHEMA_VERSION:
        raise SchemaVersionError(
            f"The database {db_path} is at schema version {found}; this worker "
            f"needs version {SCHEMA_VERSION}, which the daemon creates when it "
            "starts and `agenthandover db migrate` creates while capture is "
            "paused (the worker runs it; run it in Terminal to see why it "
            "failed).  Either the daemon has not started since the upgrade "
            "(AgentHandover.app starts it once it has Accessibility and "
            "Screen Recording: open the app and grant them; `agenthandover "
            "doctor` shows which is missing), or the daemon running is older "
            "than the worker (run `agenthandover restart`; if the error "
            "stays, reinstall the latest AgentHandover release so the daemon "
            "and the worker match)."
        )
    raise SchemaVersionError(
        f"The database {db_path} is at schema version {found}, newer than the "
        f"version {SCHEMA_VERSION} this worker understands: a newer daemon "
        "migrated it.  Fix: reinstall the latest AgentHandover release so the "
        "worker matches the daemon (downgrading the daemon alone does not "
        "undo the migration)."
    )


def wait_for_schema_version(
    db_path: str | Path,
    *,
    timeout_s: float,
    poll_s: float,
    stop: Callable[[], bool] = lambda: False,
    migrate: Callable[[], None] | None = None,
) -> None:
    """Return once *db_path* is at :data:`SCHEMA_VERSION`.

    Right after an upgrade the database can be older: *migrate* (the
    daemon's migrations, run by ``agenthandover db migrate``) is called
    once, since the daemon migrates only when it starts and it starts only
    while capture runs (a paused Mac kept a 0.4.0 database, and the worker
    waiting, for good).  Then an older version is polled for up to
    *timeout_s* (or until *stop* returns True, which returns early); a
    newer one fails at once.  Raises :class:`SchemaVersionError`.
    """
    waited = 0.0
    migrated = migrate is None
    while True:
        try:
            check_schema_version(db_path)
            return
        except SchemaVersionError:
            if database_schema_version(db_path) > SCHEMA_VERSION:
                raise
            if not migrated:
                migrated = True
                migrate()
                continue
            if waited >= timeout_s:
                raise
        if stop():
            return
        logger.info(
            "Waiting for the daemon to migrate %s to schema %d (%.0fs/%.0fs)",
            db_path, SCHEMA_VERSION, waited, timeout_s,
        )
        time.sleep(poll_s)
        waited += poll_s


def _sqlite_utc(moment: datetime) -> str:
    """*moment* in the text SQLite's ``strftime('%Y-%m-%dT%H:%M:%fZ')``
    writes, so it compares with stored timestamps the way ``'now'`` does."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _local_day_bounds(date_str: str) -> tuple[str, str]:
    """``(start, end)`` of the local calendar day *date_str* (YYYY-MM-DD)
    as UTC text that compares with stored timestamps: rows of the day are
    ``start <= timestamp < end``.

    Local midnight is resolved with the zone rules in force on that day
    (a DST change makes the day 23 or 25 hours long).  The bounds carry no
    fraction or offset: every stored timestamp is UTC (``Z`` or
    ``+00:00``), and a bare ``YYYY-MM-DDTHH:MM:SS`` sorts before any
    timestamp of that second and after every earlier one.
    """
    day = datetime.strptime(date_str, "%Y-%m-%d")
    start = day.astimezone(timezone.utc)
    end = (day + timedelta(days=1)).astimezone(timezone.utc)
    fmt = "%Y-%m-%dT%H:%M:%S"
    return start.strftime(fmt), end.strftime(fmt)


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
        # Used by the control and the inference thread, one at a time
        # (inference_runner.STATE_LOCK); SQLite itself is serialized.
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
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

    def _refresh_read_snapshot(self) -> None:
        """End any implicit read transaction so the next SELECT sees fresh data.

        In WAL mode, a read-only connection's implicit transaction starts
        on the first SELECT and holds a snapshot until committed/rolled back.
        Calling ``rollback()`` on a read-only connection is safe (no-op on
        data) but resets the snapshot so subsequent reads see writes committed
        by other connections since the last snapshot.
        """
        try:
            self._conn.rollback()
        except sqlite3.Error:
            pass

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

    def get_events_by_ids(self, event_ids: list[str]) -> list[dict]:
        """Return events matching the given IDs, ordered by timestamp.

        Uses batched ``IN (...)`` queries for efficiency.
        """
        if not event_ids:
            return []

        # SQLite has a variable limit (~999); batch if needed.
        results: list[dict] = []
        batch_size = 900
        for i in range(0, len(event_ids), batch_size):
            batch = event_ids[i : i + batch_size]
            placeholders = ",".join("?" for _ in batch)
            cur = self._conn.execute(
                f"SELECT * FROM events WHERE id IN ({placeholders}) "
                "ORDER BY timestamp ASC",
                batch,
            )
            results.extend(self._rows_to_dicts(cur.fetchall()))
        return results

    def memory_thumb_ids(self, event_ids: list[str]) -> set[str]:
        """The subset of *event_ids* that a Memory row shows as its
        thumbnail (``memory_frames.thumb_id``)."""
        found: set[str] = set()
        for i in range(0, len(event_ids), 900):
            batch = event_ids[i : i + 900]
            placeholders = ",".join("?" for _ in batch)
            cur = self._conn.execute(
                f"SELECT thumb_id FROM memory_frames WHERE thumb_id IN ({placeholders})",
                batch,
            )
            found.update(row[0] for row in cur.fetchall())
        return found

    def get_focus_session_events(self, session_id: str) -> list[dict]:
        """Return all events tagged with the given focus session ID,
        PLUS any ClipboardChange and PasteDetected events that occurred
        during the same time window (clipboard events aren't tagged with
        session_id: the pasteboard monitor and the browser extension emit
        them outside the session's capture).
        """
        # Get tagged focus events
        cur = self._conn.execute(
            "SELECT * FROM events "
            "WHERE json_extract(metadata_json, '$.focus_session_id') = ? "
            "ORDER BY timestamp ASC",
            (session_id,),
        )
        tagged = self._rows_to_dicts(cur.fetchall())
        if not tagged:
            return tagged

        # Find the time window of the focus session
        first_ts = tagged[0].get("timestamp", "")
        last_ts = tagged[-1].get("timestamp", "")
        if not first_ts or not last_ts:
            return tagged

        # Fetch clipboard events in that window (they have empty metadata)
        cur2 = self._conn.execute(
            "SELECT * FROM events "
            "WHERE json_extract(kind_json, '$.type') IN ('ClipboardChange', 'PasteDetected') "
            "  AND timestamp >= ? AND timestamp <= ? "
            "  AND json_extract(metadata_json, '$.focus_session_id') IS NULL "
            "ORDER BY timestamp ASC",
            (first_ts, last_ts),
        )
        clipboard_events = self._rows_to_dicts(cur2.fetchall())

        if not clipboard_events:
            return tagged

        # Merge and sort by timestamp
        all_events = tagged + clipboard_events
        all_events.sort(key=lambda e: e.get("timestamp", ""))
        return all_events

    # ------------------------------------------------------------------
    # Scene Annotations (v2 pipeline)
    # ------------------------------------------------------------------

    def get_unannotated_events(
        self,
        limit: int = 20,
        *,
        focus_first: bool = True,
        exclude_focus: bool = False,
    ) -> list[dict]:
        """Return events pending scene annotation, ordered for processing.

        When *focus_first* is True, events belonging to a focus session
        (identified by a non-null ``focus_session_id`` in metadata_json)
        are sorted before non-focus events so the annotation loop
        prioritises explicit recording sessions.

        When *exclude_focus* is True, focus-session events are left out
        entirely.  The passive annotation loop uses this: focus rows belong
        to the focus processor, and a batch filled with them would starve
        passive frames for as long as those rows stay pending.
        """
        focus_expr = "json_extract(metadata_json, '$.focus_session_id') IS NOT NULL"
        where = "annotation_status = 'pending'"
        if exclude_focus:
            where += f" AND NOT ({focus_expr})"
            order = "ORDER BY timestamp ASC"
        elif focus_first:
            order = (
                f"ORDER BY (CASE WHEN {focus_expr} THEN 0 ELSE 1 END) ASC, "
                "timestamp ASC"
            )
        else:
            order = "ORDER BY timestamp ASC"

        cur = self._conn.execute(
            f"SELECT * FROM events WHERE {where} {order} LIMIT ?",
            (limit,),
        )
        return self._rows_to_dicts(cur.fetchall())

    def expire_annotation_backlog(
        self, max_age_hours: float = 24.0, *, now: datetime | None = None,
    ) -> list[dict]:
        """Give up on passive frames that waited too long for annotation.

        Pending rows outside any focus session and older than
        *max_age_hours* before *now* (aware; default the current time) are
        marked ``skipped`` (the passive loop takes the oldest first, so an
        old backlog would otherwise hold back today's frames).  Returns the
        expired rows so their screenshots can be deleted.
        """
        now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        where = (
            "annotation_status = 'pending' "
            "AND json_extract(metadata_json, '$.focus_session_id') IS NULL "
            "AND julianday(timestamp) < julianday(?) - ? / 24.0"
        )
        rows = self._rows_to_dicts(self._conn.execute(
            f"SELECT id, metadata_json, artifact_ids_json FROM events WHERE {where}",
            (now_utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], max_age_hours),
        ).fetchall())
        if not rows:
            return []
        db_path = self._get_writable_path()
        if not db_path:
            return []
        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            write_conn.executemany(
                "UPDATE events SET annotation_status = 'skipped' "
                "WHERE id = ? AND annotation_status = 'pending'",
                [(r["id"],) for r in rows],
            )
            write_conn.commit()
        except sqlite3.Error as exc:
            logger.error("Failed to expire the annotation backlog: %s", exc)
            return []
        finally:
            write_conn.close()
        logger.info(
            "Expired %d passive frame(s) older than %.0fh without annotation",
            len(rows), max_age_hours,
        )
        return rows

    # ------------------------------------------------------------------
    # Worker state (small persisted values: dates, counters)
    # ------------------------------------------------------------------

    def get_state(self, key: str) -> str | None:
        """A value the worker persisted with :meth:`set_state`, or None."""
        try:
            row = self._conn.execute(
                "SELECT value FROM worker_state WHERE key = ?", (key,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        return row[0] if row else None

    def set_state(self, key: str, value: str) -> bool:
        db_path = self._get_writable_path()
        if not db_path:
            return False
        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            write_conn.execute(
                "INSERT OR REPLACE INTO worker_state (key, value) VALUES (?, ?)",
                (key, value),
            )
            write_conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.error("Failed to persist worker state %s: %s", key, exc)
            return False
        finally:
            write_conn.close()

    def pending_screenshot_paths(self) -> list[str]:
        """``metadata.screenshot_path`` of every event still pending
        annotation (passive and focus): the JPEGs that are still needed."""
        cur = self._conn.execute(
            "SELECT json_extract(metadata_json, '$.screenshot_path') FROM events "
            "WHERE annotation_status = 'pending' "
            "AND json_extract(metadata_json, '$.screenshot_path') IS NOT NULL"
        )
        return [row[0] for row in cur.fetchall() if isinstance(row[0], str) and row[0]]

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
            "  AND timestamp >= strftime('%Y-%m-%dT%H:%M:%fZ', ?, '-' || ? || ' seconds') "
            "ORDER BY timestamp DESC LIMIT ?",
            (before_timestamp, before_timestamp, str(max_age_seconds), limit),
        )
        return self._rows_to_dicts(cur.fetchall())

    def key_evidence(self, event: dict) -> dict | None:
        """The keyboard and clipboard evidence for *event*'s frame: what
        happened since the previous annotated frame of its session.

        ``{"keys": int, "pastes": int, "previous": row | None}``: the key
        presses the daemon counted (``keys_since_prev_frame`` of every
        event from just after the previous annotated frame up to and
        including *event*, skipped frames included), the browser's
        ``PasteDetected`` events in that stretch, and the previous
        annotated frame's row.  Without a previous annotated frame the
        stretch starts at the session's first event.  None when *event*
        carries no count (rows from before the daemon recorded it).
        """
        metadata = parse_json_field(event, "metadata_json")
        count = metadata.get("keys_since_prev_frame")
        if not isinstance(count, int) or isinstance(count, bool):
            return None
        timestamp = str(event.get("timestamp") or "")
        session = metadata.get("focus_session_id")
        session_expr = "json_extract(metadata_json, '$.focus_session_id') IS ?"
        row = self._conn.execute(
            "SELECT * FROM events "
            "WHERE annotation_status = 'completed' AND scene_annotation_json IS NOT NULL "
            f"  AND timestamp < ? AND {session_expr} AND id != ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (timestamp, session, event.get("id")),
        ).fetchone()
        previous = dict(row) if row is not None else None
        if previous is not None:
            start_sql, start = "timestamp > ?", previous["timestamp"]
        else:
            first = self._conn.execute(
                f"SELECT MIN(timestamp) FROM events WHERE {session_expr}", (session,),
            ).fetchone()[0]
            start_sql, start = "timestamp >= ?", first or timestamp
        keys, pastes = self._conn.execute(
            "SELECT "
            "  COALESCE(SUM(CASE WHEN json_type(metadata_json, '$.keys_since_prev_frame') "
            "    = 'integer' THEN json_extract(metadata_json, '$.keys_since_prev_frame') END), 0), "
            "  COALESCE(SUM(json_extract(kind_json, '$.type') = 'PasteDetected'), 0) "
            f"FROM events WHERE {start_sql} AND timestamp <= ?",
            (start, timestamp),
        ).fetchone()
        return {"keys": int(keys), "pastes": int(pastes), "previous": previous}

    def get_copies_before(self, timestamp: str, minutes: int = 30) -> list[dict]:
        """The ``ClipboardChange`` records (``content_hash``, and
        ``content_preview`` when the user turned it on) copied in the
        *minutes* up to *timestamp*, newest first."""
        cur = self._conn.execute(
            "SELECT kind_json FROM events "
            "WHERE json_extract(kind_json, '$.type') = 'ClipboardChange' "
            "  AND timestamp <= ? "
            "  AND timestamp >= strftime('%Y-%m-%dT%H:%M:%fZ', ?, '-' || ? || ' minutes') "
            "ORDER BY timestamp DESC",
            (timestamp, timestamp, str(minutes)),
        )
        return [parse_json_field({"kind_json": row[0]}, "kind_json") for row in cur.fetchall()]

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

    def count_passive_unannotated(self) -> int:
        """Count passive frames (outside any focus session) still pending
        annotation: the backlog the worker publishes to the daemon."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM events "
            "WHERE annotation_status = 'pending' "
            "  AND json_extract(metadata_json, '$.focus_session_id') IS NULL",
        ).fetchone()
        return row["cnt"] if row else 0

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
    # DOM snapshots (Chrome extension)
    # ------------------------------------------------------------------

    def get_dom_snapshots_near_timestamp(
        self,
        timestamp: str,
        url: str,
        tolerance_sec: float = 5.0,
    ) -> list[dict]:
        """Chrome extension DOM snapshots within *tolerance_sec* of *timestamp*
        on the host of *url* (a URL or a bare host, with or without scheme;
        empty matches any host).

        A snapshot row stores its payload in ``metadata_json``: an
        ``anchors`` set and a text ``digest`` (current extension) or a
        ``nodes`` tree (earlier versions); ``nodes`` in the result is
        whichever the row holds, for :func:`dom.flatten_dom`.

        Returns dicts with event_id, timestamp, url, nodes, digest, closest
        first.
        """
        import json as _json

        cur = self._conn.execute(
            "SELECT id, timestamp, metadata_json FROM events "
            "WHERE (json_extract(metadata_json, '$.anchors') IS NOT NULL "
            "       OR json_extract(metadata_json, '$.nodes') IS NOT NULL) "
            "  AND ABS(julianday(timestamp) - julianday(?)) * 86400.0 <= ? "
            "ORDER BY ABS(julianday(timestamp) - julianday(?)) ASC "
            "LIMIT 5",
            (timestamp, float(tolerance_sec), timestamp),
        )
        results = []
        for row in cur.fetchall():
            try:
                meta = _json.loads(row["metadata_json"])
            except (ValueError, TypeError):
                continue

            nodes = meta.get("anchors")
            if not isinstance(nodes, list):
                nodes = meta.get("nodes")
            if not isinstance(nodes, list):
                continue

            snap_url = meta.get("url", "")
            if url and not _url_match(url, snap_url):
                continue

            results.append({
                "event_id": row["id"],
                "timestamp": row["timestamp"],
                "url": snap_url,
                "nodes": nodes,
                "digest": meta.get("digest", ""),
            })

        return results

    def get_click_intents(self, start: str, end: str) -> list[dict]:
        """Chrome extension clicks between *start* and *end* (inclusive), in
        time order: dicts with event_id, timestamp, url and the click's
        ``target`` (tagName, role, ariaLabel, testId, innerText,
        composedPath)."""
        import json as _json

        cur = self._conn.execute(
            "SELECT id, timestamp, metadata_json FROM events "
            "WHERE json_extract(kind_json, '$.type') = 'ClickIntent' "
            "  AND julianday(timestamp) BETWEEN julianday(?) AND julianday(?) "
            "ORDER BY julianday(timestamp) ASC",
            (start, end),
        )
        clicks = []
        for row in cur.fetchall():
            try:
                meta = _json.loads(row["metadata_json"] or "{}")
            except (ValueError, TypeError):
                continue
            target = meta.get("target") if isinstance(meta, dict) else None
            if not isinstance(target, dict):
                continue
            clicks.append({
                "event_id": row["id"],
                "timestamp": row["timestamp"],
                "url": meta.get("url", ""),
                "target": target,
            })
        return clicks

    def get_events_between(self, start: datetime, end: datetime) -> list[dict]:
        """Every event row of any kind from *start* to *end* (inclusive),
        in time order: what happened around one observed run (its lead-in,
        its commit, the checks after it).  Compared as stored text, so the
        timestamp index serves the range."""
        self._refresh_read_snapshot()
        cur = self._conn.execute(
            "SELECT * FROM events WHERE timestamp >= ? AND timestamp <= ? "
            "ORDER BY timestamp ASC",
            (_sqlite_utc(start), _sqlite_utc(end)),
        )
        return self._rows_to_dicts(cur.fetchall())

    def get_shell_commands(self, start: str, end: str) -> list[dict]:
        """Commands the opt-in shell hook recorded between *start* and *end*
        (inclusive), in time order: dicts with event_id, timestamp, cwd and
        command (``ShellCommand`` events, already redacted by the daemon)."""
        cur = self._conn.execute(
            "SELECT id, timestamp, "
            "       json_extract(kind_json, '$.cwd') AS cwd, "
            "       json_extract(kind_json, '$.command') AS command "
            "FROM events "
            "WHERE json_extract(kind_json, '$.type') = 'ShellCommand' "
            "  AND julianday(timestamp) BETWEEN julianday(?) AND julianday(?) "
            "ORDER BY julianday(timestamp) ASC",
            (start, end),
        )
        return [
            {
                "event_id": row["id"],
                "timestamp": row["timestamp"],
                "cwd": row["cwd"] or "",
                "command": row["command"] or "",
            }
            for row in cur.fetchall()
            if row["command"]
        ]

    def get_sleep_boundaries(self, since: str | None = None) -> list[tuple[str, str]]:
        """``(slept_at, woke_at)`` of each ``SystemSleep`` event (optionally
        only those waking at or after *since*), deduplicated, in time order.
        No episode or task segment may span one."""
        sql = (
            "SELECT DISTINCT json_extract(kind_json, '$.slept_at') AS slept_at, "
            "       json_extract(kind_json, '$.woke_at') AS woke_at "
            "FROM events WHERE json_extract(kind_json, '$.type') = 'SystemSleep' "
        )
        params: tuple = ()
        if since:
            sql += "AND julianday(json_extract(kind_json, '$.woke_at')) >= julianday(?) "
            params = (since,)
        sql += "ORDER BY julianday(slept_at) ASC"
        return [
            (row["slept_at"], row["woke_at"])
            for row in self._conn.execute(sql, params).fetchall()
            if row["slept_at"] and row["woke_at"]
        ]

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
    # Write operations (via separate writable connection)
    # ------------------------------------------------------------------

    def _get_writable_path(self) -> str | None:
        """Extract the file path from the read-only connection."""
        resolved = self._conn.execute("PRAGMA database_list").fetchone()
        return resolved["file"] if resolved else None

    # ``events.processed`` values.  Anything non-zero is out of the
    # unprocessed queue; the daemon's retention purges every non-zero row.
    PROCESSED = 1
    UNPROCESSED = 0
    QUARANTINED = -1

    def _set_processed(self, event_ids: list[str], value: int, what: str) -> int:
        """Set ``processed = value`` on the given events.

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
                f"UPDATE events SET processed = ? WHERE id IN ({placeholders})",
                [value, *event_ids],
            )
            write_conn.commit()
            updated = cursor.rowcount
            logger.info("%s %d events", what, updated)
            return updated
        except sqlite3.Error as exc:
            logger.error("Failed to update events (%s): %s", what, exc)
            return 0
        finally:
            write_conn.close()

    def mark_events_processed(self, event_ids: list[str]) -> int:
        """Mark the given events as processed (``processed = 1``)."""
        return self._set_processed(event_ids, self.PROCESSED, "Marked processed:")

    def quarantine_events(self, event_ids: list[str]) -> int:
        """Take events the pipeline keeps failing on out of the queue
        (``processed = -1``) so one poison row cannot stall the worker.
        """
        return self._set_processed(event_ids, self.QUARANTINED, "Quarantined")

    # ------------------------------------------------------------------
    # Passive discovery (v2 task segmentation)
    # ------------------------------------------------------------------

    def get_annotated_events_in_window(
        self,
        *,
        hours: int = 4,
        min_timestamp: str | None = None,
        now: datetime | None = None,
    ) -> list[dict]:
        """Return all completed annotations within a time window.

        When *min_timestamp* is provided, uses it as the window start.
        When *now* (timezone-aware) is provided, the window is the *hours*
        hours up to *now*, and rows stamped after *now* are left out, so a
        replay on a virtual clock sees what the worker would have seen then.
        Otherwise, returns events from the last *hours* hours of the wall
        clock.

        Only returns events with completed annotations and non-null
        scene_annotation_json.  Used by the task segmenter.
        """
        if now is not None and not min_timestamp:
            cur = self._conn.execute(
                "SELECT * FROM events "
                "WHERE annotation_status = 'completed' "
                "  AND scene_annotation_json IS NOT NULL "
                "  AND timestamp >= ? AND timestamp <= ? "
                "ORDER BY timestamp ASC",
                (_sqlite_utc(now - timedelta(hours=hours)), _sqlite_utc(now)),
            )
        elif min_timestamp:
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
                "  AND timestamp >= strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?) "
                "ORDER BY timestamp ASC",
                (f"-{hours} hours",),
            )
        return self._rows_to_dicts(cur.fetchall())

    def get_annotated_events_for_date(self, date: str) -> list[dict]:
        """Return all annotated events of the local day *date* (YYYY-MM-DD).

        Used by the daily batch processor to aggregate a full day's
        annotated activity into a summary.  The day is the user's local
        calendar day, ``[local midnight, next local midnight)``.
        """
        self._refresh_read_snapshot()
        start, end = _local_day_bounds(date)
        cur = self._conn.execute(
            "SELECT * FROM events "
            "WHERE annotation_status = 'completed' "
            "  AND scene_annotation_json IS NOT NULL "
            "  AND timestamp >= ? AND timestamp < ? "
            "ORDER BY timestamp ASC",
            (start, end),
        )
        return self._rows_to_dicts(cur.fetchall())

    def get_events_for_procedure_window(
        self,
        start_iso: str,
        end_iso: str,
        *,
        limit: int = 500,
    ) -> list[dict]:
        """Return annotated events within a time window.

        Used by the evidence extractor to collect raw events that
        overlap with known observations of a procedure before the
        raw annotations expire at 14 days.

        Args:
            start_iso: Start timestamp (ISO 8601).
            end_iso: End timestamp (ISO 8601).
            limit: Maximum number of events to return.

        Returns:
            List of event dicts with annotations, ordered by timestamp.
        """
        self._refresh_read_snapshot()
        cur = self._conn.execute(
            "SELECT * FROM events "
            "WHERE annotation_status = 'completed' "
            "  AND scene_annotation_json IS NOT NULL "
            "  AND timestamp >= ? AND timestamp <= ? "
            "ORDER BY timestamp ASC "
            "LIMIT ?",
            (start_iso, end_iso, limit),
        )
        return self._rows_to_dicts(cur.fetchall())

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
        """Persist (upsert) a task segment from the segmenter."""
        db_path = self._get_writable_path()
        if not db_path:
            return False

        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            write_conn.execute(
                "INSERT INTO task_segments "
                "(segment_id, cluster_id, task_label, event_ids_json, "
                " frame_count, apps_json, start_time, end_time) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(segment_id) DO UPDATE SET "
                " cluster_id = excluded.cluster_id,"
                " task_label = excluded.task_label,"
                " event_ids_json = excluded.event_ids_json,"
                " frame_count = excluded.frame_count,"
                " apps_json = excluded.apps_json,"
                " start_time = excluded.start_time,"
                " end_time = excluded.end_time",
                (
                    segment_id,
                    cluster_id,
                    task_label,
                    json.dumps(event_ids),
                    len(event_ids),
                    json.dumps(apps),
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

    def get_task_segments(self) -> list[dict]:
        """All kept task segments, oldest first."""
        self._refresh_read_snapshot()
        try:
            cur = self._conn.execute(
                "SELECT segment_id, cluster_id, task_label, event_ids_json, "
                "  frame_count, apps_json, start_time, end_time, created_at "
                "FROM task_segments ORDER BY start_time ASC",
            )
            return self._rows_to_dicts(cur.fetchall())
        except sqlite3.OperationalError:
            return []

    def purge_task_segments(
        self, max_age_days: int = 14, *, now: datetime | None = None,
    ) -> int:
        """Delete segments that ended more than *max_age_days* before *now*
        (default: the wall clock).  Returns the number deleted."""
        return self._delete_older_than(
            "DELETE FROM task_segments WHERE datetime(end_time) < datetime(?)",
            (now or datetime.now(timezone.utc)) - timedelta(days=max_age_days),
        )

    def _delete_older_than(self, sql: str, cutoff: datetime) -> int:
        """Run the one-parameter DELETE *sql* with *cutoff* (UTC, in the
        text ``datetime()`` returns)."""
        db_path = self._get_writable_path()
        if not db_path:
            return 0
        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            deleted = write_conn.execute(
                sql, (cutoff.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),),
            ).rowcount
            write_conn.commit()
            return deleted
        except sqlite3.Error as exc:
            logger.error("Purge failed (%s): %s", sql.split(" WHERE")[0], exc)
            return 0
        finally:
            write_conn.close()

    # ------------------------------------------------------------------
    # Workflow observations: one row per work session.  The daemon's raw
    # retention purges events, never this table, so an observation (its
    # step signature, timing and extracted evidence) outlives its frames.
    # ------------------------------------------------------------------

    _OBSERVATION_COLUMNS = (
        "obs_id", "workflow_key", "started_utc", "started_local_date",
        "started_local_weekday", "started_local_minute", "duration_s",
        "apps_json", "hosts_json", "step_sig_json", "evidence_json",
        "source", "task_label", "event_ids_json", "promoted_slug",
        "promoted_at", "user_label", "user_is_routine", "labelled_at",
    )
    # Written by their own setters, never by an observation's upsert.
    _OBSERVATION_KEPT_COLUMNS = (
        "promoted_slug", "promoted_at", "user_label", "user_is_routine", "labelled_at",
    )

    def save_workflow_observation(self, obs: dict) -> bool:
        """Upsert one observation (keyed on ``obs_id``).  A re-recorded
        session updates its fields but keeps its promotion."""
        db_path = self._get_writable_path()
        if not db_path:
            return False
        cols = [c for c in self._OBSERVATION_COLUMNS if c not in self._OBSERVATION_KEPT_COLUMNS]
        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            write_conn.execute(
                f"INSERT INTO workflow_observations ({', '.join(cols)}) "
                f"VALUES ({', '.join('?' for _ in cols)}) "
                "ON CONFLICT(obs_id) DO UPDATE SET "
                + ", ".join(f"{c} = excluded.{c}" for c in cols if c != "obs_id"),
                tuple(obs[c] for c in cols),
            )
            write_conn.commit()
            return True
        except (sqlite3.Error, KeyError) as exc:
            logger.error("Failed to save workflow observation: %s", exc)
            return False
        finally:
            write_conn.close()

    def get_workflow_observations(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        obs_ids: list[str] | None = None,
    ) -> list[dict]:
        """Observations started in ``[since, until)`` (or those in
        *obs_ids*), oldest first."""
        self._refresh_read_snapshot()
        where: list[str] = []
        params: list[str] = []
        if since is not None:
            where.append("started_utc >= ?")
            params.append(_sqlite_utc(since))
        if until is not None:
            where.append("started_utc < ?")
            params.append(_sqlite_utc(until))
        if obs_ids is not None:
            where.append(f"obs_id IN ({', '.join('?' for _ in obs_ids)})")
            params += obs_ids
        sql = f"SELECT {', '.join(self._OBSERVATION_COLUMNS)} FROM workflow_observations"
        if where:
            sql += " WHERE " + " AND ".join(where)
        try:
            cur = self._conn.execute(sql + " ORDER BY started_utc ASC", params)
            return self._rows_to_dicts(cur.fetchall())
        except sqlite3.OperationalError:
            return []

    def set_observations_promoted(
        self, obs_ids: list[str], slug: str, *, promoted_at: datetime | None = None,
    ) -> int:
        """Attribute observations to the Skill *slug*.  With *promoted_at*
        they are the ones that promoted it (counted by the review-queue
        cap); without, later observations of an already promoted workflow."""
        db_path = self._get_writable_path()
        if not db_path or not obs_ids:
            return 0
        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            updated = write_conn.execute(
                "UPDATE workflow_observations SET promoted_slug = ?, "
                "  promoted_at = COALESCE(?, promoted_at) "
                f"WHERE obs_id IN ({', '.join('?' for _ in obs_ids)})",
                (slug, _sqlite_utc(promoted_at) if promoted_at else None, *obs_ids),
            ).rowcount
            write_conn.commit()
            return updated
        except sqlite3.Error as exc:
            logger.error("Failed to mark observations promoted: %s", exc)
            return 0
        finally:
            write_conn.close()

    def label_workflow(
        self,
        workflow_keys: list[str],
        *,
        label: str,
        is_routine: bool | None,
        labelled_at: datetime,
    ) -> int:
        """Record the user's answer to a labelling card on every
        observation of *workflow_keys*: the name they gave the workflow
        (an earlier name is kept when they gave none) and whether it is a
        routine they would hand off (None: not sure).  Returns the number
        of rows updated."""
        db_path = self._get_writable_path()
        if not db_path or not workflow_keys:
            return 0
        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            updated = write_conn.execute(
                "UPDATE workflow_observations SET "
                "  user_label = COALESCE(NULLIF(?, ''), user_label), "
                "  user_is_routine = ?, labelled_at = ? "
                f"WHERE workflow_key IN ({', '.join('?' for _ in workflow_keys)})",
                (
                    label.strip(),
                    None if is_routine is None else int(is_routine),
                    _sqlite_utc(labelled_at),
                    *workflow_keys,
                ),
            ).rowcount
            write_conn.commit()
            return updated
        except sqlite3.Error as exc:
            logger.error("Failed to label workflow observations: %s", exc)
            return 0
        finally:
            write_conn.close()

    # ------------------------------------------------------------------
    # Micro-questions (``questions``): every question asked, so each is
    # asked once and the weekly budget counts what was asked.
    # ------------------------------------------------------------------

    _QUESTION_COLUMNS = (
        "question_id", "kind", "subject", "slot", "channel", "question_json",
        "status", "answer", "asked_at", "answered_at",
    )

    def save_micro_questions(self, rows: list[dict]) -> int:
        """Record newly asked questions (``question_id`` already present:
        left as it is).  Returns how many were new."""
        db_path = self._get_writable_path()
        if not db_path or not rows:
            return 0
        cols = ("question_id", "kind", "subject", "slot", "channel", "question_json", "asked_at")
        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            added = 0
            for row in rows:
                added += write_conn.execute(
                    f"INSERT OR IGNORE INTO micro_questions ({', '.join(cols)}) "
                    f"VALUES ({', '.join('?' for _ in cols)})",
                    tuple(row[c] for c in cols),
                ).rowcount
            write_conn.commit()
            return added
        except (sqlite3.Error, KeyError) as exc:
            logger.error("Failed to save micro-questions: %s", exc)
            return 0
        finally:
            write_conn.close()

    def get_micro_questions(
        self, *, status: str | None = None, channel: str | None = None,
    ) -> list[dict]:
        """Asked questions (optionally only those with *status* and on
        *channel*), oldest first."""
        self._refresh_read_snapshot()
        where, params = [], []
        if status is not None:
            where.append("status = ?")
            params.append(status)
        if channel is not None:
            where.append("channel = ?")
            params.append(channel)
        sql = f"SELECT {', '.join(self._QUESTION_COLUMNS)} FROM micro_questions"
        if where:
            sql += " WHERE " + " AND ".join(where)
        try:
            cur = self._conn.execute(sql + " ORDER BY asked_at ASC, rowid ASC", params)
            return self._rows_to_dicts(cur.fetchall())
        except sqlite3.OperationalError:
            return []

    def count_micro_questions_since(self, since: datetime) -> int:
        """Questions asked at or after *since* (the weekly budget)."""
        self._refresh_read_snapshot()
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM micro_questions WHERE asked_at >= ?",
                (_sqlite_utc(since),),
            ).fetchone()
            return int(row[0]) if row else 0
        except sqlite3.OperationalError:
            return 0

    def close_micro_question(
        self, question_id: str, *, status: str, answer: str, at: datetime,
    ) -> bool:
        """Mark an open question ``answered`` or ``skipped``.  False when it
        was not open (unknown, or already closed)."""
        db_path = self._get_writable_path()
        if not db_path:
            return False
        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            updated = write_conn.execute(
                "UPDATE micro_questions SET status = ?, answer = ?, answered_at = ? "
                "WHERE question_id = ? AND status = 'open'",
                (status, answer, _sqlite_utc(at), question_id),
            ).rowcount
            write_conn.commit()
            return updated == 1
        except sqlite3.Error as exc:
            logger.error("Failed to close micro-question %s: %s", question_id, exc)
            return False
        finally:
            write_conn.close()

    def count_promotions_since(self, since: datetime) -> int:
        """Workflows promoted to a draft at or after *since*."""
        self._refresh_read_snapshot()
        try:
            row = self._conn.execute(
                "SELECT COUNT(DISTINCT promoted_slug) FROM workflow_observations "
                "WHERE promoted_at >= ?",
                (_sqlite_utc(since),),
            ).fetchone()
            return int(row[0]) if row else 0
        except sqlite3.OperationalError:
            return 0

    def purge_unpromoted_observations(
        self, max_age_days: int, *, now: datetime | None = None,
    ) -> int:
        """Delete observations older than *max_age_days* that never became
        part of a Skill.  Promoted ones are the Skill's evidence and stay."""
        return self._delete_older_than(
            "DELETE FROM workflow_observations "
            "WHERE promoted_slug IS NULL AND datetime(started_utc) < datetime(?)",
            (now or datetime.now(timezone.utc)) - timedelta(days=max_age_days),
        )

    # ------------------------------------------------------------------
    # Generated SOPs — track all generated SOPs with review status
    # ------------------------------------------------------------------

    def save_generated_sop(
        self,
        slug: str,
        title: str,
        source: str,
        sop_template: dict,
        confidence: float = 0.0,
        source_id: str = "",
        auto_approve: bool = True,
    ) -> str:
        """Persist a generated SOP and return its sop_id.

        If *auto_approve* is True, the SOP is immediately set to
        ``status='approved'`` with ``reviewed_at`` set to now.
        Otherwise it starts as ``status='draft'``.
        """
        sop_id = str(uuid.uuid4())
        db_path = self._get_writable_path()
        if not db_path:
            logger.error("Cannot determine DB path for SOP save")
            return sop_id

        status = "approved" if auto_approve else "draft"
        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            if auto_approve:
                write_conn.execute(
                    "INSERT INTO generated_sops "
                    "(sop_id, slug, title, source, status, sop_json, confidence, "
                    " source_id, created_at, reviewed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, "
                    " strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), "
                    " strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
                    (sop_id, slug, title, source, status,
                     json.dumps(sop_template), confidence, source_id),
                )
            else:
                write_conn.execute(
                    "INSERT INTO generated_sops "
                    "(sop_id, slug, title, source, status, sop_json, confidence, "
                    " source_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, "
                    " strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
                    (sop_id, slug, title, source, status,
                     json.dumps(sop_template), confidence, source_id),
                )
            write_conn.commit()
        except sqlite3.Error as exc:
            logger.exception("Failed to save generated SOP: %s", exc)
        finally:
            write_conn.close()

        return sop_id

    def get_generated_sops(self, status: str | None = None) -> list[dict]:
        """Return all SOPs, optionally filtered by status.

        Does NOT include ``sop_json`` in the result (too large for list views).
        """
        self._refresh_read_snapshot()
        try:
            if status is not None:
                cur = self._conn.execute(
                    "SELECT sop_id, slug, title, source, status, confidence, "
                    "  source_id, created_at, reviewed_at "
                    "FROM generated_sops WHERE status = ? "
                    "ORDER BY created_at DESC",
                    (status,),
                )
            else:
                cur = self._conn.execute(
                    "SELECT sop_id, slug, title, source, status, confidence, "
                    "  source_id, created_at, reviewed_at "
                    "FROM generated_sops ORDER BY created_at DESC",
                )
            return self._rows_to_dicts(cur.fetchall())
        except sqlite3.OperationalError:
            return []

    def get_generated_sop(self, sop_id: str) -> dict | None:
        """Return a single SOP by ID, including ``sop_json`` parsed back to dict."""
        self._refresh_read_snapshot()
        try:
            cur = self._conn.execute(
                "SELECT * FROM generated_sops WHERE sop_id = ?",
                (sop_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            result = dict(row)
            try:
                result["sop_json"] = json.loads(result["sop_json"])
            except (ValueError, TypeError):
                pass
            return result
        except sqlite3.OperationalError:
            return None

    def get_generated_sop_by_slug(self, slug: str) -> dict | None:
        """Return the most recent SOP matching *slug*, including parsed ``sop_json``."""
        self._refresh_read_snapshot()
        try:
            cur = self._conn.execute(
                "SELECT * FROM generated_sops WHERE slug = ? "
                "ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (slug,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            result = dict(row)
            try:
                result["sop_json"] = json.loads(result["sop_json"])
            except (ValueError, TypeError):
                pass
            return result
        except sqlite3.OperationalError:
            return None

    def update_sop_status(self, sop_id: str, status: str) -> bool:
        """Update SOP status to 'approved' or 'rejected' and set reviewed_at."""
        db_path = self._get_writable_path()
        if not db_path:
            return False

        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            cursor = write_conn.execute(
                "UPDATE generated_sops SET status = ?, "
                "reviewed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                "WHERE sop_id = ?",
                (status, sop_id),
            )
            write_conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as exc:
            logger.exception("Failed to update SOP status: %s", exc)
            return False
        finally:
            write_conn.close()

    def update_sop_json(self, slug: str, sop_template: dict) -> bool:
        """Replace the stored template of the newest SOP record for *slug*.

        An UPDATE, not a second INSERT: the review queue keeps one row per
        generation, and the app and the approval export read the updated
        template.  Returns False when no record exists for *slug*.
        """
        db_path = self._get_writable_path()
        if not db_path:
            return False

        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            cursor = write_conn.execute(
                "UPDATE generated_sops SET sop_json = ? WHERE sop_id = ("
                "  SELECT sop_id FROM generated_sops WHERE slug = ? "
                "  ORDER BY created_at DESC, rowid DESC LIMIT 1"
                ")",
                (json.dumps(sop_template), slug),
            )
            write_conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as exc:
            logger.exception("Failed to update SOP template: %s", exc)
            return False
        finally:
            write_conn.close()

    def get_approved_sops(self) -> list[dict]:
        """Return all approved SOPs, including parsed ``sop_json``."""
        self._refresh_read_snapshot()
        try:
            cur = self._conn.execute(
                "SELECT * FROM generated_sops WHERE status = 'approved' "
                "ORDER BY created_at DESC",
            )
            results = []
            for row in cur.fetchall():
                d = dict(row)
                try:
                    d["sop_json"] = json.loads(d["sop_json"])
                except (ValueError, TypeError):
                    pass
                results.append(d)
            return results
        except sqlite3.OperationalError:
            return []

    # ------------------------------------------------------------------
    # Failed generations — track failures for retry
    # ------------------------------------------------------------------

    def record_failed_generation(
        self,
        source: str,
        source_id: str,
        error: str,
        title: str = "",
        context: dict | None = None,
    ) -> str:
        """Record a generation failure for later retry. Returns failure_id."""
        failure_id = str(uuid.uuid4())
        db_path = self._get_writable_path()
        if not db_path:
            logger.error("Cannot determine DB path for failure record")
            return failure_id

        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            write_conn.execute(
                "INSERT INTO failed_generations "
                "(failure_id, source, source_id, title, error, context_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
                (failure_id, source, source_id, title, error,
                 json.dumps(context or {})),
            )
            write_conn.commit()
        except sqlite3.Error as exc:
            logger.exception("Failed to record generation failure: %s", exc)
        finally:
            write_conn.close()

        return failure_id

    def get_failed_generations(self, include_retried: bool = False) -> list[dict]:
        """Return failed generations, by default only unretried ones.

        Sorted by created_at DESC (newest first).
        """
        self._refresh_read_snapshot()
        try:
            if include_retried:
                cur = self._conn.execute(
                    "SELECT * FROM failed_generations "
                    "ORDER BY created_at DESC",
                )
            else:
                cur = self._conn.execute(
                    "SELECT * FROM failed_generations "
                    "WHERE retried = 0 ORDER BY created_at DESC",
                )
            return self._rows_to_dicts(cur.fetchall())
        except sqlite3.OperationalError:
            return []

    def get_failed_generation(self, failure_id: str) -> dict | None:
        """Return a single failure with parsed context_json."""
        self._refresh_read_snapshot()
        try:
            cur = self._conn.execute(
                "SELECT * FROM failed_generations WHERE failure_id = ?",
                (failure_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            result = dict(row)
            try:
                result["context_json"] = json.loads(result["context_json"])
            except (ValueError, TypeError):
                pass
            return result
        except sqlite3.OperationalError:
            return None

    def mark_failure_retried(self, failure_id: str) -> bool:
        """Mark a failure as retried. Returns True if a row was updated."""
        db_path = self._get_writable_path()
        if not db_path:
            return False

        write_conn = sqlite3.connect(db_path)
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")
            cursor = write_conn.execute(
                "UPDATE failed_generations SET retried = 1 "
                "WHERE failure_id = ?",
                (failure_id,),
            )
            write_conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as exc:
            logger.exception("Failed to mark failure retried: %s", exc)
            return False
        finally:
            write_conn.close()

    # ------------------------------------------------------------------
    # OCR text preservation (pre-expiry extraction)
    # ------------------------------------------------------------------

    def preserve_ocr_text(
        self, retention_days: int = 14,
    ) -> tuple[int, list[tuple[str, str, str]]]:
        """Extract and persist OCR text from events nearing expiry.

        Scans processed events that will expire within the next 2 days
        and saves their OCR text + visual_text_proxy to a permanent
        table that survives event purging.

        Returns (count, preserved_items) where preserved_items is a list
        of (event_id, source, text_content) tuples for downstream embedding.
        """
        import json as _json

        db_path = self._get_writable_path()
        if not db_path:
            return 0, []

        write_conn = sqlite3.connect(db_path)
        count = 0
        preserved_items: list[tuple[str, str, str]] = []
        try:
            write_conn.execute("PRAGMA busy_timeout = 5000;")

            # Find processed events nearing expiry that haven't been
            # preserved yet.  Uses the write connection for both read and
            # write to ensure consistency (read snapshot can't see our
            # own inserts within this batch).
            expiry_window = f"-{retention_days - 2} days"
            cur = write_conn.execute(
                "SELECT id, timestamp, metadata_json, "
                "  scene_annotation_json, window_json "
                "FROM events "
                "WHERE processed = 1 "
                "  AND datetime(timestamp) < datetime('now', ?) "
                "  AND id NOT IN (SELECT event_id FROM preserved_text) "
                "LIMIT 500",
                (expiry_window,),
            )
            columns = [d[0] for d in cur.description]

            for row in cur.fetchall():
                row_dict = dict(zip(columns, row))
                event_id = row_dict["id"]
                ts = row_dict["timestamp"]

                # Extract app name
                app = ""
                wj = row_dict.get("window_json", "")
                if wj:
                    try:
                        w = _json.loads(wj)
                        app = w.get("app_id", "") or w.get("app_name", "")
                    except (TypeError, _json.JSONDecodeError):
                        pass

                texts_to_save: list[tuple[str, str]] = []

                # OCR text from metadata
                meta_raw = row_dict.get("metadata_json", "{}")
                if meta_raw:
                    try:
                        meta = _json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
                        ocr = meta.get("ocr", {})
                        if isinstance(ocr, dict):
                            full_text = ocr.get("full_text", "")
                            if full_text and len(full_text) > 10:
                                texts_to_save.append(("ocr", redact(full_text[:4000])))
                    except (TypeError, _json.JSONDecodeError):
                        pass

                # Visual text proxy from annotation
                ann_raw = row_dict.get("scene_annotation_json", "")
                if ann_raw:
                    try:
                        ann = _json.loads(ann_raw) if isinstance(ann_raw, str) else ann_raw
                        proxy = ann.get("_visual_text_proxy", "")
                        if proxy and len(proxy) > 10:
                            texts_to_save.append(("visual_proxy", proxy[:4000]))
                    except (TypeError, _json.JSONDecodeError):
                        pass

                for source, text in texts_to_save:
                    try:
                        write_conn.execute(
                            "INSERT OR IGNORE INTO preserved_text "
                            "(event_id, source, timestamp, text_content, app) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (event_id, source, ts, text, app),
                        )
                        count += 1
                        preserved_items.append((event_id, source, text))
                    except sqlite3.Error:
                        pass

            write_conn.commit()
            if count > 0:
                logger.info("Preserved %d OCR/proxy texts before expiry", count)
        except sqlite3.Error as exc:
            logger.error("Failed to preserve OCR text: %s", exc)
        finally:
            write_conn.close()

        return count, preserved_items

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


def _url_match(url_a: str, url_b: str) -> bool:
    """True when both locations are on the same host (scheme optional,
    ``www.`` ignored)."""
    from agenthandover_worker.dom import url_host

    host_a = url_host(url_a)
    return bool(host_a) and host_a == url_host(url_b)
