"""Activity search and session recall over VLM annotations.

Provides full-text search (FTS5) and structured queries over the
``scene_annotation_json`` column of the daemon's events table.  The
``ActivitySearcher`` manages its own read-write SQLite connection —
independent of ``WorkerDB`` — so it can create and maintain the FTS5
virtual table without interfering with the daemon or the worker's
primary read-only connection.

Key features:
- Lazy FTS5 index creation on first search/recall
- Full-text search over what_doing, app, location fields
- Structured filtering by date, app, and time range
- Session recall with ordered timeline and active-minutes calculation
- Proper FTS5 query escaping to prevent injection
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Events within this many seconds of each other count as "active" time.
_ACTIVE_GAP_THRESHOLD_SEC = 300  # 5 minutes


@dataclass
class SearchResult:
    """A single search hit from the FTS5 index."""

    timestamp: str
    app: str
    location: str
    what_doing: str
    relevance_score: float
    event_id: str
    screenshot_id: str | None = None


@dataclass
class ActivityTimeline:
    """Ordered timeline of user activity for a date/time range."""

    entries: list[SearchResult]
    date: str
    total_active_minutes: int
    apps_used: list[str]


def _escape_fts5_query(query: str) -> str:
    """Escape a user-provided string for safe use in an FTS5 MATCH clause.

    FTS5 treats certain characters as operators (``*``, ``"``, ``(``,
    ``)``, ``:``, ``+``, ``-``, ``^``, ``NOT``, ``AND``, ``OR``,
    ``NEAR``).  We wrap each non-empty token in double quotes so they
    are interpreted as literals.  Empty/whitespace-only queries return
    an empty string.
    """
    if not query or not query.strip():
        return ""

    # Split on whitespace, drop empty tokens
    tokens = query.split()
    escaped_parts: list[str] = []
    for token in tokens:
        # Strip characters that cannot appear inside FTS5 quoted strings
        # (double-quotes would break the quoting).
        cleaned = token.replace('"', "")
        if cleaned:
            escaped_parts.append(f'"{cleaned}"')

    return " ".join(escaped_parts)


def _parse_annotation(annotation_json: str | None) -> dict:
    """Safely parse a scene_annotation_json value.

    Returns a dict with ``what_doing``, ``app``, and ``location`` keys
    (empty strings if missing or unparseable).
    """
    result = {"what_doing": "", "app": "", "location": ""}
    if not annotation_json:
        return result
    try:
        ann = json.loads(annotation_json)
    except (json.JSONDecodeError, TypeError):
        return result

    tc = ann.get("task_context") or {}
    vc = ann.get("visual_context") or {}

    # Support both flat and nested annotation formats.
    # Flat format: {"app": ..., "location": ..., "task_context": {"what_doing": ...}}
    # Nested format: {"visual_context": {"active_app": ..., "location": ...}, "task_context": ...}
    result["what_doing"] = tc.get("what_doing", "")
    result["app"] = vc.get("active_app", "") or ann.get("app", "")
    result["location"] = vc.get("location", "") or ann.get("location", "")
    return result


def _iso_to_epoch(ts: str) -> float:
    """Convert an ISO-8601 timestamp string to a UNIX epoch float.

    Handles both ``Z`` suffix and ``+00:00`` timezone offset.
    Returns 0.0 on parse failure.
    """
    if not ts:
        return 0.0
    try:
        # Normalise common variants
        cleaned = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


def _calculate_active_minutes(timestamps: list[str]) -> int:
    """Calculate total active minutes from a sorted list of ISO timestamps.

    Two consecutive events are considered "active" if the gap between them
    is <= ``_ACTIVE_GAP_THRESHOLD_SEC`` (5 minutes).  The active time for
    each such pair is the gap itself.  Gaps larger than the threshold are
    treated as idle and not counted.
    """
    if len(timestamps) < 2:
        return 0

    epochs = [_iso_to_epoch(ts) for ts in timestamps]
    total_seconds = 0.0

    for i in range(1, len(epochs)):
        gap = epochs[i] - epochs[i - 1]
        if 0 < gap <= _ACTIVE_GAP_THRESHOLD_SEC:
            total_seconds += gap

    return int(total_seconds / 60)


class ActivitySearcher:
    """Full-text + structured search over VLM annotations.

    Opens its own read-write SQLite connection for FTS5 index management.
    Does **not** import or depend on ``WorkerDB``.

    Usage::

        searcher = ActivitySearcher("/path/to/events.db")
        results = searcher.search("expired domains")
        timeline = searcher.session_recall(date="2026-03-10")
        searcher.close()
    """

    def __init__(self, db_path: str | Path) -> None:
        resolved = Path(db_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Database file does not exist: {resolved}")

        self._db_path = str(resolved)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout = 5000;")

        # Try WAL mode for concurrent reads with the daemon.
        try:
            self._conn.execute("PRAGMA journal_mode=wal;")
        except sqlite3.OperationalError:
            pass

        self._fts_ready = False
        logger.info("ActivitySearcher opened: %s", resolved)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        date: str | None = None,
        app: str | None = None,
        time_range: tuple[str, str] | None = None,
    ) -> list[SearchResult]:
        """Full-text search over VLM annotations.

        Parameters
        ----------
        query:
            Free-text search string.  Tokens are individually quoted for
            FTS5 safety.
        limit:
            Maximum number of results to return.
        date:
            Optional date filter in ``YYYY-MM-DD`` format.
        app:
            Optional app name filter (case-insensitive substring match).
        time_range:
            Optional ``(start_iso, end_iso)`` tuple to restrict the
            time window.

        Returns a list of :class:`SearchResult` ordered by relevance
        (best first).
        """
        self._ensure_fts_index()

        escaped = _escape_fts5_query(query)
        if not escaped:
            return []

        # Build the query.  We join events_fts with events to get
        # the timestamp and filter by date/time/app.
        conditions: list[str] = ["events_fts MATCH ?"]
        params: list[str | int] = [escaped]

        if date:
            # Match events whose timestamp starts with the given date.
            conditions.append("e.timestamp LIKE ?")
            params.append(f"{date}%")

        if app:
            # Case-insensitive app filter via the FTS table column.
            conditions.append("f.app LIKE ?")
            params.append(f"%{app}%")

        if time_range:
            start_ts, end_ts = time_range
            conditions.append("e.timestamp >= ?")
            params.append(start_ts)
            conditions.append("e.timestamp <= ?")
            params.append(end_ts)

        where_clause = " AND ".join(conditions)
        params.append(limit)

        sql = (
            "SELECT f.event_id, f.what_doing, f.app, f.location, "
            "  e.timestamp, rank "
            "FROM events_fts f "
            "JOIN events e ON e.id = f.event_id "
            f"WHERE {where_clause} "
            "ORDER BY rank "
            "LIMIT ?"
        )

        try:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("FTS5 search failed: %s", exc)
            return []

        if not rows:
            return []

        # Normalise FTS5 rank scores to 0–1.
        # rank() returns negative values (closer to 0 = better match).
        raw_ranks = [abs(float(r["rank"])) for r in rows]
        max_rank = max(raw_ranks) if raw_ranks else 1.0
        if max_rank == 0.0:
            max_rank = 1.0

        results: list[SearchResult] = []
        for row, raw in zip(rows, raw_ranks):
            # Invert: highest relevance → 1.0, lowest → close to 0.
            score = 1.0 - (raw / max_rank) if max_rank > 0 else 1.0
            # Ensure the best match gets 1.0 exactly.
            if raw == min(raw_ranks):
                score = 1.0

            results.append(
                SearchResult(
                    timestamp=row["timestamp"],
                    app=row["app"] or "",
                    location=row["location"] or "",
                    what_doing=row["what_doing"] or "",
                    relevance_score=round(score, 4),
                    event_id=row["event_id"],
                )
            )

        return results

    def session_recall(
        self,
        *,
        date: str | None = None,
        app: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> ActivityTimeline:
        """Build an ordered timeline of what the user did.

        Parameters
        ----------
        date:
            Optional date in ``YYYY-MM-DD`` format.  Defaults to today.
        app:
            Optional app name filter (case-insensitive substring).
        start_time:
            Optional start ISO timestamp.
        end_time:
            Optional end ISO timestamp.

        Returns an :class:`ActivityTimeline` with entries sorted
        chronologically.
        """
        self._ensure_fts_index()

        if date is None and start_time is None and end_time is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        conditions: list[str] = [
            "e.annotation_status = 'completed'",
            "e.scene_annotation_json IS NOT NULL",
        ]
        params: list[str] = []

        if date:
            conditions.append("e.timestamp LIKE ?")
            params.append(f"{date}%")

        if start_time:
            conditions.append("e.timestamp >= ?")
            params.append(start_time)

        if end_time:
            conditions.append("e.timestamp <= ?")
            params.append(end_time)

        where_clause = " AND ".join(conditions)

        sql = (
            "SELECT e.id, e.timestamp, e.scene_annotation_json "
            "FROM events e "
            f"WHERE {where_clause} "
            "ORDER BY e.timestamp ASC"
        )

        try:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("Session recall query failed: %s", exc)
            return ActivityTimeline(
                entries=[],
                date=date or "",
                total_active_minutes=0,
                apps_used=[],
            )

        entries: list[SearchResult] = []
        timestamps: list[str] = []
        apps_seen: dict[str, None] = {}  # ordered set

        for row in rows:
            parsed = _parse_annotation(row["scene_annotation_json"])
            app_name = parsed["app"]

            # Apply app filter if specified.
            if app and app.lower() not in app_name.lower():
                continue

            entries.append(
                SearchResult(
                    timestamp=row["timestamp"],
                    app=app_name,
                    location=parsed["location"],
                    what_doing=parsed["what_doing"],
                    relevance_score=1.0,  # Not a search; all equally relevant.
                    event_id=row["id"],
                )
            )
            timestamps.append(row["timestamp"])
            if app_name and app_name not in apps_seen:
                apps_seen[app_name] = None

        active_minutes = _calculate_active_minutes(timestamps)

        return ActivityTimeline(
            entries=entries,
            date=date or "",
            total_active_minutes=active_minutes,
            apps_used=list(apps_seen.keys()),
        )

    def close(self) -> None:
        """Close the underlying database connection."""
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ActivitySearcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # FTS5 index management
    # ------------------------------------------------------------------

    def _ensure_fts_index(self) -> None:
        """Create the FTS5 virtual table if it does not exist.

        Called lazily on the first ``search()`` or ``session_recall()``
        invocation.  If the table already exists (from a previous run)
        it is a fast no-op.
        """
        if self._fts_ready:
            return

        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS events_fts "
                "USING fts5(event_id, what_doing, app, location)"
            )
            self._conn.commit()
        except sqlite3.OperationalError as exc:
            logger.error("Failed to create FTS5 table: %s", exc)
            self._fts_ready = True  # Don't retry on every call.
            return

        self._populate_fts_index()
        self._fts_ready = True

    def _populate_fts_index(self) -> int:
        """Fill the FTS5 table from existing scene annotations.

        Only inserts rows that are not already present in the FTS5 table.
        Uses a SQL NOT EXISTS subquery to avoid loading all indexed IDs
        into Python memory.

        Returns the number of new entries added.
        """
        # Fetch annotated events not yet in the FTS table using SQL
        # to filter, avoiding unbounded memory from loading all IDs.
        cur = self._conn.execute(
            "SELECT e.id, e.scene_annotation_json FROM events e "
            "WHERE e.annotation_status = 'completed' "
            "  AND e.scene_annotation_json IS NOT NULL "
            "  AND NOT EXISTS ("
            "    SELECT 1 FROM events_fts f WHERE f.event_id = e.id"
            "  )"
        )

        batch: list[tuple[str, str, str, str]] = []
        for row in cur.fetchall():
            parsed = _parse_annotation(row["scene_annotation_json"])
            batch.append((
                row["id"],
                parsed["what_doing"],
                parsed["app"],
                parsed["location"],
            ))

        if not batch:
            return 0

        self._conn.executemany(
            "INSERT INTO events_fts(event_id, what_doing, app, location) "
            "VALUES (?, ?, ?, ?)",
            batch,
        )
        self._conn.commit()
        logger.info("Populated FTS5 index with %d new entries", len(batch))
        return len(batch)

    def refresh_index(self) -> int:
        """Re-scan events and add any new annotations to the FTS5 index.

        Returns the number of new entries added.  Safe to call at any time.
        """
        self._ensure_fts_index()
        return self._populate_fts_index()
