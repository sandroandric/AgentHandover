"""Forget: delete what AgentHandover keeps from a stretch of time or an app.

``forget(selector)`` deletes, for the frames the selector names, every
place AgentHandover stores what was on screen, in one place so nothing is
missed:

* ``events`` rows (window, OCR, URL, annotation, diff), their
  ``vlm_queue`` jobs and ``artifacts`` rows (with the artifact files);
* the activity search index (``events_fts`` and ``events_fts_ids``);
* ``preserved_text`` (OCR and screen text kept past the events' 14 days),
  selected by its own timestamp and app as well, since it outlives them;
* ``vector_store`` rows made from those frames: ``annotation`` (source id =
  event id), ``preserved_text`` (``<event id>:<source>``) and the
  ``daily_summary`` of each local day the frames fall on;
* ``memory_frames``: every Memory row (moment) that overlaps the range is
  forgotten whole, and with it the frames of its window inside its span;
* ``task_segments`` and ``workflow_observations`` that hold one of those
  frames (or, for a time range, started inside it): their labels say what
  was being done;
* ``annotation_cache`` rows of those windows (app and title);
* files: the frames' screenshots, their encrypted thumbnails (and, for a
  time range, any screenshot or thumbnail written inside it whose event
  is not stored yet), the daily summaries and digests of those days
  (``observations/daily``, ``observations/digests``), their entries in
  ``context/recent.json``, and the CLI's last search/recall result files.

Approved and draft Skills (``procedures``, ``generated_sops``, the
exported files) and their vectors are never touched, nor is the profile.

A selector is exactly one of: ``moment_id`` (a Memory moment as search
returns it: its rows of one window and the time they span), ``all``, a
time range ``since``/``until`` (optionally one ``app``), or an ``app``
alone (all time).  An app matches a frame's app name, its bundle id or
``pid:`` id's name, case-insensitively.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Iterable

from agenthandover_worker.annotation_cache import title_hash
from agenthandover_worker.event_helpers import display_name, parse_json_field, parse_timestamp
from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.memory import MOMENT_GAP_SECONDS, frame_window, utc_text
from agenthandover_worker.scene_annotator import find_screenshot
from agenthandover_worker.thumbnails import EXTENSION as THUMB_EXTENSION
from agenthandover_worker.thumbnails import delete_thumbnails, thumbs_dir

logger = logging.getLogger(__name__)

# The result files the CLI's search and recall fallbacks leave in the data dir.
RESULT_FILES = ("search-query-result.json", "recall-query-result.json")
# preserved_text sources (``db.preserve_ocr_text``), for their vector ids.
PRESERVED_SOURCES = ("ocr", "visual_proxy")
# Rows per IN (...) list.
CHUNK = 500


@dataclass(frozen=True)
class Selector:
    """What to forget (see the module docstring)."""

    since: datetime | None = None
    until: datetime | None = None
    app: str | None = None
    moment_id: str | None = None
    all: bool = False

    def describe(self) -> str:
        if self.all:
            return "everything"
        if self.moment_id:
            return f"moment {self.moment_id}"
        span = f"{self.since.isoformat()} to {self.until.isoformat()}" if self.since else ""
        return " and ".join(p for p in (span, f"app {self.app}" if self.app else "") if p)


def parse_time(value: Any, name: str) -> datetime:
    """An ISO-8601 time from a request (no offset: UTC); ValueError
    naming *name* when it is not one."""
    when = parse_timestamp(value) if isinstance(value, str) and value.strip() else None
    if when is None:
        raise ValueError(f"'{name}' must be an ISO-8601 time")
    return when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)


def parse_selector(body: Any) -> Selector:
    """The selector of a ``POST /memory/forget`` body; ValueError saying
    what is wrong when it is not exactly one selector."""
    if not isinstance(body, dict):
        raise ValueError("the body must be a JSON object")
    keys = set(body)
    unknown = keys - {"moment_id", "all", "since", "until", "app"}
    if unknown:
        raise ValueError(f"unknown field(s): {', '.join(sorted(unknown))}")
    if keys == {"moment_id"}:
        moment = body["moment_id"]
        if not isinstance(moment, str) or not moment.strip():
            raise ValueError("'moment_id' must be a non-empty string")
        return Selector(moment_id=moment.strip())
    if keys == {"all"}:
        if body["all"] is not True:
            raise ValueError("'all' must be true")
        return Selector(all=True)
    app = body.get("app")
    if "app" in keys and (not isinstance(app, str) or not app.strip()):
        raise ValueError("'app' must be a non-empty string")
    if keys == {"app"}:
        return Selector(app=app.strip())
    if keys in ({"since", "until"}, {"since", "until", "app"}):
        since, until = parse_time(body["since"], "since"), parse_time(body["until"], "until")
        if since > until:
            raise ValueError("'since' is after 'until'")
        return Selector(since=since, until=until, app=app.strip() if app else None)
    raise ValueError(
        "give exactly one of: moment_id; all; since and until (and optionally app); app"
    )


def _app_names(window: dict) -> set[str]:
    """The names a frame's app answers to, casefolded."""
    names = {
        str(window.get("app_id") or ""), display_name(window.get("app_id")),
        str(window.get("app_name") or ""),
    }
    return {n.strip().casefold() for n in names if n and n.strip()}


def _app_matches(app: str | None, names: Iterable[str]) -> bool:
    return app is None or app.strip().casefold() in set(names)


def _chunks(items: list, size: int = CHUNK) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _in(conn: sqlite3.Connection, sql: str, values: Iterable[Any], *params: Any) -> int:
    """Run *sql* (with one ``{in}`` placeholder list) over *values* in
    chunks; returns the rows changed."""
    changed = 0
    for chunk in _chunks(list(values)):
        changed += conn.execute(
            sql.format(placeholders=",".join("?" * len(chunk))), (*params, *chunk),
        ).rowcount
    return changed


def _epoch(text: str) -> float:
    when = parse_timestamp(text)
    if when is None:
        return 0.0
    return (when if when.tzinfo else when.replace(tzinfo=timezone.utc)).timestamp()


def moment_rows(conn: sqlite3.Connection, moment_id: str) -> list[sqlite3.Row]:
    """The Memory rows of the moment *moment_id* names, as search groups
    them (``MemoryIndex.search``): the rows of its model next to it in
    time that show the same window less than ``MOMENT_GAP_SECONDS``
    apart.  Empty when there is no such row."""
    target = conn.execute(
        "SELECT model, dim FROM memory_frames WHERE id = ?", (moment_id,),
    ).fetchone()
    if target is None:
        return []
    rows = conn.execute(
        "SELECT id, first_ts, last_ts, app, window_title FROM memory_frames "
        "WHERE model = ? AND dim = ? ORDER BY julianday(first_ts), rowid",
        (target["model"], target["dim"]),
    ).fetchall()
    at = next(i for i, row in enumerate(rows) if row["id"] == moment_id)

    def same(a: sqlite3.Row, b: sqlite3.Row) -> bool:
        return (
            (a["app"], a["window_title"]) == (b["app"], b["window_title"])
            and _epoch(b["first_ts"]) - _epoch(a["last_ts"]) < MOMENT_GAP_SECONDS
        )

    start = end = at
    while start > 0 and same(rows[start - 1], rows[start]):
        start -= 1
    while end + 1 < len(rows) and same(rows[end], rows[end + 1]):
        end += 1
    return rows[start:end + 1]


def _local_date(text: str, local_tz: tzinfo | None) -> date | None:
    when = parse_timestamp(text)
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(local_tz).date()


class _Forget:
    """One forget run: what was found, then deleted (see :func:`forget`)."""

    def __init__(
        self,
        selector: Selector,
        conn: sqlite3.Connection,
        *,
        data_dir: Path,
        knowledge_dir: Path,
        local_tz: tzinfo | None,
    ) -> None:
        self.sel = selector
        self.conn = conn
        self.data_dir = Path(data_dir)
        self.knowledge_dir = Path(knowledge_dir)
        self.local_tz = local_tz
        self.since = selector.since
        self.until = selector.until
        self.event_ids: set[str] = set()
        self.events: list[dict] = []
        self.memory: list[sqlite3.Row] = []
        self.preserved: list[sqlite3.Row] = []
        self.dates: set[str] = set()
        self.counts = {
            "events": 0, "memory_moments": 0, "thumbnails": 0, "screenshots": 0,
            "vectors": 0, "daily_summaries": 0,
        }

    # -- what is forgotten ----------------------------------------------

    def _range_sql(self, start_col: str, end_col: str | None = None) -> tuple[str, list[str]]:
        """``WHERE`` text and parameters for rows inside (or, with
        *end_col*, overlapping) the selector's range; all rows without one."""
        if self.since is None or self.until is None:
            return "1", []
        end_col = end_col or start_col
        return (
            f"julianday({start_col}) <= julianday(?) AND julianday({end_col}) >= julianday(?)",
            [utc_text(self.until), utc_text(self.since)],
        )

    def find(self) -> None:
        if self.sel.moment_id:
            rows = moment_rows(self.conn, self.sel.moment_id)
            if not rows:
                raise LookupError(f"no Memory moment {self.sel.moment_id}")
            self.since = min(parse_timestamp(r["first_ts"]) for r in rows)
            self.until = max(parse_timestamp(r["last_ts"]) for r in rows)
        app = self.sel.app

        where, params = self._range_sql("first_ts", "last_ts")
        self.memory = [
            row for row in self.conn.execute(
                "SELECT id, first_ts, last_ts, app, window_title, described_by, thumb_id "
                f"FROM memory_frames WHERE {where}",
                params,
            )
            if _app_matches(app, [row["app"].casefold()])
        ]

        where, params = self._range_sql("timestamp")
        for row in self.conn.execute(
            f"SELECT id, timestamp, window_json, metadata_json FROM events WHERE {where}", params,
        ):
            event = dict(row)
            if _app_matches(app, _app_names(parse_json_field(event, "window_json"))):
                self.events.append(event)
        # A moment is forgotten whole: the frames of its window in its span
        # (only a time range can cut through one; ``all`` and an app alone
        # already hold all of its frames).
        for moment in self.memory if self.since is not None else ():
            for row in self.conn.execute(
                "SELECT id, timestamp, window_json, metadata_json FROM events "
                "WHERE julianday(timestamp) BETWEEN julianday(?) AND julianday(?)",
                (moment["first_ts"], moment["last_ts"]),
            ):
                event = dict(row)
                if frame_window(event)[:2] == (moment["app"], moment["window_title"]):
                    self.events.append(event)
        self.events = list({e["id"]: e for e in self.events}.values())
        self.event_ids = {e["id"] for e in self.events}
        self.event_ids |= {m["id"] for m in self.memory}
        self.event_ids |= {m["described_by"] for m in self.memory if m["described_by"]}

        where, params = self._range_sql("timestamp")
        self.preserved = [
            row for row in self.conn.execute(
                f"SELECT event_id, source, timestamp, app FROM preserved_text WHERE {where}",
                params,
            )
            if _app_matches(app, {row["app"].casefold(), display_name(row["app"]).casefold()})
        ]
        self.preserved += _select_in(
            self.conn, "SELECT event_id, source, timestamp, app FROM preserved_text "
            "WHERE event_id IN ({placeholders})", self.event_ids,
        )
        self.event_ids |= {row["event_id"] for row in self.preserved}

        stamps = [e["timestamp"] for e in self.events]
        stamps += [m["first_ts"] for m in self.memory] + [m["last_ts"] for m in self.memory]
        stamps += [p["timestamp"] for p in self.preserved]
        self.dates = {d.isoformat() for d in (_local_date(s, self.local_tz) for s in stamps) if d}
        self.dates |= self._summary_dates()

    def _summary_files(self) -> list[Path]:
        observations = self.knowledge_dir / "observations"
        return sorted(
            [*(observations / "daily").glob("*.json"), *(observations / "digests").glob("*.json")]
        )

    def _summary_dates(self) -> set[str]:
        """The days whose summaries go besides those of the frames found:
        every day for ``all``; the days of a time range; for an app, the
        summaries (in the range) that name it."""
        if self.sel.all:
            return {p.stem for p in self._summary_files()}
        dates: set[str] = set()
        first = last = None
        if self.since is not None and self.until is not None:
            first = self.since.astimezone(self.local_tz).date().isoformat()
            last = self.until.astimezone(self.local_tz).date().isoformat()
        for path in self._summary_files():
            if first is not None and not first <= path.stem <= last:
                continue
            if self.sel.app is None:
                dates.add(path.stem)
                continue
            try:
                text = path.read_text(encoding="utf-8").casefold()
            except OSError:
                continue
            if self.sel.app.strip().casefold() in text:
                dates.add(path.stem)
        return dates

    # -- the database ---------------------------------------------------

    def delete_rows(self) -> list[str]:
        """Delete every row; returns the artifact files the rows named."""
        conn, ids = self.conn, sorted(self.event_ids)
        artifact_files = [
            row[0] for row in _select_in(
                conn, "SELECT file_path FROM artifacts WHERE event_id IN ({placeholders})", ids,
            )
        ]
        _in(conn, "DELETE FROM artifacts WHERE event_id IN ({placeholders})", ids)
        _in(conn, "DELETE FROM vlm_queue WHERE event_id IN ({placeholders})", ids)
        self.counts["events"] = _in(conn, "DELETE FROM events WHERE id IN ({placeholders})", ids)

        fts = [row[0] for row in _select_in(
            conn, "SELECT fts_rowid FROM events_fts_ids WHERE event_id IN ({placeholders})", ids,
        )]
        _in(conn, "DELETE FROM events_fts WHERE rowid IN ({placeholders})", fts)
        _in(conn, "DELETE FROM events_fts_ids WHERE event_id IN ({placeholders})", ids)
        _in(conn, "DELETE FROM preserved_text WHERE event_id IN ({placeholders})", ids)

        preserved_ids = {f"{p['event_id']}:{p['source']}" for p in self.preserved}
        preserved_ids |= {f"{eid}:{src}" for eid in ids for src in PRESERVED_SOURCES}
        vectors = _in(
            conn, "DELETE FROM vector_store WHERE source_type = 'annotation' "
            "AND source_id IN ({placeholders})", ids,
        )
        vectors += _in(
            conn, "DELETE FROM vector_store WHERE source_type = 'preserved_text' "
            "AND source_id IN ({placeholders})", sorted(preserved_ids),
        )
        vectors += _in(
            conn, "DELETE FROM vector_store WHERE source_type = 'daily_summary' "
            "AND source_id IN ({placeholders})", sorted(self.dates),
        )
        self.counts["vectors"] = vectors

        self.counts["memory_moments"] = _in(
            conn, "DELETE FROM memory_frames WHERE id IN ({placeholders})",
            [m["id"] for m in self.memory],
        )

        windows = {
            (display_name(w.get("app_id")), title_hash(str(w.get("title") or "")))
            for w in (parse_json_field(e, "window_json") for e in self.events)
        }
        for app, title in windows:
            conn.execute(
                "DELETE FROM annotation_cache WHERE app = ? AND title_hash = ?", (app, title),
            )

        if self.sel.all:
            # Also what outlived its event (purged by retention) and so
            # could not be matched to a frame above.
            for table in (
                "task_segments", "workflow_observations", "annotation_cache",
                "events_fts", "events_fts_ids", "preserved_text",
            ):
                conn.execute(f"DELETE FROM {table}")
            self.counts["vectors"] += conn.execute(
                "DELETE FROM vector_store WHERE source_type IN "
                "('annotation', 'preserved_text', 'daily_summary')"
            ).rowcount
        else:
            self._delete_sessions()
        return artifact_files

    def _delete_sessions(self) -> None:
        """``task_segments`` and ``workflow_observations`` holding a
        forgotten frame, or (a time range without an app) started in it."""
        for table, key, start in (
            ("task_segments", "segment_id", "start_time"),
            ("workflow_observations", "obs_id", "started_utc"),
        ):
            doomed = []
            for row in self.conn.execute(f"SELECT {key}, event_ids_json, {start} FROM {table}"):
                try:
                    held = set(json.loads(row[1] or "[]"))
                except (TypeError, ValueError):
                    held = set()
                started = parse_timestamp(row[2]) if row[2] else None
                if started is not None and started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                in_range = (
                    self.sel.app is None and started is not None
                    and self.since is not None and self.since <= started <= self.until
                )
                if in_range or held & self.event_ids:
                    doomed.append(row[0])
            _in(self.conn, f"DELETE FROM {table} WHERE {key} IN ({{placeholders}})", doomed)

    # -- files ----------------------------------------------------------

    def delete_files(self, artifact_files: list[str], screenshots_dir: Path) -> None:
        shots = 0
        for event in self.events:
            path = find_screenshot(event, screenshots_dir)
            if path is not None:
                shots += _unlink(Path(path))
        for path in artifact_files:
            _unlink(Path(path))

        thumbs = thumbs_dir(self.data_dir)
        names = set(self.event_ids) | {m["thumb_id"] for m in self.memory if m["thumb_id"]}
        removed = delete_thumbnails(sorted(names), thumbs)
        # Files written in the range whose event is not stored yet (the
        # daemon writes the files first), or every file for ``all``.
        if self.sel.app is None and (self.sel.all or self.since is not None):
            removed += _sweep(thumbs, self.sel.all, self.since, self.until, THUMB_EXTENSION)
            shots += _sweep(screenshots_dir, self.sel.all, self.since, self.until, None)
        self.counts["thumbnails"] = removed
        self.counts["screenshots"] = shots

        observations = self.knowledge_dir / "observations"
        for day in sorted(self.dates):
            self.counts["daily_summaries"] += _unlink(observations / "daily" / f"{day}.json")
            _unlink(observations / "digests" / f"{day}.json")
        self._trim_recent_context()
        for name in RESULT_FILES:
            _unlink(self.data_dir / name)

    def _trim_recent_context(self) -> None:
        path = self.knowledge_dir / "context" / "recent.json"
        try:
            recent = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        days = recent.get("last_7_days") if isinstance(recent, dict) else None
        if not isinstance(days, list):
            return
        kept = [d for d in days if not (isinstance(d, dict) and d.get("date") in self.dates)]
        if len(kept) != len(days):
            recent["last_7_days"] = kept
            KnowledgeBase._atomic_write_text(path, json.dumps(recent, indent=2, default=str))


def _select_in(conn: sqlite3.Connection, sql: str, values: Iterable[Any]) -> list[sqlite3.Row]:
    rows: list[sqlite3.Row] = []
    for chunk in _chunks(sorted(values)):
        rows += conn.execute(sql.format(placeholders=",".join("?" * len(chunk))), chunk).fetchall()
    return rows


def _unlink(path: Path) -> int:
    """1 when *path* was deleted, 0 when it was not there (or could not go)."""
    try:
        path.unlink()
        return 1
    except FileNotFoundError:
        return 0
    except OSError:
        logger.warning("Could not delete %s", path, exc_info=True)
        return 0


def _sweep(
    directory: Path, everything: bool, since: datetime | None, until: datetime | None,
    suffix: str | None,
) -> int:
    """Delete the files in *directory* (with *suffix*) modified inside
    ``[since, until]``, or all of them with *everything*."""
    if not directory.is_dir():
        return 0
    removed = 0
    for path in directory.iterdir():
        if not path.is_file() or (suffix is not None and path.suffix != suffix):
            continue
        if not everything:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if since is None or until is None or not since.timestamp() <= mtime <= until.timestamp():
                continue
        removed += _unlink(path)
    return removed


def forget(
    selector: Selector,
    *,
    db_path: Path,
    data_dir: Path,
    knowledge_dir: Path,
    screenshots_dir: Path | None = None,
    local_tz: tzinfo | None = None,
) -> dict:
    """Forget what *selector* names (see the module docstring).

    Returns the counts: ``events``, ``memory_moments``, ``thumbnails``,
    ``screenshots``, ``vectors`` and ``daily_summaries`` deleted.  Raises
    LookupError for a ``moment_id`` that names no moment.  Every database
    row goes in one transaction; the files after it commits.
    """
    conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        run = _Forget(
            selector, conn, data_dir=data_dir, knowledge_dir=knowledge_dir, local_tz=local_tz,
        )
        conn.execute("BEGIN IMMEDIATE")
        try:
            run.find()
            artifact_files = run.delete_rows()
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
    run.delete_files(artifact_files, Path(screenshots_dir or Path(data_dir) / "screenshots"))
    logger.info("Forgot %s: %s", selector.describe(), run.counts)
    return run.counts

