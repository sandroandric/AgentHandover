"""Memory: every kept frame embedded, so what was on screen can be searched.

:class:`MemoryWriter` embeds each kept frame (its screenshot and
``"title: <app> — <window title> | text: "``) with ``[memory] model`` at
``[memory] dimensions`` into ``memory_frames`` (migration v008), as its own
low-priority job on the inference thread (P2, ``stages/memory.py``),
independent of annotation: frames that are never annotated (battery, the
hourly budget, the 24-hour backlog expiry) still enter Memory.

The writer walks the daemon's events in insertion order (``rowid``) from a
cursor persisted in ``worker_state`` (:data:`CURSOR_STATE_KEY`), one frame
at a time:

* frames Memory never keeps are passed over: a privacy zone's blocked or
  metadata-only window, system UI (notification banners, the Dock), a
  focus recording's frames that are not keyframes (the focus processor
  marks them ``skipped``), frames whose JPEG is gone;
* a focus recording's frame waits until the focus processor has decided
  whether it is a keyframe (its row leaves ``pending``);
* a frame of the same window as the latest row, less than
  :data:`MERGE_GAP_SECONDS` after it and with cosine >=
  :data:`MERGE_COSINE` to its vector, extends that row (``last_ts``,
  ``frames``) instead of adding one: one row per moment;
* the row's description is the annotation of one of its frames when one
  exists (:func:`annotation_description`, redacted), else the frame's
  screen text (``SceneAnnotator._build_visual_text_proxy``, redacted);
  an annotation that lands later replaces the screen text
  (:meth:`MemoryWriter.annotated`).

Embed before delete: until the writer has passed a frame, every path that
deletes screenshots after annotation (the scene annotator, the backlog
expiry, the startup orphan sweep) leaves its JPEG alone
(:meth:`MemoryWriter.holds`), and once the writer has embedded a frame
whose annotation is done it deletes the JPEG itself.  The wait is bounded:
a frame older than :data:`HOLD_HOURS` is given up on (not held, not
embedded).  Blocked and metadata-only frames are deleted at once by the
annotation path, as before; they are never held.

Embedding waits (frames stay held) while ``[memory]`` is paused, while the
model scheduler's memory/swap guard or a generation phase stands in the
way (:meth:`ModelScheduler.embedding_blocker`), while
``[memory] max_frames_per_hour`` is spent, and after an Ollama failure (a
backoff).  While paused (``<data dir>/status/memory-pause.json``) the
cursor moves past every frame: what is captured during a pause is never
embedded.

:class:`MemoryIndex` is the search side: the rows' vectors in a numpy
matrix, loaded lazily and refreshed incrementally, scored by dot product
against the query's vector, with a small boost for rows whose description
holds every word of the query, and adjacent rows of one window grouped
into moments.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from agenthandover_worker.dom import url_host
from agenthandover_worker.event_helpers import display_name, parse_json_field, parse_timestamp
from agenthandover_worker.model_profiles import EMBEDDING_PROFILES
from agenthandover_worker.ollama import OllamaClient, OllamaError
from agenthandover_worker.privacy_zones import ObservationTier, is_system_ui
from agenthandover_worker.redaction import redact
from agenthandover_worker.scene_annotator import SceneAnnotator, find_screenshot

logger = logging.getLogger(__name__)

# worker_state key: the rowid of the last event the writer has passed.
CURSOR_STATE_KEY = "memory_cursor"
# Pause state, written by the app and the CLI: {"paused_until": ISO | null}.
PAUSE_FILE = "memory-pause.json"
THUMB_SUFFIX = ".ahthumb"

# A frame extends the latest row when it shows the same window this close
# to it, at most this long after it.
MERGE_COSINE = 0.97
MERGE_GAP_SECONDS = 600.0
# How long Memory waits for a frame (a focus recording's processing, the
# model coming back, the hourly budget) before giving up on it.  Longer
# than the annotation backlog's 24 hours, so a frame the backlog expiry
# gives up on is embedded first; well inside the daemon's 7-day screenshot
# TTL.
HOLD_HOURS = 48.0
# Frames embedded per job: one job stays a few seconds, so P0 work (a
# stopped recording) never waits long behind it.
FRAMES_PER_JOB = 8
# Events read per job (frames passed over count against this, not the above).
EVENTS_PER_JOB = 200
# Per embedding request.
EMBED_TIMEOUT_SECONDS = 60.0
# After an Ollama failure: wait this long, doubling to the cap.
BACKOFF_SECONDS = 60.0
BACKOFF_MAX_SECONDS = 1800.0
# Descriptions are search results' text, not documents.
DESCRIPTION_MAX_CHARS = 1000
# Search: added to the score of a row whose description holds every word.
KEYWORD_BOOST = 0.05
# Search: rows of one window this close in time are one moment.
MOMENT_GAP_SECONDS = MERGE_GAP_SECONDS


# ---------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------


def pause_file(data_dir: Path) -> Path:
    return Path(data_dir) / "status" / PAUSE_FILE


def paused(data_dir: Path, now: datetime) -> bool:
    """True while ``memory-pause.json`` says Memory is paused at *now*.

    A missing or unreadable file is not paused; ``"paused_until": null``
    is paused until resumed."""
    try:
        state = json.loads(pause_file(data_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(state, dict) or "paused_until" not in state:
        return False
    until = state["paused_until"]
    if until is None:
        return True
    when = parse_timestamp(until)
    if when is None:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return now < when


def thumbs_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "thumbs"


def utc_text(moment: datetime) -> str:
    """*moment* as the UTC text memory rows store (millisecond precision,
    ``Z``), which sorts and compares as time."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def frame_window(event: dict) -> tuple[str, str, str]:
    """``(app, window title, url host)`` of a frame, the title redacted."""
    window = parse_json_field(event, "window_json")
    metadata = parse_json_field(event, "metadata_json")
    app = display_name(window.get("app_id")) or str(window.get("app_name") or "")
    title = redact(str(window.get("title") or ""))
    url = metadata.get("url")
    return app, title, url_host(url) if isinstance(url, str) else ""


def embed_text(app: str, title: str) -> str:
    """The text a frame is embedded with, beside its screenshot."""
    return f"title: {app} — {title} | text: "


def annotation_description(annotation: dict | None) -> str:
    """A frame's description from its annotation: what the user was doing,
    the app and the place in it, redacted ("" when it says nothing)."""
    if not isinstance(annotation, dict):
        return ""
    task = annotation.get("task_context")
    doing = str(task.get("what_doing") or "").strip() if isinstance(task, dict) else ""
    where = [
        str(annotation.get(key) or "").strip() for key in ("app", "location")
    ]
    where = [w for w in where if w]
    if not doing and not where:
        return ""
    text = doing + (f" ({', '.join(where)})" if where else "") if doing else ", ".join(where)
    return redact(text)[:DESCRIPTION_MAX_CHARS]


def proxy_description(event: dict) -> str:
    """A frame's description from its screen text, redacted."""
    proxy = SceneAnnotator._build_visual_text_proxy(event, None) or ""
    return proxy[:DESCRIPTION_MAX_CHARS]


def _completed_annotation(event: dict) -> dict | None:
    if event.get("annotation_status") != "completed":
        return None
    raw = event.get("scene_annotation_json")
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _focus_session(event: dict) -> str | None:
    value = parse_json_field(event, "metadata_json").get("focus_session_id")
    return str(value) if value else None


def _pack(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype="<f4").tobytes()


def _unit(vector: list[float] | np.ndarray) -> np.ndarray | None:
    v = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if v.ndim != 1 or v.size == 0 or not np.isfinite(norm) or norm == 0.0:
        return None
    return v / norm


def ollama_embedder(
    host: str, model: str, dimensions: int, *, prefix: str = "",
) -> Callable[[list[str | dict]], list[list[float]]]:
    """``items -> vectors`` through Ollama's ``/api/embed``: one request
    for the items given, truncated to *dimensions* and unit length."""
    client = OllamaClient(host)

    def embed(items: list[str | dict]) -> list[list[float]]:
        return client.embed(
            model, items, prefix=prefix, dimensions=dimensions,
            timeout=EMBED_TIMEOUT_SECONDS,
        )

    return embed


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


@dataclass
class _Latest:
    """The latest row, the one a frame may extend."""

    id: str
    app: str
    title: str
    last_ts: datetime
    vector: np.ndarray
    described: bool
    has_thumb: bool


class MemoryWriter:
    """Embeds kept frames into ``memory_frames`` (see the module docstring).

    *embed* turns ``[{"text", "image"}]`` into unit vectors of *config*'s
    dimensions (:func:`ollama_embedder` in the worker); *blocker* returns
    why no model may run now, or None; *privacy_checker* is the worker's
    ``PrivacyZoneChecker`` (None: no zones).
    """

    def __init__(
        self,
        db_path: str | Path,
        config: Any,
        *,
        embed: Callable[[list[str | dict]], list[list[float]]],
        screenshots_dir: Path,
        data_dir: Path,
        privacy_checker: Any = None,
        blocker: Callable[[], str | None] = lambda: None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._embed = embed
        self.screenshots_dir = Path(screenshots_dir)
        self.data_dir = Path(data_dir)
        self._privacy = privacy_checker
        self._blocker = blocker
        self._now = now
        self._monotonic = monotonic
        self._conn = sqlite3.connect(
            str(db_path), isolation_level=None, check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._latest: _Latest | None = None
        self._latest_loaded = False
        self._embedded_at: deque[float] = deque()
        self._failures = 0
        self._retry_at = 0.0
        # Why the last run stopped short of the newest frame (None: it did not).
        self.waiting: str | None = None

    def close(self) -> None:
        self._conn.close()

    # -- state ----------------------------------------------------------

    def active(self) -> bool:
        """True while Memory is on and not paused."""
        return bool(self.config.enabled) and not paused(self.data_dir, self._now())

    def cursor(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM worker_state WHERE key = ?", (CURSOR_STATE_KEY,),
        ).fetchone()
        try:
            return int(row[0]) if row else 0
        except (TypeError, ValueError):
            return 0

    def _set_cursor(self, rowid: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO worker_state (key, value) VALUES (?, ?)",
            (CURSOR_STATE_KEY, str(rowid)),
        )

    def _max_rowid(self) -> int:
        return int(self._conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM events").fetchone()[0])

    def _cutoff(self) -> datetime:
        return self._now() - timedelta(hours=HOLD_HOURS)

    # -- what Memory keeps ----------------------------------------------

    def _never_kept(self, event: dict) -> bool:
        """Frames Memory never embeds: privacy-zoned windows, system UI,
        a focus recording's dropped (non-key) frames."""
        status = event.get("annotation_status")
        if status in ("privacy_blocked", "metadata_only"):
            return True
        if is_system_ui(event):
            return True
        if self._privacy is not None and self._privacy.check_event(event) in (
            ObservationTier.BLOCKED, ObservationTier.METADATA_ONLY,
        ):
            return True
        return _focus_session(event) is not None and status == "skipped"

    def holds(self, event: dict) -> bool:
        """True while Memory still needs *event*'s screenshot: deleting it
        now would lose the frame before it is embedded."""
        if not self.active():
            return False
        row = self._conn.execute(
            "SELECT rowid, * FROM events WHERE id = ?", (event.get("id"),),
        ).fetchone()
        if row is None:
            return False
        stored = dict(row)
        if stored["rowid"] <= self.cursor():
            return False
        when = parse_timestamp(stored.get("timestamp"))
        if when is None or when < self._cutoff():
            return False
        return not self._never_kept(stored)

    def held_screenshot_paths(self) -> set[str]:
        """The JPEGs Memory still needs (for the startup orphan sweep)."""
        if not self.active():
            return set()
        rows = self._conn.execute(
            "SELECT rowid, * FROM events WHERE rowid > ? "
            "AND json_extract(metadata_json, '$.screenshot_path') IS NOT NULL",
            (self.cursor(),),
        ).fetchall()
        cutoff = self._cutoff()
        held: set[str] = set()
        for row in rows:
            event = dict(row)
            when = parse_timestamp(event.get("timestamp"))
            if when is None or when < cutoff or self._never_kept(event):
                continue
            path = parse_json_field(event, "metadata_json").get("screenshot_path")
            if isinstance(path, str) and path:
                held.add(str(Path(path).resolve()))
        return held

    # -- budget ---------------------------------------------------------

    def _budget_spent(self) -> bool:
        hour_ago = self._monotonic() - 3600.0
        while self._embedded_at and self._embedded_at[0] <= hour_ago:
            self._embedded_at.popleft()
        return len(self._embedded_at) >= self.config.max_frames_per_hour

    def _embed_blocker(self, *, probe: bool) -> str | None:
        """Why the next frame may not be embedded now, or None.  *probe*
        also asks the model scheduler (memory, swap, generation), once per
        run: its probes start processes."""
        if self._monotonic() < self._retry_at:
            return "Ollama failed; retrying after a backoff"
        if self._budget_spent():
            return f"{self.config.max_frames_per_hour} frames embedded in the last hour"
        return self._blocker() if probe else None

    # -- the walk -------------------------------------------------------

    def run(self) -> dict:
        """Embed the frames waiting, up to :data:`FRAMES_PER_JOB`.

        Returns counts: ``embedded``, ``merged`` (of those, frames that
        extended a row), ``passed`` (frames passed over: not kept, gone or
        given up on), and ``waiting`` (why the walk stopped short, or None).
        """
        stats: dict[str, Any] = {"embedded": 0, "merged": 0, "passed": 0, "waiting": None}
        if not self.config.enabled:
            return stats
        if paused(self.data_dir, self._now()):
            # Frames captured while paused are never embedded.
            self._set_cursor(self._max_rowid())
            self.waiting = None
            return stats
        cursor = self.cursor()
        newest = self._max_rowid()
        if cursor > newest:
            # The events table lost its newest rows (forgotten): rowids
            # will be handed out again from there.
            cursor = newest
            self._set_cursor(cursor)
        rows = self._conn.execute(
            "SELECT rowid, * FROM events WHERE rowid > ? "
            "AND json_extract(metadata_json, '$.screenshot_path') IS NOT NULL "
            "ORDER BY rowid LIMIT ?",
            (cursor, EVENTS_PER_JOB),
        ).fetchall()
        cutoff = self._cutoff()
        for row in rows:
            event = dict(row)
            rowid = int(event.pop("rowid"))
            when = parse_timestamp(event.get("timestamp"))
            if when is not None and when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            path = find_screenshot(event, self.screenshots_dir)
            if when is None or path is None or self._never_kept(event):
                self._pass(rowid, stats)
                continue
            if when < cutoff:
                # Waited too long: given up on, its JPEG freed if nothing
                # else needs it.
                self._release(event, path)
                self._pass(rowid, stats)
                continue
            if _focus_session(event) is not None and event.get("annotation_status") == "pending":
                stats["waiting"] = "a focus recording's frames are not processed yet"
                break
            if stats["embedded"] >= FRAMES_PER_JOB:
                break
            blocker = self._embed_blocker(probe=stats["embedded"] == 0)
            if blocker is not None:
                stats["waiting"] = blocker
                break
            try:
                merged = self._embed_frame(rowid, event, when, path)
            except OllamaError as exc:
                self._failures += 1
                delay = min(BACKOFF_SECONDS * 2 ** (self._failures - 1), BACKOFF_MAX_SECONDS)
                self._retry_at = self._monotonic() + delay
                stats["waiting"] = f"Ollama failed ({exc}); retrying in {delay:.0f} s"
                logger.warning("Memory embedding deferred: %s", stats["waiting"])
                break
            self._failures = 0
            self._embedded_at.append(self._monotonic())
            stats["embedded"] += 1
            stats["merged"] += int(merged)
            self._release(event, path)
        if stats["waiting"] != self.waiting and stats["waiting"] is not None:
            logger.info("Memory waiting: %s", stats["waiting"])
        self.waiting = stats["waiting"]
        if stats["embedded"]:
            logger.info(
                "Memory: %d frame(s) embedded (%d extended a moment), %d passed over",
                stats["embedded"], stats["merged"], stats["passed"],
            )
        return stats

    def _pass(self, rowid: int, stats: dict) -> None:
        self._set_cursor(rowid)
        stats["passed"] += 1

    def _release(self, event: dict, path: str) -> None:
        """Delete a frame's JPEG once its annotation (or the focus
        processor) no longer needs it."""
        if event.get("annotation_status") == "pending":
            return
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete screenshot %s", path, exc_info=True)

    def _embed_frame(self, rowid: int, event: dict, when: datetime, path: str) -> bool:
        """Embed one frame and write it (a new row, or the latest row
        extended) with the cursor, in one transaction.  True when merged."""
        app, title, host = frame_window(event)
        image = Path(path).read_bytes()
        vectors = self._embed([{"text": embed_text(app, title), "image": image}])
        vector = _unit(vectors[0]) if vectors else None
        if vector is None or vector.size != self.config.dimensions:
            raise _BadVector(
                f"{self.config.model} returned a vector of "
                f"{0 if vector is None else vector.size} values, not {self.config.dimensions}"
            )
        annotation = _completed_annotation(event)
        described = annotation_description(annotation)
        latest = self._load_latest()
        event_id = str(event["id"])
        has_thumb = (thumbs_dir(self.data_dir) / f"{event_id}{THUMB_SUFFIX}").is_file()
        merge = (
            latest is not None
            and latest.app == app
            and latest.title == title
            and 0.0 <= (when - latest.last_ts).total_seconds() < MERGE_GAP_SECONDS
            and float(np.dot(latest.vector, vector)) >= MERGE_COSINE
        )
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            if merge:
                assert latest is not None
                sets = ["frames = frames + 1", "last_ts = MAX(last_ts, ?)"]
                params: list[Any] = [utc_text(when)]
                if described and not latest.described:
                    sets += ["description = ?", "described_by = ?"]
                    params += [described, event_id]
                if has_thumb and not latest.has_thumb:
                    sets.append("thumb_id = ?")
                    params.append(event_id)
                self._conn.execute(
                    f"UPDATE memory_frames SET {', '.join(sets)} WHERE id = ?",
                    (*params, latest.id),
                )
            else:
                self._conn.execute(
                    "INSERT OR REPLACE INTO memory_frames (id, first_ts, last_ts, frames, "
                    "app, window_title, url_host, description, described_by, embedding, "
                    "model, dim, thumb_id, focus_session_id) "
                    "VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event_id, utc_text(when), utc_text(when), app, title, host,
                        described or proxy_description(event),
                        event_id if described else None,
                        _pack(vector), self.config.model, self.config.dimensions,
                        event_id if has_thumb else None, _focus_session(event),
                    ),
                )
            self._set_cursor(rowid)
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        if merge:
            assert latest is not None
            latest.last_ts = max(latest.last_ts, when)
            latest.described = latest.described or bool(described)
            if has_thumb and latest.has_thumb:
                # The moment shows its first thumbnail; this one is a copy.
                _unlink(thumbs_dir(self.data_dir) / f"{event_id}{THUMB_SUFFIX}")
            latest.has_thumb = latest.has_thumb or has_thumb
        else:
            self._latest = _Latest(
                id=event_id, app=app, title=title, last_ts=when, vector=vector,
                described=bool(described), has_thumb=has_thumb,
            )
        return merge

    def _load_latest(self) -> _Latest | None:
        if self._latest_loaded:
            return self._latest
        self._latest_loaded = True
        row = self._conn.execute(
            "SELECT id, app, window_title, last_ts, embedding, dim, described_by, thumb_id "
            "FROM memory_frames WHERE model = ? AND dim = ? "
            "ORDER BY last_ts DESC LIMIT 1",
            (self.config.model, self.config.dimensions),
        ).fetchone()
        if row is None:
            return None
        when = parse_timestamp(row["last_ts"])
        vector = _unit(np.frombuffer(row["embedding"], dtype="<f4"))
        if when is None or vector is None:
            return None
        self._latest = _Latest(
            id=row["id"], app=row["app"], title=row["window_title"], last_ts=when,
            vector=vector, described=row["described_by"] is not None,
            has_thumb=row["thumb_id"] is not None,
        )
        return self._latest

    # -- annotations ----------------------------------------------------

    def annotated(self, event: dict, annotation: dict) -> bool:
        """An annotation landed for *event*: it describes the moment the
        frame belongs to, unless an annotation already does.  True when a
        row changed."""
        described = annotation_description(annotation)
        when = parse_timestamp(event.get("timestamp"))
        if not described or when is None:
            return False
        ts = utc_text(when)
        cur = self._conn.execute(
            "UPDATE memory_frames SET description = ?, described_by = ? "
            "WHERE id = (SELECT id FROM memory_frames WHERE first_ts <= ? AND last_ts >= ? "
            "ORDER BY first_ts DESC LIMIT 1) AND described_by IS NULL",
            (described, str(event.get("id") or ""), ts, ts),
        )
        changed = cur.rowcount > 0
        if changed and self._latest is not None and self._latest.last_ts >= when:
            self._latest.described = True
        return changed

    # -- retention ------------------------------------------------------

    def purge(self) -> dict:
        """Daily: delete rows past ``[memory] retention_days`` with their
        thumbnails, and thumbnails past ``[memory] thumbnail_days``.
        Returns the counts."""
        now = self._now()
        rows_cutoff = utc_text(now - timedelta(days=self.config.retention_days))
        thumbs = thumbs_dir(self.data_dir)
        old = self._conn.execute(
            "SELECT id, thumb_id FROM memory_frames WHERE last_ts < ?", (rows_cutoff,),
        ).fetchall()
        for row in old:
            if row["thumb_id"]:
                _unlink(thumbs / f"{row['thumb_id']}{THUMB_SUFFIX}")
        self._conn.execute("DELETE FROM memory_frames WHERE last_ts < ?", (rows_cutoff,))
        if self._latest is not None and any(r["id"] == self._latest.id for r in old):
            self._latest = None
            self._latest_loaded = False

        expired_thumbs = 0
        thumb_cutoff = (now - timedelta(days=self.config.thumbnail_days)).timestamp()
        if thumbs.is_dir():
            gone: list[str] = []
            for path in thumbs.iterdir():
                if path.suffix != THUMB_SUFFIX or not path.is_file():
                    continue
                try:
                    if path.stat().st_mtime >= thumb_cutoff:
                        continue
                    path.unlink()
                except OSError:
                    logger.debug("Could not expire thumbnail %s", path, exc_info=True)
                    continue
                gone.append(path.stem)
                expired_thumbs += 1
            self._conn.executemany(
                "UPDATE memory_frames SET thumb_id = NULL WHERE thumb_id = ?",
                [(name,) for name in gone],
            )
            if self._latest is not None and self._latest.id in gone:
                self._latest.has_thumb = False
        if old or expired_thumbs:
            logger.info(
                "Memory retention: %d row(s) older than %d days, %d thumbnail(s) "
                "older than %d days deleted",
                len(old), self.config.retention_days, expired_thumbs,
                self.config.thumbnail_days,
            )
        return {"rows": len(old), "thumbnails": expired_thumbs}


class _BadVector(OllamaError):
    """The model answered with no vector or one of the wrong size."""


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.debug("Could not delete %s", path, exc_info=True)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _epoch(text: str) -> float:
    when = parse_timestamp(text)
    if when is None:
        return 0.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.timestamp()


_WORD = re.compile(r"\w{3,}", re.UNICODE)


class MemoryIndex:
    """Memory search over ``memory_frames`` rows of one model and size.

    *embed_query* turns the query text into a unit vector of *dim* values
    (:func:`query_embedder` in the product).  The rows' vectors are read
    into a float32 matrix on the first search and refreshed on each:
    rows added or extended since (``created_at``, ``last_ts``) are read
    again; when rows were deleted the matrix is read whole.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        model: str,
        dim: int,
        embed_query: Callable[[str], list[float]],
    ) -> None:
        self.model = model
        self.dim = dim
        self._embed_query = embed_query
        self._conn = sqlite3.connect(
            f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._reset()

    def close(self) -> None:
        self._conn.close()

    def _reset(self) -> None:
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        self._first = np.zeros(0)
        self._last = np.zeros(0)
        self._app: list[str] = []
        self._codes: dict[tuple[str, str], int] = {}
        self._window = np.zeros(0, dtype=np.int64)
        self._seen_created = ""
        self._seen_last = ""

    def __len__(self) -> int:
        return len(self._ids)

    def refresh(self) -> int:
        """Bring the matrix up to date with the table; returns the rows read."""
        where = "model = ? AND dim = ?"
        params: list[Any] = [self.model, self.dim]
        if self._ids:
            where += " AND (created_at >= ? OR last_ts >= ?)"
            params += [self._seen_created, self._seen_last]
        rows = self._conn.execute(
            "SELECT id, first_ts, last_ts, app, window_title, embedding, created_at "
            f"FROM memory_frames WHERE {where}",
            params,
        ).fetchall()
        self._upsert(rows)
        total = self._conn.execute(
            "SELECT count(*) FROM memory_frames WHERE model = ? AND dim = ?",
            (self.model, self.dim),
        ).fetchone()[0]
        if total != len(self._ids):
            # Rows were deleted (retention, forget): read them all again.
            self._reset()
            rows = self._conn.execute(
                "SELECT id, first_ts, last_ts, app, window_title, embedding, created_at "
                "FROM memory_frames WHERE model = ? AND dim = ?",
                (self.model, self.dim),
            ).fetchall()
            self._upsert(rows)
        return len(rows)

    def _upsert(self, rows: list[sqlite3.Row]) -> None:
        new_vectors: list[np.ndarray] = []
        new_first: list[float] = []
        new_last: list[float] = []
        new_window: list[int] = []
        for row in rows:
            vector = np.frombuffer(row["embedding"], dtype="<f4")
            if vector.size != self.dim:
                continue
            first, last = _epoch(row["first_ts"]), _epoch(row["last_ts"])
            key = (row["app"], row["window_title"])
            code = self._codes.setdefault(key, len(self._codes))
            self._seen_created = max(self._seen_created, row["created_at"])
            self._seen_last = max(self._seen_last, row["last_ts"])
            at = self._pos.get(row["id"])
            if at is not None:
                if at < self._first.size:
                    self._first[at], self._last[at] = first, last
                continue
            self._pos[row["id"]] = len(self._ids)
            self._ids.append(row["id"])
            self._app.append(row["app"])
            new_vectors.append(vector)
            new_first.append(first)
            new_last.append(last)
            new_window.append(code)
        if not new_vectors:
            return
        self._matrix = np.vstack([self._matrix, np.stack(new_vectors)])
        self._first = np.concatenate([self._first, new_first])
        self._last = np.concatenate([self._last, new_last])
        self._window = np.concatenate([self._window, np.asarray(new_window, dtype=np.int64)])

    def _keyword_rows(self, query: str) -> set[str]:
        """Rows whose description holds every word (3+ letters) of *query*."""
        words = list(dict.fromkeys(w.lower() for w in _WORD.findall(query)))
        if not words:
            return set()
        clauses = " AND ".join(["lower(description) LIKE ? ESCAPE '\\'"] * len(words))
        patterns = [
            "%" + w.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            for w in words
        ]
        rows = self._conn.execute(
            f"SELECT id FROM memory_frames WHERE model = ? AND dim = ? AND {clauses}",
            (self.model, self.dim, *patterns),
        ).fetchall()
        return {r[0] for r in rows}

    def search(
        self,
        query: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        app: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """The moments that best match *query*, best first.

        *since*/*until* keep moments that overlap the range; *app* keeps
        one app (case-insensitive).  Each moment is a run of rows of one
        window less than :data:`MOMENT_GAP_SECONDS` apart: ``id`` (its
        best row), ``first_ts``, ``last_ts``, ``frames``, ``app``,
        ``window_title``, ``url_host``, ``description`` and ``thumb_id``
        (of its best row; the first thumbnail kept when that row has
        none), ``score`` and ``rows`` (the row ids, in time order).
        """
        if not query.strip() or limit <= 0:
            return []
        self.refresh()
        if not self._ids:
            return []
        q = _unit(self._embed_query(query))
        if q is None or q.size != self.dim:
            raise ValueError(f"query vector is not {self.dim} values")
        mask = np.ones(len(self._ids), dtype=bool)
        if since is not None:
            mask &= self._last >= since.timestamp()
        if until is not None:
            mask &= self._first <= until.timestamp()
        if app:
            wanted = app.casefold()
            mask &= np.fromiter((a.casefold() == wanted for a in self._app), bool, len(self._app))
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            return []
        scores = self._matrix @ q
        boosted = self._keyword_rows(query)
        if boosted:
            for row_id in boosted:
                at = self._pos.get(row_id)
                if at is not None:
                    scores[at] += KEYWORD_BOOST
        order = idx[np.argsort(self._first[idx], kind="stable")]
        same = (
            (self._window[order][1:] == self._window[order][:-1])
            & (self._first[order][1:] - self._last[order][:-1] < MOMENT_GAP_SECONDS)
        )
        starts = np.concatenate([[0], np.nonzero(~same)[0] + 1])
        ends = np.concatenate([starts[1:], [order.size]])
        ordered_scores = scores[order]
        best = np.maximum.reduceat(ordered_scores, starts)
        top = np.argsort(-best, kind="stable")[:limit]
        groups = [order[starts[g]:ends[g]] for g in top]
        return self._moments(groups, scores)

    def _moments(self, groups: list[np.ndarray], scores: np.ndarray) -> list[dict]:
        ids = [self._ids[i] for group in groups for i in group]
        rows: dict[str, sqlite3.Row] = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            for row in self._conn.execute(
                "SELECT id, first_ts, last_ts, frames, app, window_title, url_host, "
                f"description, thumb_id FROM memory_frames WHERE id IN ({','.join('?' * len(chunk))})",
                chunk,
            ):
                rows[row["id"]] = row
        moments: list[dict] = []
        for group in groups:
            members = [self._ids[i] for i in group if self._ids[i] in rows]
            if not members:
                continue
            best_at = max(group, key=lambda i: scores[i])
            best = rows.get(self._ids[best_at]) or rows[members[0]]
            thumb = best["thumb_id"] or next(
                (rows[m]["thumb_id"] for m in members if rows[m]["thumb_id"]), None,
            )
            moments.append({
                "id": best["id"],
                "first_ts": min(rows[m]["first_ts"] for m in members),
                "last_ts": max(rows[m]["last_ts"] for m in members),
                "frames": sum(int(rows[m]["frames"]) for m in members),
                "app": best["app"],
                "window_title": best["window_title"],
                "url_host": best["url_host"],
                "description": best["description"],
                "thumb_id": thumb,
                "score": round(float(scores[best_at]), 4),
                "rows": members,
            })
        return moments


def query_embedder(host: str, model: str, dim: int) -> Callable[[str], list[float]]:
    """``query -> vector`` with the model's query prefix (its profile)."""
    profile = EMBEDDING_PROFILES.get(model)
    embed = ollama_embedder(
        host, model, dim, prefix=profile.query_prefix if profile else "",
    )
    return lambda query: embed([query])[0]
