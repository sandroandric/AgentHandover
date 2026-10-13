"""Memory over the query API: search, status, pause, resume and forget.

``MemoryService`` answers the ``/memory/*`` endpoints of
:mod:`~agenthandover_worker.query_api` (the contract the app and the CLI
build against).  Each method takes the request body and returns
``(HTTP status, JSON object)``:

* ``search`` — ``{"query", "since"?, "until"?, "app"?, "limit"? 1..100
  (20)}`` → ``{"moments": [...], "model"}``; 503 ``memory_off`` while
  ``[memory] enabled = false``; 503 ``model_unavailable`` (with ``detail``)
  when the query cannot be embedded.  One :class:`~memory.MemoryIndex`
  serves every search (it refreshes itself from the table).
* ``status`` — what Memory holds and whether it runs.
* ``pause`` — ``{"minutes"?: int > 0 | null}`` writes the pause file
  (null or absent: until resumed); ``resume`` deletes it; both answer the
  status.
* ``forget`` — exactly one selector (:func:`memory_forget.parse_selector`)
  → the counts.  It runs under the worker's state lock, so no job is
  half-way through the rows it deletes.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Callable

from agenthandover_worker import memory, memory_forget
from agenthandover_worker.config import load_config
from agenthandover_worker.inference_runner import STATE_LOCK
from agenthandover_worker.ollama import OllamaError
from agenthandover_worker.thumbnails import EXTENSION as THUMB_EXTENSION
from agenthandover_worker.thumbnails import thumbs_dir

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 20
MAX_LIMIT = 100
# How long a forget waits for the worker's running job to let go of the
# state lock (a model call releases it at once).
FORGET_LOCK_TIMEOUT_SECONDS = 120.0
# The query's embedding: an app or a terminal is waiting.
QUERY_EMBED_TIMEOUT_SECONDS = 15.0

Reply = tuple[int, dict]


def iso(moment: datetime | None) -> str | None:
    """*moment* as ``YYYY-MM-DDTHH:MM:SSZ`` (None stays None)."""
    if moment is None:
        return None
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class MemoryService:
    """The ``/memory/*`` endpoints (module docstring).

    *config* is the worker's ``Config`` (``[memory]`` as the worker
    started with it); ``agents_access`` is read again from *config_path*
    on each status, as the MCP server does.  *writer* is the worker's
    ``MemoryWriter`` (its ``waiting`` and the latest row it extends);
    *vector_kb* the worker's vector store (its index is dropped after a
    forget).  *embed_query* defaults to Ollama with the model's query
    prefix.
    """

    def __init__(
        self,
        config: Any,
        *,
        db_path: Path,
        data_dir: Path,
        knowledge_dir: Path,
        config_path: Path | None = None,
        writer: Any = None,
        vector_kb: Any = None,
        embed_query: Callable[[str], list[float]] | None = None,
        local_tz: tzinfo | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.config = config
        self.memory = config.memory
        self.db_path = Path(db_path)
        self.data_dir = Path(data_dir)
        self.knowledge_dir = Path(knowledge_dir)
        self.config_path = config_path if config_path is not None else config.path
        self.writer = writer
        self.vector_kb = vector_kb
        self._embed_query = embed_query or memory.query_embedder(
            config.vlm.ollama_host, self.memory.model, self.memory.dimensions,
            timeout=QUERY_EMBED_TIMEOUT_SECONDS,
        )
        self.local_tz = local_tz
        self._now = now
        self._index: memory.MemoryIndex | None = None
        self._index_lock = threading.Lock()

    def close(self) -> None:
        if self._index is not None:
            self._index.close()

    # -- search ---------------------------------------------------------

    def _memory_index(self) -> memory.MemoryIndex:
        if self._index is None:
            self._index = memory.MemoryIndex(
                self.db_path, model=self.memory.model, dim=self.memory.dimensions,
                embed_query=self._embed_query,
            )
        return self._index

    def moments(self, query: str, *, keyword: bool = False, **filters: Any) -> list[dict]:
        """The index's moments for *query* (``MemoryIndex.search``; with
        *keyword*, ``memory.keyword_search``), under the index lock.
        *filters*: ``since``, ``until``, ``app``, ``limit``."""
        with self._index_lock:
            index = self._memory_index()
            if keyword:
                return memory.keyword_search(index, query, **filters)
            return index.search(query, **filters)

    def search(self, body: Any) -> Reply:
        if not isinstance(body, dict):
            return 400, {"error": "the body must be a JSON object"}
        query = body.get("query")
        if not isinstance(query, str) or not query.strip():
            return 400, {"error": "'query' must be a non-empty string"}
        limit = body.get("limit", DEFAULT_LIMIT)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
            return 400, {"error": f"'limit' must be an integer from 1 to {MAX_LIMIT}"}
        app = body.get("app")
        if app is not None and (not isinstance(app, str) or not app.strip()):
            return 400, {"error": "'app' must be a non-empty string"}
        try:
            since = memory_forget.parse_time(body["since"], "since") if body.get("since") is not None else None
            until = memory_forget.parse_time(body["until"], "until") if body.get("until") is not None else None
        except ValueError as exc:
            return 400, {"error": str(exc)}
        if since is not None and until is not None and since > until:
            return 400, {"error": "'since' is after 'until'"}
        if not self.memory.enabled:
            return 503, {"error": "memory_off"}
        try:
            moments = self.moments(query, since=since, until=until, app=app, limit=limit)
        except (OllamaError, ConnectionError, TimeoutError, ValueError) as exc:
            logger.warning("Memory search: the query could not be embedded: %s", exc)
            return 503, {"error": "model_unavailable", "detail": str(exc)}
        for moment in moments:
            moment.pop("rows", None)
        return 200, {"moments": moments, "model": self.memory.model}

    # -- status, pause, resume ------------------------------------------

    def _agents_access(self) -> bool:
        if self.config_path is None:
            return bool(self.memory.agents_access)
        return bool(load_config(self.config_path).memory.agents_access)

    def status(self) -> Reply:
        paused, until = memory.pause_until(self.data_dir, self._now())
        conn = sqlite3.connect(f"{self.db_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            moments, frames, oldest, newest, row_bytes = conn.execute(
                "SELECT count(*), coalesce(sum(frames), 0), min(first_ts), max(last_ts), "
                "coalesce(sum(length(embedding) + length(description) + length(window_title) "
                "+ length(app) + length(url_host) + length(id)), 0) FROM memory_frames"
            ).fetchone()
        finally:
            conn.close()
        thumbs, thumb_bytes = 0, 0
        directory = thumbs_dir(self.data_dir)
        if directory.is_dir():
            for path in directory.glob(f"*{THUMB_EXTENSION}"):
                try:
                    thumb_bytes += path.stat().st_size
                    thumbs += 1
                except OSError:
                    continue
        return 200, {
            "enabled": bool(self.memory.enabled),
            "paused": paused,
            "paused_until": iso(until),
            "agents_access": self._agents_access(),
            "model": self.memory.model,
            "moments": moments,
            "frames": frames,
            "oldest": iso(memory_forget.parse_time(oldest, "oldest")) if oldest else None,
            "newest": iso(memory_forget.parse_time(newest, "newest")) if newest else None,
            "thumbnails": thumbs,
            "disk_bytes": row_bytes + thumb_bytes,
            "waiting": getattr(self.writer, "waiting", None),
        }

    def pause(self, body: Any) -> Reply:
        if body is None:
            body = {}
        if not isinstance(body, dict) or set(body) - {"minutes"}:
            return 400, {"error": "the body is {\"minutes\": a whole number > 0, or null}"}
        minutes = body.get("minutes")
        if minutes is not None and (
            isinstance(minutes, bool) or not isinstance(minutes, int) or minutes <= 0
        ):
            return 400, {"error": "'minutes' must be a whole number > 0, or null"}
        until = None if minutes is None else self._now() + timedelta(minutes=minutes)
        memory.write_pause(self.data_dir, until)
        logger.info("Memory paused %s", f"until {iso(until)}" if until else "until resumed")
        return self.status()

    def resume(self) -> Reply:
        memory.clear_pause(self.data_dir)
        logger.info("Memory resumed")
        return self.status()

    # -- forget ---------------------------------------------------------

    def forget(self, body: Any) -> Reply:
        try:
            selector = memory_forget.parse_selector(body)
        except ValueError as exc:
            return 400, {"error": str(exc)}
        if not STATE_LOCK.acquire(timeout=FORGET_LOCK_TIMEOUT_SECONDS):
            return 503, {"error": "busy", "detail": "the worker is busy; try again in a minute"}
        try:
            counts = memory_forget.forget(
                selector, db_path=self.db_path, data_dir=self.data_dir,
                knowledge_dir=self.knowledge_dir, local_tz=self.local_tz,
            )
            # What was read before the forget is read again.
            if self.writer is not None:
                self.writer.forgotten()
            if self.vector_kb is not None:
                self.vector_kb.forgotten()
        except LookupError as exc:
            return 404, {"error": str(exc)}
        finally:
            STATE_LOCK.release()
        with self._index_lock:
            if self._index is not None:
                self._index.reload()
        return 200, counts
