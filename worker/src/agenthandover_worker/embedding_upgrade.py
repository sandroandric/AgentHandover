"""Upgrade everything: re-embed the vector store with EmbeddingGemma 2, then
switch ``[embedding] model`` to it.

An upgrader who answers the Memory offer with "Upgrade everything" (the app
writes ``"choice": "everything"`` into ``<data dir>/memory-offer.json``,
:data:`~agenthandover_worker.model_profiles.MEMORY_OFFER_FILE`) moves every
text path (Skill search, dedup, the procedure matcher, the task segmenter,
the session linker) from nomic-embed-text to :data:`TARGET_MODEL`.  New
installs never need this: setup writes ``[embedding] model`` with it on an
empty database.

* **Staging.**  ``vector_store``'s key is the source, so a source cannot
  hold an old and a new vector at once.  The re-embed writes the new
  model's vectors into ``vector_store_next`` (migration v009, same ``id``),
  while every search, link and match keeps using ``vector_store`` with the
  old model: no ranking ever mixes models.
* **Work.**  A low-priority job on the inference thread
  (``stages/upgrade.py``, P2), :data:`ROWS_PER_JOB` rows per job, waiting
  while the model scheduler's memory guard or a generation phase stands in
  the way (``ModelScheduler.embedding_blocker``).  Each ``vector_store``
  row is re-embedded from its source's text, rebuilt by the builder that
  made it (:meth:`EmbeddingUpgrade.source_text`), with the new model's
  prefix of the same kind (document, or clustering for the task
  segmenter's frame vectors).  A source whose text can no longer be rebuilt
  (its event expired, its procedure was deleted) is staged without a
  vector, dropped at the switch and counted.
* **Resume.**  The staging table is the progress: a ``vector_store`` row is
  done when ``vector_store_next`` holds its id with ``from_hash`` equal to
  its ``text_hash``.  A restart, Ollama going away mid-way, or a source
  changing meanwhile (its row's ``text_hash`` moves on) only means the rows
  not done are taken next.
* **Switch.**  When none is left, one transaction replaces ``vector_store``'s
  rows by the staged ones (the old model's rows go with it), empties the
  staging table and records ``"switched"`` in ``worker_state``
  (:data:`STATE_KEY`).  Then the worker's objects switch in process
  (``context.use_embedding_model``), ``[embedding] model`` is written into
  config.toml (``model_profiles.switch_embedding_file``: backup, TOML
  re-check, every other byte kept) and the state becomes ``"done"``.  A
  restart between the transaction and the config write finishes the
  switch before the first tick (``context.start``).  The worker is not
  restarted for it (its launchd job does not restart a clean exit).
* **Failure.**  Ollama older than 0.40 or without the model: state
  ``"failed"`` with an error saying what to do, nothing switched, asked
  again every :data:`RETRY_SECONDS`.  Ollama not answering: the state keeps
  the error and the job backs off.  Either way the user stays on the old
  model with nothing broken.

:meth:`EmbeddingUpgrade.status` is ``worker-status.json``'s
``"embedding_upgrade"``: ``{"state": "pending" | "running" | "failed" |
"done", "done", "total", "dropped", "error"}`` (``"switched"`` shows as
``"running"``).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

from agenthandover_worker.model_profiles import (
    MEMORY_MODEL,
    MEMORY_OFFER_FILE,
    MINIMUM_OLLAMA_VERSION_FOR_MEMORY,
    embedding_profile,
    set_embedding_model,
    switch_embedding_file,
    version_at_least,
)
from agenthandover_worker.ollama import OllamaClient, OllamaError
from agenthandover_worker.vector_kb import (
    _encode_embedding,
    _normalize,
    _quantize,
    _text_hash,
)

logger = logging.getLogger(__name__)

TARGET_MODEL = MEMORY_MODEL
# The memory-offer answer that asks for this upgrade.
CHOICE = "everything"
# ``worker_state`` key: {"state", "error", "dropped"}.
STATE_KEY = "embedding_upgrade"

ROWS_PER_JOB = 64
# Texts per ``/api/embed`` request.
BATCH_SIZE = 32
EMBED_TIMEOUT_SECONDS = 120.0
# EmbeddingGemma is released soon after each job (memory.EMBED_KEEP_ALIVE).
EMBED_KEEP_ALIVE = "1m"
# Ollama not answering: first wait, doubled per failure up to the maximum.
BACKOFF_SECONDS = 60.0
BACKOFF_MAX_SECONDS = 900.0
# Ollama too old or without the model: asked again this often.
RETRY_SECONDS = 600.0

# The vector_store rows not re-embedded yet: no staged row for their
# current text.
_PENDING = (
    "SELECT v.id, v.source_type, v.source_id, v.text_hash FROM vector_store v "
    "LEFT JOIN vector_store_next n ON n.id = v.id AND n.from_hash = v.text_hash "
    "WHERE n.id IS NULL ORDER BY v.rowid LIMIT ?"
)
# Staged rows whose vector_store row is gone.
_ORPHANS = "DELETE FROM vector_store_next WHERE id NOT IN (SELECT id FROM vector_store)"


def offer_choice(data_dir: Path) -> str | None:
    """The Memory offer's answer in ``memory-offer.json``, or None."""
    try:
        state = json.loads((Path(data_dir) / MEMORY_OFFER_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    choice = state.get("choice") if isinstance(state, dict) else None
    return choice if isinstance(choice, str) else None


class EmbeddingUpgrade:
    """The re-embed and the switch (module docstring).

    *knowledge_base* rebuilds procedure, daily summary and profile texts;
    events and preserved text are read from *db_path*.  *model* is the
    ``[embedding] model`` in use; *config_file* the config.toml the switch
    writes.  *preflight* returns why the new model cannot run (Ollama too
    old, the model not downloaded) or None, and raises
    :class:`OllamaError` when Ollama does not answer; the default asks
    Ollama.  *client* (default: an ``OllamaClient`` for *ollama_host*)
    embeds.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        knowledge_base: Any,
        data_dir: Path,
        config_file: Path,
        ollama_host: str,
        model: str,
        blocker: Callable[[], str | None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        rows_per_job: int = ROWS_PER_JOB,
        batch_size: int = BATCH_SIZE,
        preflight: Callable[[], str | None] | None = None,
        client: Any = None,
    ) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA busy_timeout = 5000;")
        self._kb = knowledge_base
        self.data_dir = Path(data_dir)
        self.config_file = Path(config_file)
        self.model = model
        self._client = client or OllamaClient(ollama_host)
        self._blocker = blocker
        self._monotonic = monotonic
        self.rows_per_job = rows_per_job
        self.batch_size = batch_size
        self._preflight = preflight or self._ollama_problem
        self._ready = False
        self._retry_at = 0.0
        self._failures = 0
        # Why the last job did nothing (the scheduler's guard), or None.
        self.waiting: str | None = None
        self._status: dict | None = None
        self._refresh_status()

    # -- the trigger and the state ------------------------------------

    def wanted(self) -> bool:
        """The user asked for the upgrade and it has not happened yet.  Once
        done it is never wanted again: a user who sets ``[embedding] model``
        back by hand keeps their choice."""
        if self.model == TARGET_MODEL or offer_choice(self.data_dir) != CHOICE:
            return False
        return (self.saved_state() or {}).get("state") != "done"

    def saved_state(self) -> dict | None:
        row = self._conn.execute(
            "SELECT value FROM worker_state WHERE key = ?", (STATE_KEY,),
        ).fetchone()
        if row is None:
            return None
        try:
            saved = json.loads(row[0])
        except ValueError:
            return None
        return saved if isinstance(saved, dict) else None

    def _set_state(self, state: str, error: str | None = None, dropped: int | None = None) -> None:
        saved = self.saved_state() or {}
        value = {
            "state": state,
            "error": error,
            "dropped": saved.get("dropped", 0) if dropped is None else dropped,
        }
        if saved != value:
            self._conn.execute(
                "INSERT OR REPLACE INTO worker_state (key, value) VALUES (?, ?)",
                (STATE_KEY, json.dumps(value, sort_keys=True)),
            )
        self._refresh_status()

    def status(self) -> dict | None:
        """``worker-status.json``'s ``"embedding_upgrade"``, or None when
        the upgrade was never asked for (the module docstring)."""
        return self._status

    def _refresh_status(self) -> None:
        saved = self.saved_state()
        if saved is None and not self.wanted():
            self._status = None
            return
        state = (saved or {}).get("state", "pending")
        if state == "switched":
            done = total = self._count("SELECT COUNT(*) FROM vector_store")
        else:
            total = self._count("SELECT COUNT(*) FROM vector_store")
            done = (
                total if state == "done" else self._count(
                    "SELECT COUNT(*) FROM vector_store v JOIN vector_store_next n "
                    "ON n.id = v.id AND n.from_hash = v.text_hash"
                )
            )
        self._status = {
            "state": "running" if state == "switched" else state,
            "done": done,
            "total": total,
            "dropped": int((saved or {}).get("dropped", 0)),
            "error": (saved or {}).get("error"),
        }

    def _count(self, sql: str) -> int:
        return int(self._conn.execute(sql).fetchone()[0])

    # -- the job ------------------------------------------------------

    def step(self) -> bool:
        """Re-embed up to :attr:`rows_per_job` rows; once none is left,
        replace ``vector_store``'s rows by the staged ones.  True when the
        rows are switched (now or by an earlier run) and the caller must
        switch the worker's objects and call :meth:`complete`."""
        if self.model == TARGET_MODEL:
            return False
        saved = self.saved_state() or {}
        if saved.get("state") == "switched":
            return True
        if not self.wanted():
            self._abandon(saved)
            return False
        # A source deleted since it was staged (forgotten, purged, a
        # deleted procedure): its new vector goes too.
        self._conn.execute(_ORPHANS)
        if self._monotonic() < self._retry_at:
            return False
        self.waiting = self._blocker() if self._blocker is not None else None
        if self.waiting is not None:
            return False
        if not self._ready:
            try:
                problem = self._preflight()
            except OllamaError as exc:
                self._back_off(saved, f"Ollama is not answering: {exc}")
                return False
            if problem is not None:
                if saved.get("error") != problem:
                    logger.warning("Embedding upgrade waits: %s", problem)
                self._set_state("failed", problem)
                self._retry_at = self._monotonic() + RETRY_SECONDS
                return False
            self._ready = True

        rows = self._conn.execute(_PENDING, (self.rows_per_job,)).fetchall()
        if not rows:
            return self._switch_rows()
        try:
            self._embed(rows)
        except OllamaError as exc:
            # Ask again before the next attempt: the model may be gone.
            self._ready = False
            self._back_off(saved, f"Ollama failed: {exc}")
            return False
        self._failures = 0
        if saved.get("state") != "running":
            logger.info(
                "Embedding upgrade: re-embedding %d vectors with %s in the background",
                self._status["total"] if self._status else 0, TARGET_MODEL,
            )
        self._set_state("running")
        logger.debug(
            "Embedding upgrade: %d of %d vectors re-embedded",
            self._status["done"], self._status["total"],
        )
        return False

    def _back_off(self, saved: dict, error: str) -> None:
        self._failures += 1
        delay = min(BACKOFF_SECONDS * 2 ** (self._failures - 1), BACKOFF_MAX_SECONDS)
        self._retry_at = self._monotonic() + delay
        logger.warning("Embedding upgrade deferred %.0f s: %s", delay, error)
        state = saved.get("state")
        self._set_state(state if state in ("running", "failed") else "pending", error)

    def _embed(self, rows: list[tuple]) -> None:
        """Embed *rows*' rebuilt texts with the new model and stage them;
        sources with no text are staged without a vector."""
        profile = embedding_profile(TARGET_MODEL)
        texts: dict[str, list[tuple[tuple, str]]] = {}
        gone: list[tuple] = []
        for row in rows:
            built = self.source_text(row[1], row[2])
            if built is None:
                gone.append(row)
            else:
                text, prefix = built
                texts.setdefault(prefix, []).append((row, text))
        vectors: list[tuple[tuple, str, str, list[float]]] = []
        for prefix, items in texts.items():
            embeddings = self._client.embed(
                TARGET_MODEL, [text for _, text in items],
                prefix=prefix, dimensions=profile.dimensions,
                batch_size=self.batch_size, timeout=EMBED_TIMEOUT_SECONDS,
                keep_alive=EMBED_KEEP_ALIVE,
            )
            for (row, text), emb in zip(items, embeddings):
                if emb:
                    vectors.append((row, text, prefix, emb))
                else:
                    logger.warning(
                        "Embedding upgrade: %s gave no vector for %s %s; dropped",
                        TARGET_MODEL, row[1], row[2],
                    )
                    gone.append(row)
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            for row, text, prefix, emb in vectors:
                unit = _normalize(emb)
                qvec, scale = _quantize(unit)
                self._stage(
                    row, (_encode_embedding(unit), TARGET_MODEL, len(unit),
                          _text_hash(text), prefix, qvec, scale),
                )
            for row in gone:
                self._stage(row, (None, TARGET_MODEL, 0, "", "", None, None))
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        if gone:
            logger.info(
                "Embedding upgrade: %d source(s) whose text is gone will be dropped",
                len(gone),
            )

    def _stage(self, row: tuple, values: tuple) -> None:
        vid, source_type, source_id, from_hash = row
        self._conn.execute(
            "INSERT OR REPLACE INTO vector_store_next "
            "(id, source_type, source_id, from_hash, embedding, model, dim, "
            " text_hash, prefix, qvec, qscale) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (vid, source_type, source_id, from_hash, *values),
        )

    def _switch_rows(self) -> bool:
        """The switch's transaction (module docstring).  False (nothing
        changed) when a row turned out not done, or when config.toml could
        not take the new model afterwards (state ``failed``)."""
        problem = self._config_problem()
        if problem is not None:
            self._set_state("failed", problem)
            self._retry_at = self._monotonic() + RETRY_SECONDS
            return False
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            # Sources deleted since they were staged stay deleted.
            self._conn.execute(_ORPHANS)
            if self._conn.execute(_PENDING, (1,)).fetchone() is not None:
                self._conn.execute("ROLLBACK")
                return False
            dropped = self._count("SELECT COUNT(*) FROM vector_store_next WHERE embedding IS NULL")
            self._conn.execute("DELETE FROM vector_store")
            self._conn.execute(
                "INSERT INTO vector_store "
                "(id, source_type, source_id, embedding, model, dim, text_hash, "
                " prefix, qvec, qscale) "
                "SELECT id, source_type, source_id, embedding, model, dim, text_hash, "
                " prefix, qvec, qscale FROM vector_store_next WHERE embedding IS NOT NULL"
            )
            self._conn.execute("DELETE FROM vector_store_next")
            self._conn.execute(
                "INSERT OR REPLACE INTO worker_state (key, value) VALUES (?, ?)",
                (STATE_KEY, json.dumps(
                    {"state": "switched", "error": None, "dropped": dropped}, sort_keys=True,
                )),
            )
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._refresh_status()
        logger.info(
            "Embedding upgrade: vector store switched to %s (%d vectors, %d dropped)",
            TARGET_MODEL, self._status["total"], dropped,
        )
        return True

    def _config_problem(self) -> str | None:
        """Why config.toml cannot take ``[embedding] model`` = the new
        model, or None (checked before the rows switch)."""
        try:
            text = self.config_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError) as exc:
            return f"cannot read {self.config_file}: {exc}"
        if set_embedding_model(text, TARGET_MODEL) is None:
            return (
                f"{self.config_file} cannot be edited safely: set "
                f'[embedding] model = "{TARGET_MODEL}" by hand'
            )
        return None

    def complete(self) -> bool:
        """After the rows switched and the worker's objects with them:
        write ``[embedding] model`` into config.toml and record ``done``.
        False (state stays switched, the error recorded, retried by the
        next job or the next start) when the file cannot be written."""
        try:
            backup = switch_embedding_file(self.config_file, TARGET_MODEL)
        except (OSError, ValueError) as exc:
            logger.error("Embedding upgrade: could not write %s: %s", self.config_file, exc)
            self._set_state("switched", f"could not write {self.config_file}: {exc}")
            return False
        self.model = TARGET_MODEL
        self._set_state("done")
        logger.info(
            "Embedding upgrade done: [embedding] model = %s in %s%s",
            TARGET_MODEL, self.config_file, f" (backup: {backup})" if backup else "",
        )
        return True

    def _abandon(self, saved: dict) -> None:
        """The offer's answer is no longer "everything": the staged rows
        go and the state is forgotten."""
        if not saved or saved.get("state") == "done":
            return
        self._conn.execute("DELETE FROM vector_store_next")
        self._conn.execute("DELETE FROM worker_state WHERE key = ?", (STATE_KEY,))
        self._refresh_status()
        logger.info("Embedding upgrade abandoned: the Memory offer's answer changed")

    def _ollama_problem(self) -> str | None:
        version = self._client.version()
        if not version_at_least(version, MINIMUM_OLLAMA_VERSION_FOR_MEMORY):
            return (
                f"{TARGET_MODEL} needs Ollama {MINIMUM_OLLAMA_VERSION_FOR_MEMORY} "
                f"or newer (this Mac has {version or 'an unknown version'}): update Ollama"
            )
        if not self._client.has_model(TARGET_MODEL):
            return f"{TARGET_MODEL} is not downloaded: download it (ollama pull {TARGET_MODEL})"
        return None

    # -- the texts ----------------------------------------------------

    def source_text(self, source_type: str, source_id: str) -> tuple[str, str] | None:
        """``(text, prefix)`` a source is embedded from with the new model,
        built by the same code that embedded it, or None when it cannot be
        rebuilt.  The prefix is the new model's document prefix, or its
        clustering prefix for the task segmenter's frame vectors."""
        from agenthandover_worker import main as wm
        from agenthandover_worker.stages import daily
        from agenthandover_worker.task_segmenter import AnnotatedFrame, frame_text

        profile = embedding_profile(TARGET_MODEL)
        text = ""
        prefix = profile.doc_prefix
        if source_type == "procedure":
            procedure = self._kb.get_procedure(source_id)
            if procedure:
                text = wm._procedure_text(procedure)[:4000]
        elif source_type == "daily_summary":
            summary = self._kb.get_daily_summary(source_id)
            if summary:
                tasks = summary.get("tasks") or []
                text = daily.summary_text(
                    source_id,
                    int(summary.get("task_count") or 0),
                    float(summary.get("active_hours") or 0.0),
                    [t.get("intent") or "" for t in tasks if isinstance(t, dict)],
                )[:4000]
        elif source_type == "profile":
            profile_doc = self._kb.get_profile()
            if profile_doc:
                text = daily._profile_text(profile_doc)[:4000]
        elif source_type == "preserved_text":
            event_id, _, source = source_id.rpartition(":")
            row = self._conn.execute(
                "SELECT text_content FROM preserved_text WHERE event_id = ? AND source = ?",
                (event_id, source),
            ).fetchone()
            if row is not None:
                text = (row[0] or "")[:2000]
        elif source_type == "annotation":
            self._conn.row_factory = sqlite3.Row
            try:
                event = self._conn.execute(
                    "SELECT * FROM events WHERE id = ?", (source_id,),
                ).fetchone()
            finally:
                self._conn.row_factory = None
            frame = AnnotatedFrame.from_event(dict(event)) if event is not None else None
            if frame is not None:
                text = frame_text(frame)
                prefix = profile.cluster_prefix
        if not text:
            return None
        return text, prefix

    def close(self) -> None:
        self._conn.close()
