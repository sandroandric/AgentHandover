"""Knowledge base manager for ~/.agenthandover/knowledge/.

Procedures live in the ``procedures`` table of the daemon's database
(created by ``crates/storage/src/migrations/v007_procedures.sql``); every
save also writes the procedure, byte for byte as stored, to
``procedures/<slug>.json``: the export the app and file-reading agents
read.  The other artifacts are JSON files in the directory.  Every file
is written atomically (tmp+fsync+rename) to prevent corruption on crash.

Directory structure::

    ~/.agenthandover/knowledge/
        procedures/          # export of the procedures table (one JSON per slug)
        observations/
            daily/           # daily summaries (YYYY-MM-DD.json)
            patterns/        # detected patterns (chains, recurrence)
        context/             # rolling context (recent.json, etc.)
        profile.json         # inferred user profile
        decisions.json       # extracted decision rules
        triggers.json        # detected triggers and recurrence
        constraints.json     # per-procedure and global guardrails
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from agenthandover_worker import paths
from agenthandover_worker.paths import KNOWLEDGE_BASE_DIR as DEFAULT_KNOWLEDGE_DIR

logger = logging.getLogger(__name__)

# Sub-directory names
_PROCEDURES_DIR = "procedures"
_OBSERVATIONS_DIR = "observations"
_DAILY_DIR = "daily"
_PATTERNS_DIR = "patterns"
_CONTEXT_DIR = "context"

# Top-level singleton files
_PROFILE_FILE = "profile.json"
_DECISIONS_FILE = "decisions.json"
_TRIGGERS_FILE = "triggers.json"
_CONSTRAINTS_FILE = "constraints.json"

# worker_state key set when a knowledge base's procedure files were
# imported into the procedures table (its value: when).
_IMPORT_MARKER = "procedures_imported_at"


def _sanitize_slug(slug: str) -> str:
    """Sanitize a slug to prevent path traversal.

    Strips directory separators, ``..`` components, and null bytes.
    Returns only the basename of the slug.
    """
    # Remove null bytes
    slug = slug.replace("\x00", "")
    # Take only the final component (strips any path separators)
    slug = Path(slug).name
    # Reject pure-dot names
    if not slug or slug in (".", ".."):
        return "unknown"
    return slug


class KnowledgeBase:
    """The knowledge base: procedures in the daemon's database (*db_path*,
    default ``paths.db_path()``), exported to and alongside the other
    artifacts in *root* (default ``~/.agenthandover/knowledge``)."""

    def __init__(self, root: Path | None = None, db_path: Path | None = None) -> None:
        self._root = Path(root) if root is not None else DEFAULT_KNOWLEDGE_DIR
        self._db_path = Path(db_path) if db_path is not None else paths.db_path()
        # list_procedures() cache: slug -> (stored text, parsed procedure)
        self._procedure_cache: dict[str, tuple[str, dict]] = {}
        # One connection, opened on first use and shared by the threads
        # that use the knowledge base (control, inference, query API) one
        # at a time under _lock.
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def db_path(self) -> Path:
        return self._db_path

    # ------------------------------------------------------------------
    # Directory structure
    # ------------------------------------------------------------------

    def ensure_structure(self) -> None:
        """Create the full knowledge directory tree if it doesn't exist."""
        dirs = [
            self._root,
            self._root / _PROCEDURES_DIR,
            self._root / _OBSERVATIONS_DIR,
            self._root / _OBSERVATIONS_DIR / _DAILY_DIR,
            self._root / _OBSERVATIONS_DIR / _PATTERNS_DIR,
            self._root / _CONTEXT_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.debug("Knowledge base structure ensured at %s", self._root)

    # ------------------------------------------------------------------
    # Procedures
    # ------------------------------------------------------------------

    def _connection(self, *, required: bool) -> sqlite3.Connection | None:
        """The read-write connection (call under ``_lock``), or None when
        the daemon has not created the database yet (*required*: raise
        instead).  Never creates the file: the daemon owns the database
        and its schema."""
        if self._conn is None:
            if not self._db_path.is_file():
                if required:
                    raise FileNotFoundError(
                        f"No AgentHandover database at {self._db_path}: the "
                        "daemon creates it on its first start (agenthandover start)."
                    )
                return None
            self._conn = sqlite3.connect(
                f"{self._db_path.resolve().as_uri()}?mode=rw",
                uri=True, timeout=10, check_same_thread=False,
            )
        return self._conn

    def _read(self, sql: str, params: tuple = ()) -> list[tuple]:
        """The rows of *sql*; none before the daemon created the database.
        Each read sees every write committed before it (no transaction
        is held between statements)."""
        with self._lock:
            conn = self._connection(required=False)
            return [] if conn is None else conn.execute(sql, params).fetchall()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """The connection inside one transaction (committed on success)."""
        with self._lock:
            conn = self._connection(required=True)
            with conn:
                yield conn

    def _procedure_path(self, slug: str) -> Path:
        return self._root / _PROCEDURES_DIR / f"{slug}.json"

    def _store_procedure(self, slug: str, procedure: dict, text: str) -> Path:
        """Write *text* (the JSON of *procedure*) as the row of *slug* and
        as its export file."""
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO procedures (slug, title, lifecycle_state, procedure_json, "
                "  updated_at) "
                "VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')) "
                "ON CONFLICT(slug) DO UPDATE SET title = excluded.title, "
                "  lifecycle_state = excluded.lifecycle_state, "
                "  procedure_json = excluded.procedure_json, "
                "  updated_at = excluded.updated_at",
                (
                    slug,
                    str(procedure.get("title") or ""),
                    str(procedure.get("lifecycle_state") or ""),
                    text,
                ),
            )
            # Inside the transaction and the lock: a failed export rolls
            # the row back, and two saves of one slug cannot leave the
            # file on the older text.  (A crash between the two is repaired
            # by export_procedures at the next start.)
            path = self._procedure_path(slug)
            self._atomic_write_text(path, text)
        return path

    def save_procedure(self, procedure: dict) -> Path:
        """Save a v3 procedure: its row in the ``procedures`` table, then
        its export file ``procedures/<id>.json``.

        The procedure is keyed by its ``id`` (slug).  Overwrites any
        existing procedure with the same id.

        Every save recomputes the procedure's learning ledger
        (``procedure["knowledge"]``, :func:`ledger.refresh_knowledge`) in
        place, so what is stored always says what is known, hypothesised
        and unknown about the procedure as saved.

        Returns the path to the export file.
        """
        from agenthandover_worker.ledger import refresh_knowledge

        refresh_knowledge(procedure)
        slug = _sanitize_slug(
            procedure.get("id", procedure.get("slug", "unknown"))
        )
        path = self._store_procedure(slug, procedure, _procedure_text(procedure))
        logger.info("Saved procedure: %s", path)
        return path

    def get_procedure(self, slug: str) -> dict | None:
        """Load a procedure by slug, or return None if not found."""
        rows = self._read(
            "SELECT procedure_json FROM procedures WHERE slug = ?", (_sanitize_slug(slug),),
        )
        return _parse_procedure(slug, rows[0][0]) if rows else None

    def list_procedures(self) -> list[dict]:
        """Every procedure, in the order of their export file names.

        Parsed rows are cached by their stored text: a call that finds no
        procedure changed parses nothing.  The returned dicts are shared
        with the cache, so callers must not modify them (edit a copy from
        :meth:`get_procedure` and save it instead).
        """
        # Ordered as the files were listed before procedures moved into the
        # database (by file name), so every consumer sees the same order.
        rows = self._read(
            "SELECT slug, procedure_json FROM procedures ORDER BY slug || '.json'"
        )
        cache = self._procedure_cache
        fresh: dict[str, tuple[str, dict]] = {}
        results = []
        for slug, text in rows:
            hit = cache.get(slug)
            data = hit[1] if hit is not None and hit[0] == text else _parse_procedure(slug, text)
            if data is not None:
                fresh[slug] = (text, data)
                results.append(data)
        self._procedure_cache = fresh
        return results

    def backfill_procedures(self) -> int:
        """Repair procedures saved before the schema field-mapping fix.

        Rewrites, in the table and the export, each procedure that
        ``procedure_schema.backfill_procedure_fields`` changes (empty
        ``description`` filled from ``task_description``; empty step
        app/location/input/verify filled from ``parameters``).

        Returns the number of procedures rewritten.
        """
        from agenthandover_worker.procedure_schema import backfill_procedure_fields

        rows = self._read("SELECT slug, procedure_json FROM procedures")
        count = 0
        for slug, text in rows:
            data = _parse_procedure(slug, text)
            if data is not None and backfill_procedure_fields(data):
                self._store_procedure(slug, data, _procedure_text(data))
                count += 1
        logger.info("Backfilled %d procedure(s) in %s", count, self._db_path)
        return count

    def delete_procedure(self, slug: str) -> bool:
        """Delete a procedure (its row and its export file).  Returns True
        if it existed."""
        slug = _sanitize_slug(slug)
        deleted = False
        with self._lock:
            conn = self._connection(required=False)
            if conn is not None:
                with conn:
                    deleted = conn.execute(
                        "DELETE FROM procedures WHERE slug = ?", (slug,),
                    ).rowcount > 0
            path = self._procedure_path(slug)
            if path.is_file():
                path.unlink()
                deleted = True
        if deleted:
            logger.info("Deleted procedure: %s", slug)
        return deleted

    def import_procedure_files(self) -> int:
        """Import the procedure files of a knowledge base from before
        procedures moved into the database (v0.4), once.

        Each ``procedures/*.json`` holding a JSON object becomes the row
        of its file name, with the file's text stored verbatim, so the
        file stays the procedure's export byte for byte.  A slug already
        in the table keeps its row.  The import and the ``worker_state``
        marker commit together: after it, no file is imported again (a
        procedure deleted later stays deleted).  Unreadable files are
        logged and left where they are.  Returns the number imported.
        """
        proc_dir = self._root / _PROCEDURES_DIR
        with self._transaction() as conn:
            if conn.execute(
                "SELECT 1 FROM worker_state WHERE key = ?", (_IMPORT_MARKER,),
            ).fetchone():
                return 0
            imported = 0
            files = sorted(proc_dir.glob("*.json")) if proc_dir.is_dir() else []
            for path in files:
                if path.name.startswith("."):
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                    data = json.loads(text)
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    logger.warning("Procedure file %s not imported: %s", path, exc)
                    continue
                if not isinstance(data, dict):
                    logger.warning("Procedure file %s not imported: not a JSON object", path)
                    continue
                imported += conn.execute(
                    "INSERT OR IGNORE INTO procedures "
                    "(slug, title, lifecycle_state, procedure_json) VALUES (?, ?, ?, ?)",
                    (
                        path.stem,
                        str(data.get("title") or ""),
                        str(data.get("lifecycle_state") or ""),
                        text,
                    ),
                ).rowcount
            conn.execute(
                "INSERT INTO worker_state (key, value) VALUES (?, ?)",
                (_IMPORT_MARKER, datetime.now(timezone.utc).isoformat()),
            )
        logger.info(
            "Imported %d procedure file(s) from %s into %s",
            imported, proc_dir, self._db_path,
        )
        return imported

    def export_procedures(self) -> int:
        """Rewrite each procedure's export file that is missing or differs
        from its row (a save interrupted between the two writes).  Files
        with no row are left alone.  Returns the number written."""
        written = 0
        with self._lock:  # no save runs between a row's read and its file write
            for slug, text in self._read("SELECT slug, procedure_json FROM procedures"):
                path = self._procedure_path(slug)
                try:
                    current = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    current = None
                if current != text:
                    self._atomic_write_text(path, text)
                    written += 1
        if written:
            logger.info("Re-exported %d procedure file(s) to %s", written, self._root)
        return written

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------

    def get_profile(self) -> dict:
        """Load the user profile, or return empty defaults."""
        data = self._read_json(self._root / _PROFILE_FILE)
        if data is None:
            return {
                "tools": {},
                "working_hours": {},
                "accounts": [],
                "communication_style": {},
                "updated_at": None,
            }
        return data

    def update_profile(self, updates: dict) -> None:
        """Merge *updates* into the existing profile."""
        profile = self.get_profile()
        profile.update(updates)
        profile["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.atomic_write_json(self._root / _PROFILE_FILE, profile)
        logger.info("Profile updated")

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    def get_decisions(self) -> dict:
        """Load decision rules, or return empty defaults."""
        data = self._read_json(self._root / _DECISIONS_FILE)
        if data is None:
            return {"decision_sets": [], "updated_at": None}
        return data

    def update_decisions(self, decisions: dict) -> None:
        """Replace decision rules."""
        decisions["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.atomic_write_json(self._root / _DECISIONS_FILE, decisions)
        logger.info("Decisions updated")

    # ------------------------------------------------------------------
    # Triggers
    # ------------------------------------------------------------------

    def get_triggers(self) -> dict:
        """Load triggers/recurrence data, or return empty defaults."""
        data = self._read_json(self._root / _TRIGGERS_FILE)
        if data is None:
            return {"recurrence": [], "chains": [], "updated_at": None}
        return data

    def update_triggers(self, triggers: dict) -> None:
        """Replace triggers data."""
        triggers["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.atomic_write_json(self._root / _TRIGGERS_FILE, triggers)
        logger.info("Triggers updated")

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def get_constraints(self) -> dict:
        """Load constraints, or return empty defaults."""
        data = self._read_json(self._root / _CONSTRAINTS_FILE)
        if data is None:
            return {
                "global": {},
                "per_procedure": {},
                "updated_at": None,
            }
        return data

    def update_constraints(self, constraints: dict) -> None:
        """Replace constraints data."""
        constraints["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.atomic_write_json(self._root / _CONSTRAINTS_FILE, constraints)
        logger.info("Constraints updated")

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def get_context(self, name: str) -> dict:
        """Load a named context file (e.g. 'recent'), or return empty dict."""
        path = self._root / _CONTEXT_DIR / f"{name}.json"
        data = self._read_json(path)
        return data if data is not None else {}

    def update_context(self, name: str, data: dict) -> None:
        """Write a named context file."""
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        path = self._root / _CONTEXT_DIR / f"{name}.json"
        self.atomic_write_json(path, data)

    # ------------------------------------------------------------------
    # Observations — daily summaries
    # ------------------------------------------------------------------

    def save_daily_summary(self, date: str, summary: dict) -> Path:
        """Save a daily summary.  *date* must be YYYY-MM-DD."""
        path = self._root / _OBSERVATIONS_DIR / _DAILY_DIR / f"{date}.json"
        summary["date"] = date
        summary["saved_at"] = datetime.now(timezone.utc).isoformat()
        self.atomic_write_json(path, summary)
        logger.info("Daily summary saved: %s", date)
        return path

    def get_daily_summary(self, date: str) -> dict | None:
        """Load a daily summary by date string (YYYY-MM-DD)."""
        path = self._root / _OBSERVATIONS_DIR / _DAILY_DIR / f"{date}.json"
        return self._read_json(path)

    def list_daily_summaries(self, limit: int = 30) -> list[str]:
        """Return the most recent *limit* daily summary dates (newest first)."""
        daily_dir = self._root / _OBSERVATIONS_DIR / _DAILY_DIR
        if not daily_dir.is_dir():
            return []
        dates = sorted(
            (p.stem for p in daily_dir.glob("*.json")),
            reverse=True,
        )
        return dates[:limit]

    def load_daily_summaries(self, limit: int = 30) -> list[dict]:
        """Load the most recent *limit* daily summaries as full dicts.

        Convenience wrapper around :meth:`list_daily_summaries` +
        :meth:`get_daily_summary` to avoid N+1 boilerplate in callers.
        Returns a list of summary dicts (newest first), skipping any
        that fail to load.
        """
        dates = self.list_daily_summaries(limit=limit)
        summaries: list[dict] = []
        for d in dates:
            s = self.get_daily_summary(d)
            if s is not None:
                summaries.append(s)
        return summaries

    # ------------------------------------------------------------------
    # Observations — patterns
    # ------------------------------------------------------------------

    def save_pattern(self, pattern_type: str, data: dict) -> Path:
        """Save a detected pattern (e.g. 'chains', 'recurrence')."""
        path = (
            self._root / _OBSERVATIONS_DIR / _PATTERNS_DIR
            / f"{pattern_type}.json"
        )
        data["pattern_type"] = pattern_type
        data["saved_at"] = datetime.now(timezone.utc).isoformat()
        self.atomic_write_json(path, data)
        logger.info("Pattern saved: %s", pattern_type)
        return path

    def get_pattern(self, pattern_type: str) -> dict | None:
        """Load a pattern by type name."""
        path = (
            self._root / _OBSERVATIONS_DIR / _PATTERNS_DIR
            / f"{pattern_type}.json"
        )
        return self._read_json(path)

    # ------------------------------------------------------------------
    # Atomic JSON I/O
    # ------------------------------------------------------------------

    def atomic_write_json(self, path: Path, data: dict) -> None:
        """Write *data* as JSON to *path* using tmp+fsync+rename.

        This is the same pattern used by ``main.py`` for crash-safe
        writes: write to a temporary file in the same directory,
        fsync, then rename (atomic on POSIX).

        This is a public method — modules that persist state through
        the knowledge base directory should call this directly rather
        than reimplementing the pattern.
        """
        self._atomic_write_text(path, json.dumps(data, indent=2, default=str))

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        """Write *text* to *path* using tmp+fsync+rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".kb_"
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp_path, str(path))
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _read_json(self, path: Path) -> dict | None:
        """Read a JSON file, returning None if missing or invalid."""
        if not path.is_file():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.warning("Expected dict in %s, got %s", path, type(data).__name__)
                return None
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return None


def _procedure_text(procedure: dict) -> str:
    """A procedure as stored and exported (the format of every knowledge
    base file)."""
    return json.dumps(procedure, indent=2, default=str)


def _parse_procedure(slug: str, text: str) -> dict | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Procedure %s is not valid JSON: %s", slug, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("Procedure %s is not a JSON object", slug)
        return None
    return data
