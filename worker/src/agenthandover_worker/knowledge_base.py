"""Knowledge base manager for ~/.agenthandover/knowledge/.

Manages the persistent, file-based knowledge store that any AI agent can
read.  All artifacts are stored as JSON files using atomic writes
(tmp+fsync+rename) to prevent corruption on crash.

Directory structure::

    ~/.agenthandover/knowledge/
        procedures/          # v3 machine procedures (one JSON per slug)
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
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_KNOWLEDGE_DIR = Path.home() / ".agenthandover" / "knowledge"

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
    """Manages ~/.agenthandover/knowledge/ — atomic JSON read/write for all artifacts."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = Path(root) if root is not None else DEFAULT_KNOWLEDGE_DIR

    @property
    def root(self) -> Path:
        return self._root

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

    def save_procedure(self, procedure: dict) -> Path:
        """Save a v3 procedure JSON to the procedures directory.

        The file is named by the procedure's ``id`` (slug).  Overwrites
        any existing procedure with the same id.

        Returns the path to the written file.
        """
        slug = _sanitize_slug(
            procedure.get("id", procedure.get("slug", "unknown"))
        )
        path = self._root / _PROCEDURES_DIR / f"{slug}.json"
        self.atomic_write_json(path, procedure)
        logger.info("Saved procedure: %s", path)
        return path

    def get_procedure(self, slug: str) -> dict | None:
        """Load a procedure by slug, or return None if not found."""
        path = self._root / _PROCEDURES_DIR / f"{_sanitize_slug(slug)}.json"
        return self._read_json(path)

    def list_procedures(self) -> list[dict]:
        """List all procedures with summary info."""
        proc_dir = self._root / _PROCEDURES_DIR
        if not proc_dir.is_dir():
            return []
        results = []
        for p in sorted(proc_dir.glob("*.json")):
            data = self._read_json(p)
            if data is not None:
                results.append(data)
        return results

    def delete_procedure(self, slug: str) -> bool:
        """Delete a procedure by slug.  Returns True if deleted."""
        path = self._root / _PROCEDURES_DIR / f"{_sanitize_slug(slug)}.json"
        if path.is_file():
            path.unlink()
            logger.info("Deleted procedure: %s", slug)
            return True
        return False

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
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".kb_"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, default=str)
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
