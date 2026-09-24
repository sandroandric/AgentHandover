"""Trigger spool: the one control channel from the app and CLI to the worker.

Every producer (SwiftUI app, Rust CLI) writes one file per request to
``<state_dir>/triggers/<kind>-<uuid>.json`` via tmp + rename.  The worker
drains the directory once per poll tick, oldest first, and dispatches each
payload to the handler registered for its ``kind``.

One file per request replaces the old single-slot ``<kind>-trigger.json``
files, where a second click before the worker's next tick overwrote the
first request and the user's action was silently lost.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Mapping

logger = logging.getLogger(__name__)

SPOOL_DIRNAME = "triggers"

# Every kind a producer may write.  The legacy single-slot file for a kind
# was ``<kind>-trigger.json`` in the state dir.
KINDS: tuple[str, ...] = (
    "retry",
    "approve",
    "failed-query",
    "search-query",
    "recall-query",
    "trust-accept",
    "trust-dismiss",
    "staleness-reviewed",
    "lifecycle-promote",
    "merge",
    "drift-reviewed",
    "lifecycle-dismiss",
    "export",
    "micro-answer",
)

_FILENAME_RE = re.compile(
    r"^(?P<kind>.+)-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}"
    r"-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\.json$"
)


def spool_dir(state_dir: Path) -> Path:
    return state_dir / SPOOL_DIRNAME


def kind_of(path: Path) -> str | None:
    """Return the kind encoded in a spool filename, or None if malformed."""
    match = _FILENAME_RE.match(path.name)
    return match.group("kind") if match else None


def write_trigger(state_dir: Path, kind: str, payload: dict) -> Path:
    """Write one trigger into the spool atomically and return its path."""
    directory = spool_dir(state_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{kind}-{uuid.uuid4()}.json"
    fd, tmp = tempfile.mkstemp(dir=str(directory), prefix=".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def pending(state_dir: Path) -> list[Path]:
    """Spooled trigger files, oldest first (mtime, then name)."""
    directory = spool_dir(state_dir)
    if not directory.is_dir():
        return []
    files = []
    for path in directory.glob("*.json"):
        try:
            files.append((path.stat().st_mtime_ns, path.name, path))
        except FileNotFoundError:
            continue
    return [path for _, _, path in sorted(files)]


def drain(
    state_dir: Path,
    handlers: Mapping[str, Callable[[dict], None]],
) -> int:
    """Consume every spooled trigger; return how many were consumed.

    Each file is removed before its handler runs, so a handler that crashes
    the worker cannot wedge the spool into a restart loop; a handler's
    exception is logged and the drain continues with the next file.
    Unreadable files and unknown kinds are logged and removed.
    """
    consumed = 0
    for path in pending(state_dir):
        kind = kind_of(path)
        try:
            with open(path) as f:
                payload = json.load(f)
        except FileNotFoundError:
            continue
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            logger.warning("Discarding unreadable trigger %s", path.name)
            payload = None
        path.unlink(missing_ok=True)
        consumed += 1

        if not isinstance(payload, dict):
            if payload is not None:
                logger.warning("Discarding non-object trigger %s", path.name)
            continue
        handler = handlers.get(kind) if kind else None
        if handler is None:
            logger.warning("Discarding trigger with unknown kind: %s", path.name)
            continue
        try:
            handler(payload)
        except Exception:
            logger.exception("Trigger handler for %s failed", path.name)
    return consumed


def migrate_legacy(state_dir: Path) -> int:
    """Move leftover single-slot ``<kind>-trigger.json`` files into the spool.

    Runs once at worker startup so a request written by the previous
    version just before an upgrade is still honoured.
    """
    moved = 0
    for kind in KINDS:
        legacy = state_dir / f"{kind}-trigger.json"
        if not legacy.is_file():
            continue
        directory = spool_dir(state_dir)
        directory.mkdir(parents=True, exist_ok=True)
        os.replace(legacy, directory / f"{kind}-{uuid.uuid4()}.json")
        moved += 1
    if moved:
        logger.info("Moved %d legacy trigger file(s) into the spool", moved)
    return moved
