"""Encrypted Memory thumbnails (v0.6): expiry and deletion.

The daemon writes ``<data dir>/thumbs/<event_id>.ahthumb`` (AES-256-GCM
under the app's Keychain key) for each kept frame while ``[memory]`` is on;
the app decrypts them for display.  The worker never reads them: it only
deletes them, once they are older than ``[memory] thumbnail_days`` (daily
maintenance) and when the user forgets a range (``delete_thumbnails``).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from agenthandover_worker import paths
from agenthandover_worker.event_helpers import parse_timestamp

if TYPE_CHECKING:
    from agenthandover_worker.db import WorkerDB

logger = logging.getLogger("agenthandover_worker")

EXTENSION = ".ahthumb"
DEFAULT_THUMBNAIL_DAYS = 7
# The daemon writes a thumbnail just before it hands the frame's event to
# its storage writer: a file this young may name an event not inserted yet.
ORPHAN_MIN_AGE_SECONDS = 300


def thumbs_dir() -> Path:
    """``<data dir>/thumbs``."""
    return paths.data_dir() / "thumbs"


def thumbnail_days(raw_config: dict[str, Any]) -> int:
    """``[memory] thumbnail_days`` from the parsed config.toml: a whole
    number of days >= 0 (0 keeps none), else the default (7)."""
    memory = raw_config.get("memory")
    value = memory.get("thumbnail_days") if isinstance(memory, dict) else None
    if value is None:
        return DEFAULT_THUMBNAIL_DAYS
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        logger.warning(
            "[memory] thumbnail_days = %r is not a whole number of days >= 0; using %d",
            value, DEFAULT_THUMBNAIL_DAYS,
        )
        return DEFAULT_THUMBNAIL_DAYS
    return value


def delete_thumbnails(event_ids: Iterable[str], directory: Path | None = None) -> int:
    """Delete the thumbnails of *event_ids*.  Returns how many were deleted."""
    directory = directory if directory is not None else thumbs_dir()
    removed = 0
    for event_id in event_ids:
        # Event ids are UUIDs; never follow anything else out of the dir.
        if not event_id or "/" in event_id or event_id.startswith("."):
            continue
        try:
            (directory / f"{event_id}{EXTENSION}").unlink()
            removed += 1
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Could not delete thumbnail of %s", event_id, exc_info=True)
    return removed


def expire_thumbnails(
    db: "WorkerDB",
    *,
    days: int,
    directory: Path | None = None,
    now: datetime | None = None,
) -> int:
    """Delete thumbnails older than *days*: by the file's mtime, or by its
    event's timestamp, and those whose event no longer exists (purged or
    forgotten).  A temp file a crash left behind goes too.  Files younger
    than ``ORPHAN_MIN_AGE_SECONDS`` are kept unless expired by age.
    Returns how many files were deleted."""
    directory = directory if directory is not None else thumbs_dir()
    if not directory.is_dir():
        return 0
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    cutoff_epoch = cutoff.timestamp()
    settled_epoch = now.timestamp() - ORPHAN_MIN_AGE_SECONDS

    expired: list[str] = []
    candidates: dict[str, float] = {}  # event id -> mtime
    removed = 0
    for path in directory.iterdir():
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if path.name.endswith(f"{EXTENSION}.tmp") and path.name.startswith("."):
            if mtime <= settled_epoch:
                path.unlink(missing_ok=True)
                removed += 1
            continue
        if path.suffix != EXTENSION or not path.is_file():
            continue
        if mtime <= cutoff_epoch:
            expired.append(path.stem)
        else:
            candidates[path.stem] = mtime

    if candidates:
        stamps = {
            row["id"]: parse_timestamp(row.get("timestamp"))
            for row in db.get_events_by_ids(list(candidates))
        }
        for event_id, mtime in candidates.items():
            if event_id in stamps:
                stamp = stamps[event_id]
                if stamp is not None and stamp <= cutoff:
                    expired.append(event_id)
            elif mtime <= settled_epoch:
                expired.append(event_id)

    removed += delete_thumbnails(expired, directory)
    if removed:
        logger.info("Removed %d Memory thumbnail(s) from %s", removed, directory)
    return removed
