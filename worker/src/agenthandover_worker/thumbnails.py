"""Encrypted Memory thumbnails (2.0): expiry and deletion.

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
from typing import TYPE_CHECKING, Iterable

from agenthandover_worker import paths
from agenthandover_worker.event_helpers import parse_timestamp

if TYPE_CHECKING:
    from agenthandover_worker.db import WorkerDB

logger = logging.getLogger("agenthandover_worker")

EXTENSION = ".ahthumb"
# The daemon writes a thumbnail just before it hands the frame's event to
# its storage writer: a file this young may name an event not inserted yet.
ORPHAN_MIN_AGE_SECONDS = 300


def thumbs_dir(data_dir: Path | None = None) -> Path:
    """``<data dir>/thumbs``."""
    return Path(data_dir if data_dir is not None else paths.data_dir()) / "thumbs"


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
    event's timestamp, and those that neither an event nor a Memory row
    names any more (purged or forgotten; a Memory row outlives its raw
    event, so the row keeps its thumbnail).  A temp file a crash left
    behind goes too.  Files younger than ``ORPHAN_MIN_AGE_SECONDS`` are
    kept unless expired by age.  Returns how many files were deleted."""
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
        shown = db.memory_thumb_ids(
            [event_id for event_id in candidates if event_id not in stamps]
        )
        for event_id, mtime in candidates.items():
            if event_id in stamps:
                stamp = stamps[event_id]
                if stamp is not None and stamp <= cutoff:
                    expired.append(event_id)
            elif event_id not in shown and mtime <= settled_epoch:
                expired.append(event_id)

    removed += delete_thumbnails(expired, directory)
    if removed:
        logger.info("Removed %d Memory thumbnail(s) from %s", removed, directory)
    return removed
