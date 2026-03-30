"""Cleanup module — time-based record expiration.

Provides functions to clean up stale data from the worker's
operational records, such as clipboard preview records older
than a configurable TTL (default 24 hours).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def purge_old_clipboard_previews(
    events: list[dict],
    ttl_hours: float = 24.0,
    now: datetime | None = None,
) -> list[dict]:
    """Remove clipboard preview records older than *ttl_hours*.

    Clipboard preview events are ``ClipboardChange`` events that carry
    preview content.  This function filters them out when they are
    older than the TTL to comply with privacy requirements.

    Parameters
    ----------
    events:
        List of event dicts to filter.
    ttl_hours:
        Maximum age in hours for clipboard preview records.
        Records older than this are removed.
    now:
        Current time for testing.  Defaults to ``datetime.now(utc)``.

    Returns
    -------
    list[dict]
        Events with stale clipboard previews removed.
    """
    if not events:
        return []

    if now is None:
        now = datetime.now(timezone.utc)

    cutoff = now - timedelta(hours=ttl_hours)
    kept: list[dict] = []

    for event in events:
        if _is_expired_clipboard_preview(event, cutoff):
            eid = event.get("id", "unknown")
            logger.debug("Purging expired clipboard preview: %s", eid)
            continue
        kept.append(event)

    purged_count = len(events) - len(kept)
    if purged_count > 0:
        logger.info(
            "Purged %d clipboard preview(s) older than %.1f hours",
            purged_count,
            ttl_hours,
        )

    return kept


def _is_expired_clipboard_preview(event: dict, cutoff: datetime) -> bool:
    """Check if event is an expired clipboard preview record."""
    # Must be a ClipboardChange event
    kind_json = event.get("kind_json", "")
    if not kind_json:
        return False

    try:
        parsed = json.loads(kind_json) if isinstance(kind_json, str) else kind_json
    except (json.JSONDecodeError, TypeError):
        return False

    if not isinstance(parsed, dict) or "ClipboardChange" not in parsed:
        return False

    # Must have a preview in metadata
    metadata_json = event.get("metadata_json", "")
    if metadata_json:
        try:
            metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        if not isinstance(metadata, dict):
            metadata = {}

        # Only purge if metadata contains a preview field
        if "content_preview" not in metadata and "preview" not in metadata:
            return False
    else:
        return False

    # Check timestamp against cutoff
    ts_raw = event.get("timestamp")
    if not ts_raw or not isinstance(ts_raw, str):
        return False

    try:
        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return False

    return ts < cutoff
