"""Clipboard Copy-Paste Linker — match paste hashes to copy hashes.

Implements section 5.7 of the AgentHandover spec: identify clipboard
copy-paste pairs by matching SHA-256 content hashes within a
configurable time window (default 30 minutes).

Event format (``crates/common/src/event.rs``).  The kind and its payload
live in ``kind_json``::

    kind_json = {
        "type": "ClipboardChange",
        "content_types": [...],
        "byte_size": N,
        "content_hash": "sha256...",
        "high_entropy": bool,
    }

    kind_json = {
        "type": "PasteDetected",
        "matched_copy_hash": "sha256..." | null,
    }

The daemon emits ``PasteDetected`` for a ``paste`` event the browser
extension saw; it hashes the pasteboard at that moment (the same hash the
clipboard poller stores on ``ClipboardChange``) and sets
``matched_copy_hash`` when that copy was recorded within 30 minutes
(``crates/daemon/src/capture/paste.rs``).  A paste with no match carries
``null`` and is never linked.

``metadata_json`` holds unrelated fields (e.g. ``focus_session_id``)
and is not read here.

``timestamp`` is ISO 8601 format.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ClipboardLink:
    """A matched copy-paste pair."""

    copy_event_id: str
    paste_event_id: str
    content_hash: str
    time_delta_seconds: float


class ClipboardLinker:
    """Scan events for copy-paste pairs based on hash matching within a time window.

    Parameters
    ----------
    window_minutes:
        Maximum elapsed time (in minutes) between a copy and a paste
        for the pair to be linked.  Default 30 minutes.
    """

    def __init__(self, window_minutes: float = 30.0) -> None:
        self.window_minutes = window_minutes

    def find_links(self, events: list[dict]) -> list[ClipboardLink]:
        """Scan *events* chronologically and return copy-paste links.

        For each ``PasteDetected`` event whose ``matched_copy_hash`` equals
        the ``content_hash`` of a prior ``ClipboardChange`` event within the
        time window, a ``ClipboardLink`` is emitted.  When several copies
        share the hash, the most recent copy before the paste is linked.
        """
        if not events:
            return []

        # content_hash -> [(event_id, timestamp)] in event order
        copy_index: dict[str, list[tuple[str, datetime]]] = {}
        links: list[ClipboardLink] = []

        for event in events:
            kind_json = self._parse_kind_json(event)
            if kind_json is None:
                continue
            kind = kind_json.get("type")
            ts = self._parse_timestamp(event)
            if ts is None:
                continue
            eid = event.get("id", "")

            if kind == "ClipboardChange":
                content_hash = kind_json.get("content_hash")
                if content_hash:
                    copy_index.setdefault(str(content_hash), []).append((eid, ts))

            elif kind == "PasteDetected":
                content_hash = kind_json.get("matched_copy_hash")
                if not content_hash:
                    continue
                content_hash = str(content_hash)
                for copy_eid, copy_ts in reversed(copy_index.get(content_hash, [])):
                    delta = (ts - copy_ts).total_seconds()
                    if delta < 0:
                        continue  # a copy after the paste
                    if delta <= self.window_minutes * 60:
                        links.append(ClipboardLink(
                            copy_event_id=copy_eid,
                            paste_event_id=eid,
                            content_hash=content_hash,
                            time_delta_seconds=delta,
                        ))
                    break

        return links

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_kind_json(event: dict) -> dict | None:
        """Parse ``kind_json`` into a dict.  Returns None on any failure."""
        kind_json = event.get("kind_json", "")
        if not kind_json:
            return None
        try:
            parsed = (
                json.loads(kind_json)
                if isinstance(kind_json, str)
                else kind_json
            )
        except (json.JSONDecodeError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _parse_timestamp(event: dict) -> datetime | None:
        raw = event.get("timestamp")
        if not raw or not isinstance(raw, str):
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
