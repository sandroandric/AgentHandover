"""Clipboard Copy-Paste Linker — match paste hashes to copy hashes.

Implements section 5.7 of the AgentHandover spec: identify clipboard
copy-paste pairs by matching SHA-256 content hashes within a
configurable time window (default 30 minutes).

Event format (from daemon's event model).  Both the kind AND the
clipboard payload live inside ``kind_json``::

    kind_json = {
        "type": "ClipboardChange",
        "content_types": [...],
        "byte_size": N,
        "content_hash": "sha256...",
        "high_entropy": bool,
    }

    kind_json = {
        "type": "PasteDetected",
        "content_hash": "sha256...",
        "target_app": "...",
        "byte_size": N,
    }

``metadata_json`` holds unrelated fields (e.g. ``focus_session_id``)
and should NEVER be read here.  Historically these helpers read
``metadata_json`` because the event schema changed and nobody updated
the worker — meaning no clipboard copy/paste link has been produced
in passive discovery since that migration.  See tests in
``test_clipboard_linker.py`` for the regression coverage.

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

        For each ``PasteDetected`` event whose ``content_hash`` matches
        a prior ``ClipboardChange`` event within the time window, a
        ``ClipboardLink`` is emitted.  When multiple copies share the
        same hash, the most recent copy before the paste is linked.
        """
        if not events:
            return []

        # Map content_hash -> list of (event_id, timestamp, content_length) ordered by time
        # We store all copies so we can pick the most recent one for a paste
        copy_index: dict[str, list[tuple[str, datetime, int]]] = {}
        links: list[ClipboardLink] = []

        for event in events:
            kind = self._extract_kind(event)
            ts = self._parse_timestamp(event)
            eid = event.get("id", "")

            if kind == "ClipboardChange":
                content_hash = self._extract_hash(event)
                if content_hash and ts:
                    content_length = self._extract_content_length(event)
                    copy_index.setdefault(content_hash, []).append((eid, ts, content_length))

            elif kind == "PasteDetected":
                content_hash = self._extract_hash(event)
                if not content_hash or not ts:
                    continue

                copies = copy_index.get(content_hash)
                if not copies:
                    continue

                paste_length = self._extract_content_length(event)

                # Find the most recent copy that is within the time window
                best_copy: tuple[str, datetime] | None = None
                for copy_eid, copy_ts, copy_len in reversed(copies):
                    delta = (ts - copy_ts).total_seconds()
                    if delta < 0:
                        # Paste before copy — skip
                        continue
                    if delta <= self.window_minutes * 60:
                        # Secondary verification: if both have content_length,
                        # they should match (skip with warning if they differ)
                        if paste_length > 0 and copy_len > 0:
                            if abs(paste_length - copy_len) > max(paste_length, copy_len) * 0.1:
                                logger.warning(
                                    "Hash match but content_length mismatch: "
                                    "copy=%d paste=%d (copy_eid=%s, paste_eid=%s)",
                                    copy_len, paste_length, copy_eid, eid,
                                )
                                continue
                        best_copy = (copy_eid, copy_ts, copy_len)
                        break

                if best_copy is not None:
                    copy_eid, copy_ts, _ = best_copy
                    delta_seconds = (ts - copy_ts).total_seconds()
                    links.append(
                        ClipboardLink(
                            copy_event_id=copy_eid,
                            paste_event_id=eid,
                            content_hash=content_hash,
                            time_delta_seconds=delta_seconds,
                        )
                    )

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
    def _extract_kind(event: dict) -> str:
        """Extract the event kind name from ``kind_json``.

        The daemon serializes events as ``{"type": "ClipboardChange", ...}``.
        Return the VALUE of the ``type`` field — NOT the first key
        (historical bug: ``next(iter(parsed))`` returned ``"type"`` which
        never matched any event kind in this module).
        """
        parsed = ClipboardLinker._parse_kind_json(event)
        if parsed is None:
            return ""
        kind = parsed.get("type", "")
        return str(kind) if kind else ""

    @staticmethod
    def _extract_content_length(event: dict) -> int:
        """Extract ``byte_size`` from ``kind_json``.

        Historical bug: this was reading ``metadata_json`` which is always
        ``{}`` for clipboard events, so the length was always 0 and the
        content-length safety check in ``find_links`` never actually
        compared real lengths.
        """
        parsed = ClipboardLinker._parse_kind_json(event)
        if parsed is None:
            return 0
        byte_size = parsed.get("byte_size", 0)
        try:
            return int(byte_size)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _extract_hash(event: dict) -> str:
        """Extract ``content_hash`` from ``kind_json``.

        Historical bug: this was reading ``metadata_json`` which doesn't
        contain ``content_hash``, so no copy/paste pair has ever been
        linked in the passive pipeline since the event schema migration.
        """
        parsed = ClipboardLinker._parse_kind_json(event)
        if parsed is None:
            return ""
        h = parsed.get("content_hash", "")
        return str(h) if h else ""

    @staticmethod
    def _parse_timestamp(event: dict) -> datetime | None:
        raw = event.get("timestamp")
        if not raw or not isinstance(raw, str):
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
