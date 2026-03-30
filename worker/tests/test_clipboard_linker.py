"""Tests for agenthandover_worker.clipboard_linker.

Covers hash matching, time-window expiry, most-recent-copy selection,
and edge cases (empty input, copy without paste).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone

from agenthandover_worker.clipboard_linker import ClipboardLink, ClipboardLinker


def _ts(dt: datetime) -> str:
    """Format a datetime as the ISO 8601 string the daemon produces."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _hash(content: str) -> str:
    """Produce a deterministic SHA-256 hex digest for test content."""
    return hashlib.sha256(content.encode()).hexdigest()


def _make_clipboard_event(
    *,
    kind: str,
    content_hash: str,
    timestamp: str,
    event_id: str | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    """Build a ClipboardChange or PasteDetected event dict."""
    eid = event_id or str(uuid.uuid4())
    metadata: dict = {"content_hash": content_hash}
    if kind == "ClipboardChange":
        metadata["content_types"] = ["text/plain"]
        metadata["byte_size"] = 42
    elif kind == "PasteDetected":
        metadata["target_app"] = "com.apple.TextEdit"
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "id": eid,
        "timestamp": timestamp,
        "kind_json": json.dumps({kind: {}}),
        "window_json": json.dumps({"app_id": "com.example.App", "title": "Test"}),
        "metadata_json": json.dumps(metadata),
        "display_topology_json": "[]",
        "primary_display_id": "main",
        "processed": 0,
    }


# ------------------------------------------------------------------
# 1. Matching copy-paste
# ------------------------------------------------------------------


class TestMatchingCopyPaste:
    def test_matching_copy_paste(self) -> None:
        """Copy then paste with same hash within window → 1 link."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        h = _hash("hello world")

        events = [
            _make_clipboard_event(
                kind="ClipboardChange",
                content_hash=h,
                timestamp=_ts(base),
                event_id="copy-1",
            ),
            _make_clipboard_event(
                kind="PasteDetected",
                content_hash=h,
                timestamp=_ts(base + timedelta(minutes=5)),
                event_id="paste-1",
            ),
        ]

        linker = ClipboardLinker()
        links = linker.find_links(events)

        assert len(links) == 1
        link = links[0]
        assert link.copy_event_id == "copy-1"
        assert link.paste_event_id == "paste-1"
        assert link.content_hash == h
        assert abs(link.time_delta_seconds - 300.0) < 1.0


# ------------------------------------------------------------------
# 2. No match — different hash
# ------------------------------------------------------------------


class TestNoMatchDifferentHash:
    def test_no_match_different_hash(self) -> None:
        """Different hashes → no link."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        h1 = _hash("content A")
        h2 = _hash("content B")

        events = [
            _make_clipboard_event(
                kind="ClipboardChange",
                content_hash=h1,
                timestamp=_ts(base),
            ),
            _make_clipboard_event(
                kind="PasteDetected",
                content_hash=h2,
                timestamp=_ts(base + timedelta(minutes=5)),
            ),
        ]

        linker = ClipboardLinker()
        links = linker.find_links(events)

        assert len(links) == 0


# ------------------------------------------------------------------
# 3. Expired window
# ------------------------------------------------------------------


class TestExpiredWindow:
    def test_expired_window(self) -> None:
        """Paste 31 minutes after copy → no link."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        h = _hash("data")

        events = [
            _make_clipboard_event(
                kind="ClipboardChange",
                content_hash=h,
                timestamp=_ts(base),
            ),
            _make_clipboard_event(
                kind="PasteDetected",
                content_hash=h,
                timestamp=_ts(base + timedelta(minutes=31)),
            ),
        ]

        linker = ClipboardLinker()
        links = linker.find_links(events)

        assert len(links) == 0


# ------------------------------------------------------------------
# 4. Within window
# ------------------------------------------------------------------


class TestWithinWindow:
    def test_within_window(self) -> None:
        """Paste 29 minutes after copy → 1 link."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        h = _hash("data")

        events = [
            _make_clipboard_event(
                kind="ClipboardChange",
                content_hash=h,
                timestamp=_ts(base),
            ),
            _make_clipboard_event(
                kind="PasteDetected",
                content_hash=h,
                timestamp=_ts(base + timedelta(minutes=29)),
            ),
        ]

        linker = ClipboardLinker()
        links = linker.find_links(events)

        assert len(links) == 1
        assert abs(links[0].time_delta_seconds - (29 * 60)) < 1.0


# ------------------------------------------------------------------
# 5. Multiple copies — links to most recent
# ------------------------------------------------------------------


class TestMultipleCopiesLinksMostRecent:
    def test_multiple_copies_links_most_recent(self) -> None:
        """Two copies with same hash, one paste → links to most recent copy."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        h = _hash("shared content")

        events = [
            _make_clipboard_event(
                kind="ClipboardChange",
                content_hash=h,
                timestamp=_ts(base),
                event_id="copy-old",
            ),
            _make_clipboard_event(
                kind="ClipboardChange",
                content_hash=h,
                timestamp=_ts(base + timedelta(minutes=10)),
                event_id="copy-new",
            ),
            _make_clipboard_event(
                kind="PasteDetected",
                content_hash=h,
                timestamp=_ts(base + timedelta(minutes=15)),
                event_id="paste-1",
            ),
        ]

        linker = ClipboardLinker()
        links = linker.find_links(events)

        assert len(links) == 1
        link = links[0]
        assert link.copy_event_id == "copy-new"
        assert link.paste_event_id == "paste-1"
        # 15 min - 10 min = 5 min = 300 seconds
        assert abs(link.time_delta_seconds - 300.0) < 1.0


# ------------------------------------------------------------------
# 6. Empty events
# ------------------------------------------------------------------


class TestEmptyEvents:
    def test_empty_events(self) -> None:
        """No events → no links."""
        linker = ClipboardLinker()
        assert linker.find_links([]) == []


# ------------------------------------------------------------------
# 7. Copy without paste
# ------------------------------------------------------------------


class TestCopyWithoutPaste:
    def test_copy_without_paste(self) -> None:
        """Only copies → no links."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        h1 = _hash("stuff A")
        h2 = _hash("stuff B")

        events = [
            _make_clipboard_event(
                kind="ClipboardChange",
                content_hash=h1,
                timestamp=_ts(base),
            ),
            _make_clipboard_event(
                kind="ClipboardChange",
                content_hash=h2,
                timestamp=_ts(base + timedelta(minutes=5)),
            ),
        ]

        linker = ClipboardLinker()
        links = linker.find_links(events)

        assert len(links) == 0
