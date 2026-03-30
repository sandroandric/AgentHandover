"""Tests for the clipboard preview cleanup module.

Covers TTL-based purging of clipboard preview records.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from agenthandover_worker.cleanup import purge_old_clipboard_previews


def _make_clipboard_event(
    *,
    timestamp: str,
    event_id: str = "ev-1",
    has_preview: bool = True,
) -> dict:
    metadata: dict = {
        "content_hash": "abc123",
        "content_types": ["text/plain"],
        "byte_size": 42,
    }
    if has_preview:
        metadata["content_preview"] = "Hello world..."

    return {
        "id": event_id,
        "timestamp": timestamp,
        "kind_json": json.dumps({"ClipboardChange": {}}),
        "window_json": json.dumps({"app_id": "com.example.App", "title": "Test"}),
        "metadata_json": json.dumps(metadata),
        "display_topology_json": "[]",
        "primary_display_id": "main",
        "processed": 0,
    }


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ------------------------------------------------------------------
# 1. Old clipboard previews are purged
# ------------------------------------------------------------------


class TestPurgeOldClipboardPreviews:
    def test_old_preview_purged(self) -> None:
        now = datetime(2026, 2, 16, 12, 0, 0, tzinfo=timezone.utc)
        old_ts = _ts(now - timedelta(hours=25))

        events = [_make_clipboard_event(timestamp=old_ts, event_id="old")]
        result = purge_old_clipboard_previews(events, ttl_hours=24.0, now=now)

        assert len(result) == 0

    def test_recent_preview_kept(self) -> None:
        now = datetime(2026, 2, 16, 12, 0, 0, tzinfo=timezone.utc)
        recent_ts = _ts(now - timedelta(hours=1))

        events = [_make_clipboard_event(timestamp=recent_ts, event_id="recent")]
        result = purge_old_clipboard_previews(events, ttl_hours=24.0, now=now)

        assert len(result) == 1
        assert result[0]["id"] == "recent"


# ------------------------------------------------------------------
# 2. Non-clipboard events are not affected
# ------------------------------------------------------------------


class TestNonClipboardEventsKept:
    def test_focus_event_kept(self) -> None:
        now = datetime(2026, 2, 16, 12, 0, 0, tzinfo=timezone.utc)
        old_ts = _ts(now - timedelta(hours=48))

        event = {
            "id": "focus-1",
            "timestamp": old_ts,
            "kind_json": json.dumps({"FocusChange": {}}),
            "window_json": "{}",
            "metadata_json": "{}",
            "display_topology_json": "[]",
            "primary_display_id": "main",
            "processed": 0,
        }
        result = purge_old_clipboard_previews([event], ttl_hours=24.0, now=now)

        assert len(result) == 1
        assert result[0]["id"] == "focus-1"


# ------------------------------------------------------------------
# 3. Clipboard events without preview are kept
# ------------------------------------------------------------------


class TestClipboardWithoutPreviewKept:
    def test_no_preview_kept(self) -> None:
        now = datetime(2026, 2, 16, 12, 0, 0, tzinfo=timezone.utc)
        old_ts = _ts(now - timedelta(hours=48))

        events = [_make_clipboard_event(
            timestamp=old_ts,
            event_id="no-preview",
            has_preview=False,
        )]
        result = purge_old_clipboard_previews(events, ttl_hours=24.0, now=now)

        assert len(result) == 1


# ------------------------------------------------------------------
# 4. Empty input
# ------------------------------------------------------------------


class TestPurgeEmptyInput:
    def test_empty_returns_empty(self) -> None:
        result = purge_old_clipboard_previews([])
        assert result == []


# ------------------------------------------------------------------
# 5. Mixed events
# ------------------------------------------------------------------


class TestPurgeMixed:
    def test_mixed_old_and_recent(self) -> None:
        now = datetime(2026, 2, 16, 12, 0, 0, tzinfo=timezone.utc)
        old_ts = _ts(now - timedelta(hours=30))
        recent_ts = _ts(now - timedelta(hours=2))

        events = [
            _make_clipboard_event(timestamp=old_ts, event_id="old"),
            _make_clipboard_event(timestamp=recent_ts, event_id="recent"),
        ]
        result = purge_old_clipboard_previews(events, ttl_hours=24.0, now=now)

        assert len(result) == 1
        assert result[0]["id"] == "recent"


# ------------------------------------------------------------------
# 6. Custom TTL
# ------------------------------------------------------------------


class TestCustomTTL:
    def test_custom_short_ttl(self) -> None:
        now = datetime(2026, 2, 16, 12, 0, 0, tzinfo=timezone.utc)
        # 2 hours old, TTL = 1 hour
        old_ts = _ts(now - timedelta(hours=2))

        events = [_make_clipboard_event(timestamp=old_ts)]
        result = purge_old_clipboard_previews(events, ttl_hours=1.0, now=now)

        assert len(result) == 0

    def test_custom_long_ttl(self) -> None:
        now = datetime(2026, 2, 16, 12, 0, 0, tzinfo=timezone.utc)
        # 2 hours old, TTL = 48 hours
        old_ts = _ts(now - timedelta(hours=2))

        events = [_make_clipboard_event(timestamp=old_ts)]
        result = purge_old_clipboard_previews(events, ttl_hours=48.0, now=now)

        assert len(result) == 1
