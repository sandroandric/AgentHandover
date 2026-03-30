"""Tests for agenthandover_worker.negative_demo.

Covers undo detection, cancel-click detection, discard-changes detection,
lookback window limiting, case-insensitive patterns, and normal passthrough.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

from agenthandover_worker.negative_demo import NegativeDemoPruner


def _ts(dt: datetime) -> str:
    """Format a datetime as the ISO 8601 string the daemon produces."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _make_event(
    *,
    app_id: str = "com.example.App",
    kind: str = "FocusChange",
    metadata: dict | None = None,
    timestamp: str | None = None,
    event_id: str | None = None,
) -> dict:
    """Build a minimal event dict for pruner tests."""
    eid = event_id or str(uuid.uuid4())
    window = {"app_id": app_id, "title": "Test"}

    return {
        "id": eid,
        "timestamp": timestamp or _ts(datetime.now(timezone.utc)),
        "kind_json": json.dumps({kind: {}}),
        "window_json": json.dumps(window),
        "metadata_json": json.dumps(metadata or {}),
        "display_topology_json": "[]",
        "primary_display_id": "main",
        "processed": 0,
    }


# ------------------------------------------------------------------
# 1. Undo marks preceding events
# ------------------------------------------------------------------


class TestUndoMarksPrecedingEvents:
    def test_undo_marks_preceding_events(self) -> None:
        """Ctrl+Z marks the trigger and recent preceding events as negative."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        app = "com.apple.TextEdit"

        events = [
            _make_event(app_id=app, kind="KeyPress", metadata={"key": "a"}, timestamp=_ts(base)),
            _make_event(
                app_id=app, kind="KeyPress", metadata={"key": "b"}, timestamp=_ts(base + timedelta(seconds=1))
            ),
            _make_event(
                app_id=app, kind="KeyPress", metadata={"key": "c"}, timestamp=_ts(base + timedelta(seconds=2))
            ),
            _make_event(
                app_id=app,
                kind="KeyPress",
                metadata={"shortcut": "ctrl+z"},
                timestamp=_ts(base + timedelta(seconds=3)),
            ),
        ]

        pruner = NegativeDemoPruner()
        result = pruner.prune(events)

        # The undo event and preceding events should be negative
        assert len(result.negative_events) >= 2  # at least the undo + some preceding
        assert events[3] in result.negative_events  # undo itself
        # At least one preceding event should be negative
        preceding_negative = [e for e in result.negative_events if e["id"] != events[3]["id"]]
        assert len(preceding_negative) >= 1


# ------------------------------------------------------------------
# 2. Cancel click marks negative
# ------------------------------------------------------------------


class TestCancelClickMarksNegative:
    def test_cancel_click_marks_negative(self) -> None:
        """Click on 'Cancel' button marks modal events negative."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        app = "com.apple.Finder"

        events = [
            _make_event(
                app_id=app,
                kind="ClickIntent",
                metadata={"element_text": "Open File"},
                timestamp=_ts(base),
            ),
            _make_event(
                app_id=app,
                kind="FocusChange",
                metadata={"dialog": "file_picker"},
                timestamp=_ts(base + timedelta(seconds=2)),
            ),
            _make_event(
                app_id=app,
                kind="ClickIntent",
                metadata={"element_text": "Cancel"},
                timestamp=_ts(base + timedelta(seconds=5)),
            ),
        ]

        pruner = NegativeDemoPruner()
        result = pruner.prune(events)

        # The Cancel click and at least some preceding events should be negative
        assert events[2] in result.negative_events
        assert len(result.negative_events) >= 2

        # Check reasons
        cancel_id = events[2]["id"]
        assert result.prune_reasons[cancel_id] == "cancel_click"


# ------------------------------------------------------------------
# 3. Discard changes marks negative
# ------------------------------------------------------------------


class TestDiscardChangesMarksNegative:
    def test_discard_changes_marks_negative(self) -> None:
        """'Discard changes' click marks negative."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        app = "com.microsoft.VSCode"

        events = [
            _make_event(
                app_id=app,
                kind="KeyPress",
                metadata={"key": "x"},
                timestamp=_ts(base),
            ),
            _make_event(
                app_id=app,
                kind="KeyPress",
                metadata={"key": "y"},
                timestamp=_ts(base + timedelta(seconds=1)),
            ),
            _make_event(
                app_id=app,
                kind="ClickIntent",
                metadata={"element_text": "Discard changes"},
                timestamp=_ts(base + timedelta(seconds=3)),
            ),
        ]

        pruner = NegativeDemoPruner()
        result = pruner.prune(events)

        assert events[2] in result.negative_events
        assert len(result.negative_events) >= 2

        discard_id = events[2]["id"]
        assert result.prune_reasons[discard_id] == "cancel_click"


# ------------------------------------------------------------------
# 4. Normal events pass through
# ------------------------------------------------------------------


class TestNormalEventsPassThrough:
    def test_normal_events_pass_through(self) -> None:
        """No negative markers → all events are positive."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)

        events = [
            _make_event(kind="FocusChange", timestamp=_ts(base)),
            _make_event(kind="FocusChange", timestamp=_ts(base + timedelta(seconds=1))),
            _make_event(kind="FocusChange", timestamp=_ts(base + timedelta(seconds=2))),
        ]

        pruner = NegativeDemoPruner()
        result = pruner.prune(events)

        assert len(result.positive_events) == 3
        assert len(result.negative_events) == 0
        assert len(result.prune_reasons) == 0


# ------------------------------------------------------------------
# 5. Empty events
# ------------------------------------------------------------------


class TestEmptyEvents:
    def test_empty_events(self) -> None:
        """Empty → empty result."""
        pruner = NegativeDemoPruner()
        result = pruner.prune([])

        assert result.positive_events == []
        assert result.negative_events == []
        assert result.prune_reasons == {}


# ------------------------------------------------------------------
# 6. Lookback window limited
# ------------------------------------------------------------------


class TestLookbackWindowLimited:
    def test_lookback_event_limit(self) -> None:
        """Only marks back 10 events, not entire history."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        app = "com.apple.TextEdit"

        # 20 events all within 30s, then an undo
        events = []
        for i in range(20):
            events.append(
                _make_event(
                    app_id=app,
                    kind="KeyPress",
                    metadata={"key": chr(ord("a") + (i % 26))},
                    timestamp=_ts(base + timedelta(seconds=i)),
                )
            )
        # Undo at second 20
        events.append(
            _make_event(
                app_id=app,
                kind="KeyPress",
                metadata={"shortcut": "cmd+z"},
                timestamp=_ts(base + timedelta(seconds=20)),
            )
        )

        pruner = NegativeDemoPruner()
        result = pruner.prune(events)

        # Should mark undo + at most 10 preceding = 11 max
        assert len(result.negative_events) <= 11
        # At least the undo itself
        assert events[-1] in result.negative_events
        # The earliest events (index 0-9) should be positive
        # since lookback only goes 10 events back from index 20
        assert events[0] in result.positive_events

    def test_lookback_time_limit(self) -> None:
        """Only marks back 30 seconds, not events older than that."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        app = "com.apple.TextEdit"

        events = [
            # Old event: 60 seconds before undo
            _make_event(
                app_id=app,
                kind="KeyPress",
                metadata={"key": "old"},
                timestamp=_ts(base),
            ),
            # Recent event: 5 seconds before undo
            _make_event(
                app_id=app,
                kind="KeyPress",
                metadata={"key": "recent"},
                timestamp=_ts(base + timedelta(seconds=55)),
            ),
            # Undo at 60s
            _make_event(
                app_id=app,
                kind="KeyPress",
                metadata={"shortcut": "ctrl+z"},
                timestamp=_ts(base + timedelta(seconds=60)),
            ),
        ]

        pruner = NegativeDemoPruner()
        result = pruner.prune(events)

        # The old event should be positive (>30s before undo)
        assert events[0] in result.positive_events
        # The recent event and undo should be negative
        assert events[1] in result.negative_events
        assert events[2] in result.negative_events


# ------------------------------------------------------------------
# 7. Case-insensitive patterns
# ------------------------------------------------------------------


class TestCaseInsensitivePatterns:
    def test_case_insensitive_patterns(self) -> None:
        """'CANCEL', 'Cancel', 'cancel' are all detected."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        app = "com.apple.Finder"

        variants = ["CANCEL", "Cancel", "cancel", "CaNcEl"]

        for variant in variants:
            events = [
                _make_event(
                    app_id=app,
                    kind="FocusChange",
                    timestamp=_ts(base),
                ),
                _make_event(
                    app_id=app,
                    kind="ClickIntent",
                    metadata={"element_text": variant},
                    timestamp=_ts(base + timedelta(seconds=1)),
                ),
            ]

            pruner = NegativeDemoPruner()
            result = pruner.prune(events)

            assert events[1] in result.negative_events, (
                f"Failed to detect cancel variant: {variant!r}"
            )
