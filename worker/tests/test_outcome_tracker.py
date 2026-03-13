"""Tests for the outcome tracker module."""

from __future__ import annotations

import json

import pytest

from oc_apprentice_worker.outcome_tracker import (
    DetectedOutcome,
    OutcomeTracker,
    _event_annotation,
    _event_app,
    _event_kind,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_event(
    kind: str = "FocusChange",
    app: str = "Chrome",
    what_doing: str = "",
    location: str = "",
) -> dict:
    """Create a mock event dict."""
    annotation = {
        "task_context": {"what_doing": what_doing},
        "visual_context": {"active_app": app, "location": location},
    }
    return {
        "kind_json": json.dumps({kind: {}}),
        "window_json": json.dumps({"app": app, "title": f"{app} Window"}),
        "metadata_json": "{}",
        "scene_annotation_json": json.dumps(annotation),
    }


def make_bare_event(kind: str = "FocusChange", app: str = "") -> dict:
    """Create a minimal event without annotations."""
    result: dict = {
        "kind_json": json.dumps({kind: {}}),
        "metadata_json": "{}",
    }
    if app:
        result["window_json"] = json.dumps({"app": app, "title": ""})
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tracker() -> OutcomeTracker:
    return OutcomeTracker()


# ---------------------------------------------------------------------------
# Empty / no events
# ---------------------------------------------------------------------------


class TestEmptyEvents:
    """Tests with empty or minimal event lists."""

    def test_empty_list(self, tracker: OutcomeTracker) -> None:
        result = tracker.detect_outcomes([])
        assert result == []

    def test_single_generic_event(self, tracker: OutcomeTracker) -> None:
        events = [make_event(kind="FocusChange", app="Chrome")]
        result = tracker.detect_outcomes(events)
        # No clipboard, no file app, no comm, 0 URLs, no data entry kw
        assert result == []


# ---------------------------------------------------------------------------
# Clipboard transfer detection
# ---------------------------------------------------------------------------


class TestClipboardTransfer:
    """Tests for clipboard / data transfer detection."""

    def test_clipboard_between_different_apps(
        self, tracker: OutcomeTracker
    ) -> None:
        events = [
            make_event(kind="FocusChange", app="Chrome"),
            make_event(kind="FocusChange", app="Chrome"),
            make_event(kind="ClipboardChange", app="Chrome"),
            make_event(kind="FocusChange", app="TextEdit"),
            make_event(kind="FocusChange", app="TextEdit"),
        ]
        result = tracker.detect_outcomes(events)
        transfer = next(
            (o for o in result if o.type == "data_transfer"), None
        )
        assert transfer is not None
        assert transfer.confidence == 0.7
        assert "Chrome" in transfer.description
        assert "TextEdit" in transfer.description

    def test_clipboard_same_app(self, tracker: OutcomeTracker) -> None:
        events = [
            make_event(kind="FocusChange", app="Chrome"),
            make_event(kind="ClipboardChange", app="Chrome"),
            make_event(kind="FocusChange", app="Chrome"),
        ]
        result = tracker.detect_outcomes(events)
        transfer = next(
            (o for o in result if o.type == "data_transfer"), None
        )
        assert transfer is not None
        # Same app before and after -> lower confidence
        assert transfer.confidence == 0.5
        assert "Clipboard was used" in transfer.description

    def test_no_clipboard_events(self, tracker: OutcomeTracker) -> None:
        events = [
            make_event(kind="FocusChange", app="Chrome"),
            make_event(kind="FocusChange", app="TextEdit"),
        ]
        result = tracker.detect_outcomes(events)
        transfer = next(
            (o for o in result if o.type == "data_transfer"), None
        )
        assert transfer is None

    def test_clipboard_no_surrounding_apps(
        self, tracker: OutcomeTracker
    ) -> None:
        events = [
            make_bare_event(kind="ClipboardChange"),
        ]
        result = tracker.detect_outcomes(events)
        transfer = next(
            (o for o in result if o.type == "data_transfer"), None
        )
        # Clipboard event exists, but no apps around it
        assert transfer is not None
        assert transfer.confidence == 0.5

    def test_clipboard_multiple_apps_before_and_after(
        self, tracker: OutcomeTracker
    ) -> None:
        events = [
            make_event(kind="FocusChange", app="Safari"),
            make_event(kind="FocusChange", app="Chrome"),
            make_event(kind="ClipboardChange", app="Chrome"),
            make_event(kind="FocusChange", app="TextEdit"),
            make_event(kind="FocusChange", app="VS Code"),
        ]
        result = tracker.detect_outcomes(events)
        transfer = next(
            (o for o in result if o.type == "data_transfer"), None
        )
        assert transfer is not None
        assert transfer.confidence == 0.7


# ---------------------------------------------------------------------------
# File activity detection
# ---------------------------------------------------------------------------


class TestFileActivity:
    """Tests for file creation / modification detection."""

    def test_finder_activity(self, tracker: OutcomeTracker) -> None:
        events = [make_event(kind="FocusChange", app="Finder")]
        result = tracker.detect_outcomes(events)
        file_out = next(
            (o for o in result if o.type == "file_created"), None
        )
        assert file_out is not None
        assert "Finder" in file_out.description
        assert file_out.confidence == 0.6

    def test_vs_code_activity(self, tracker: OutcomeTracker) -> None:
        events = [make_event(kind="FocusChange", app="VS Code")]
        result = tracker.detect_outcomes(events)
        file_out = next(
            (o for o in result if o.type == "file_created"), None
        )
        assert file_out is not None
        assert "VS Code" in file_out.description

    def test_xcode_activity(self, tracker: OutcomeTracker) -> None:
        events = [make_event(kind="FocusChange", app="Xcode")]
        result = tracker.detect_outcomes(events)
        file_out = next(
            (o for o in result if o.type == "file_created"), None
        )
        assert file_out is not None

    def test_no_file_apps(self, tracker: OutcomeTracker) -> None:
        events = [make_event(kind="FocusChange", app="Chrome")]
        result = tracker.detect_outcomes(events)
        file_out = next(
            (o for o in result if o.type == "file_created"), None
        )
        assert file_out is None

    def test_annotation_save_keyword(self, tracker: OutcomeTracker) -> None:
        events = [
            make_event(
                kind="FocusChange",
                app="Chrome",
                what_doing="save the document",
            )
        ]
        result = tracker.detect_outcomes(events)
        file_out = next(
            (o for o in result if o.type == "file_created"), None
        )
        assert file_out is not None
        assert "save the document" in file_out.description

    def test_annotation_download_keyword(
        self, tracker: OutcomeTracker
    ) -> None:
        events = [
            make_event(
                kind="FocusChange",
                app="Chrome",
                what_doing="download report",
            )
        ]
        result = tracker.detect_outcomes(events)
        file_out = next(
            (o for o in result if o.type == "file_created"), None
        )
        assert file_out is not None

    def test_annotation_export_keyword(
        self, tracker: OutcomeTracker
    ) -> None:
        events = [
            make_event(
                kind="FocusChange",
                app="Chrome",
                what_doing="export as PDF",
            )
        ]
        result = tracker.detect_outcomes(events)
        file_out = next(
            (o for o in result if o.type == "file_created"), None
        )
        assert file_out is not None

    def test_file_app_uses_last_event(
        self, tracker: OutcomeTracker
    ) -> None:
        events = [
            make_event(kind="FocusChange", app="VS Code"),
            make_event(kind="FocusChange", app="Xcode"),
        ]
        result = tracker.detect_outcomes(events)
        file_out = next(
            (o for o in result if o.type == "file_created"), None
        )
        assert file_out is not None
        # Uses last file-app event
        assert "Xcode" in file_out.description


# ---------------------------------------------------------------------------
# Communication detection
# ---------------------------------------------------------------------------


class TestCommunication:
    """Tests for communication / message-sent detection."""

    def test_slack_send(self, tracker: OutcomeTracker) -> None:
        events = [
            make_event(
                kind="FocusChange",
                app="Slack",
                what_doing="send a message to the team",
            )
        ]
        result = tracker.detect_outcomes(events)
        comm = next(
            (o for o in result if o.type == "communication_sent"), None
        )
        assert comm is not None
        assert "Slack" in comm.description
        assert comm.confidence == 0.75

    def test_mail_reply(self, tracker: OutcomeTracker) -> None:
        events = [
            make_event(
                kind="FocusChange",
                app="Mail",
                what_doing="reply to client email",
            )
        ]
        result = tracker.detect_outcomes(events)
        comm = next(
            (o for o in result if o.type == "communication_sent"), None
        )
        assert comm is not None
        assert "Mail" in comm.description

    def test_comm_app_no_keywords(self, tracker: OutcomeTracker) -> None:
        events = [
            make_event(
                kind="FocusChange",
                app="Slack",
                what_doing="browsing channels",
            )
        ]
        result = tracker.detect_outcomes(events)
        comm = next(
            (o for o in result if o.type == "communication_sent"), None
        )
        # "browsing channels" doesn't match send/reply/compose/message/chat
        assert comm is None

    def test_non_comm_app_ignored(self, tracker: OutcomeTracker) -> None:
        events = [
            make_event(
                kind="FocusChange",
                app="Chrome",
                what_doing="sending a message",
            )
        ]
        result = tracker.detect_outcomes(events)
        comm = next(
            (o for o in result if o.type == "communication_sent"), None
        )
        # Chrome is not in the comm_apps set
        assert comm is None

    def test_discord_chat(self, tracker: OutcomeTracker) -> None:
        events = [
            make_event(
                kind="FocusChange",
                app="Discord",
                what_doing="chatting with friends",
            )
        ]
        result = tracker.detect_outcomes(events)
        comm = next(
            (o for o in result if o.type == "communication_sent"), None
        )
        assert comm is not None
        assert "Discord" in comm.description


# ---------------------------------------------------------------------------
# Navigation completion detection
# ---------------------------------------------------------------------------


class TestNavigationCompletion:
    """Tests for navigation workflow completion."""

    def test_multiple_urls_detected(self, tracker: OutcomeTracker) -> None:
        events = [
            make_event(
                kind="FocusChange",
                app="Chrome",
                location="https://example.com/page1",
            ),
            make_event(
                kind="FocusChange",
                app="Chrome",
                location="https://example.com/page2",
            ),
        ]
        result = tracker.detect_outcomes(events)
        nav = next(
            (o for o in result if o.type == "navigation_completed"), None
        )
        assert nav is not None
        assert nav.confidence == 0.5
        assert "2 pages" in nav.description
        assert nav.verification["pages_visited"] == 2

    def test_three_urls(self, tracker: OutcomeTracker) -> None:
        events = [
            make_event(
                kind="FocusChange",
                app="Chrome",
                location="https://a.com",
            ),
            make_event(
                kind="FocusChange",
                app="Chrome",
                location="https://b.com",
            ),
            make_event(
                kind="FocusChange",
                app="Chrome",
                location="https://c.com",
            ),
        ]
        result = tracker.detect_outcomes(events)
        nav = next(
            (o for o in result if o.type == "navigation_completed"), None
        )
        assert nav is not None
        assert "3 pages" in nav.description
        assert nav.verification["check"] == "ended at https://c.com"

    def test_single_url_no_navigation(
        self, tracker: OutcomeTracker
    ) -> None:
        events = [
            make_event(
                kind="FocusChange",
                app="Chrome",
                location="https://example.com",
            ),
        ]
        result = tracker.detect_outcomes(events)
        nav = next(
            (o for o in result if o.type == "navigation_completed"), None
        )
        assert nav is None

    def test_non_http_locations_ignored(
        self, tracker: OutcomeTracker
    ) -> None:
        events = [
            make_event(
                kind="FocusChange",
                app="Chrome",
                location="/Users/test/file.txt",
            ),
            make_event(
                kind="FocusChange",
                app="Chrome",
                location="file:///path",
            ),
        ]
        result = tracker.detect_outcomes(events)
        nav = next(
            (o for o in result if o.type == "navigation_completed"), None
        )
        assert nav is None


# ---------------------------------------------------------------------------
# Data entry detection
# ---------------------------------------------------------------------------


class TestDataEntry:
    """Tests for data entry detection."""

    def test_fill_keyword(self, tracker: OutcomeTracker) -> None:
        events = [
            make_event(
                kind="FocusChange",
                app="Chrome",
                what_doing="fill out the registration form",
            )
        ]
        result = tracker.detect_outcomes(events)
        entry = next(
            (o for o in result if o.type == "data_entry"), None
        )
        assert entry is not None
        assert entry.confidence == 0.6

    def test_enter_keyword(self, tracker: OutcomeTracker) -> None:
        events = [
            make_event(
                kind="FocusChange",
                app="Chrome",
                what_doing="enter the login credentials",
            )
        ]
        result = tracker.detect_outcomes(events)
        entry = next(
            (o for o in result if o.type == "data_entry"), None
        )
        assert entry is not None

    def test_type_keyword(self, tracker: OutcomeTracker) -> None:
        events = [
            make_event(
                kind="FocusChange",
                app="Chrome",
                what_doing="type the search query",
            )
        ]
        result = tracker.detect_outcomes(events)
        entry = next(
            (o for o in result if o.type == "data_entry"), None
        )
        assert entry is not None

    def test_form_keyword(self, tracker: OutcomeTracker) -> None:
        events = [
            make_event(
                kind="FocusChange",
                app="Chrome",
                what_doing="complete the form submission",
            )
        ]
        result = tracker.detect_outcomes(events)
        entry = next(
            (o for o in result if o.type == "data_entry"), None
        )
        assert entry is not None

    def test_no_data_entry_keywords(
        self, tracker: OutcomeTracker
    ) -> None:
        events = [
            make_event(
                kind="FocusChange",
                app="Chrome",
                what_doing="browsing the web",
            )
        ]
        result = tracker.detect_outcomes(events)
        entry = next(
            (o for o in result if o.type == "data_entry"), None
        )
        assert entry is None


# ---------------------------------------------------------------------------
# Multiple outcomes from one task
# ---------------------------------------------------------------------------


class TestMultipleOutcomes:
    """Tests for multiple outcomes detected from one event set."""

    def test_clipboard_and_file_activity(
        self, tracker: OutcomeTracker
    ) -> None:
        events = [
            make_event(kind="FocusChange", app="Chrome"),
            make_event(kind="ClipboardChange", app="Chrome"),
            make_event(kind="FocusChange", app="VS Code"),
        ]
        result = tracker.detect_outcomes(events)
        types = {o.type for o in result}
        assert "data_transfer" in types
        assert "file_created" in types

    def test_comm_and_navigation(self, tracker: OutcomeTracker) -> None:
        events = [
            make_event(
                kind="FocusChange",
                app="Chrome",
                location="https://slack.com/channel",
            ),
            make_event(
                kind="FocusChange",
                app="Chrome",
                location="https://slack.com/thread",
            ),
            make_event(
                kind="FocusChange",
                app="Slack",
                what_doing="send update message",
            ),
        ]
        result = tracker.detect_outcomes(events)
        types = {o.type for o in result}
        assert "navigation_completed" in types
        assert "communication_sent" in types


# ---------------------------------------------------------------------------
# Events without annotations
# ---------------------------------------------------------------------------


class TestEventsWithoutAnnotations:
    """Tests for events that lack annotations or have partial data."""

    def test_no_annotations_at_all(self, tracker: OutcomeTracker) -> None:
        events = [
            make_bare_event(kind="FocusChange", app="Chrome"),
            make_bare_event(kind="FocusChange", app="Chrome"),
        ]
        result = tracker.detect_outcomes(events)
        assert result == []

    def test_no_window_json(self, tracker: OutcomeTracker) -> None:
        events = [
            {
                "kind_json": json.dumps({"FocusChange": {}}),
                "metadata_json": "{}",
            },
        ]
        result = tracker.detect_outcomes(events)
        assert result == []

    def test_invalid_annotation_json(
        self, tracker: OutcomeTracker
    ) -> None:
        events = [
            {
                "kind_json": json.dumps({"FocusChange": {}}),
                "window_json": json.dumps({"app": "Chrome"}),
                "metadata_json": "{}",
                "scene_annotation_json": "not valid json{{{",
            },
        ]
        result = tracker.detect_outcomes(events)
        # Should not crash — annotation returns None gracefully
        assert result == []


# ---------------------------------------------------------------------------
# Private helper tests
# ---------------------------------------------------------------------------


class TestEventHelpers:
    """Tests for the private helper functions."""

    def test_event_kind_normal(self) -> None:
        event = {"kind_json": json.dumps({"FocusChange": {}})}
        assert _event_kind(event) == "FocusChange"

    def test_event_kind_empty(self) -> None:
        event = {"kind_json": "{}"}
        assert _event_kind(event) == ""

    def test_event_kind_invalid_json(self) -> None:
        event = {"kind_json": "not json"}
        assert _event_kind(event) == ""

    def test_event_kind_dict_input(self) -> None:
        event = {"kind_json": {"ClipboardChange": {}}}
        assert _event_kind(event) == "ClipboardChange"

    def test_event_kind_missing_key(self) -> None:
        event = {}
        assert _event_kind(event) == ""

    def test_event_app_normal(self) -> None:
        event = {"window_json": json.dumps({"app": "Chrome", "title": "T"})}
        assert _event_app(event) == "Chrome"

    def test_event_app_no_window(self) -> None:
        event = {}
        assert _event_app(event) == ""

    def test_event_app_invalid_json(self) -> None:
        event = {"window_json": "bad"}
        assert _event_app(event) == ""

    def test_event_app_dict_input(self) -> None:
        event = {"window_json": {"app": "Finder"}}
        assert _event_app(event) == "Finder"

    def test_event_app_none_value(self) -> None:
        event = {"window_json": None}
        assert _event_app(event) == ""

    def test_event_annotation_normal(self) -> None:
        ann = {"task_context": {"what_doing": "testing"}}
        event = {"scene_annotation_json": json.dumps(ann)}
        result = _event_annotation(event)
        assert result == ann

    def test_event_annotation_none(self) -> None:
        event = {}
        assert _event_annotation(event) is None

    def test_event_annotation_invalid_json(self) -> None:
        event = {"scene_annotation_json": "{{bad"}
        assert _event_annotation(event) is None

    def test_event_annotation_non_dict(self) -> None:
        event = {"scene_annotation_json": json.dumps([1, 2, 3])}
        assert _event_annotation(event) is None

    def test_event_annotation_dict_input(self) -> None:
        ann = {"task_context": {"what_doing": "test"}}
        event = {"scene_annotation_json": ann}
        assert _event_annotation(event) == ann


# ---------------------------------------------------------------------------
# DetectedOutcome dataclass
# ---------------------------------------------------------------------------


class TestDetectedOutcomeDataclass:
    """Verify dataclass structure."""

    def test_fields(self) -> None:
        o = DetectedOutcome(
            type="file_created",
            description="Created a file",
            verification={"check": "file exists"},
            confidence=0.8,
        )
        assert o.type == "file_created"
        assert o.description == "Created a file"
        assert o.verification == {"check": "file exists"}
        assert o.confidence == 0.8
