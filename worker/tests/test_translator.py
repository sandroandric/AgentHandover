"""Tests for agenthandover_worker.translator.

Covers intent extraction, UI anchor resolution priority cascade,
parameter extraction, batch translation with context, and edge cases.
"""

from __future__ import annotations

import json
import uuid

from agenthandover_worker.translator import SemanticTranslator, TranslationResult, UIAnchor


def _make_event(
    *,
    kind: str = "ClickIntent",
    metadata: dict | None = None,
    app_id: str = "com.apple.Safari",
    window_title: str = "Test Window",
    event_id: str | None = None,
    timestamp: str = "2026-02-16T10:00:00.000Z",
) -> dict:
    """Build a minimal event dict for translator tests."""
    eid = event_id or str(uuid.uuid4())
    window = {"app_id": app_id, "title": window_title}

    return {
        "id": eid,
        "timestamp": timestamp,
        "kind_json": json.dumps({kind: {}}),
        "window_json": json.dumps(window),
        "metadata_json": json.dumps(metadata or {}),
        "display_topology_json": "[]",
        "primary_display_id": "main",
        "processed": 0,
    }


# ------------------------------------------------------------------
# 1. Click event translation — full metadata
# ------------------------------------------------------------------


class TestClickEventTranslation:
    def test_click_event_translation(self) -> None:
        """ClickIntent with full metadata translates to 'click' intent with anchor."""
        event = _make_event(
            kind="ClickIntent",
            metadata={
                "target": {
                    "tagName": "button",
                    "role": "button",
                    "ariaLabel": "Submit review",
                    "testId": "submit-review-btn",
                    "innerText": "Submit",
                    "composedPath": ["button#submit", "div.actions", "form.review"],
                },
                "url": "https://github.com/repo/pull/42",
                "x": 450,
                "y": 320,
            },
            window_title="Pull Request #42",
        )

        translator = SemanticTranslator()
        result = translator.translate_event(event)

        assert isinstance(result, TranslationResult)
        assert result.intent == "click"
        assert result.raw_event_id == event["id"]

        # Should have a UI anchor (ARIA label is highest priority)
        assert result.target is not None
        assert result.target.method == "aria_label"
        assert "Submit review" in result.target.selector
        assert result.target.confidence_contribution >= 0.40

        # Pre-state should contain window title and URL
        assert result.pre_state["window_title"] == "Pull Request #42"
        assert result.pre_state["url"] == "https://github.com/repo/pull/42"

        # Parameters should capture button info
        assert result.parameters.get("button_text") == "Submit"
        assert result.parameters.get("element_type") == "button"


# ------------------------------------------------------------------
# 2. Focus change → navigate intent
# ------------------------------------------------------------------


class TestFocusChangeTranslation:
    def test_focus_change_translation(self) -> None:
        """FocusChange maps to 'navigate' intent."""
        event = _make_event(
            kind="FocusChange",
            metadata={"url": "https://docs.google.com/doc/123"},
            window_title="Project Notes - Google Docs",
        )

        translator = SemanticTranslator()
        result = translator.translate_event(event)

        assert result.intent == "navigate"
        assert result.pre_state["window_title"] == "Project Notes - Google Docs"
        assert result.pre_state["url"] == "https://docs.google.com/doc/123"


# ------------------------------------------------------------------
# 3. App switch → switch_app intent
# ------------------------------------------------------------------


class TestAppSwitchTranslation:
    def test_app_switch_translation(self) -> None:
        """AppSwitch maps to 'switch_app' intent."""
        event = _make_event(
            kind="AppSwitch",
            metadata={"app_name": "Visual Studio Code"},
            app_id="com.microsoft.VSCode",
            window_title="main.py - AgentHandover",
        )

        translator = SemanticTranslator()
        result = translator.translate_event(event)

        assert result.intent == "switch_app"
        assert result.pre_state["app_id"] == "com.microsoft.VSCode"
        assert result.parameters.get("app_name") == "Visual Studio Code"


# ------------------------------------------------------------------
# 4. ARIA label resolution (highest priority)
# ------------------------------------------------------------------


class TestAriaLabelResolution:
    def test_aria_label_resolution(self) -> None:
        """ARIA label is used when present, giving highest confidence."""
        event = _make_event(
            kind="ClickIntent",
            metadata={
                "target": {
                    "tagName": "button",
                    "ariaLabel": "Close dialog",
                    "innerText": "X",
                    "testId": "close-btn",
                    "role": "button",
                },
            },
        )

        translator = SemanticTranslator()
        result = translator.translate_event(event)

        assert result.target is not None
        assert result.target.method == "aria_label"
        assert result.target.confidence_contribution >= 0.40
        assert "Close dialog" in result.target.selector

    def test_aria_label_short_gets_lower_confidence(self) -> None:
        """Short ARIA labels (<=3 chars) get 0.40, not 0.45."""
        event = _make_event(
            kind="ClickIntent",
            metadata={
                "target": {"ariaLabel": "OK"},
            },
        )

        translator = SemanticTranslator()
        result = translator.translate_event(event)

        assert result.target is not None
        assert result.target.method == "aria_label"
        assert result.target.confidence_contribution == 0.40


# ------------------------------------------------------------------
# 5. Inner text fallback (when no ARIA)
# ------------------------------------------------------------------


class TestInnerTextFallback:
    def test_inner_text_fallback(self) -> None:
        """When no ARIA label or test ID, falls back to innerText."""
        event = _make_event(
            kind="ClickIntent",
            metadata={
                "target": {
                    "tagName": "a",
                    "innerText": "View all comments",
                },
            },
        )

        translator = SemanticTranslator()
        result = translator.translate_event(event)

        assert result.target is not None
        assert result.target.method == "inner_text"
        assert result.target.confidence_contribution >= 0.25
        assert "view all comments" in result.target.selector

    def test_inner_text_normalization(self) -> None:
        """Inner text is trimmed, spaces collapsed, and lowercased."""
        event = _make_event(
            kind="ClickIntent",
            metadata={
                "target": {
                    "tagName": "button",
                    "innerText": "  Submit   Review  ",
                },
            },
        )

        translator = SemanticTranslator()
        result = translator.translate_event(event)

        assert result.target is not None
        assert result.target.method == "inner_text"
        assert "submit review" in result.target.selector


# ------------------------------------------------------------------
# 6. Test ID resolution
# ------------------------------------------------------------------


class TestTestIdResolution:
    def test_test_id_resolution(self) -> None:
        """data-testid is resolved when no ARIA label is present."""
        event = _make_event(
            kind="ClickIntent",
            metadata={
                "target": {
                    "tagName": "button",
                    "testId": "submit-review-btn",
                    "innerText": "Submit",
                    "role": "button",
                },
            },
        )

        translator = SemanticTranslator()
        result = translator.translate_event(event)

        assert result.target is not None
        assert result.target.method == "test_id"
        assert result.target.confidence_contribution >= 0.35
        assert "submit-review-btn" in result.target.selector

    def test_test_id_with_hash_lower_confidence(self) -> None:
        """Test IDs containing hash-like segments get lower confidence."""
        event = _make_event(
            kind="ClickIntent",
            metadata={
                "target": {
                    "tagName": "div",
                    "testId": "card-a1b2c3d4e5f6g7h8",
                },
            },
        )

        translator = SemanticTranslator()
        result = translator.translate_event(event)

        assert result.target is not None
        assert result.target.method == "test_id"
        assert result.target.confidence_contribution == 0.35


# ------------------------------------------------------------------
# 7. No metadata fallback (minimal event)
# ------------------------------------------------------------------


class TestNoMetadataFallback:
    def test_no_metadata_fallback_with_app_context(self) -> None:
        """Event with no target info but valid window_json → app_context anchor.

        The ``_try_app_context()`` fallback extracts app_id and window_title
        from ``window_json`` to build a low-confidence (0.10-0.15) anchor
        for native app events that have no DOM metadata.
        """
        event = _make_event(
            kind="FocusChange",
            metadata={},
        )

        translator = SemanticTranslator()
        result = translator.translate_event(event)

        # app_context fallback provides an anchor from window_json
        assert result.target is not None
        assert result.target.method == "app_context"
        assert result.target.confidence_contribution == 0.15
        assert "com.apple.Safari" in result.target.selector
        assert result.intent == "navigate"

    def test_no_app_context_no_anchor(self) -> None:
        """Event with empty window_json → truly no anchor."""
        event = {
            "id": str(uuid.uuid4()),
            "timestamp": "2026-02-16T10:00:00.000Z",
            "kind_json": json.dumps({"FocusChange": {}}),
            "window_json": "{}",
            "metadata_json": json.dumps({}),
            "display_topology_json": "[]",
            "primary_display_id": "main",
            "processed": 0,
        }

        translator = SemanticTranslator()
        result = translator.translate_event(event)

        assert result.target is None
        assert result.intent == "navigate"

    def test_empty_metadata_json(self) -> None:
        """Event with empty metadata_json string + empty window → no anchor, no crash."""
        event = {
            "id": str(uuid.uuid4()),
            "timestamp": "2026-02-16T10:00:00.000Z",
            "kind_json": json.dumps({"ClickIntent": {}}),
            "window_json": "{}",
            "metadata_json": "",
            "display_topology_json": "[]",
            "primary_display_id": "main",
            "processed": 0,
        }

        translator = SemanticTranslator()
        result = translator.translate_event(event)

        assert result.target is None
        assert result.intent == "click"


# ------------------------------------------------------------------
# 8. Batch translation with context building
# ------------------------------------------------------------------


class TestBatchTranslation:
    def test_batch_translation(self) -> None:
        """Multiple events with context flowing between them."""
        events = [
            _make_event(
                kind="FocusChange",
                metadata={"url": "https://github.com/repo"},
                window_title="GitHub - Repo",
                app_id="com.apple.Safari",
                timestamp="2026-02-16T10:00:00.000Z",
                event_id="event-1",
            ),
            _make_event(
                kind="ClickIntent",
                metadata={
                    "target": {
                        "tagName": "button",
                        "ariaLabel": "Create pull request",
                    },
                    "url": "https://github.com/repo/compare",
                },
                window_title="Compare changes",
                app_id="com.apple.Safari",
                timestamp="2026-02-16T10:00:05.000Z",
                event_id="event-2",
            ),
            _make_event(
                kind="AppSwitch",
                metadata={"app_name": "Slack"},
                app_id="com.tinyspeck.slackmacgap",
                window_title="Slack - General",
                timestamp="2026-02-16T10:00:10.000Z",
                event_id="event-3",
            ),
        ]

        translator = SemanticTranslator()
        results = translator.translate_batch(events)

        assert len(results) == 3

        # First event: navigate
        assert results[0].intent == "navigate"
        assert results[0].raw_event_id == "event-1"

        # Second event: click with ARIA anchor
        assert results[1].intent == "click"
        assert results[1].target is not None
        assert results[1].target.method == "aria_label"

        # Third event: switch_app
        assert results[2].intent == "switch_app"
        assert results[2].parameters.get("app_name") == "Slack"

    def test_batch_context_propagation(self) -> None:
        """Context from earlier events flows into later pre-states."""
        events = [
            _make_event(
                kind="FocusChange",
                metadata={"url": "https://example.com/page1"},
                window_title="Page 1",
                app_id="com.apple.Safari",
                timestamp="2026-02-16T10:00:00.000Z",
            ),
            # Second event has no URL/title in its own metadata
            _make_event(
                kind="ClickIntent",
                metadata={
                    "target": {"tagName": "button", "ariaLabel": "Next"},
                },
                window_title="Page 1",
                app_id="com.apple.Safari",
                timestamp="2026-02-16T10:00:01.000Z",
            ),
        ]

        translator = SemanticTranslator()
        results = translator.translate_batch(events)

        # Second event should have the URL from the first event's context
        # since it did not have its own URL
        assert results[1].pre_state.get("window_title") == "Page 1"


# ------------------------------------------------------------------
# 9. Vision bbox fallback
# ------------------------------------------------------------------


class TestVisionBboxFallback:
    def test_vision_bbox_fallback(self) -> None:
        """When only coordinates are available, vision_bbox is used."""
        event = _make_event(
            kind="ClickIntent",
            metadata={
                "x": 450,
                "y": 320,
            },
        )

        translator = SemanticTranslator()
        result = translator.translate_event(event)

        assert result.target is not None
        assert result.target.method == "vision_bbox"
        assert result.target.confidence_contribution == 0.10
        assert "450" in result.target.selector
        assert "320" in result.target.selector


# ------------------------------------------------------------------
# 10. Role + position resolution
# ------------------------------------------------------------------


class TestRolePositionResolution:
    def test_role_position_resolution(self) -> None:
        """Role + tag + path gives role_position anchor."""
        event = _make_event(
            kind="ClickIntent",
            metadata={
                "target": {
                    "role": "button",
                    "tagName": "div",
                    "composedPath": ["div.actions", "section.toolbar"],
                },
            },
        )

        translator = SemanticTranslator()
        result = translator.translate_event(event)

        assert result.target is not None
        assert result.target.method == "role_position"
        assert result.target.confidence_contribution >= 0.15
        assert "role=button" in result.target.selector


# ------------------------------------------------------------------
# 11. All event kinds map correctly
# ------------------------------------------------------------------


class TestAllEventKindsMapped:
    def test_all_kinds_map(self) -> None:
        """Every known event kind maps to its expected intent."""
        kind_to_intent = {
            "ClickIntent": "click",
            "FocusChange": "navigate",
            "AppSwitch": "switch_app",
            "DwellSnapshot": "read",
            "ScrollReadSnapshot": "scroll_read",
            "ClipboardChange": "copy",
            "PasteDetected": "paste",
            "WindowTitleChange": "navigate",
            "KeyPress": "type",
            "SecureFieldFocus": "secure_focus",
        }

        translator = SemanticTranslator()
        for kind, expected_intent in kind_to_intent.items():
            event = _make_event(kind=kind)
            result = translator.translate_event(event)
            assert result.intent == expected_intent, (
                f"Kind {kind!r} should map to {expected_intent!r}, got {result.intent!r}"
            )

    def test_unknown_kind(self) -> None:
        """Unrecognized event kinds map to 'unknown'."""
        event = _make_event(kind="SomeFutureEventKind")
        translator = SemanticTranslator()
        result = translator.translate_event(event)
        assert result.intent == "unknown"
