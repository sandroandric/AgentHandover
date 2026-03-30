"""Tests for the SemanticStep data model.

Covers:
1. Basic creation with all fields
2. to_dict serialization
3. from_dict deserialization
4. to_sop_step simplified output
5. Evidence fields with all anchor types
6. Negative marking flag
7. Default values
8. Full round-trip (to_dict -> from_dict preserves all fields)
9. Edge cases: missing optional fields, empty parameters
10. Timestamp handling: ISO strings with Z suffix, timezone-aware
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agenthandover_worker.models.semantic_step import Evidence, SemanticStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_full_step() -> SemanticStep:
    """Create a SemanticStep with every field populated."""
    return SemanticStep(
        step_id="step-001",
        episode_id="ep-abc",
        step_index=3,
        intent="click",
        target_description="Submit button in review form",
        target_selector="[aria-label='Submit review']",
        parameters={"text": "LGTM", "modifier": "ctrl"},
        pre_state={
            "window_title": "PR #123",
            "url": "https://github.com/org/repo/pull/123",
            "app_id": "com.google.Chrome",
        },
        post_state={
            "window_title": "PR #123 — merged",
            "url": "https://github.com/org/repo/pull/123",
        },
        confidence=0.85,
        confidence_reasons=["dom_anchor_match", "aria_label_found"],
        decision="accept",
        evidence=Evidence(
            dom_anchor="button.submit-review",
            ax_path="window > dialog > button[Submit review]",
            vision_bbox={"x": 100, "y": 200, "width": 80, "height": 30},
            screenshot_id="ss-123",
            url="https://github.com/org/repo/pull/123",
            window_title="PR #123",
        ),
        raw_event_id="evt-789",
        timestamp=datetime(2026, 2, 16, 10, 30, 0, tzinfo=timezone.utc),
        is_negative=False,
    )


# ---------------------------------------------------------------------------
# Test 1: Basic creation with all fields
# ---------------------------------------------------------------------------


class TestCreateSemanticStep:
    def test_all_fields_populated(self) -> None:
        step = _make_full_step()

        assert step.step_id == "step-001"
        assert step.episode_id == "ep-abc"
        assert step.step_index == 3
        assert step.intent == "click"
        assert step.target_description == "Submit button in review form"
        assert step.target_selector == "[aria-label='Submit review']"
        assert step.parameters == {"text": "LGTM", "modifier": "ctrl"}
        assert step.pre_state["app_id"] == "com.google.Chrome"
        assert step.post_state["window_title"] == "PR #123 — merged"
        assert step.confidence == 0.85
        assert len(step.confidence_reasons) == 2
        assert step.decision == "accept"
        assert step.evidence.dom_anchor == "button.submit-review"
        assert step.raw_event_id == "evt-789"
        assert step.timestamp == datetime(2026, 2, 16, 10, 30, 0, tzinfo=timezone.utc)
        assert step.is_negative is False

    def test_required_fields_only(self) -> None:
        step = SemanticStep(
            step_id="s1",
            episode_id="e1",
            step_index=0,
            intent="navigate",
            target_description="Address bar",
        )
        assert step.step_id == "s1"
        assert step.intent == "navigate"
        assert step.confidence == 0.0
        assert step.decision == "reject"


# ---------------------------------------------------------------------------
# Test 2: to_dict serialization
# ---------------------------------------------------------------------------


class TestToDictSerialization:
    def test_serializes_all_fields(self) -> None:
        step = _make_full_step()
        d = step.to_dict()

        assert d["step_id"] == "step-001"
        assert d["episode_id"] == "ep-abc"
        assert d["step_index"] == 3
        assert d["intent"] == "click"
        assert d["target_description"] == "Submit button in review form"
        assert d["target_selector"] == "[aria-label='Submit review']"
        assert d["parameters"] == {"text": "LGTM", "modifier": "ctrl"}
        assert d["confidence"] == 0.85
        assert d["confidence_reasons"] == ["dom_anchor_match", "aria_label_found"]
        assert d["decision"] == "accept"
        assert d["raw_event_id"] == "evt-789"
        assert d["is_negative"] is False

    def test_timestamp_serialized_as_iso_string(self) -> None:
        step = _make_full_step()
        d = step.to_dict()
        assert isinstance(d["timestamp"], str)
        assert "2026-02-16" in d["timestamp"]

    def test_none_timestamp_serialized_as_none(self) -> None:
        step = SemanticStep(
            step_id="s1",
            episode_id="e1",
            step_index=0,
            intent="click",
            target_description="Button",
        )
        d = step.to_dict()
        assert d["timestamp"] is None

    def test_evidence_serialized_as_dict(self) -> None:
        step = _make_full_step()
        d = step.to_dict()
        assert isinstance(d["evidence"], dict)
        assert d["evidence"]["dom_anchor"] == "button.submit-review"
        assert d["evidence"]["vision_bbox"]["width"] == 80

    def test_confidence_reasons_is_list_copy(self) -> None:
        """Verify the serialized list is a copy, not a reference."""
        step = _make_full_step()
        d = step.to_dict()
        d["confidence_reasons"].append("mutated")
        assert len(step.confidence_reasons) == 2  # original unchanged


# ---------------------------------------------------------------------------
# Test 3: from_dict deserialization
# ---------------------------------------------------------------------------


class TestFromDictDeserialization:
    def test_reconstructs_from_dict(self) -> None:
        original = _make_full_step()
        d = original.to_dict()
        restored = SemanticStep.from_dict(d)

        assert restored.step_id == original.step_id
        assert restored.episode_id == original.episode_id
        assert restored.step_index == original.step_index
        assert restored.intent == original.intent
        assert restored.target_description == original.target_description
        assert restored.target_selector == original.target_selector
        assert restored.confidence == original.confidence
        assert restored.decision == original.decision
        assert restored.is_negative == original.is_negative

    def test_handles_z_suffix_timestamp(self) -> None:
        d = {
            "step_id": "s1",
            "episode_id": "e1",
            "step_index": 0,
            "intent": "click",
            "target_description": "Btn",
            "timestamp": "2026-02-16T10:30:00Z",
        }
        step = SemanticStep.from_dict(d)
        assert step.timestamp is not None
        assert step.timestamp.year == 2026
        assert step.timestamp.tzinfo is not None

    def test_handles_missing_optional_fields(self) -> None:
        d = {
            "step_id": "s1",
            "episode_id": "e1",
            "step_index": 0,
            "intent": "scroll",
            "target_description": "Page body",
        }
        step = SemanticStep.from_dict(d)
        assert step.target_selector is None
        assert step.parameters == {}
        assert step.pre_state == {}
        assert step.post_state == {}
        assert step.confidence == 0.0
        assert step.confidence_reasons == []
        assert step.decision == "reject"
        assert step.raw_event_id == ""
        assert step.timestamp is None
        assert step.is_negative is False

    def test_handles_none_timestamp(self) -> None:
        d = {
            "step_id": "s1",
            "episode_id": "e1",
            "step_index": 0,
            "intent": "click",
            "target_description": "Btn",
            "timestamp": None,
        }
        step = SemanticStep.from_dict(d)
        assert step.timestamp is None

    def test_handles_invalid_timestamp_gracefully(self) -> None:
        d = {
            "step_id": "s1",
            "episode_id": "e1",
            "step_index": 0,
            "intent": "click",
            "target_description": "Btn",
            "timestamp": "not-a-date",
        }
        step = SemanticStep.from_dict(d)
        assert step.timestamp is None


# ---------------------------------------------------------------------------
# Test 4: to_sop_step simplified output
# ---------------------------------------------------------------------------


class TestToSopStep:
    def test_produces_simplified_dict(self) -> None:
        step = _make_full_step()
        sop = step.to_sop_step()

        assert sop == {
            "step": "click",
            "target": "Submit button in review form",
            "selector": "[aria-label='Submit review']",
            "parameters": {"text": "LGTM", "modifier": "ctrl"},
            "confidence": 0.85,
        }

    def test_sop_step_has_exactly_five_keys(self) -> None:
        step = _make_full_step()
        sop = step.to_sop_step()
        assert set(sop.keys()) == {"step", "target", "selector", "parameters", "confidence"}

    def test_sop_step_with_no_selector(self) -> None:
        step = SemanticStep(
            step_id="s1",
            episode_id="e1",
            step_index=0,
            intent="scroll",
            target_description="Page body",
        )
        sop = step.to_sop_step()
        assert sop["selector"] is None
        assert sop["step"] == "scroll"


# ---------------------------------------------------------------------------
# Test 5: Evidence fields with all anchor types
# ---------------------------------------------------------------------------


class TestEvidenceFields:
    def test_all_evidence_fields_populated(self) -> None:
        ev = Evidence(
            dom_anchor="div#main > button.submit",
            ax_path="AXApplication > AXWindow > AXButton[Submit]",
            vision_bbox={"x": 50, "y": 100, "width": 120, "height": 40},
            screenshot_id="artifact-ss-456",
            url="https://example.com/form",
            window_title="Form — Example",
        )
        assert ev.dom_anchor == "div#main > button.submit"
        assert ev.ax_path == "AXApplication > AXWindow > AXButton[Submit]"
        assert ev.vision_bbox["width"] == 120
        assert ev.screenshot_id == "artifact-ss-456"
        assert ev.url == "https://example.com/form"
        assert ev.window_title == "Form — Example"

    def test_evidence_to_dict(self) -> None:
        ev = Evidence(
            dom_anchor="input#email",
            url="https://login.example.com",
        )
        d = ev.to_dict()
        assert d["dom_anchor"] == "input#email"
        assert d["url"] == "https://login.example.com"
        assert d["ax_path"] is None
        assert d["vision_bbox"] is None
        assert d["screenshot_id"] is None
        assert d["window_title"] is None

    def test_evidence_from_dict(self) -> None:
        d = {
            "dom_anchor": "a.nav-link",
            "ax_path": "window > nav > link[Home]",
            "vision_bbox": {"x": 10, "y": 20, "width": 60, "height": 25},
            "screenshot_id": "ss-99",
            "url": "https://app.com",
            "window_title": "Dashboard",
        }
        ev = Evidence.from_dict(d)
        assert ev.dom_anchor == "a.nav-link"
        assert ev.ax_path == "window > nav > link[Home]"
        assert ev.vision_bbox["height"] == 25
        assert ev.screenshot_id == "ss-99"

    def test_evidence_round_trip(self) -> None:
        original = Evidence(
            dom_anchor="span.label",
            ax_path="form > label[Email]",
            vision_bbox={"x": 0, "y": 0, "width": 200, "height": 16},
            screenshot_id="ss-1",
            url="https://x.com",
            window_title="Sign Up",
        )
        restored = Evidence.from_dict(original.to_dict())
        assert restored.dom_anchor == original.dom_anchor
        assert restored.ax_path == original.ax_path
        assert restored.vision_bbox == original.vision_bbox
        assert restored.screenshot_id == original.screenshot_id
        assert restored.url == original.url
        assert restored.window_title == original.window_title


# ---------------------------------------------------------------------------
# Test 6: Negative marking
# ---------------------------------------------------------------------------


class TestNegativeMarking:
    def test_default_is_not_negative(self) -> None:
        step = SemanticStep(
            step_id="s1",
            episode_id="e1",
            step_index=0,
            intent="click",
            target_description="OK",
        )
        assert step.is_negative is False

    def test_marked_negative_serializes(self) -> None:
        step = SemanticStep(
            step_id="s1",
            episode_id="e1",
            step_index=0,
            intent="click",
            target_description="Cancel button",
            is_negative=True,
        )
        d = step.to_dict()
        assert d["is_negative"] is True

    def test_negative_from_dict(self) -> None:
        d = {
            "step_id": "s1",
            "episode_id": "e1",
            "step_index": 0,
            "intent": "click",
            "target_description": "Cancel",
            "is_negative": True,
        }
        step = SemanticStep.from_dict(d)
        assert step.is_negative is True


# ---------------------------------------------------------------------------
# Test 7: Default values
# ---------------------------------------------------------------------------


class TestDefaultValues:
    def test_defaults_are_sensible(self) -> None:
        step = SemanticStep(
            step_id="s1",
            episode_id="e1",
            step_index=0,
            intent="type",
            target_description="Search box",
        )
        assert step.target_selector is None
        assert step.parameters == {}
        assert step.pre_state == {}
        assert step.post_state == {}
        assert step.confidence == 0.0
        assert step.confidence_reasons == []
        assert step.decision == "reject"
        assert isinstance(step.evidence, Evidence)
        assert step.evidence.dom_anchor is None
        assert step.raw_event_id == ""
        assert step.timestamp is None
        assert step.is_negative is False

    def test_evidence_defaults_all_none(self) -> None:
        ev = Evidence()
        assert ev.dom_anchor is None
        assert ev.ax_path is None
        assert ev.vision_bbox is None
        assert ev.screenshot_id is None
        assert ev.url is None
        assert ev.window_title is None


# ---------------------------------------------------------------------------
# Test 8: Full round-trip (to_dict -> from_dict preserves all fields)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_full_round_trip_preserves_all_fields(self) -> None:
        original = _make_full_step()
        d = original.to_dict()
        restored = SemanticStep.from_dict(d)

        assert restored.step_id == original.step_id
        assert restored.episode_id == original.episode_id
        assert restored.step_index == original.step_index
        assert restored.intent == original.intent
        assert restored.target_description == original.target_description
        assert restored.target_selector == original.target_selector
        assert restored.parameters == original.parameters
        assert restored.pre_state == original.pre_state
        assert restored.post_state == original.post_state
        assert restored.confidence == original.confidence
        assert restored.confidence_reasons == original.confidence_reasons
        assert restored.decision == original.decision
        assert restored.raw_event_id == original.raw_event_id
        assert restored.is_negative == original.is_negative

        # Timestamp comparison (ISO round-trip may lose sub-microsecond precision
        # but should preserve to the second)
        assert restored.timestamp is not None
        assert original.timestamp is not None
        assert restored.timestamp.year == original.timestamp.year
        assert restored.timestamp.month == original.timestamp.month
        assert restored.timestamp.day == original.timestamp.day
        assert restored.timestamp.hour == original.timestamp.hour
        assert restored.timestamp.minute == original.timestamp.minute
        assert restored.timestamp.second == original.timestamp.second

        # Evidence round-trip
        assert restored.evidence.dom_anchor == original.evidence.dom_anchor
        assert restored.evidence.ax_path == original.evidence.ax_path
        assert restored.evidence.vision_bbox == original.evidence.vision_bbox
        assert restored.evidence.screenshot_id == original.evidence.screenshot_id
        assert restored.evidence.url == original.evidence.url
        assert restored.evidence.window_title == original.evidence.window_title

    def test_double_round_trip(self) -> None:
        """Two round-trips should be identical."""
        original = _make_full_step()
        d1 = original.to_dict()
        r1 = SemanticStep.from_dict(d1)
        d2 = r1.to_dict()
        r2 = SemanticStep.from_dict(d2)

        assert d1 == d2
        assert r2.step_id == original.step_id
        assert r2.confidence == original.confidence

    def test_minimal_step_round_trip(self) -> None:
        """Round-trip a step with only required fields."""
        step = SemanticStep(
            step_id="min-1",
            episode_id="ep-1",
            step_index=0,
            intent="scroll",
            target_description="Viewport",
        )
        restored = SemanticStep.from_dict(step.to_dict())
        assert restored.step_id == "min-1"
        assert restored.intent == "scroll"
        assert restored.target_selector is None
        assert restored.parameters == {}
        assert restored.timestamp is None
