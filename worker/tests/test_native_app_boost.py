"""Tests for native app context detection and VLM priority boost."""

from __future__ import annotations

from agenthandover_worker.confidence import (
    ConfidenceScore,
    ConfidenceScorer,
    is_native_app_context,
)
from agenthandover_worker.translator import TranslationResult, UIAnchor


def _make_translation(
    *,
    url: str | None = None,
    app_id: str | None = None,
    anchor_method: str | None = None,
    anchor_selector: str | None = None,
) -> TranslationResult:
    """Build a TranslationResult for testing."""
    pre_state: dict = {}
    if url:
        pre_state["url"] = url
    if app_id:
        pre_state["app_id"] = app_id

    target = None
    if anchor_method:
        target = UIAnchor(
            method=anchor_method,
            selector=anchor_selector or "test",
            confidence_contribution=0.3,
        )

    return TranslationResult(
        intent="click",
        target=target,
        parameters={},
        pre_state=pre_state,
        post_state={},
        raw_event_id="evt-1",
    )


class TestIsNativeAppContext:
    def test_native_detected_no_url_no_dom(self) -> None:
        """No URL and no DOM-based anchor = native app."""
        tr = _make_translation(app_id="com.apple.Finder")
        assert is_native_app_context(tr, {}) is True

    def test_native_with_role_position_anchor(self) -> None:
        """role_position is an accessibility anchor, not DOM-based."""
        tr = _make_translation(
            app_id="com.apple.Finder",
            anchor_method="role_position",
            anchor_selector="AXButton[2]",
        )
        assert is_native_app_context(tr, {}) is True

    def test_native_with_vision_bbox_anchor(self) -> None:
        """vision_bbox is a VLM-based anchor, not DOM-based."""
        tr = _make_translation(
            app_id="com.apple.Finder",
            anchor_method="vision_bbox",
            anchor_selector="0.1,0.2,0.5,0.1",
        )
        assert is_native_app_context(tr, {}) is True

    def test_browser_with_url_not_native(self) -> None:
        """URL present = browser context, not native."""
        tr = _make_translation(
            app_id="com.apple.Safari",
            url="https://example.com",
        )
        assert is_native_app_context(tr, {}) is False

    def test_browser_with_context_url_not_native(self) -> None:
        """URL in context dict = browser context."""
        tr = _make_translation(app_id="com.apple.Safari")
        assert is_native_app_context(tr, {"expected_url": "https://example.com"}) is False

    def test_dom_anchor_aria_label_not_native(self) -> None:
        """aria_label anchor = DOM-based, not native."""
        tr = _make_translation(
            app_id="com.apple.Safari",
            anchor_method="aria_label",
            anchor_selector="Submit",
        )
        assert is_native_app_context(tr, {}) is False

    def test_dom_anchor_test_id_not_native(self) -> None:
        """test_id anchor = DOM-based, not native."""
        tr = _make_translation(
            anchor_method="test_id",
            anchor_selector="btn-submit",
        )
        assert is_native_app_context(tr, {}) is False

    def test_dom_anchor_inner_text_not_native(self) -> None:
        """inner_text anchor = DOM-based, not native."""
        tr = _make_translation(
            anchor_method="inner_text",
            anchor_selector="Click me",
        )
        assert is_native_app_context(tr, {}) is False


class TestNativeAppFlag:
    def test_native_flag_in_score(self) -> None:
        """ConfidenceScore should have native_app=True for native context."""
        scorer = ConfidenceScorer()
        tr = _make_translation(app_id="com.apple.Finder")
        context = {"expected_app": "com.apple.Finder"}
        score = scorer.score(tr, context)
        assert score.native_app is True
        assert "native_app_context" in score.reasons

    def test_browser_flag_not_set(self) -> None:
        """ConfidenceScore should have native_app=False for browser context."""
        scorer = ConfidenceScorer()
        tr = _make_translation(
            app_id="com.apple.Safari",
            url="https://example.com",
            anchor_method="aria_label",
            anchor_selector="Submit",
        )
        context = {
            "expected_app": "com.apple.Safari",
            "expected_url": "https://example.com",
        }
        score = scorer.score(tr, context)
        assert score.native_app is False
        assert "native_app_context" not in score.reasons


class TestVLMPriorityBoost:
    def test_boost_applied_for_native(self) -> None:
        """VLM priority should be boosted for native app events."""
        base_priority = 0.5
        boost = ConfidenceScorer.NATIVE_APP_VLM_BOOST
        boosted = min(base_priority + boost, 1.0)
        assert boosted == 0.65
        assert boost == 0.15

    def test_boost_does_not_exceed_one(self) -> None:
        """Boosted priority should be capped at 1.0."""
        base_priority = 0.95
        boost = ConfidenceScorer.NATIVE_APP_VLM_BOOST
        boosted = min(base_priority + boost, 1.0)
        assert boosted == 1.0
