"""Tests for the evidence dict added to ConfidenceScore.

Covers evidence building from ARIA, test_id, inner_text, role_position,
and vision_bbox anchors, plus state extraction.
"""

from __future__ import annotations

from agenthandover_worker.confidence import ConfidenceScore, ConfidenceScorer
from agenthandover_worker.translator import TranslationResult, UIAnchor


def _make_translation(
    *,
    intent: str = "click",
    anchor_method: str | None = "aria_label",
    anchor_confidence: float = 0.45,
    anchor_selector: str = "[aria-label='Submit']",
    anchor_evidence: dict | None = None,
    pre_state: dict | None = None,
    post_state: dict | None = None,
    event_id: str = "test-event-1",
) -> TranslationResult:
    target = None
    if anchor_method is not None:
        target = UIAnchor(
            method=anchor_method,
            selector=anchor_selector,
            confidence_contribution=anchor_confidence,
            raw_evidence=anchor_evidence or {},
        )

    return TranslationResult(
        intent=intent,
        target=target,
        parameters={},
        pre_state=pre_state or {},
        post_state=post_state or {},
        raw_event_id=event_id,
    )


# ------------------------------------------------------------------
# 1. Evidence dict has all expected keys
# ------------------------------------------------------------------


class TestEvidenceDictKeys:
    def test_evidence_has_all_keys(self) -> None:
        translation = _make_translation()
        scorer = ConfidenceScorer()
        score = scorer.score(translation, {})

        expected_keys = {
            "dom_anchor", "ax_path", "vision_bbox",
            "screenshot_id", "url", "window_title",
        }
        assert set(score.evidence.keys()) == expected_keys


# ------------------------------------------------------------------
# 2. ARIA label produces dom_anchor
# ------------------------------------------------------------------


class TestEvidenceAriaLabel:
    def test_aria_label_sets_dom_anchor(self) -> None:
        translation = _make_translation(
            anchor_method="aria_label",
            anchor_selector="[aria-label='Submit']",
        )
        scorer = ConfidenceScorer()
        score = scorer.score(translation, {})

        assert score.evidence["dom_anchor"] == "[aria-label='Submit']"
        assert score.evidence["ax_path"] is None
        assert score.evidence["vision_bbox"] is None


# ------------------------------------------------------------------
# 3. Test ID produces dom_anchor
# ------------------------------------------------------------------


class TestEvidenceTestId:
    def test_test_id_sets_dom_anchor(self) -> None:
        translation = _make_translation(
            anchor_method="test_id",
            anchor_selector="[data-testid='submit-btn']",
        )
        scorer = ConfidenceScorer()
        score = scorer.score(translation, {})

        assert score.evidence["dom_anchor"] == "[data-testid='submit-btn']"


# ------------------------------------------------------------------
# 4. Inner text produces dom_anchor
# ------------------------------------------------------------------


class TestEvidenceInnerText:
    def test_inner_text_sets_dom_anchor(self) -> None:
        translation = _make_translation(
            anchor_method="inner_text",
            anchor_selector="text='submit'",
        )
        scorer = ConfidenceScorer()
        score = scorer.score(translation, {})

        assert score.evidence["dom_anchor"] == "text='submit'"


# ------------------------------------------------------------------
# 5. Role position produces ax_path
# ------------------------------------------------------------------


class TestEvidenceRolePosition:
    def test_role_position_sets_ax_path(self) -> None:
        translation = _make_translation(
            anchor_method="role_position",
            anchor_selector="[role=button, tag=button]",
        )
        scorer = ConfidenceScorer()
        score = scorer.score(translation, {})

        assert score.evidence["ax_path"] == "[role=button, tag=button]"
        assert score.evidence["dom_anchor"] is None


# ------------------------------------------------------------------
# 6. Vision bbox produces vision_bbox evidence
# ------------------------------------------------------------------


class TestEvidenceVisionBbox:
    def test_vision_bbox_sets_evidence(self) -> None:
        translation = _make_translation(
            anchor_method="vision_bbox",
            anchor_selector="bbox(100,200)",
            anchor_evidence={"x": 100, "y": 200},
        )
        scorer = ConfidenceScorer()
        score = scorer.score(translation, {})

        assert score.evidence["vision_bbox"] == {"x": 100, "y": 200}
        assert score.evidence["dom_anchor"] is None
        assert score.evidence["ax_path"] is None


# ------------------------------------------------------------------
# 7. URL and window_title from pre_state
# ------------------------------------------------------------------


class TestEvidenceFromPreState:
    def test_url_and_title_from_pre_state(self) -> None:
        translation = _make_translation(
            pre_state={
                "url": "https://github.com/repo",
                "window_title": "Pull Request #42",
            },
        )
        scorer = ConfidenceScorer()
        score = scorer.score(translation, {})

        assert score.evidence["url"] == "https://github.com/repo"
        assert score.evidence["window_title"] == "Pull Request #42"


# ------------------------------------------------------------------
# 8. Screenshot ID from context
# ------------------------------------------------------------------


class TestEvidenceScreenshotId:
    def test_screenshot_id_from_context(self) -> None:
        translation = _make_translation()
        scorer = ConfidenceScorer()
        score = scorer.score(translation, {"screenshot_id": "scr-abc-123"})

        assert score.evidence["screenshot_id"] == "scr-abc-123"

    def test_no_screenshot_id_when_not_in_context(self) -> None:
        translation = _make_translation()
        scorer = ConfidenceScorer()
        score = scorer.score(translation, {})

        assert score.evidence["screenshot_id"] is None


# ------------------------------------------------------------------
# 9. No anchor produces no dom_anchor or ax_path
# ------------------------------------------------------------------


class TestEvidenceNoAnchor:
    def test_no_anchor_produces_empty_evidence(self) -> None:
        translation = _make_translation(anchor_method=None)
        scorer = ConfidenceScorer()
        score = scorer.score(translation, {})

        assert score.evidence["dom_anchor"] is None
        assert score.evidence["ax_path"] is None
        assert score.evidence["vision_bbox"] is None
