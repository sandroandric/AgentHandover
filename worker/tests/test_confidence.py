"""Tests for agenthandover_worker.confidence.

Covers the three-component additive scoring model, threshold decisions,
reason generation, and edge cases.
"""

from __future__ import annotations

from agenthandover_worker.confidence import ConfidenceScore, ConfidenceScorer
from agenthandover_worker.translator import TranslationResult, UIAnchor


def _make_translation(
    *,
    intent: str = "click",
    anchor_method: str | None = "aria_label",
    anchor_confidence: float = 0.45,
    pre_state: dict | None = None,
    post_state: dict | None = None,
    parameters: dict | None = None,
    event_id: str = "test-event-1",
) -> TranslationResult:
    """Build a TranslationResult for scoring tests."""
    target = None
    if anchor_method is not None:
        target = UIAnchor(
            method=anchor_method,
            selector=f"[{anchor_method}]",
            confidence_contribution=anchor_confidence,
        )

    return TranslationResult(
        intent=intent,
        target=target,
        parameters=parameters or {},
        pre_state=pre_state or {},
        post_state=post_state or {},
        raw_event_id=event_id,
    )


# ------------------------------------------------------------------
# 1. High confidence → accept
# ------------------------------------------------------------------


class TestHighConfidenceAccept:
    def test_high_confidence_accept(self) -> None:
        """Score >= 0.85 → decision 'accept'."""
        translation = _make_translation(
            anchor_method="aria_label",
            anchor_confidence=0.45,
            pre_state={
                "window_title": "GitHub - PR",
                "url": "https://github.com/repo",
                "app_id": "com.apple.Safari",
            },
        )
        context = {
            "expected_title": "GitHub - PR",
            "expected_url": "https://github.com/repo",
            "expected_app": "com.apple.Safari",
            "clipboard_link": True,
        }

        scorer = ConfidenceScorer()
        score = scorer.score(translation, context)

        # UI: 0.45 + State: 0.35 + Provenance: 0.10 = 0.90
        assert score.total >= 0.85
        assert score.decision == "accept"
        assert "strong_ui_anchor" in score.reasons
        assert "state_matches" in score.reasons


# ------------------------------------------------------------------
# 2. Medium confidence → accept_flagged
# ------------------------------------------------------------------


class TestMediumConfidenceFlagged:
    def test_medium_confidence_flagged(self) -> None:
        """Score between 0.60 and 0.85 → decision 'accept_flagged'."""
        translation = _make_translation(
            anchor_method="inner_text",
            anchor_confidence=0.30,
            pre_state={
                "window_title": "Dashboard",
                "url": "https://app.example.com",
            },
        )
        context = {
            "expected_title": "Dashboard",
            "expected_url": "https://app.example.com",
            "dwell_snapshot": True,
        }

        scorer = ConfidenceScorer()
        score = scorer.score(translation, context)

        # UI: 0.30 + State: 0.25 + Provenance: 0.10 = 0.65
        assert 0.60 <= score.total < 0.85
        assert score.decision == "accept_flagged"


# ------------------------------------------------------------------
# 3. Low confidence → reject
# ------------------------------------------------------------------


class TestLowConfidenceReject:
    def test_low_confidence_reject(self) -> None:
        """Score < 0.60 → decision 'reject'."""
        translation = _make_translation(
            anchor_method="vision_bbox",
            anchor_confidence=0.10,
            pre_state={"window_title": "Unknown Window"},
        )
        context = {
            "expected_title": "Completely Different",
        }

        scorer = ConfidenceScorer()
        score = scorer.score(translation, context)

        # UI: 0.10 + State: 0.0 (mismatch) + Provenance: 0.0 = 0.10
        assert score.total < 0.60
        assert score.decision == "reject"
        assert "strong_ui_anchor" not in score.reasons


# ------------------------------------------------------------------
# 4. UI anchor scoring — ARIA gives 0.40+
# ------------------------------------------------------------------


class TestUIAnchorScoring:
    def test_aria_anchor_score(self) -> None:
        """ARIA anchor gives ui_anchor_score of 0.45."""
        translation = _make_translation(
            anchor_method="aria_label",
            anchor_confidence=0.45,
        )

        scorer = ConfidenceScorer()
        score = scorer.score(translation, {})

        assert score.ui_anchor_score == 0.45
        assert "strong_ui_anchor" in score.reasons

    def test_inner_text_anchor_score(self) -> None:
        """Inner text anchor gives a lower UI score."""
        translation = _make_translation(
            anchor_method="inner_text",
            anchor_confidence=0.30,
        )

        scorer = ConfidenceScorer()
        score = scorer.score(translation, {})

        assert score.ui_anchor_score == 0.30
        # 0.30 < 0.35, so no "strong_ui_anchor" reason
        assert "strong_ui_anchor" not in score.reasons

    def test_ui_score_capped_at_045(self) -> None:
        """UI anchor score never exceeds 0.45 even if contribution is higher."""
        translation = _make_translation(
            anchor_method="aria_label",
            anchor_confidence=0.99,  # artificially high
        )

        scorer = ConfidenceScorer()
        score = scorer.score(translation, {})

        assert score.ui_anchor_score == 0.45


# ------------------------------------------------------------------
# 5. State match scoring — matching title/URL/app
# ------------------------------------------------------------------


class TestStateMatchScoring:
    def test_all_state_matches(self) -> None:
        """All three state dimensions matching gives 0.35."""
        translation = _make_translation(
            anchor_method=None,
            pre_state={
                "window_title": "My App",
                "url": "https://example.com",
                "app_id": "com.example.App",
            },
        )
        context = {
            "expected_title": "My App",
            "expected_url": "https://example.com",
            "expected_app": "com.example.App",
        }

        scorer = ConfidenceScorer()
        score = scorer.score(translation, context)

        assert score.state_match_score == 0.35
        assert "state_matches" in score.reasons

    def test_partial_state_match(self) -> None:
        """Only title matches → 0.15 state score."""
        translation = _make_translation(
            anchor_method=None,
            pre_state={
                "window_title": "My App",
                "url": "https://other.com",
            },
        )
        context = {
            "expected_title": "My App",
            "expected_url": "https://example.com",
        }

        scorer = ConfidenceScorer()
        score = scorer.score(translation, context)

        assert score.state_match_score == 0.15

    def test_no_state_context(self) -> None:
        """No expected state in context → state_match_score = 0."""
        translation = _make_translation(
            anchor_method=None,
            pre_state={"window_title": "App"},
        )

        scorer = ConfidenceScorer()
        score = scorer.score(translation, {})

        assert score.state_match_score == 0.0


# ------------------------------------------------------------------
# 6. Provenance scoring — clipboard + dwell
# ------------------------------------------------------------------


class TestProvenanceScoring:
    def test_both_provenance_signals(self) -> None:
        """Clipboard link + dwell snapshot → 0.20 provenance."""
        translation = _make_translation(anchor_method=None)
        context = {
            "clipboard_link": True,
            "dwell_snapshot": True,
        }

        scorer = ConfidenceScorer()
        score = scorer.score(translation, context)

        assert score.provenance_score == 0.20
        assert "provenance_confirmed" in score.reasons

    def test_single_provenance_signal(self) -> None:
        """Only clipboard link → 0.10 provenance."""
        translation = _make_translation(anchor_method=None)
        context = {"clipboard_link": True}

        scorer = ConfidenceScorer()
        score = scorer.score(translation, context)

        assert score.provenance_score == 0.10
        # 0.10 < 0.15, so no "provenance_confirmed" reason
        assert "provenance_confirmed" not in score.reasons

    def test_no_provenance(self) -> None:
        """No provenance signals → 0.0."""
        translation = _make_translation(anchor_method=None)

        scorer = ConfidenceScorer()
        score = scorer.score(translation, {})

        assert score.provenance_score == 0.0


# ------------------------------------------------------------------
# 7. No anchor → very low score
# ------------------------------------------------------------------


class TestNoAnchorLowScore:
    def test_no_anchor_low_score(self) -> None:
        """No UI anchor means ui_anchor_score=0, leading to low total."""
        translation = _make_translation(anchor_method=None)

        scorer = ConfidenceScorer()
        score = scorer.score(translation, {})

        assert score.ui_anchor_score == 0.0
        assert score.total < 0.60
        assert score.decision == "reject"

    def test_no_anchor_with_full_state_and_provenance(self) -> None:
        """Even with perfect state + provenance, no anchor limits score."""
        translation = _make_translation(
            anchor_method=None,
            pre_state={
                "window_title": "App",
                "url": "https://app.com",
                "app_id": "com.test.App",
            },
        )
        context = {
            "expected_title": "App",
            "expected_url": "https://app.com",
            "expected_app": "com.test.App",
            "clipboard_link": True,
            "dwell_snapshot": True,
        }

        scorer = ConfidenceScorer()
        score = scorer.score(translation, context)

        # 0.0 + 0.35 + 0.20 = 0.55 → reject
        assert score.total == 0.55
        assert score.decision == "reject"


# ------------------------------------------------------------------
# 8. All components at maximum → perfect score
# ------------------------------------------------------------------


class TestAllComponentsMax:
    def test_all_components_max(self) -> None:
        """Perfect evidence: ARIA + full state match + full provenance = 1.0."""
        translation = _make_translation(
            anchor_method="aria_label",
            anchor_confidence=0.45,
            pre_state={
                "window_title": "Target App",
                "url": "https://target.com/page",
                "app_id": "com.target.App",
            },
        )
        context = {
            "expected_title": "Target App",
            "expected_url": "https://target.com/page",
            "expected_app": "com.target.App",
            "clipboard_link": True,
            "dwell_snapshot": True,
        }

        scorer = ConfidenceScorer()
        score = scorer.score(translation, context)

        # 0.45 + 0.35 + 0.20 = 1.00
        assert score.total == 1.0
        assert score.ui_anchor_score == 0.45
        assert score.state_match_score == 0.35
        assert score.provenance_score == 0.20
        assert score.decision == "accept"

        assert "strong_ui_anchor" in score.reasons
        assert "state_matches" in score.reasons
        assert "provenance_confirmed" in score.reasons

    def test_score_capped_at_one(self) -> None:
        """Total confidence never exceeds 1.0 even with high sub-scores."""
        # Artificially inflate the anchor confidence beyond cap
        # (the scorer caps it at 0.45)
        translation = _make_translation(
            anchor_method="aria_label",
            anchor_confidence=0.45,
            pre_state={
                "window_title": "X",
                "url": "https://x.com",
                "app_id": "x",
            },
        )
        context = {
            "expected_title": "X",
            "expected_url": "https://x.com",
            "expected_app": "x",
            "clipboard_link": True,
            "dwell_snapshot": True,
        }

        scorer = ConfidenceScorer()
        score = scorer.score(translation, context)

        assert score.total <= 1.0


# ------------------------------------------------------------------
# 9. Threshold boundary tests
# ------------------------------------------------------------------


class TestThresholdBoundaries:
    def test_exact_accept_boundary(self) -> None:
        """Score exactly 0.85 → accept."""
        # ARIA 0.45 + title match 0.15 + url match 0.10 + app match 0.10 + clipboard 0.10 = 0.90
        # We need exactly 0.85: ARIA 0.45 + title 0.15 + url 0.10 + clipboard 0.10 + dwell 0.10 = 0.90
        # Try: ARIA 0.45 + title 0.15 + url 0.10 + clipboard 0.10 + no dwell = 0.80 nope
        # ARIA 0.45 + title 0.15 + url 0.10 + app 0.10 + clipboard 0.10 - nope thats 0.90
        # inner_text 0.35 + title 0.15 + url 0.10 + app 0.10 + clipboard 0.10 + dwell 0.10 = 0.90
        # test_id 0.45 + title 0.15 + url 0.10 + clipboard 0.10 + dwell 0.10 = 0.90
        # test_id 0.45 + title 0.15 + url 0.10 + clipboard 0.10 = 0.80 nope
        # test_id 0.45 + title 0.15 + url 0.10 + app 0.10 + clipboard 0.10 = 0.90
        # ARIA 0.45 + title 0.15 + url 0.10 + clipboard 0.10 + dwell 0.10 = 0.90
        # Let's try: ARIA 0.40 + title 0.15 + url 0.10 + app 0.10 + clipboard 0.10 = 0.85
        translation = _make_translation(
            anchor_method="aria_label",
            anchor_confidence=0.40,
            pre_state={
                "window_title": "App",
                "url": "https://app.com",
                "app_id": "com.app",
            },
        )
        context = {
            "expected_title": "App",
            "expected_url": "https://app.com",
            "expected_app": "com.app",
            "clipboard_link": True,
        }

        scorer = ConfidenceScorer()
        score = scorer.score(translation, context)

        assert score.total == 0.85
        assert score.decision == "accept"

    def test_just_below_accept(self) -> None:
        """Score just below 0.85 → accept_flagged."""
        translation = _make_translation(
            anchor_method="inner_text",
            anchor_confidence=0.35,
            pre_state={
                "window_title": "App",
                "url": "https://app.com",
                "app_id": "com.app",
            },
        )
        context = {
            "expected_title": "App",
            "expected_url": "https://app.com",
            "expected_app": "com.app",
            # no provenance → 0.35 + 0.35 = 0.70
        }

        scorer = ConfidenceScorer()
        score = scorer.score(translation, context)

        assert score.total == 0.70
        assert score.decision == "accept_flagged"

    def test_exact_flag_boundary(self) -> None:
        """Score exactly 0.60 → accept_flagged."""
        # vision_bbox 0.10 + title 0.15 + url 0.10 + app 0.10 + clipboard 0.10 + dwell 0.10 = 0.65
        # inner_text 0.25 + title 0.15 + url 0.10 + clipboard 0.10 = 0.60
        translation = _make_translation(
            anchor_method="inner_text",
            anchor_confidence=0.25,
            pre_state={
                "window_title": "X",
                "url": "https://x.com",
            },
        )
        context = {
            "expected_title": "X",
            "expected_url": "https://x.com",
            "clipboard_link": True,
        }

        scorer = ConfidenceScorer()
        score = scorer.score(translation, context)

        assert score.total == 0.60
        assert score.decision == "accept_flagged"
