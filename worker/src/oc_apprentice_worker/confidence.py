"""Confidence Scoring Engine — three-component additive confidence score.

Implements section 9.2 of the OpenMimic spec: each semantic step receives
a numeric confidence score in [0.0, 1.0] composed of three independent
components:

  UI anchor resolution  (0.0–0.45)
    Found target via role/name/aria-label/testid, stable across sessions.

  State match           (0.0–0.35)
    Current UI state matches expected preconditions (title, URL, app).

  Provenance            (0.0–0.20)
    Clipboard link or dwell snapshot supports where data came from.

Threshold behavior (learning stage):
  >= 0.85 : accept as strong semantic step
  0.60–0.85 : accept but mark "needs more examples"  (accept_flagged)
  <  0.60 : reject; enqueue VLM job if enabled
"""

from __future__ import annotations

from dataclasses import dataclass, field

from oc_apprentice_worker.translator import TranslationResult, UIAnchor


@dataclass
class ConfidenceScore:
    """Full confidence breakdown for a translated semantic step."""

    total: float  # 0.0-1.0
    ui_anchor_score: float  # 0.0-0.45
    state_match_score: float  # 0.0-0.35
    provenance_score: float  # 0.0-0.20
    reasons: list[str] = field(default_factory=list)
    decision: str = "reject"  # "accept", "accept_flagged", "reject"
    evidence: dict = field(default_factory=dict)
    # Evidence dict contains: dom_anchor, ax_path, vision_bbox,
    # screenshot_id, url, window_title


class ConfidenceScorer:
    """Compute confidence scores for translated semantic steps.

    The scorer evaluates three independent axes — UI anchor quality,
    state matching, and data provenance — then sums them (capped at 1.0)
    and maps the total to a decision.
    """

    ACCEPT_THRESHOLD: float = 0.85
    FLAG_THRESHOLD: float = 0.60

    def score(
        self,
        translation: TranslationResult,
        context: dict,
    ) -> ConfidenceScore:
        """Compute confidence score for a translation result.

        Parameters
        ----------
        translation:
            The semantic translation to score.
        context:
            Environment context that carries expected state and provenance
            signals.  Recognized keys:

            - ``expected_title`` — expected window title before the action
            - ``expected_url`` — expected URL before the action
            - ``expected_app`` — expected app identifier
            - ``clipboard_link`` — ``True`` when a clipboard copy-paste
              link has been established for this step
            - ``dwell_snapshot`` — ``True`` when a dwell (reading) snapshot
              supports this step's provenance
        """
        # Reject unknown intents — they should not enter SOPs
        if translation.intent == "unknown":
            return ConfidenceScore(
                total=0.0,
                ui_anchor_score=0.0,
                state_match_score=0.0,
                provenance_score=0.0,
                reasons=["unknown_intent"],
                decision="reject",
                evidence=self._build_evidence(translation, context),
            )

        ui_score = self._score_ui_anchor(translation.target)
        state_score = self._score_state_match(translation, context)
        provenance = self._score_provenance(translation, context)

        total = max(0.0, min(ui_score + state_score + provenance, 1.0))

        reasons: list[str] = []
        if ui_score >= 0.35:
            reasons.append("strong_ui_anchor")
        if state_score >= 0.25:
            reasons.append("state_matches")
        if provenance >= 0.15:
            reasons.append("provenance_confirmed")

        decision = self._decide(total)

        # Build evidence dict from translation and context
        evidence = self._build_evidence(translation, context)

        return ConfidenceScore(
            total=total,
            ui_anchor_score=ui_score,
            state_match_score=state_score,
            provenance_score=provenance,
            reasons=reasons,
            decision=decision,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Scoring components
    # ------------------------------------------------------------------

    def _score_ui_anchor(self, anchor: UIAnchor | None) -> float:
        """Score based on UI anchor quality. Max 0.45.

        If no anchor was resolved the score is 0.0.  Otherwise the
        anchor's ``confidence_contribution`` is used directly (it was
        already set by the translator's resolution methods).
        """
        if anchor is None:
            return 0.0
        return min(anchor.confidence_contribution, 0.45)

    def _score_state_match(
        self,
        translation: TranslationResult,
        context: dict,
    ) -> float:
        """Score based on pre/post state matching expectations. Max 0.35.

        Awards partial credit for each matching dimension:
          - Window title matches expected_title: +0.15
          - URL matches expected_url:            +0.10
          - App matches expected_app:            +0.10
        """
        score = 0.0

        # Window title match
        pre_title = translation.pre_state.get("window_title")
        expected_title = context.get("expected_title")
        if pre_title and expected_title:
            if pre_title == expected_title:
                score += 0.15

        # URL match
        pre_url = translation.pre_state.get("url")
        expected_url = context.get("expected_url")
        if pre_url and expected_url:
            if pre_url == expected_url:
                score += 0.10

        # App match
        pre_app = translation.pre_state.get("app_id")
        expected_app = context.get("expected_app")
        if pre_app and expected_app:
            if pre_app == expected_app:
                score += 0.10

        return min(score, 0.35)

    def _score_provenance(
        self,
        translation: TranslationResult,
        context: dict,
    ) -> float:
        """Score based on data provenance signals. Max 0.20.

        Awards partial credit for each provenance signal:
          - Clipboard copy-paste link established: +0.10
          - Dwell (reading) snapshot supports step: +0.10
        """
        score = 0.0

        if context.get("clipboard_link"):
            score += 0.10

        if context.get("dwell_snapshot"):
            score += 0.10

        return min(score, 0.20)

    # ------------------------------------------------------------------
    # Evidence building
    # ------------------------------------------------------------------

    def _build_evidence(
        self,
        translation: TranslationResult,
        context: dict,
    ) -> dict:
        """Build evidence dict from translation and context.

        Collects dom_anchor, ax_path, vision_bbox, screenshot_id,
        url, and window_title from the translation's target, pre_state,
        and scoring context.
        """
        evidence: dict = {
            "dom_anchor": None,
            "ax_path": None,
            "vision_bbox": None,
            "screenshot_id": None,
            "url": None,
            "window_title": None,
        }

        # Extract from UI anchor
        if translation.target is not None:
            if translation.target.method in ("aria_label", "test_id", "inner_text"):
                evidence["dom_anchor"] = translation.target.selector
            elif translation.target.method == "role_position":
                evidence["ax_path"] = translation.target.selector
            elif translation.target.method == "vision_bbox":
                evidence["vision_bbox"] = translation.target.raw_evidence

        # Extract from pre_state
        evidence["url"] = translation.pre_state.get("url")
        evidence["window_title"] = translation.pre_state.get("window_title")

        # Extract screenshot_id from context if available
        evidence["screenshot_id"] = context.get("screenshot_id")

        return evidence

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _decide(self, total: float) -> str:
        """Map total confidence to a decision string.

        - ``"accept"`` when total >= ACCEPT_THRESHOLD (0.85)
        - ``"accept_flagged"`` when total >= FLAG_THRESHOLD (0.60)
        - ``"reject"`` otherwise
        """
        if total >= self.ACCEPT_THRESHOLD:
            return "accept"
        if total >= self.FLAG_THRESHOLD:
            return "accept_flagged"
        return "reject"
