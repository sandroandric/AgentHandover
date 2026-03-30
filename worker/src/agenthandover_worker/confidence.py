"""Confidence Scoring Engine — v1 and v2 confidence scoring.

v1 (section 9.2 of the AgentHandover spec): each semantic step receives a
numeric confidence score in [0.0, 1.0] composed of three independent
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

v2 (VLM-based pipeline): SOP-level confidence score based on four signals:

  Demonstration count   (weight 0.30)
    1 demo = 0.50, 2 demos = 0.70, 3+ demos = 0.90

  Step consistency      (weight 0.30)
    Fraction of steps present in all demonstrations.

  Annotation quality    (weight 0.20)
    Average fraction of non-empty fields in scene annotations.

  Variable detection    (weight 0.20)
    Fraction of varying values correctly detected as {{variables}}.

  Focus Recording gets a flat +0.10 bonus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agenthandover_worker.translator import TranslationResult, UIAnchor

logger = logging.getLogger(__name__)


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
    native_app: bool = False  # True when event comes from a native (non-web) app


def is_native_app_context(translation: TranslationResult, context: dict) -> bool:
    """Detect if this translation comes from a native (non-web) app context.

    Returns True when there is no URL AND no DOM-based UI anchor
    (aria_label, test_id, inner_text). Native app events lack the rich
    selectors that browser DOM provides, making VLM-based identification
    more valuable.
    """
    # If there's a URL, this is a web context
    if translation.pre_state.get("url"):
        return False
    if context.get("expected_url"):
        return False

    # If the UI anchor is DOM-based, this is a web context
    if translation.target is not None:
        dom_methods = {"aria_label", "test_id", "inner_text"}
        if translation.target.method in dom_methods:
            return False

    return True


class ConfidenceScorer:
    """Compute confidence scores for translated semantic steps.

    The scorer evaluates three independent axes — UI anchor quality,
    state matching, and data provenance — then sums them (capped at 1.0)
    and maps the total to a decision.
    """

    ACCEPT_THRESHOLD: float = 0.85
    FLAG_THRESHOLD: float = 0.60
    NATIVE_APP_VLM_BOOST: float = 0.15

    # Native app events lack DOM-level selectors (aria-label, test-id,
    # inner_text), so they can never reach web-level confidence scores.
    # Use lower thresholds for native contexts so that app-level anchors
    # (app name, window title) + VLM boost can still produce accepted
    # translations.
    NATIVE_ACCEPT_THRESHOLD: float = 0.40
    NATIVE_FLAG_THRESHOLD: float = 0.15

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

        # Apply VLM confidence boost if available (from completed VLM job)
        vlm_boost = float(context.get("vlm_boost", 0.0))

        total = max(0.0, min(ui_score + state_score + provenance + vlm_boost, 1.0))

        reasons: list[str] = []
        if ui_score >= 0.35:
            reasons.append("strong_ui_anchor")
        if state_score >= 0.25:
            reasons.append("state_matches")
        if provenance >= 0.15:
            reasons.append("provenance_confirmed")
        if vlm_boost > 0.0:
            reasons.append(f"vlm_boost_{vlm_boost:.2f}")

        # Detect native app context
        native_app = is_native_app_context(translation, context)
        if native_app:
            reasons.append("native_app_context")

        decision = self._decide(total, native_app=native_app)

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
            native_app=native_app,
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

    def _decide(self, total: float, *, native_app: bool = False) -> str:
        """Map total confidence to a decision string.

        For web contexts:
        - ``"accept"`` when total >= ACCEPT_THRESHOLD (0.85)
        - ``"accept_flagged"`` when total >= FLAG_THRESHOLD (0.60)
        - ``"reject"`` otherwise

        For native app contexts (no DOM-level selectors available):
        - ``"accept"`` when total >= NATIVE_ACCEPT_THRESHOLD (0.50)
        - ``"accept_flagged"`` when total >= NATIVE_FLAG_THRESHOLD (0.30)
        - ``"reject"`` otherwise
        """
        accept_t = self.NATIVE_ACCEPT_THRESHOLD if native_app else self.ACCEPT_THRESHOLD
        flag_t = self.NATIVE_FLAG_THRESHOLD if native_app else self.FLAG_THRESHOLD

        if total >= accept_t:
            return "accept"
        if total >= flag_t:
            return "accept_flagged"
        return "reject"


# ======================================================================
# v2 SOP-Level Confidence Scoring
# ======================================================================


@dataclass
class V2ConfidenceBreakdown:
    """Detailed confidence breakdown for a v2 SOP."""

    total: float  # 0.0-1.0 (clamped)
    demo_count_score: float  # 0.0-0.30  (weighted)
    step_consistency_score: float  # 0.0-0.30  (weighted)
    annotation_quality_score: float  # 0.0-0.20  (weighted)
    variable_detection_score: float  # 0.0-0.20  (weighted)
    focus_bonus: float  # 0.0 or 0.10
    reasons: list[str] = field(default_factory=list)


# Weights for each scoring component
_W_DEMO_COUNT = 0.30
_W_STEP_CONSISTENCY = 0.30
_W_ANNOTATION_QUALITY = 0.20
_W_VARIABLE_DETECTION = 0.20
_FOCUS_BONUS = 0.10

# Annotation fields considered for quality scoring.  These are the keys
# inside a scene annotation JSON that, when populated, indicate the VLM
# extracted useful information.
_ANNOTATION_QUALITY_FIELDS = (
    "app",
    "location",
    "visible_content",
    "ui_state",
    "task_context",
)


def _score_demo_count(demo_count: int) -> float:
    """Raw score [0, 1] for the number of demonstrations.

    1 demo → 0.50,  2 demos → 0.70,  3+ demos → 0.90.
    Linear interpolation above 3 capped at 1.0.
    """
    if demo_count <= 0:
        return 0.0
    if demo_count == 1:
        return 0.50
    if demo_count == 2:
        return 0.70
    # 3+ demos: 0.90 base + 0.025 per extra demo, capped at 1.0
    return min(0.90 + (demo_count - 3) * 0.025, 1.0)


def _score_step_consistency(
    sop_steps: list[dict],
    demonstrations: list[list[dict]] | None,
) -> float:
    """Raw score [0, 1] for step consistency across demonstrations.

    Measures what fraction of the SOP's steps appear in every
    demonstration.  If no demonstration data is available, returns 1.0
    for single-demo (focus) or 0.5 as a conservative fallback.
    """
    if not sop_steps:
        return 0.0

    if demonstrations is None or len(demonstrations) <= 1:
        # Single demonstration — all steps came from the same demo
        return 1.0

    # For each SOP step, check how many demos contain a similar action
    # by comparing the step's "action" or "step" key.
    step_actions = []
    for s in sop_steps:
        action = s.get("step", s.get("action", ""))
        step_actions.append(action.lower().strip())

    if not step_actions:
        return 0.0

    consistent_count = 0
    for action in step_actions:
        if not action:
            continue
        # Count how many demos have a frame whose what_doing or action
        # semantically matches this step.  We use simple substring
        # overlap since full embedding comparison is expensive here.
        present_in = 0
        for demo in demonstrations:
            for frame in demo:
                ann = frame.get("annotation", {})
                tc = ann.get("task_context", {})
                what_doing = str(tc.get("what_doing", "")).lower()
                diff = frame.get("diff", {})
                diff_desc = str(diff.get("step_description", "")).lower() if isinstance(diff, dict) else ""
                # Simple keyword overlap
                action_words = set(action.split())
                if action_words & set(what_doing.split()):
                    present_in += 1
                    break
                if action_words & set(diff_desc.split()):
                    present_in += 1
                    break

        if present_in >= len(demonstrations):
            consistent_count += 1

    return consistent_count / len(step_actions) if step_actions else 0.0


def _score_annotation_quality(
    annotations: list[dict] | None,
) -> float:
    """Raw score [0, 1] for annotation completeness.

    Measures the average fraction of expected fields that are non-empty
    across all annotations.
    """
    if not annotations:
        return 0.5  # Conservative default

    total_ratio = 0.0
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        filled = 0
        for f in _ANNOTATION_QUALITY_FIELDS:
            val = ann.get(f)
            if val:
                # Non-empty dict counts as filled
                if isinstance(val, dict) and not val:
                    continue
                filled += 1
        total_ratio += filled / len(_ANNOTATION_QUALITY_FIELDS)

    return total_ratio / len(annotations) if annotations else 0.5


def _score_variable_detection(
    sop_variables: list[dict],
    demonstrations: list[list[dict]] | None,
) -> float:
    """Raw score [0, 1] for variable detection quality.

    For single-demo (focus) SOPs: score based on whether variables were
    identified at all (a good sign the VLM found parameterizable values).
    For multi-demo SOPs: score based on ratio of detected variables to
    total varying values across demonstrations.
    """
    if demonstrations is None or len(demonstrations) <= 1:
        # Focus mode: any variable detection is good
        if sop_variables:
            return min(0.50 + len(sop_variables) * 0.10, 1.0)
        return 0.3  # No variables found but that might be correct

    # Multi-demo: compare frame diff inputs across demonstrations to
    # count how many values actually differ.
    varying_values: set[str] = set()
    # Collect all input field names that appear across demos
    field_values: dict[str, set[str]] = {}
    for demo in demonstrations:
        for frame in demo:
            diff = frame.get("diff", {})
            if not isinstance(diff, dict):
                continue
            for inp in diff.get("inputs", []):
                if isinstance(inp, dict):
                    fn = inp.get("field", "")
                    val = str(inp.get("value", ""))
                    if fn:
                        field_values.setdefault(fn, set()).add(val)

    # A field varies if it has 2+ distinct values across demos
    for fn, vals in field_values.items():
        if len(vals) >= 2:
            varying_values.add(fn)

    if not varying_values:
        # No varying values detected — could mean the task is constant
        return 0.8 if not sop_variables else 1.0

    # Score: what fraction of varying values are captured as variables?
    detected_var_names = {v.get("name", "").lower() for v in sop_variables}
    matched = 0
    for vv in varying_values:
        if vv.lower() in detected_var_names:
            matched += 1
        else:
            # Partial match: check if any variable name contains the field
            for dn in detected_var_names:
                if vv.lower() in dn or dn in vv.lower():
                    matched += 1
                    break

    return matched / len(varying_values) if varying_values else 1.0


def compute_v2_confidence(
    sop_template: dict,
    *,
    demonstrations: list[list[dict]] | None = None,
    annotations: list[dict] | None = None,
    is_focus: bool = False,
) -> V2ConfidenceBreakdown:
    """Compute v2 SOP-level confidence score.

    Parameters
    ----------
    sop_template:
        The generated SOP template dict (with steps, variables, etc.).
    demonstrations:
        List of demonstration timelines (each a list of frame dicts).
        None for focus mode (single demo).
    annotations:
        List of raw annotation dicts used for quality scoring.
        None to use a conservative default.
    is_focus:
        True if this is a focus recording (gets +0.10 bonus).

    Returns
    -------
    V2ConfidenceBreakdown with total score and component breakdown.
    """
    demo_count = sop_template.get("episode_count", 1)
    if demonstrations is not None:
        demo_count = max(demo_count, len(demonstrations))

    steps = sop_template.get("steps", [])
    variables = sop_template.get("variables", [])

    # Compute raw scores (0-1 each)
    raw_demo = _score_demo_count(demo_count)
    raw_consistency = _score_step_consistency(steps, demonstrations)
    raw_quality = _score_annotation_quality(annotations)
    raw_variables = _score_variable_detection(variables, demonstrations)

    # Apply weights
    w_demo = raw_demo * _W_DEMO_COUNT
    w_consistency = raw_consistency * _W_STEP_CONSISTENCY
    w_quality = raw_quality * _W_ANNOTATION_QUALITY
    w_variables = raw_variables * _W_VARIABLE_DETECTION
    bonus = _FOCUS_BONUS if is_focus else 0.0

    total = min(w_demo + w_consistency + w_quality + w_variables + bonus, 1.0)

    # Build reasons
    reasons: list[str] = []
    reasons.append(f"demos={demo_count}")
    if raw_consistency >= 0.80:
        reasons.append("high_step_consistency")
    elif raw_consistency >= 0.50:
        reasons.append("moderate_step_consistency")
    else:
        reasons.append("low_step_consistency")
    if raw_quality >= 0.80:
        reasons.append("high_annotation_quality")
    if raw_variables >= 0.80:
        reasons.append("good_variable_detection")
    if is_focus:
        reasons.append("focus_recording_bonus")

    return V2ConfidenceBreakdown(
        total=round(total, 4),
        demo_count_score=round(w_demo, 4),
        step_consistency_score=round(w_consistency, 4),
        annotation_quality_score=round(w_quality, 4),
        variable_detection_score=round(w_variables, 4),
        focus_bonus=bonus,
        reasons=reasons,
    )
