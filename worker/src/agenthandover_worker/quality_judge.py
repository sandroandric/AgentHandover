"""Semantic quality judge for AgentHandover procedures.

Uses the shared LLM reasoner to assess whether a procedure is complete,
grounded in evidence, actionable, and free of gaps before allowing
lifecycle promotions.

If the LLM is unavailable or over budget, returns a default-pass
assessment so that promotions are never blocked by LLM downtime.
If the LLM abstains (insufficient evidence), returns passed=True
with abstained=True so that uncertainty does not block promotions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from agenthandover_worker.llm_reasoning import LLMReasoner, ReasoningResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class QualityAssessment:
    """Result of a quality assessment on a procedure."""

    passed: bool
    score: float            # 0.0-1.0
    reasons: list[str]
    gaps: list[str]
    abstained: bool = False
    provenance: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# QualityJudge
# ---------------------------------------------------------------------------


class QualityJudge:
    """Assess procedure quality using LLM reasoning before promotions.

    Args:
        llm_reasoner: The shared LLM reasoning interface.
    """

    def __init__(self, llm_reasoner: LLMReasoner) -> None:
        self._reasoner = llm_reasoner

    def assess(
        self,
        procedure: dict,
        target_state: str,
    ) -> QualityAssessment:
        """Assess whether a procedure is ready for promotion.

        Args:
            procedure: The full procedure dict from the knowledge base.
            target_state: The lifecycle state being promoted to.

        Returns:
            A QualityAssessment indicating pass/fail with reasons and gaps.

            Fail-open for low-stakes promotions (observed→draft, draft→reviewed).
            Fail-CLOSED for high-stakes promotions (→agent_ready): if the LLM
            is unavailable, the promotion is blocked because agent_ready is the
            trust boundary where agents can execute.
        """
        # Promotions to agent_ready are high-stakes — the agent trust boundary.
        # LLM must actually assess these; fail-closed if unavailable.
        is_high_stakes = target_state == "agent_ready"
        self._last_target_state = target_state  # for _parse_assessment

        prompt = self._build_assessment_prompt(procedure, target_state)

        try:
            result = self._reasoner.reason_json(
                prompt,
                system=(
                    "You are a quality reviewer for workflow procedures. "
                    "Evaluate completeness, evidence grounding, actionability, "
                    "and gaps. Respond with valid JSON only."
                ),
                caller="quality_judge.assess",
            )
        except Exception:
            logger.debug("Quality judge LLM call failed", exc_info=True)
            if is_high_stakes:
                return QualityAssessment(
                    passed=False, score=0.0,
                    reasons=["Quality judge unavailable — cannot promote to agent_ready without assessment"],
                    gaps=["LLM quality assessment required for agent_ready promotion"],
                )
            return self._default_pass("LLM call raised exception")

        if not result.success:
            if is_high_stakes:
                return QualityAssessment(
                    passed=False, score=0.0,
                    reasons=[f"Quality judge failed: {result.error or 'unknown'} — cannot promote to agent_ready"],
                    gaps=["LLM quality assessment required for agent_ready promotion"],
                )
            return self._default_pass(
                result.error or "LLM call unsuccessful",
            )

        if result.abstained:
            if is_high_stakes:
                return QualityAssessment(
                    passed=False, score=0.0,
                    reasons=["Quality judge abstained — insufficient evidence to promote to agent_ready"],
                    gaps=["More evidence needed before agent_ready promotion"],
                    abstained=True,
                    provenance=self._reasoner.make_provenance(
                        result, caller="quality_judge.assess",
                    ),
                )
            return QualityAssessment(
                passed=True, score=0.0,
                reasons=["LLM abstained — insufficient evidence to judge"],
                gaps=[], abstained=True,
                provenance=self._reasoner.make_provenance(
                    result, caller="quality_judge.assess",
                ),
            )

        return self._parse_assessment(result)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_assessment_prompt(
        self,
        procedure: dict,
        target_state: str,
    ) -> str:
        """Build the assessment prompt from procedure data."""
        title = procedure.get("title", procedure.get("id", "Untitled"))
        steps = procedure.get("steps", [])
        evidence = procedure.get("evidence", {})
        episode_count = procedure.get("episode_count", 0) or evidence.get("total_observations", 0)
        confidence = procedure.get("confidence_avg", 0.0)

        # Summarise steps
        step_lines = []
        for i, step in enumerate(steps, 1):
            action = step.get("action", step.get("step", ""))
            app = step.get("app", "")
            step_lines.append(f"  {i}. [{app}] {action}")
        steps_text = "\n".join(step_lines) if step_lines else "  (no steps)"

        # Summarise evidence
        observation_count = len(evidence.get("observations", []))
        contradictions = evidence.get("contradictions", [])
        contradiction_text = (
            f"Contradictions: {len(contradictions)}"
            if contradictions
            else "No contradictions"
        )

        return (
            f"Evaluate this procedure for promotion to '{target_state}'.\n\n"
            f"Title: {title}\n"
            f"Episodes observed: {episode_count}\n"
            f"Average confidence: {confidence:.2f}\n"
            f"Evidence observations: {observation_count}\n"
            f"{contradiction_text}\n\n"
            f"Steps:\n{steps_text}\n\n"
            "Evaluate:\n"
            "1. Is the procedure complete (all steps documented)?\n"
            "2. Is it grounded in evidence (steps match observations)?\n"
            "3. Would an agent be able to execute it (actionable, specific)?\n"
            "4. Are there gaps (missing decisions, unclear conditions)?\n\n"
            'Respond with JSON: {"passed": bool, "score": 0.0-1.0, '
            '"reasons": [...], "gaps": [...]}\n'
            "If you cannot determine quality, respond with: "
            '{"abstained": true, "reason": "..."}'
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_assessment(self, result: ReasoningResult) -> QualityAssessment:
        """Parse a successful LLM JSON response into a QualityAssessment."""
        data = result.value

        if not isinstance(data, dict):
            return self._default_pass("LLM response was not a dict")

        # Handle abstention in the parsed JSON — same fail-closed
        # rule as the raw-abstention path above.
        if data.get("abstained", False):
            is_high_stakes = self._last_target_state == "agent_ready"
            return QualityAssessment(
                passed=not is_high_stakes,
                score=0.0,
                reasons=[
                    data.get("reason", "LLM abstained")
                    + (" — blocked for agent_ready" if is_high_stakes else "")
                ],
                gaps=(
                    ["More evidence needed before agent_ready promotion"]
                    if is_high_stakes else []
                ),
                abstained=True,
                provenance=self._reasoner.make_provenance(
                    result, caller="quality_judge.assess",
                ),
            )

        passed = bool(data.get("passed", True))
        score = float(data.get("score", 0.0))
        reasons = data.get("reasons", [])
        gaps = data.get("gaps", [])

        # Ensure types
        if not isinstance(reasons, list):
            reasons = [str(reasons)] if reasons else []
        if not isinstance(gaps, list):
            gaps = [str(gaps)] if gaps else []

        return QualityAssessment(
            passed=passed,
            score=max(0.0, min(1.0, score)),
            reasons=[str(r) for r in reasons],
            gaps=[str(g) for g in gaps],
            abstained=False,
            provenance=self._reasoner.make_provenance(
                result, caller="quality_judge.assess",
            ),
        )

    # ------------------------------------------------------------------
    # Default pass (fail-open)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_pass(reason: str) -> QualityAssessment:
        """Return a default-pass assessment when LLM is unavailable.

        Promotions should never be blocked by LLM downtime.
        """
        return QualityAssessment(
            passed=True,
            score=0.0,
            reasons=[f"Default pass: {reason}"],
            gaps=[],
            abstained=False,
            provenance={},
        )
