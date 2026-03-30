"""Decision rule extraction from multi-observation comparison.

Analyzes variation points across multiple observations of the same
procedure to infer the rules that govern different choices.  This is
a heuristic-only implementation that uses pattern matching to identify
common decision patterns without requiring VLM.

When an ``LLMReasoner`` is provided, generic heuristic conditions
(e.g. "when choosing X over Y") are refined via a targeted Qwen call
that considers observation context to produce more specific conditions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agenthandover_worker.knowledge_base import KnowledgeBase

if TYPE_CHECKING:
    from agenthandover_worker.llm_reasoning import LLMReasoner

logger = logging.getLogger(__name__)


@dataclass
class DecisionRule:
    """A single inferred decision rule."""

    condition: str  # "price < 50 AND domain_authority > 20"
    action: str  # "buy"
    observed_count: int
    description: str


@dataclass
class DecisionSet:
    """A set of decision rules for a specific step in a procedure."""

    procedure_slug: str
    applies_to_step: str
    rules: list[DecisionRule]
    confidence: float
    inferred_from_observations: int


class DecisionExtractor:
    """Extract decision rules from multi-observation comparison.

    Analyzes variation points across multiple observations of the same
    procedure to infer the rules that govern different choices.

    This is a heuristic-only implementation.  Uses pattern matching
    to identify common decision patterns without requiring VLM.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        llm_reasoner: "LLMReasoner | None" = None,
    ) -> None:
        self._kb = kb
        self._llm_reasoner = llm_reasoner

    def extract_decisions(
        self,
        slug: str,
        observations: list[dict] | None = None,
        evidence: dict | None = None,
    ) -> list[DecisionSet]:
        """Extract decision rules for a procedure.

        Args:
            slug: Procedure slug.
            observations: List of observation dicts, each containing:
                - steps: list of step dicts from one observation
                - context: dict with any context (inputs, outputs, etc.)
                If None, returns empty (needs raw observation data).

        Returns:
            List of DecisionSets with inferred rules.
        """
        if observations is None or len(observations) < 2:
            return []

        variations = self._find_variation_points(observations)

        decision_sets: list[DecisionSet] = []
        for variation in variations:
            rules = self._infer_rules(variation, evidence=evidence)
            if rules:
                decision_sets.append(
                    DecisionSet(
                        procedure_slug=slug,
                        applies_to_step=variation["step_id"],
                        rules=rules,
                        confidence=variation["confidence"],
                        inferred_from_observations=len(observations),
                    )
                )

        return decision_sets

    def _find_variation_points(self, observations: list[dict]) -> list[dict]:
        """Find steps where different observations show different behavior.

        A variation point is a step where the action, input, or target
        differs across observations.
        """
        if not observations:
            return []

        # Use first observation as reference
        ref_steps = observations[0].get("steps", [])
        variations: list[dict] = []

        for step_idx, ref_step in enumerate(ref_steps):
            step_id = ref_step.get("step_id", f"step_{step_idx + 1}")

            # Collect what each observation does at this step
            actions: dict[str, int] = {}
            inputs: dict[str, int] = {}
            contexts: list[dict] = []

            for obs in observations:
                obs_steps = obs.get("steps", [])
                if step_idx >= len(obs_steps):
                    continue
                obs_step = obs_steps[step_idx]
                obs_action = obs_step.get("action", "").strip().lower()
                obs_input = obs_step.get("input", "").strip()

                actions[obs_action] = actions.get(obs_action, 0) + 1
                if obs_input:
                    inputs[obs_input] = inputs.get(obs_input, 0) + 1

                contexts.append(obs.get("context", {}))

            # Check for variation
            has_action_variation = len(actions) > 1
            has_input_variation = len(inputs) > 1

            if has_action_variation or has_input_variation:
                total = sum(actions.values())
                max_count = max(actions.values())
                confidence = max_count / max(total, 1)

                variations.append(
                    {
                        "step_id": step_id,
                        "step_index": step_idx,
                        "actions": actions,
                        "inputs": inputs,
                        "contexts": contexts,
                        "confidence": round(confidence, 3),
                        "type": "action" if has_action_variation else "input",
                    }
                )

        return variations

    def _infer_rules(
        self,
        variation: dict,
        evidence: dict | None = None,
    ) -> list[DecisionRule]:
        """Infer decision rules from a variation point.

        Uses heuristic pattern matching:
        - If actions differ, create a rule per unique action
        - If inputs differ, create rules based on input patterns
        """
        rules: list[DecisionRule] = []

        if variation["type"] == "action":
            for action, count in variation["actions"].items():
                # Try to infer condition from context
                condition = self._infer_condition_from_contexts(
                    action, variation["contexts"], variation["actions"],
                    variation=variation,
                    evidence=evidence,
                )
                rules.append(
                    DecisionRule(
                        condition=condition,
                        action=action,
                        observed_count=count,
                        description=f"Observed {count} times at {variation['step_id']}",
                    )
                )
        elif variation["type"] == "input":
            for input_val, count in variation["inputs"].items():
                condition = f"input_value == '{input_val}'"
                rules.append(
                    DecisionRule(
                        condition=condition,
                        action=f"use input '{input_val}'",
                        observed_count=count,
                        description=f"Input '{input_val}' used {count} times",
                    )
                )

        return rules

    def _infer_condition_from_contexts(
        self,
        action: str,
        contexts: list[dict],
        all_actions: dict[str, int],
        variation: dict | None = None,
        evidence: dict | None = None,
    ) -> str:
        """Try to infer what condition leads to a specific action.

        Heuristic: look for distinguishing context fields.
        When the heuristic produces a generic fallback string and an
        ``LLMReasoner`` is available, attempts LLM-based refinement.
        """
        # Simple: if there are only 2 actions, frame as if/else
        if len(all_actions) == 2:
            actions_list = list(all_actions.keys())
            other = actions_list[0] if actions_list[1] == action else actions_list[1]
            heuristic = f"when choosing '{action}' over '{other}'"
        else:
            heuristic = f"when action is '{action}'"

        # Check whether heuristic is generic and LLM can improve it
        _GENERIC_MARKERS = ("when choosing", "when action is")
        is_generic = any(marker in heuristic for marker in _GENERIC_MARKERS)

        if is_generic and self._llm_reasoner is not None:
            llm_condition = self._infer_condition_with_llm(
                action, contexts, all_actions, variation or {},
                evidence=evidence,
            )
            if llm_condition is not None:
                return llm_condition

        return heuristic

    def _infer_condition_with_llm(
        self,
        action: str,
        contexts: list[dict],
        all_actions: dict[str, int],
        variation: dict,
        evidence: dict | None = None,
    ) -> str | None:
        """Use LLM to infer a specific condition for an action choice.

        Returns a condition string, or ``None`` if the LLM fails,
        is over budget, or abstains.
        """
        alternatives = [a for a in all_actions if a != action]
        context_summary = "; ".join(
            str(ctx) for ctx in contexts[:5] if ctx
        ) or "no additional context"

        prompt = (
            f"What condition determines when the user does '{action}' "
            f"vs '{', '.join(alternatives)}'? "
            f"Context from observations: {context_summary}. "
            f"Be specific — describe the observable condition, not just "
            f"the choice."
        )

        # Include evidence signals when available
        if evidence:
            evidence_parts = []
            selection = evidence.get("selection_signals", {})
            if selection:
                high_engagement = selection.get("high_engagement_locations", [])
                if high_engagement:
                    evidence_parts.append(
                        f"High engagement locations: {', '.join(str(loc) for loc in high_engagement[:3])}"
                    )
            content = evidence.get("content_produced", {})
            content_patterns = content.get("types", [])
            if content_patterns:
                evidence_parts.append(
                    f"Content patterns: {', '.join(content_patterns[:3])}"
                )
            if evidence_parts:
                prompt += f" Evidence: {'; '.join(evidence_parts)}."

        try:
            result = self._llm_reasoner.reason_text(
                prompt=prompt,
                caller="decision_extractor._infer_condition_with_llm",
            )
            if result.success and not result.abstained and result.value:
                return str(result.value)
        except Exception:
            logger.debug(
                "LLM condition inference failed for action '%s'",
                action,
                exc_info=True,
            )

        return None

    def save_decisions(self, decision_sets: list[DecisionSet]) -> None:
        """Save extracted decision sets to the knowledge base."""
        decisions = self._kb.get_decisions()

        existing_sets = decisions.get("decision_sets", [])

        for ds in decision_sets:
            # Serialize the DecisionSet to a dict
            ds_dict = {
                "procedure_slug": ds.procedure_slug,
                "applies_to_step": ds.applies_to_step,
                "rules": [
                    {
                        "condition": r.condition,
                        "action": r.action,
                        "observed_count": r.observed_count,
                        "description": r.description,
                    }
                    for r in ds.rules
                ],
                "confidence": ds.confidence,
                "inferred_from_observations": ds.inferred_from_observations,
            }

            # Update or append
            updated = False
            for i, existing in enumerate(existing_sets):
                if (
                    existing.get("procedure_slug") == ds.procedure_slug
                    and existing.get("applies_to_step") == ds.applies_to_step
                ):
                    existing_sets[i] = ds_dict
                    updated = True
                    break

            if not updated:
                existing_sets.append(ds_dict)

        decisions["decision_sets"] = existing_sets
        self._kb.update_decisions(decisions)
