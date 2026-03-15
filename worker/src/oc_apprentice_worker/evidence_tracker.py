"""Evidence tracker — links procedures to their supporting observations.

Every procedure should be traceable back to the specific observations
that led to its creation.  This module tracks:

- Which focus/passive sessions contributed to a procedure
- Per-step consistency across demonstrations
- Contradictions between observations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class ObservationEvidence:
    """A single observation that supports a procedure."""

    date: str
    type: str  # "focus" or "passive"
    duration_minutes: int
    session_id: str
    event_count: int


@dataclass
class StepEvidence:
    """Evidence for a single step across observations."""

    step_id: str
    observed_count: int
    consistent_count: int
    contradictions: list[dict] = field(default_factory=list)

    @property
    def consistency_ratio(self) -> float:
        """Ratio of consistent observations (0.0-1.0)."""
        if self.observed_count == 0:
            return 0.0
        return self.consistent_count / self.observed_count


class EvidenceTracker:
    """Track and build evidence linking procedures to observations."""

    def __init__(self, knowledge_base: "KnowledgeBase") -> None:
        from oc_apprentice_worker.knowledge_base import KnowledgeBase

        self._kb: KnowledgeBase = knowledge_base

    def build_evidence(self, slug: str) -> dict:
        """Build the evidence section for a procedure.

        Loads the procedure from the knowledge base, returns its
        evidence dict (or empty evidence if not found).
        """
        proc = self._kb.get_procedure(slug)
        if proc is None:
            return {
                "observations": [],
                "step_evidence": [],
                "contradictions": [],
                "total_observations": 0,
            }
        return proc.get("evidence", {
            "observations": [],
            "step_evidence": [],
            "contradictions": [],
            "total_observations": 0,
        })

    def add_observation(self, slug: str, evidence: ObservationEvidence) -> None:
        """Add an observation record to a procedure's evidence.

        Updates the procedure in the knowledge base with the new
        observation appended to its evidence list.
        """
        proc = self._kb.get_procedure(slug)
        if proc is None:
            logger.warning(
                "Cannot add observation: procedure '%s' not found", slug
            )
            return

        ev = proc.setdefault("evidence", {
            "observations": [],
            "step_evidence": [],
            "contradictions": [],
            "total_observations": 0,
        })
        ev["observations"].append({
            "date": evidence.date,
            "type": evidence.type,
            "duration_minutes": evidence.duration_minutes,
            "session_id": evidence.session_id,
            "event_count": evidence.event_count,
        })
        ev["total_observations"] = len(ev["observations"])

        # Update staleness.last_observed
        staleness = proc.setdefault("staleness", {})
        staleness["last_observed"] = datetime.now(timezone.utc).isoformat()

        self._kb.save_procedure(proc)
        logger.info(
            "Added observation to '%s' (total: %d)",
            slug, ev["total_observations"],
        )

    def compute_step_evidence(
        self,
        slug: str,
        demos: list[list[dict]],
        embeddings: dict[str, list[float]] | None = None,
    ) -> list[StepEvidence]:
        """Compute per-step evidence from multiple demonstrations.

        Each demonstration is a list of step dicts (from different
        observations of the same procedure).  We compare step actions
        across demos to find consistency and contradictions.

        Args:
            slug: Procedure slug (for logging).
            demos: List of demonstrations.  Each demo is a list of step
                dicts with at least ``action`` and ``step_id`` keys.

        Returns:
            List of StepEvidence, one per unique step_id found.
        """
        if not demos:
            return []

        # Collect all unique step_ids from the first (reference) demo
        reference = demos[0]
        step_ids = [s.get("step_id", f"step_{i+1}") for i, s in enumerate(reference)]

        results: list[StepEvidence] = []

        for step_idx, step_id in enumerate(step_ids):
            observed = 0
            consistent = 0
            contradictions: list[dict] = []

            ref_action = (
                reference[step_idx].get("action", "") if step_idx < len(reference) else ""
            )

            for demo_idx, demo in enumerate(demos):
                if step_idx >= len(demo):
                    continue
                observed += 1
                demo_action = demo[step_idx].get("action", "")
                emb_a = embeddings.get(ref_action) if embeddings else None
                emb_b = embeddings.get(demo_action) if embeddings else None
                if _actions_match(ref_action, demo_action, embedding_a=emb_a, embedding_b=emb_b):
                    consistent += 1
                else:
                    contradictions.append({
                        "demo_index": demo_idx,
                        "expected": ref_action,
                        "actual": demo_action,
                        "step_id": step_id,
                    })

            results.append(StepEvidence(
                step_id=step_id,
                observed_count=observed,
                consistent_count=consistent,
                contradictions=contradictions,
            ))

        return results

    def update_step_evidence(
        self,
        slug: str,
        step_evidence: list[StepEvidence],
    ) -> None:
        """Persist computed step evidence into the procedure."""
        proc = self._kb.get_procedure(slug)
        if proc is None:
            return

        ev = proc.setdefault("evidence", {
            "observations": [],
            "step_evidence": [],
            "contradictions": [],
            "total_observations": 0,
        })

        ev["step_evidence"] = [
            {
                "step_id": se.step_id,
                "observed_count": se.observed_count,
                "consistent_count": se.consistent_count,
                "contradictions": se.contradictions,
            }
            for se in step_evidence
        ]

        # Merge all contradictions
        all_contradictions: list[dict] = []
        for se in step_evidence:
            all_contradictions.extend(se.contradictions)
        ev["contradictions"] = all_contradictions

        self._kb.save_procedure(proc)


def _actions_match(
    a: str,
    b: str,
    embedding_a: list[float] | None = None,
    embedding_b: list[float] | None = None,
    threshold: float = 0.60,
) -> bool:
    """Check if two action strings are semantically equivalent."""
    if embedding_a and embedding_b:
        from oc_apprentice_worker.task_segmenter import _cosine_similarity
        return _cosine_similarity(embedding_a, embedding_b) >= threshold
    return a.strip().lower() == b.strip().lower()
