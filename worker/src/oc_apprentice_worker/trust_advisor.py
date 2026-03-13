"""Trust promotion advisor for OpenMimic procedures.

Analyzes execution history and suggests trust-level promotions for
procedures that have demonstrated reliability.  **Never auto-promotes** —
every suggestion must be explicitly accepted by the human operator.

Trust levels (ascending capability):

1. observe
2. suggest
3. draft
4. execute_with_approval
5. autonomous

Suggestions are persisted at ``{kb_root}/observations/trust_suggestions.json``.
Execution stats are read from ``{kb_root}/observations/executions.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import uuid

from oc_apprentice_worker.constraint_manager import TrustLevel
from oc_apprentice_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

TRUST_LEVELS = [level.value for level in TrustLevel]


@dataclass
class TrustSuggestion:
    """A suggestion to promote a procedure's trust level."""

    procedure_slug: str
    current_level: str
    suggested_level: str
    reason: str
    evidence: dict  # {"observations": N, "success_rate": float, "last_failure": str|None}
    suggested_at: str
    dismissed: bool = False
    accepted: bool = False


class TrustAdvisor:
    """Suggests trust promotions. NEVER auto-promotes — human must accept."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        min_observations: int = 3,
        min_success_rate: float = 0.90,
    ) -> None:
        self._kb = knowledge_base
        self._min_observations = min_observations
        self._min_success_rate = min_success_rate
        self._suggestions: list[TrustSuggestion] = []
        self._load_suggestions()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_all(self) -> list[TrustSuggestion]:
        """Evaluate all known procedures and return new suggestions."""
        procedures = self._kb.list_procedures()
        new_suggestions: list[TrustSuggestion] = []
        for proc in procedures:
            slug = proc.get("id", proc.get("slug", ""))
            if not slug:
                continue
            suggestion = self.evaluate_procedure(slug)
            if suggestion is not None:
                new_suggestions.append(suggestion)
        return new_suggestions

    def evaluate_procedure(self, slug: str) -> TrustSuggestion | None:
        """Evaluate a single procedure and return a suggestion if warranted.

        Returns ``None`` if the procedure does not qualify for promotion
        (insufficient observations, low success rate, already at max level,
        or a pending suggestion already exists).
        """
        # Skip if there's already a pending (non-dismissed, non-accepted) suggestion
        for s in self._suggestions:
            if s.procedure_slug == slug and not s.dismissed and not s.accepted:
                return None

        current_level = self._get_current_trust_level(slug)
        next_level = self._next_level(current_level)
        if next_level is None:
            return None  # Already at maximum level

        stats = self._get_execution_stats(slug)
        observations = stats.get("total", 0)
        if observations < self._min_observations:
            return None  # Not enough data

        successes = stats.get("successes", 0)
        success_rate = successes / observations if observations > 0 else 0.0
        if success_rate < self._min_success_rate:
            return None  # Success rate too low

        last_failure = stats.get("last_failure", None)

        reason = (
            f"Procedure '{slug}' has {observations} observations with "
            f"{success_rate:.0%} success rate. Ready for promotion from "
            f"'{current_level}' to '{next_level}'."
        )

        suggestion = TrustSuggestion(
            procedure_slug=slug,
            current_level=current_level,
            suggested_level=next_level,
            reason=reason,
            evidence={
                "observations": observations,
                "success_rate": success_rate,
                "last_failure": last_failure,
            },
            suggested_at=datetime.now(timezone.utc).isoformat(),
        )
        self._suggestions.append(suggestion)
        self._save_suggestions()
        return suggestion

    def get_suggestions(
        self, include_dismissed: bool = False
    ) -> list[TrustSuggestion]:
        """Return pending suggestions, optionally including dismissed ones."""
        if include_dismissed:
            return list(self._suggestions)
        return [s for s in self._suggestions if not s.dismissed]

    def accept_suggestion(self, procedure_slug: str) -> bool:
        """Accept a pending suggestion and apply the trust level promotion.

        Returns ``True`` if a suggestion was found and accepted.
        """
        for s in self._suggestions:
            if (
                s.procedure_slug == procedure_slug
                and not s.dismissed
                and not s.accepted
            ):
                # Apply the promotion to the procedure in the KB
                proc = self._kb.get_procedure(procedure_slug)
                if proc is not None:
                    proc.setdefault("constraints", {})[
                        "trust_level"
                    ] = s.suggested_level
                    self._kb.save_procedure(proc)

                s.accepted = True
                self._save_suggestions()
                logger.info(
                    "Trust suggestion accepted: %s -> %s",
                    procedure_slug,
                    s.suggested_level,
                )
                return True
        return False

    def dismiss_suggestion(self, procedure_slug: str) -> bool:
        """Dismiss a pending suggestion.

        Returns ``True`` if a suggestion was found and dismissed.
        """
        for s in self._suggestions:
            if (
                s.procedure_slug == procedure_slug
                and not s.dismissed
                and not s.accepted
            ):
                s.dismissed = True
                self._save_suggestions()
                logger.info(
                    "Trust suggestion dismissed: %s", procedure_slug
                )
                return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_level(self, current: str) -> str | None:
        """Return the next trust level above *current*, or None if at max."""
        try:
            idx = TRUST_LEVELS.index(current)
        except ValueError:
            # Unknown level — treat as observe
            idx = 0
        if idx >= len(TRUST_LEVELS) - 1:
            return None
        return TRUST_LEVELS[idx + 1]

    def _get_current_trust_level(self, slug: str) -> str:
        """Read the current trust level from the procedure in the KB."""
        proc = self._kb.get_procedure(slug)
        if proc is None:
            return "observe"
        return proc.get("constraints", {}).get("trust_level", "observe")

    def _get_execution_stats(self, slug: str) -> dict:
        """Read execution stats for a procedure from executions.json.

        Returns a dict with keys: ``total``, ``successes``, ``failures``,
        ``last_failure``.  Returns zeros if no stats are available.
        """
        path = self._kb.root / "observations" / "executions.json"
        if not path.is_file():
            return {"total": 0, "successes": 0, "failures": 0, "last_failure": None}
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"total": 0, "successes": 0, "failures": 0, "last_failure": None}

        proc_stats = data.get("procedures", {}).get(slug, {})
        return {
            "total": proc_stats.get("total", 0),
            "successes": proc_stats.get("successes", 0),
            "failures": proc_stats.get("failures", 0),
            "last_failure": proc_stats.get("last_failure", None),
        }

    def _load_suggestions(self) -> None:
        """Load suggestions from persistent storage."""
        path = self._kb.root / "observations" / "trust_suggestions.json"
        if not path.is_file():
            self._suggestions = []
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._suggestions = []
            return

        self._suggestions = []
        for item in data.get("suggestions", []):
            self._suggestions.append(
                TrustSuggestion(
                    procedure_slug=item["procedure_slug"],
                    current_level=item["current_level"],
                    suggested_level=item["suggested_level"],
                    reason=item["reason"],
                    evidence=item["evidence"],
                    suggested_at=item["suggested_at"],
                    dismissed=item.get("dismissed", False),
                    accepted=item.get("accepted", False),
                )
            )

    def _save_suggestions(self) -> None:
        """Persist suggestions to trust_suggestions.json using atomic write."""
        data = {
            "suggestions": [
                {
                    "procedure_slug": s.procedure_slug,
                    "current_level": s.current_level,
                    "suggested_level": s.suggested_level,
                    "reason": s.reason,
                    "evidence": s.evidence,
                    "suggested_at": s.suggested_at,
                    "dismissed": s.dismissed,
                    "accepted": s.accepted,
                }
                for s in self._suggestions
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self._kb.root / "observations" / "trust_suggestions.json"
        self._kb.atomic_write_json(path, data)
