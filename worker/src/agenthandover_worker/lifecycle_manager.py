"""Lifecycle state machine for procedures.

Manages the 7-state lifecycle independently from trust levels.
Lifecycle answers 'what maturity state is this procedure in?'
Trust answers 'what is the agent allowed to do with it?'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum

from agenthandover_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class ProcedureLifecycle(str, Enum):
    OBSERVED = "observed"
    DRAFT = "draft"
    REVIEWED = "reviewed"
    VERIFIED = "verified"
    AGENT_READY = "agent_ready"
    STALE = "stale"
    ARCHIVED = "archived"


_VALID_TRANSITIONS: dict[ProcedureLifecycle, frozenset[ProcedureLifecycle]] = {
    ProcedureLifecycle.OBSERVED: frozenset({ProcedureLifecycle.DRAFT, ProcedureLifecycle.ARCHIVED}),
    ProcedureLifecycle.DRAFT: frozenset({ProcedureLifecycle.REVIEWED, ProcedureLifecycle.STALE, ProcedureLifecycle.ARCHIVED}),
    ProcedureLifecycle.REVIEWED: frozenset({ProcedureLifecycle.VERIFIED, ProcedureLifecycle.DRAFT, ProcedureLifecycle.STALE}),
    ProcedureLifecycle.VERIFIED: frozenset({ProcedureLifecycle.AGENT_READY, ProcedureLifecycle.REVIEWED, ProcedureLifecycle.STALE}),
    ProcedureLifecycle.AGENT_READY: frozenset({ProcedureLifecycle.STALE, ProcedureLifecycle.VERIFIED, ProcedureLifecycle.ARCHIVED}),
    ProcedureLifecycle.STALE: frozenset({ProcedureLifecycle.DRAFT, ProcedureLifecycle.REVIEWED, ProcedureLifecycle.ARCHIVED}),
    ProcedureLifecycle.ARCHIVED: frozenset({ProcedureLifecycle.DRAFT}),
}

# States eligible for auto-demotion to STALE when freshness drops
_AUTO_STALE_STATES = frozenset({ProcedureLifecycle.DRAFT, ProcedureLifecycle.AGENT_READY})
_MIN_FRESHNESS = 0.3


class InvalidTransitionError(ValueError):
    """Raised when attempting an invalid lifecycle transition."""


@dataclass
class LifecycleTransition:
    from_state: str
    to_state: str
    trigger: str
    actor: str
    timestamp: str
    reason: str


class LifecycleManager:
    def __init__(self, kb: KnowledgeBase) -> None:
        self._kb = kb

    def get_state(self, slug: str) -> ProcedureLifecycle:
        """Get current lifecycle state. Returns OBSERVED if field absent."""
        proc = self._kb.get_procedure(slug)
        if proc is None:
            return ProcedureLifecycle.OBSERVED
        state_str = proc.get("lifecycle_state", "observed")
        try:
            return ProcedureLifecycle(state_str)
        except ValueError:
            return ProcedureLifecycle.OBSERVED

    def transition(
        self, slug: str, to_state: ProcedureLifecycle,
        trigger: str, actor: str = "system", reason: str = "",
    ) -> bool:
        """Execute a lifecycle transition. Raises InvalidTransitionError if not valid."""
        proc = self._kb.get_procedure(slug)
        if proc is None:
            logger.warning("Cannot transition nonexistent procedure: %s", slug)
            return False

        current = ProcedureLifecycle(proc.get("lifecycle_state", "observed"))

        if to_state not in _VALID_TRANSITIONS.get(current, frozenset()):
            raise InvalidTransitionError(
                f"Cannot transition {slug} from {current.value} to {to_state.value}. "
                f"Valid targets: {sorted(s.value for s in _VALID_TRANSITIONS.get(current, frozenset()))}"
            )

        now_iso = datetime.now(timezone.utc).isoformat()
        transition = LifecycleTransition(
            from_state=current.value,
            to_state=to_state.value,
            trigger=trigger,
            actor=actor,
            timestamp=now_iso,
            reason=reason,
        )

        proc["lifecycle_state"] = to_state.value
        history = proc.setdefault("lifecycle_history", [])
        history.append(asdict(transition))
        self._kb.save_procedure(proc)

        logger.info(
            "Lifecycle transition: %s %s -> %s (trigger=%s, actor=%s)",
            slug, current.value, to_state.value, trigger, actor,
        )
        return True

    def can_transition(self, slug: str, to_state: ProcedureLifecycle) -> bool:
        """Check if transition is valid without mutating."""
        current = self.get_state(slug)
        return to_state in _VALID_TRANSITIONS.get(current, frozenset())

    def check_auto_transitions(self) -> list[tuple[str, ProcedureLifecycle]]:
        """Check for procedures that should be auto-transitioned to STALE.
        Returns list of (slug, proposed_new_state) without applying.
        """
        from agenthandover_worker.staleness_detector import procedure_freshness

        proposals: list[tuple[str, ProcedureLifecycle]] = []
        for proc in self._kb.list_procedures():
            state_str = proc.get("lifecycle_state", "observed")
            try:
                state = ProcedureLifecycle(state_str)
            except ValueError:
                continue
            if state not in _AUTO_STALE_STATES:
                continue
            freshness = procedure_freshness(proc)
            if freshness < _MIN_FRESHNESS:
                proposals.append((proc.get("id", proc.get("slug", "")), ProcedureLifecycle.STALE))
        return proposals

    def apply_auto_transitions(self) -> list[tuple[str, str, str]]:
        """Apply all auto-transitions. Returns list of (slug, old_state, new_state)."""
        proposals = self.check_auto_transitions()
        applied: list[tuple[str, str, str]] = []
        for slug, new_state in proposals:
            old_state = self.get_state(slug).value
            try:
                self.transition(
                    slug, new_state,
                    trigger="freshness_decay",
                    actor="system",
                    reason=f"Freshness dropped below {_MIN_FRESHNESS}",
                )
                applied.append((slug, old_state, new_state.value))
            except InvalidTransitionError:
                logger.debug("Auto-transition failed for %s", slug, exc_info=True)
        return applied

    def get_transition_history(self, slug: str) -> list[LifecycleTransition]:
        """Get lifecycle transition history for a procedure."""
        proc = self._kb.get_procedure(slug)
        if proc is None:
            return []
        history = proc.get("lifecycle_history", [])
        return [
            LifecycleTransition(
                from_state=h.get("from_state", ""),
                to_state=h.get("to_state", ""),
                trigger=h.get("trigger", ""),
                actor=h.get("actor", ""),
                timestamp=h.get("timestamp", ""),
                reason=h.get("reason", ""),
            )
            for h in history
        ]
