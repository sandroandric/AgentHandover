"""Per-procedure and global constraint management for agent safety.

Manages trust levels, guardrails, and execution constraints that
determine what an AI agent is allowed to do with a given procedure.
All state is persisted through the :class:`KnowledgeBase`.

Trust levels (ascending capability):

1. **observe** — watch only, no interaction.
2. **suggest** — can suggest actions to the user.
3. **draft** — can draft actions (e.g. compose an email) but not send.
4. **execute_with_approval** — can execute, but must get user approval first.
5. **autonomous** — fully autonomous execution allowed.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from oc_apprentice_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Trust levels for procedure execution, from most restrictive to least."""

    OBSERVE = "observe"
    SUGGEST = "suggest"
    DRAFT = "draft"
    EXECUTE_WITH_APPROVAL = "execute_with_approval"
    AUTONOMOUS = "autonomous"

    @classmethod
    def from_string(cls, s: str) -> TrustLevel:
        """Parse a trust level from its string value.

        Returns :attr:`OBSERVE` for unrecognised strings.
        """
        for member in cls:
            if member.value == s:
                return member
        return cls.OBSERVE


class ConstraintManager:
    """Manage per-procedure and global constraints for agent safety."""

    def __init__(self, kb: KnowledgeBase) -> None:
        self._kb = kb

    # ------------------------------------------------------------------
    # Constraints — get / set
    # ------------------------------------------------------------------

    def get_constraints(self, slug: str | None = None) -> dict:
        """Get constraints for a procedure or global constraints.

        If *slug* is given, returns merged global + per-procedure constraints
        (per-procedure values override global).
        If *slug* is ``None``, returns global constraints only.
        """
        global_constraints = self._kb.get_constraints()

        if slug is None:
            return global_constraints.get("global", {})

        # Merge global + per-procedure (per-procedure wins on conflict).
        per_proc = global_constraints.get("per_procedure", {}).get(slug, {})
        merged = {**global_constraints.get("global", {}), **per_proc}
        return merged

    def set_constraint(
        self,
        slug: str | None,
        key: str,
        value: Any,
    ) -> None:
        """Set a constraint value.

        Args:
            slug: Procedure slug, or *None* for a global constraint.
            key: Constraint key (e.g. ``"max_spend_usd_without_approval"``).
            value: Constraint value.
        """
        constraints = self._kb.get_constraints()

        if slug is None:
            constraints.setdefault("global", {})[key] = value
        else:
            constraints.setdefault("per_procedure", {}).setdefault(slug, {})[
                key
            ] = value

        self._kb.update_constraints(constraints)

    # ------------------------------------------------------------------
    # Trust levels
    # ------------------------------------------------------------------

    def get_trust_level(self, slug: str) -> TrustLevel:
        """Get the trust level for a procedure.

        Returns :attr:`TrustLevel.OBSERVE` if the procedure does not exist
        or has no trust level set.
        """
        proc = self._kb.get_procedure(slug)
        if proc is None:
            return TrustLevel.OBSERVE

        trust_str = proc.get("constraints", {}).get("trust_level", "observe")
        return TrustLevel.from_string(trust_str)

    def set_trust_level(self, slug: str, level: TrustLevel) -> None:
        """Set the trust level for a procedure."""
        proc = self._kb.get_procedure(slug)
        if proc is None:
            logger.warning(
                "Cannot set trust level: procedure '%s' not found", slug
            )
            return

        proc.setdefault("constraints", {})["trust_level"] = level.value
        self._kb.save_procedure(proc)

    # ------------------------------------------------------------------
    # Execution checks
    # ------------------------------------------------------------------

    def check_execution_allowed(self, slug: str) -> tuple[bool, str]:
        """Check if autonomous execution is allowed for a procedure.

        Returns:
            ``(allowed, reason)`` tuple.
        """
        trust = self.get_trust_level(slug)

        if trust == TrustLevel.AUTONOMOUS:
            return True, "autonomous execution allowed"

        if trust == TrustLevel.EXECUTE_WITH_APPROVAL:
            return False, "requires approval before execution"

        if trust == TrustLevel.DRAFT:
            return False, "can only draft actions, not execute"

        if trust == TrustLevel.SUGGEST:
            return False, "can only suggest actions, not execute"

        return False, "observe-only mode — no execution allowed"

    # ------------------------------------------------------------------
    # Guardrails
    # ------------------------------------------------------------------

    def get_guardrails(self, slug: str) -> list[dict]:
        """Get guardrails for a procedure.

        Returns an empty list if the procedure does not exist or has no
        guardrails.
        """
        proc = self._kb.get_procedure(slug)
        if proc is None:
            return []
        return proc.get("constraints", {}).get("guardrails", [])

    def add_guardrail(self, slug: str, guardrail: dict) -> None:
        """Add a guardrail to a procedure.

        No-op if the procedure does not exist.
        """
        proc = self._kb.get_procedure(slug)
        if proc is None:
            return

        proc.setdefault("constraints", {}).setdefault("guardrails", []).append(
            guardrail
        )
        self._kb.save_procedure(proc)
