"""Escalation handler for execution failures and deviations.

Rules:
1. First failure with retry strategy -> RETRY
2. Exhausted retries -> ABORT_NOTIFY
3. 3+ failures in 7 days -> DEMOTE (agent_ready -> stale)
4. Deviations count as soft failures toward demotion threshold
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING

from oc_apprentice_worker.knowledge_base import KnowledgeBase

if TYPE_CHECKING:
    from oc_apprentice_worker.lifecycle_manager import LifecycleManager

logger = logging.getLogger(__name__)

_DEMOTION_THRESHOLD = 3  # failures in window
_DEMOTION_WINDOW_DAYS = 7


class EscalationDecision(str, Enum):
    """Possible escalation decisions after a failure or deviation."""

    RETRY = "retry"
    ABORT_NOTIFY = "abort_notify"
    DEMOTE = "demote"


@dataclass
class EscalationResult:
    """Outcome of an escalation decision."""

    decision: EscalationDecision
    reason: str
    retry_count: int = 0
    max_retries: int = 0
    demoted: bool = False


class EscalationHandler:
    """Decide what to do when a procedure execution fails or deviates.

    The handler inspects the procedure's retry strategy and recent failure
    history to determine whether to retry, abort with notification, or
    demote the procedure in the lifecycle.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        lifecycle_manager: "LifecycleManager | None" = None,
    ) -> None:
        self._kb = kb
        self._lifecycle = lifecycle_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_failure(
        self, slug: str, execution_id: str, error: str
    ) -> EscalationResult:
        """Handle an execution failure.  Returns escalation decision."""
        proc = self._kb.get_procedure(slug)
        if proc is None:
            return EscalationResult(
                decision=EscalationDecision.ABORT_NOTIFY,
                reason=f"Procedure '{slug}' not found",
            )

        # Check retry strategy from first step's on_failure
        steps = proc.get("steps", [])
        retry_strategy = None
        max_retries = 0
        for step in steps:
            of = step.get("on_failure", {})
            if of.get("strategy") == "retry":
                retry_strategy = "retry"
                max_retries = of.get("max_retries", 1)
                break

        # Count recent failures
        recent = self.get_recent_failures(slug)

        # Check demotion threshold first (highest priority)
        if recent >= _DEMOTION_THRESHOLD:
            demoted = self._apply_demotion(slug)
            return EscalationResult(
                decision=EscalationDecision.DEMOTE,
                reason=(
                    f"{recent} failures in {_DEMOTION_WINDOW_DAYS} days "
                    f"exceeds threshold of {_DEMOTION_THRESHOLD}"
                ),
                demoted=demoted,
            )

        # Check retry possibility
        if retry_strategy == "retry" and max_retries > 0:
            return EscalationResult(
                decision=EscalationDecision.RETRY,
                reason=f"Retry strategy available (max {max_retries})",
                retry_count=0,
                max_retries=max_retries,
            )

        return EscalationResult(
            decision=EscalationDecision.ABORT_NOTIFY,
            reason=f"No retry strategy; error: {error[:200]}",
        )

    def handle_deviation(
        self, slug: str, execution_id: str, deviation_detail: str
    ) -> EscalationResult:
        """Handle an execution deviation.  Treated as soft failure."""
        recent = self.get_recent_failures(slug)

        if recent >= _DEMOTION_THRESHOLD:
            demoted = self._apply_demotion(slug)
            return EscalationResult(
                decision=EscalationDecision.DEMOTE,
                reason=f"{recent} failures/deviations in {_DEMOTION_WINDOW_DAYS} days",
                demoted=demoted,
            )

        return EscalationResult(
            decision=EscalationDecision.ABORT_NOTIFY,
            reason=f"Deviation: {deviation_detail[:200]}",
        )

    def get_recent_failures(
        self, slug: str, window_days: int = _DEMOTION_WINDOW_DAYS
    ) -> int:
        """Count FAILED + DEVIATED executions in the last *window_days*."""
        exec_path = self._kb.root / "observations" / "executions.json"
        if not exec_path.exists():
            return 0

        try:
            with open(exec_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return 0

        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=window_days)
        ).isoformat()
        count = 0
        for record in data.get("records", []):
            if record.get("procedure_slug") != slug:
                continue
            status = record.get("status", "")
            if status not in ("failed", "deviated"):
                continue
            completed = record.get(
                "completed_at", record.get("started_at", "")
            )
            if completed >= cutoff:
                count += 1
        return count

    def check_demotion_threshold(self, slug: str) -> bool:
        """Check if a procedure has hit the demotion threshold."""
        return self.get_recent_failures(slug) >= _DEMOTION_THRESHOLD

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_demotion(self, slug: str) -> bool:
        """Demote procedure from agent_ready to stale."""
        if self._lifecycle is None:
            logger.warning("No lifecycle manager -- cannot demote %s", slug)
            return False
        try:
            from oc_apprentice_worker.lifecycle_manager import ProcedureLifecycle

            current = self._lifecycle.get_state(slug)
            if current in (ProcedureLifecycle.AGENT_READY, ProcedureLifecycle.DRAFT):
                self._lifecycle.transition(
                    slug,
                    ProcedureLifecycle.STALE,
                    trigger="repeated_failures",
                    actor="system",
                    reason=(
                        f"{_DEMOTION_THRESHOLD}+ failures "
                        f"in {_DEMOTION_WINDOW_DAYS} days"
                    ),
                )
                return True
        except Exception:
            logger.debug("Demotion failed for %s", slug, exc_info=True)
        return False
