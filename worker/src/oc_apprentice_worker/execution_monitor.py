"""Execution monitor — tracks agent execution of procedures.

Observes agent-produced events and tracks execution success, failure, and
deviation against known procedures stored in the KnowledgeBase.

Execution history is persisted to::

    {kb_root}/observations/executions.json

using the same atomic write pattern (tmp+fsync+rename) as KnowledgeBase.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DEVIATED = "deviated"
    ABORTED = "aborted"


@dataclass
class ExecutionStep:
    step_id: str
    expected_action: str
    actual_action: str | None = None
    status: ExecutionStatus = ExecutionStatus.IN_PROGRESS
    started_at: str | None = None
    completed_at: str | None = None
    deviation_detail: str | None = None


@dataclass
class ExecutionRecord:
    execution_id: str
    procedure_slug: str
    agent_id: str
    status: ExecutionStatus
    started_at: str
    completed_at: str | None = None
    steps: list[ExecutionStep] = field(default_factory=list)
    outcomes: list[dict] = field(default_factory=list)
    deviations: list[dict] = field(default_factory=list)
    error: str | None = None
    metadata: dict | None = None


def _record_to_dict(record: ExecutionRecord) -> dict:
    """Serialize an ExecutionRecord to a JSON-safe dict."""
    d = asdict(record)
    d["status"] = record.status.value
    for step_dict, step_obj in zip(d["steps"], record.steps):
        step_dict["status"] = step_obj.status.value
    return d


def _dict_to_record(d: dict) -> ExecutionRecord:
    """Deserialize a dict into an ExecutionRecord."""
    steps = []
    for s in d.get("steps", []):
        steps.append(ExecutionStep(
            step_id=s["step_id"],
            expected_action=s["expected_action"],
            actual_action=s.get("actual_action"),
            status=ExecutionStatus(s.get("status", "in_progress")),
            started_at=s.get("started_at"),
            completed_at=s.get("completed_at"),
            deviation_detail=s.get("deviation_detail"),
        ))
    return ExecutionRecord(
        execution_id=d["execution_id"],
        procedure_slug=d["procedure_slug"],
        agent_id=d["agent_id"],
        status=ExecutionStatus(d["status"]),
        started_at=d["started_at"],
        completed_at=d.get("completed_at"),
        steps=steps,
        outcomes=d.get("outcomes", []),
        deviations=d.get("deviations", []),
        error=d.get("error"),
        metadata=d.get("metadata"),
    )


_EXECUTIONS_FILE = "executions.json"


class ExecutionMonitor:
    """Monitors agent execution of procedures and tracks outcomes."""

    def __init__(self, knowledge_base, escalation_handler=None) -> None:
        self._kb = knowledge_base
        self._escalation = escalation_handler
        self._active_executions: dict[str, ExecutionRecord] = {}
        self._history: list[ExecutionRecord] = []
        self._load_history()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_execution(
        self, procedure_slug: str, agent_id: str = "unknown"
    ) -> str:
        """Start monitoring an execution.  Returns execution_id (UUID)."""
        execution_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Build expected steps from the procedure in the KB
        procedure = self._kb.get_procedure(procedure_slug)
        steps: list[ExecutionStep] = []
        if procedure is not None:
            raw_steps = procedure.get("steps", [])
            for i, s in enumerate(raw_steps):
                expected_action = (
                    s.get("action")
                    or s.get("step")
                    or f"step_{i}"
                )
                steps.append(ExecutionStep(
                    step_id=str(i),
                    expected_action=expected_action,
                    started_at=now,
                ))

        record = ExecutionRecord(
            execution_id=execution_id,
            procedure_slug=procedure_slug,
            agent_id=agent_id,
            status=ExecutionStatus.IN_PROGRESS,
            started_at=now,
            steps=steps,
        )
        self._active_executions[execution_id] = record
        logger.info(
            "Started execution %s for procedure %s (agent=%s, steps=%d)",
            execution_id, procedure_slug, agent_id, len(steps),
        )
        return execution_id

    def record_step(
        self, execution_id: str, step_id: str, actual_action: str
    ) -> None:
        """Record what the agent actually did for a step.

        Compares with expected action from the procedure.  If the
        execution_id is unknown, logs a warning and returns.
        """
        record = self._active_executions.get(execution_id)
        if record is None:
            logger.warning(
                "record_step: unknown execution_id %s", execution_id
            )
            return

        now = datetime.now(timezone.utc).isoformat()

        # Find the matching step
        step = self._find_step(record, step_id)
        if step is None:
            # Dynamic step — agent did something we didn't expect
            step = ExecutionStep(
                step_id=step_id,
                expected_action="(dynamic)",
                started_at=now,
            )
            record.steps.append(step)

        step.actual_action = actual_action
        step.completed_at = now
        step.status = ExecutionStatus.COMPLETED

        # Check for deviation: expected vs actual
        if (
            step.expected_action != "(dynamic)"
            and not self._actions_equivalent(step.expected_action, actual_action)
        ):
            step.status = ExecutionStatus.DEVIATED
            detail = (
                f"Expected '{step.expected_action}', "
                f"got '{actual_action}'"
            )
            step.deviation_detail = detail
            record.deviations.append({
                "step_id": step_id,
                "detail": detail,
                "at": now,
            })
            logger.info(
                "Deviation in execution %s step %s: %s",
                execution_id, step_id, detail,
            )

    def record_deviation(
        self, execution_id: str, step_id: str, detail: str
    ) -> None:
        """Record a deviation from the expected procedure."""
        record = self._active_executions.get(execution_id)
        if record is None:
            logger.warning(
                "record_deviation: unknown execution_id %s", execution_id
            )
            return

        now = datetime.now(timezone.utc).isoformat()

        step = self._find_step(record, step_id)
        if step is not None:
            step.status = ExecutionStatus.DEVIATED
            step.deviation_detail = detail

        record.deviations.append({
            "step_id": step_id,
            "detail": detail,
            "at": now,
        })
        logger.info(
            "Deviation recorded for execution %s step %s: %s",
            execution_id, step_id, detail,
        )

    def complete_execution(
        self,
        execution_id: str,
        outcomes: list[dict] | None = None,
    ) -> ExecutionRecord:
        """Mark execution as completed, record outcomes, move to history."""
        record = self._active_executions.get(execution_id)
        if record is None:
            raise KeyError(f"No active execution with id {execution_id}")

        now = datetime.now(timezone.utc).isoformat()
        record.status = ExecutionStatus.COMPLETED
        record.completed_at = now
        if outcomes:
            record.outcomes = outcomes

        # If any step deviated, mark overall as DEVIATED
        if record.deviations:
            record.status = ExecutionStatus.DEVIATED

        if self._escalation is not None and record.status == ExecutionStatus.DEVIATED:
            try:
                dev_detail = "; ".join(d.get("detail", "") for d in record.deviations[:3])
                esc_result = self._escalation.handle_deviation(
                    record.procedure_slug, execution_id, dev_detail,
                )
                record.metadata = record.metadata or {}
                record.metadata["escalation"] = {
                    "decision": esc_result.decision.value,
                    "reason": esc_result.reason,
                    "demoted": esc_result.demoted,
                }
            except Exception:
                logger.debug("Escalation check failed", exc_info=True)

        self._finalize(execution_id, record)
        return record

    def fail_execution(
        self, execution_id: str, error: str
    ) -> ExecutionRecord:
        """Mark execution as failed."""
        record = self._active_executions.get(execution_id)
        if record is None:
            raise KeyError(f"No active execution with id {execution_id}")

        now = datetime.now(timezone.utc).isoformat()
        record.status = ExecutionStatus.FAILED
        record.completed_at = now
        record.error = error

        # Escalation check
        if self._escalation is not None:
            try:
                esc_result = self._escalation.handle_failure(
                    record.procedure_slug, execution_id, error,
                )
                record.metadata = record.metadata or {}
                record.metadata["escalation"] = {
                    "decision": esc_result.decision.value,
                    "reason": esc_result.reason,
                    "demoted": esc_result.demoted,
                }
            except Exception:
                logger.debug("Escalation check failed", exc_info=True)

        self._finalize(execution_id, record)
        return record

    def abort_execution(self, execution_id: str) -> ExecutionRecord:
        """Mark execution as aborted."""
        record = self._active_executions.get(execution_id)
        if record is None:
            raise KeyError(f"No active execution with id {execution_id}")

        now = datetime.now(timezone.utc).isoformat()
        record.status = ExecutionStatus.ABORTED
        record.completed_at = now

        self._finalize(execution_id, record)
        return record

    def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        """Get an execution record by ID (active or history)."""
        # Check active first
        record = self._active_executions.get(execution_id)
        if record is not None:
            return record
        # Check history
        for rec in self._history:
            if rec.execution_id == execution_id:
                return rec
        return None

    def get_history(
        self,
        procedure_slug: str | None = None,
        limit: int = 50,
    ) -> list[ExecutionRecord]:
        """Get execution history, optionally filtered by procedure.

        Returns records ordered newest-first, capped at *limit*.
        """
        records = self._history
        if procedure_slug is not None:
            records = [
                r for r in records
                if r.procedure_slug == procedure_slug
            ]
        # newest first (by started_at)
        records = sorted(records, key=lambda r: r.started_at, reverse=True)
        return records[:limit]

    def get_success_rate(self, procedure_slug: str) -> dict:
        """Get success/failure/deviation statistics for a procedure.

        Returns::

            {
                "total": N,
                "completed": N,
                "failed": N,
                "deviated": N,
                "aborted": N,
                "success_rate": float,
            }
        """
        relevant = [
            r for r in self._history
            if r.procedure_slug == procedure_slug
        ]
        total = len(relevant)
        completed = sum(
            1 for r in relevant if r.status == ExecutionStatus.COMPLETED
        )
        failed = sum(
            1 for r in relevant if r.status == ExecutionStatus.FAILED
        )
        deviated = sum(
            1 for r in relevant if r.status == ExecutionStatus.DEVIATED
        )
        aborted = sum(
            1 for r in relevant if r.status == ExecutionStatus.ABORTED
        )
        success_rate = completed / total if total > 0 else 0.0
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "deviated": deviated,
            "aborted": aborted,
            "success_rate": success_rate,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _actions_equivalent(self, expected: str, actual: str) -> bool:
        """Check if two action strings are equivalent.

        Uses case-insensitive string comparison (existing behavior).
        Can be extended to use embedding-based similarity in the future.
        """
        return expected.strip().lower() == actual.strip().lower()

    def _find_step(
        self, record: ExecutionRecord, step_id: str
    ) -> ExecutionStep | None:
        """Find a step by step_id within a record."""
        for step in record.steps:
            if step.step_id == step_id:
                return step
        return None

    def _finalize(
        self, execution_id: str, record: ExecutionRecord
    ) -> None:
        """Move a record from active to history and persist."""
        del self._active_executions[execution_id]
        self._history.append(record)
        self._save_history()
        logger.info(
            "Execution %s finalized as %s",
            execution_id, record.status.value,
        )

    def _executions_path(self) -> Path:
        """Path to the executions history file."""
        return (
            self._kb.root / "observations" / _EXECUTIONS_FILE
        )

    def _load_history(self) -> None:
        """Load execution history from {kb_root}/observations/executions.json."""
        path = self._executions_path()
        if not path.is_file():
            self._history = []
            return
        try:
            with open(path) as f:
                data = json.load(f)
            records_raw = data.get("records", [])
            self._history = [_dict_to_record(d) for d in records_raw]
            logger.debug(
                "Loaded %d execution records from %s",
                len(self._history), path,
            )
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning("Failed to load execution history: %s", exc)
            self._history = []

    def _save_history(self) -> None:
        """Persist execution history to KB using atomic write."""
        path = self._executions_path()
        payload = {
            "records": [_record_to_dict(r) for r in self._history],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._kb.atomic_write_json(path, payload)
