"""Skill improver — writes a finished agent run back onto its procedure.

When an agent reports ``report_execution_complete`` (MCP), the run is
already a row of the ``runs`` table (``execution_monitor``).  This module
updates the procedure from it:

  * ``execution_stats`` — counts over every finished run of the procedure,
    read from the table (the table is the source, the procedure a copy);
  * ``execution_confidence`` — how well agent runs go, kept apart from
    ``confidence_avg`` (how well the synthesis is grounded in what was
    observed; nothing here changes it);
  * ``staleness.last_confirmed`` — when a run got to the end (completed
    or deviated);
  * ``observed_alternatives`` — one entry per deviated step of the run;
  * drift signals for deviated and failed runs, and the timing average
    for completed runs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def execution_confidence(stats: dict) -> float:
    """Laplace-smoothed success over finished runs; 0.5 with no runs.

    A deviated run counts half; aborted runs carry no signal.
    """
    completed = stats.get("completed", 0)
    deviated = stats.get("deviated", 0)
    failed = stats.get("failed", 0)
    return round((completed + 0.5 * deviated + 1) / (completed + deviated + failed + 2), 4)


class SkillImprover:
    """Processes finished runs into procedure updates."""

    def __init__(self, kb, monitor) -> None:
        self._kb = kb
        self._monitor = monitor

    def process_execution(self, record) -> list[str]:
        """Update the run's procedure; returns what changed."""
        slug = record.procedure_slug
        proc = self._kb.get_procedure(slug)
        if proc is None:
            return []

        status = record.status.value if hasattr(record.status, "value") else str(record.status)
        now = datetime.now(timezone.utc).isoformat()
        improvements: list[str] = []

        stats = self._monitor.get_success_rate(slug)
        proc["execution_stats"] = stats
        proc["execution_confidence"] = execution_confidence(stats)
        improvements.append("updated_execution_stats")

        staleness = proc.setdefault("staleness", {})
        if status in ("completed", "deviated"):
            staleness["last_confirmed"] = record.completed_at or now
            improvements.append("confirmed_fresh")

        if status == "completed":
            if self._update_timing(proc, record):
                improvements.append("updated_timing")
        elif status == "deviated":
            staleness.setdefault("drift_signals", []).append({
                "type": "execution_deviation",
                "detail": f"{len(record.deviations)} step(s) deviated",
                "first_seen": now,
            })
            improvements.append("added_drift_signal")
            improvements += self._add_alternatives(proc, record, now)
        elif status == "failed":
            staleness.setdefault("drift_signals", []).append({
                "type": "execution_failure",
                "detail": f"Agent execution failed: {record.error or 'unknown'}",
                "first_seen": now,
            })
            improvements.append("added_drift_signal")

        self._kb.save_procedure(proc)
        logger.info(
            "Skill '%s' updated from execution %s: %s",
            slug, record.execution_id[:8], improvements,
        )
        return improvements

    @staticmethod
    def _add_alternatives(proc: dict, record, now: str) -> list[str]:
        """One ``observed_alternatives`` entry per deviated step of the run."""
        alternatives = proc.setdefault("observed_alternatives", [])
        seen = {(a.get("execution_id"), a.get("step_id")) for a in alternatives}
        for dev in record.deviations or []:
            step_id = dev.get("step_id", "")
            if not step_id or (record.execution_id, step_id) in seen:
                continue
            seen.add((record.execution_id, step_id))
            alternatives.append({
                "step_id": step_id,
                "observed_action": dev.get("actual_action") or dev.get("detail", ""),
                "detail": dev.get("detail", ""),
                "execution_id": record.execution_id,
                "timestamp": dev.get("at") or now,
            })

        per_step: dict[str, int] = {}
        for alt in alternatives:
            per_step[alt.get("step_id", "")] = per_step.get(alt.get("step_id", ""), 0) + 1
        return [
            f"branch_suggested:{sid}" for sid, n in per_step.items() if sid and n >= 2
        ]

    @staticmethod
    def _update_timing(proc: dict, record) -> bool:
        """Fold the run's duration into ``workflow_rhythm`` (EMA)."""
        if not (record.completed_at and record.started_at):
            return False
        try:
            start = datetime.fromisoformat(record.started_at)
            end = datetime.fromisoformat(record.completed_at)
        except (ValueError, TypeError):
            return False
        duration = (end - start).total_seconds() / 60.0
        rhythm = proc.setdefault("workflow_rhythm", {})
        prev = rhythm.get("avg_duration_minutes")
        rhythm["avg_duration_minutes"] = (
            round(prev * 0.7 + duration * 0.3, 1) if prev else round(duration, 1)
        )
        return True
