"""Skill improver — processes execution results to improve procedures.

When an agent executes a Skill and reports back (via MCP or REST),
this module updates the procedure based on the outcome:
  * Success: boost confidence, confirm freshness, update timing
  * Deviation: track alternatives, suggest branches, add drift signals
  * Failure: reduce confidence, add drift signal, trigger demotion check
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SkillImprover:
    """Processes execution results into procedure improvements."""

    def __init__(self, kb, lifecycle_manager=None) -> None:
        self._kb = kb
        self._lifecycle = lifecycle_manager

    def process_execution(self, record) -> list[str]:
        """Process a completed execution record.

        Returns a list of improvement descriptions made.
        """
        improvements: list[str] = []
        slug = record.procedure_slug
        proc = self._kb.get_procedure(slug)
        if proc is None:
            return improvements

        status = record.status
        status_str = status.value if hasattr(status, "value") else str(status)

        if status_str == "completed":
            improvements += self._handle_success(proc, record)
        elif status_str == "deviated":
            improvements += self._handle_deviation(proc, record)
        elif status_str in ("failed", "error"):
            improvements += self._handle_failure(proc, record)

        if improvements:
            self._kb.save_procedure(proc)
            logger.info(
                "Skill '%s' improved from execution %s: %s",
                slug, record.execution_id[:8], improvements,
            )

        return improvements

    def _handle_success(self, proc: dict, record) -> list[str]:
        """Success: boost confidence, confirm freshness, update timing."""
        improvements = []
        now = datetime.now(timezone.utc).isoformat()

        # Confirm freshness
        staleness = proc.setdefault("staleness", {})
        staleness["last_confirmed"] = now
        improvements.append("confirmed_fresh")

        # Boost confidence (small increment, cap at 1.0)
        current = proc.get("confidence_avg", 0.5)
        proc["confidence_avg"] = min(1.0, round(current + 0.02, 4))
        trend = staleness.setdefault("confidence_trend", [])
        trend.append(proc["confidence_avg"])
        if len(trend) > 50:
            trend[:] = trend[-50:]
        improvements.append("boosted_confidence")

        # Update timing via EMA if available
        if record.completed_at and record.started_at:
            try:
                start = datetime.fromisoformat(record.started_at)
                end = datetime.fromisoformat(record.completed_at)
                duration = (end - start).total_seconds() / 60.0
                rhythm = proc.setdefault("workflow_rhythm", {})
                prev = rhythm.get("avg_duration_minutes")
                if prev:
                    rhythm["avg_duration_minutes"] = round(prev * 0.7 + duration * 0.3, 1)
                else:
                    rhythm["avg_duration_minutes"] = round(duration, 1)
                improvements.append("updated_timing")
            except (ValueError, TypeError):
                pass

        # Track execution stats
        self._update_stats(proc, "success", record)

        return improvements

    def _handle_deviation(self, proc: dict, record) -> list[str]:
        """Deviation: track alternatives, suggest branches."""
        improvements = []
        now = datetime.now(timezone.utc).isoformat()

        # Add drift signal
        staleness = proc.setdefault("staleness", {})
        drift = staleness.setdefault("drift_signals", [])
        dev_count = len(record.deviations) if record.deviations else 0
        drift.append({
            "type": "execution_deviation",
            "detail": f"{dev_count} step(s) deviated",
            "first_seen": now,
        })
        improvements.append("added_drift_signal")

        # Track observed alternatives
        alternatives = proc.setdefault("observed_alternatives", [])
        for dev in (record.deviations or []):
            step_id = dev.get("step_id", "")
            detail = dev.get("detail", "")
            if step_id and detail:
                alternatives.append({
                    "step_id": step_id,
                    "observed_action": detail,
                    "execution_id": record.execution_id,
                    "timestamp": now,
                })

        # If same alternative seen 2+ times, suggest branch
        step_alts: dict[str, list[str]] = {}
        for alt in alternatives:
            sid = alt.get("step_id", "")
            action = alt.get("observed_action", "")
            if sid:
                step_alts.setdefault(sid, []).append(action)
        for sid, actions in step_alts.items():
            if len(actions) >= 2:
                improvements.append(f"branch_suggested:{sid}")

        # Track stats
        self._update_stats(proc, "deviated", record)

        return improvements

    def _handle_failure(self, proc: dict, record) -> list[str]:
        """Failure: reduce confidence, add drift signal."""
        improvements = []
        now = datetime.now(timezone.utc).isoformat()

        # Add drift signal
        staleness = proc.setdefault("staleness", {})
        drift = staleness.setdefault("drift_signals", [])
        drift.append({
            "type": "execution_failure",
            "detail": f"Agent execution failed: {record.error or 'unknown'}",
            "first_seen": now,
        })

        # Reduce confidence
        current = proc.get("confidence_avg", 0.5)
        proc["confidence_avg"] = max(0.0, round(current - 0.05, 4))
        trend = staleness.setdefault("confidence_trend", [])
        trend.append(proc["confidence_avg"])
        improvements.append("reduced_confidence")

        # Track stats
        self._update_stats(proc, "failed", record)

        return improvements

    @staticmethod
    def _update_stats(proc: dict, outcome: str, record) -> None:
        """Update execution_stats on the procedure."""
        stats = proc.setdefault("execution_stats", {
            "total": 0, "success": 0, "failed": 0, "deviated": 0,
        })
        stats["total"] = stats.get("total", 0) + 1
        stats[outcome] = stats.get(outcome, 0) + 1
        stats["last_executed"] = (
            record.completed_at or datetime.now(timezone.utc).isoformat()
        )
        total = stats["total"]
        if total > 0:
            stats["success_rate"] = round(stats.get("success", 0) / total, 3)
