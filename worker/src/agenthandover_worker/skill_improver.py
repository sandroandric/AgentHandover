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
    for completed runs;
  * the learning ledger (``proc["knowledge"]``, :mod:`ledger`), through
    :func:`feed_ledger`: a run is evidence about the slots it exercised.

How a run feeds the ledger (``source="agent"``, ``run`` evidence):

  * A *completed* run (no step deviated) that reported every step as
    written adds a ``run`` pointer to the ``steps`` slot while it is a
    ``hypothesis``; one that reported every step a decision applies at
    does the same for ``decisions``.  :data:`RUN_CONFIRMATIONS` such runs
    make the slot ``known``.  A slot already ``known`` or ``unknown`` is
    left as it is (a run does not answer what nobody knows).
  * A deviation at a step a ``hypothesis`` decision applies at demotes
    ``decisions`` to ``unknown`` (the run's pointer is its evidence) and
    queues a question for the user: an unanswered entry in
    ``agent_clarifications`` (category ``decision``) naming the step and
    what the agent did.  The packet shows it as an open question, the
    daily digest lists it, and the user's answer makes the slot known.

A decision *applies at* the step a branch names (``step_index`` from
divergence mining, ``position`` from variant detection).  Selection
criteria and branches that name no step apply at every step.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from agenthandover_worker.ledger import refresh_knowledge, set_slot
from agenthandover_worker.procedure_schema import procedure_steps, step_key

logger = logging.getLogger(__name__)


def execution_confidence(stats: dict) -> float:
    """Laplace-smoothed success over finished runs; 0.5 with no runs.

    A deviated run counts half; aborted runs carry no signal.
    """
    completed = stats.get("completed", 0)
    deviated = stats.get("deviated", 0)
    failed = stats.get("failed", 0)
    return round((completed + 0.5 * deviated + 1) / (completed + deviated + failed + 2), 4)


#: Completed runs that make a ``hypothesis`` slot ``known``.
RUN_CONFIRMATIONS = 3
#: ``origin.kind`` of a question queued from an agent run.
RUN_QUESTION = "agent_run"


def _decision_steps(proc: dict) -> set[str] | None:
    """The ids of the steps the procedure's decisions apply at; None when
    a decision names no step (it applies at every step)."""
    steps = proc.get("steps") or []
    keys = {id(s): step_key(s, i) for i, s in enumerate(procedure_steps(proc))}
    if any(isinstance(sc, dict) and sc.get("criterion") for sc in proc.get("selection_criteria") or []):
        return None
    ids: set[str] = set()
    for branch in proc.get("branches") or []:
        if not isinstance(branch, dict):
            continue
        index = branch.get("step_index", branch.get("position"))
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            return None
        # Divergence mining puts a detour after the last step at len(steps).
        index = min(index, len(steps) - 1)
        if index >= 0 and id(steps[index]) in keys:
            ids.add(keys[id(steps[index])])
    # A decision the procedure holds without any branch (a user's answer)
    # names no step either.
    return ids or None


def _add_run(proc: dict, slot: str, entry: dict, execution_id: str) -> str | None:
    """Add the run's pointer to the ``hypothesis`` *entry* of *slot*;
    ``known`` once :data:`RUN_CONFIRMATIONS` runs confirm it.  Returns the
    change, None when the run was already counted."""
    pointer = {"kind": "run", "id": execution_id}
    evidence = [dict(e) for e in entry.get("evidence") or []]
    if pointer in evidence:
        return None
    evidence.append(pointer)
    runs = sum(1 for e in evidence if e.get("kind") == "run")
    status = "known" if runs >= RUN_CONFIRMATIONS else "hypothesis"
    set_slot(proc, slot, status=status, source="agent", evidence=evidence,
             value_ref=entry.get("value_ref"))
    return f"ledger_{status}:{slot}"


def _queue_question(proc: dict, record, deviation: dict, now: str) -> None:
    """An unanswered ``decision`` question about *deviation* (once per run
    and step)."""
    step_id = deviation.get("step_id", "")
    origin = {"kind": RUN_QUESTION, "id": record.execution_id, "step_id": step_id}
    clarifications = proc.setdefault("agent_clarifications", [])
    if any(isinstance(c, dict) and c.get("origin") == origin for c in clarifications):
        return
    step = next((s for s in record.steps if s.step_id == step_id), None)
    expected = step.expected_action if step is not None else ""
    did = deviation.get("actual_action") or deviation.get("detail") or "something else"
    at = f'step [{step_id}] ("{expected}")' if expected else f"step [{step_id}]"
    clarifications.append({
        "category": "decision",
        "question": (
            f"An agent running this Skill did not follow {at}: it did \"{did}\". "
            "How do you decide what to do at this step?"
        ),
        "answer": "",
        "status": "unanswered",
        "context": deviation.get("detail", ""),
        "origin": origin,
        "asked_at": now,
    })


def feed_ledger(proc: dict, record, now: str) -> list[str]:
    """Write what the finished run *record* shows onto the ledger of
    *proc*, in place (see the module docstring).  Returns what changed."""
    status = record.status.value if hasattr(record.status, "value") else str(record.status)
    ledger = refresh_knowledge(proc)
    decisions = ledger["decisions"]
    at_steps = _decision_steps(proc)
    changes: list[str] = []

    if record.deviations and decisions["status"] == "hypothesis":
        hit = [
            d for d in record.deviations
            if at_steps is None or d.get("step_id") in at_steps
        ]
        if hit:
            set_slot(proc, "decisions", status="unknown", source="agent",
                     evidence=[{"kind": "run", "id": record.execution_id}])
            for deviation in hit:
                _queue_question(proc, record, deviation, now)
            changes += ["ledger_unknown:decisions", "question_queued"]
    elif status == "completed":
        done = {
            s.step_id for s in record.steps
            if (s.status.value if hasattr(s.status, "value") else str(s.status)) == "completed"
        }
        written = {step_key(s, i) for i, s in enumerate(procedure_steps(proc))}
        if ledger["steps"]["status"] == "hypothesis" and written and written <= done:
            changes.append(_add_run(proc, "steps", ledger["steps"], record.execution_id))
        needed = written if at_steps is None else at_steps
        if decisions["status"] == "hypothesis" and needed and needed <= done:
            changes.append(_add_run(proc, "decisions", decisions, record.execution_id))
    return [c for c in changes if c]


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

        improvements += feed_ledger(proc, record, now)
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
