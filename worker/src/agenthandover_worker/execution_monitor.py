"""Execution monitor — tracks agent execution of procedures.

An agent reports a run through the MCP server (``report_execution_start``,
``report_step_result``, ``report_execution_complete``); this module keeps
each run as one row of the ``runs`` table of the daemon's database (the
knowledge base's ``db_path``; the daemon creates the table, migration
v006).

A run is written when it starts and after every step, so the MCP server
(one process per agent session) and the worker (which reads the table
for trust suggestions and the query API) always see the same runs.

Step ids are the ids the packet prints (``procedure_schema.step_key``).
An agent may report a step by that id or by its position (``step_2``,
``2``); both land on the same step.  At most one deviation is kept per
step and run.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from agenthandover_worker.procedure_schema import procedure_steps, step_key

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


# A step the procedure did not list (the agent reported an unknown id).
DYNAMIC_ACTION = "(dynamic)"

# Where runs were kept before they moved into the daemon's database
# (development builds of v0.5), relative to the knowledge base root.
_OLD_RUNS_DB = Path("observations") / "runs.db"
_LEGACY_JSON = "executions.json"

# Two actions are the same when their normalised token sets overlap by at
# least this Jaccard index ("Open Chrome and go to example.com" ==
# "opened chrome, went to example.com"; "Click Submit" != "Press Enter").
ACTION_SIMILARITY_THRESHOLD = 0.5

_STOPWORDS = frozenset(
    "a an the to on in into of and or for with from at by then this that it "
    "its is be your my".split()
)


def _action_tokens(text: str) -> set[str]:
    tokens = set()
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in _STOPWORDS:
            continue
        # Light stemming so tense and plurals compare equal.
        for suffix in ("ing", "ed", "es", "s"):
            if len(tok) > len(suffix) + 2 and tok.endswith(suffix):
                tok = tok[: -len(suffix)]
                break
        tokens.add(tok)
    return tokens


def actions_equivalent(expected: str, actual: str) -> bool:
    """True when *actual* describes the same action as *expected*."""
    a, b = _action_tokens(expected), _action_tokens(actual)
    if not a or not b:
        return expected.strip().lower() == actual.strip().lower()
    return len(a & b) / len(a | b) >= ACTION_SIMILARITY_THRESHOLD


def _ordinal(step_id: str) -> int | None:
    """The 1-based position a step id names: ``step_2``, ``Step 2``, ``2``."""
    text = step_id.strip().lower()
    text = re.sub(r"^step[\s_\-#]*", "", text)
    return int(text) if text.isdigit() else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _step_to_dict(step: ExecutionStep) -> dict:
    return {
        "step_id": step.step_id,
        "expected_action": step.expected_action,
        "actual_action": step.actual_action,
        "status": step.status.value,
        "started_at": step.started_at,
        "completed_at": step.completed_at,
        "deviation_detail": step.deviation_detail,
    }


def _dict_to_step(s: dict) -> ExecutionStep:
    return ExecutionStep(
        step_id=str(s["step_id"]),
        expected_action=s["expected_action"],
        actual_action=s.get("actual_action"),
        status=ExecutionStatus(s.get("status", "in_progress")),
        started_at=s.get("started_at"),
        completed_at=s.get("completed_at"),
        deviation_detail=s.get("deviation_detail"),
    )


class ExecutionMonitor:
    """Records agent runs of procedures in the ``runs`` table."""

    def __init__(self, knowledge_base) -> None:
        self._kb = knowledge_base
        self._path = Path(knowledge_base.db_path)
        self._import_old_runs_db()
        self._import_legacy_json()

    @property
    def db_path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_execution(
        self, procedure_slug: str, agent_id: str = "unknown"
    ) -> str:
        """Start a run.  Returns its execution_id (UUID)."""
        execution_id = str(uuid.uuid4())
        now = _now()
        procedure = self._kb.get_procedure(procedure_slug)
        steps = [
            ExecutionStep(
                step_id=step_key(s, i),
                expected_action=str(
                    s.get("action") or s.get("step") or s.get("description")
                    or step_key(s, i)
                ),
                started_at=now,
            )
            for i, s in enumerate(procedure_steps(procedure or {}))
        ]
        record = ExecutionRecord(
            execution_id=execution_id,
            procedure_slug=procedure_slug,
            agent_id=agent_id,
            status=ExecutionStatus.IN_PROGRESS,
            started_at=now,
            steps=steps,
        )
        self._save(record)
        logger.info(
            "Started execution %s for procedure %s (agent=%s, steps=%d)",
            execution_id, procedure_slug, agent_id, len(steps),
        )
        return execution_id

    def record_step(
        self,
        execution_id: str,
        step_id: str,
        actual_action: str = "",
        *,
        deviated: bool = False,
        notes: str = "",
    ) -> ExecutionStep | None:
        """Record the agent's report for one step; returns the matched step.

        The step deviates when the agent says so (*deviated*) or when it
        describes an action (*actual_action*) that is not equivalent to
        the expected one.  An empty *actual_action* claims nothing about
        what was done.  Returns ``None`` for an unknown or finished run.
        """
        record = self._load_active(execution_id, "record_step")
        if record is None:
            return None
        now = _now()
        step = self._find_step(record, step_id)
        if step is None:
            step = ExecutionStep(
                step_id=str(step_id).strip(),
                expected_action=DYNAMIC_ACTION,
                started_at=now,
            )
            record.steps.append(step)

        step.actual_action = actual_action or None
        step.completed_at = now
        step.status = ExecutionStatus.COMPLETED

        mismatch = (
            bool(actual_action)
            and step.expected_action != DYNAMIC_ACTION
            and not actions_equivalent(step.expected_action, actual_action)
        )
        if deviated or mismatch:
            if notes:
                detail = notes
            elif actual_action:
                detail = f"Expected '{step.expected_action}', got '{actual_action}'"
            else:
                detail = "The agent reported a deviation without details"
            self._set_deviation(record, step, detail, now)
        else:
            # A later report for the same step supersedes an earlier one.
            step.deviation_detail = None
            record.deviations = [
                d for d in record.deviations if d.get("step_id") != step.step_id
            ]

        self._save(record)
        return step

    def record_deviation(
        self, execution_id: str, step_id: str, detail: str
    ) -> None:
        """Record a deviation from the expected procedure."""
        record = self._load_active(execution_id, "record_deviation")
        if record is None:
            return
        now = _now()
        step = self._find_step(record, step_id)
        if step is None:
            step = ExecutionStep(
                step_id=str(step_id).strip(),
                expected_action=DYNAMIC_ACTION,
                started_at=now,
                completed_at=now,
            )
            record.steps.append(step)
        self._set_deviation(record, step, detail, now)
        self._save(record)

    def complete_execution(
        self,
        execution_id: str,
        outcomes: list[dict] | None = None,
    ) -> ExecutionRecord:
        """Finish a run as completed (or deviated, when any step deviated)."""
        record = self._require_active(execution_id)
        record.status = (
            ExecutionStatus.DEVIATED if record.deviations
            else ExecutionStatus.COMPLETED
        )
        record.completed_at = _now()
        if outcomes:
            record.outcomes = outcomes
        return self._finalize(record)

    def fail_execution(
        self, execution_id: str, error: str
    ) -> ExecutionRecord:
        """Finish a run as failed."""
        record = self._require_active(execution_id)
        record.status = ExecutionStatus.FAILED
        record.completed_at = _now()
        record.error = error
        return self._finalize(record)

    def abort_execution(self, execution_id: str) -> ExecutionRecord:
        """Finish a run as aborted."""
        record = self._require_active(execution_id)
        record.status = ExecutionStatus.ABORTED
        record.completed_at = _now()
        return self._finalize(record)

    def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        """The run with *execution_id*, active or finished."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE execution_id = ?", (execution_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get_history(
        self,
        procedure_slug: str | None = None,
        limit: int = 50,
    ) -> list[ExecutionRecord]:
        """Finished runs, newest first, optionally for one procedure."""
        sql = "SELECT * FROM runs WHERE status != ?"
        params: list = [ExecutionStatus.IN_PROGRESS.value]
        if procedure_slug is not None:
            sql += " AND procedure_slug = ?"
            params.append(procedure_slug)
        sql += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_success_rate(self, procedure_slug: str) -> dict:
        """Outcome counts over the finished runs of a procedure.

        Returns ``total``, ``completed``, ``failed``, ``deviated``,
        ``aborted``, ``deviations`` (deviated steps over all runs),
        ``success_rate`` (completed / total), ``last_run`` and
        ``last_failure`` (``completed_at`` timestamps or ``None``).
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT status, completed_at, deviations_json FROM runs "
                "WHERE procedure_slug = ? AND status != ?",
                (procedure_slug, ExecutionStatus.IN_PROGRESS.value),
            ).fetchall()
        counts = {s.value: 0 for s in ExecutionStatus if s is not ExecutionStatus.IN_PROGRESS}
        deviations = 0
        last_run: str | None = None
        last_failure: str | None = None
        for row in rows:
            counts[row["status"]] += 1
            deviations += len(json.loads(row["deviations_json"]))
            done = row["completed_at"]
            if done and (last_run is None or done > last_run):
                last_run = done
            if row["status"] == ExecutionStatus.FAILED.value and done and (
                last_failure is None or done > last_failure
            ):
                last_failure = done
        total = len(rows)
        return {
            "total": total,
            **counts,
            "deviations": deviations,
            "success_rate": counts["completed"] / total if total else 0.0,
            "last_run": last_run,
            "last_failure": last_failure,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_step(
        self, record: ExecutionRecord, step_id: str
    ) -> ExecutionStep | None:
        """The step *step_id* names: its exact id, else its position.

        ``step_2``, ``Step 2`` and ``2`` all name the second step the
        procedure listed (the packet numbers steps from 1).
        """
        wanted = str(step_id).strip()
        for step in record.steps:
            if step.step_id == wanted:
                return step
        ordinal = _ordinal(wanted)
        expected = [s for s in record.steps if s.expected_action != DYNAMIC_ACTION]
        if ordinal is not None and 1 <= ordinal <= len(expected):
            return expected[ordinal - 1]
        return None

    @staticmethod
    def _set_deviation(
        record: ExecutionRecord, step: ExecutionStep, detail: str, at: str
    ) -> None:
        """Mark *step* deviated; one deviation entry per step and run."""
        step.status = ExecutionStatus.DEVIATED
        step.deviation_detail = detail
        record.deviations = [
            d for d in record.deviations if d.get("step_id") != step.step_id
        ]
        record.deviations.append({
            "step_id": step.step_id,
            "detail": detail,
            "actual_action": step.actual_action,
            "at": at,
        })
        logger.info(
            "Deviation in execution %s step %s: %s",
            record.execution_id, step.step_id, detail,
        )

    def _load_active(self, execution_id: str, caller: str) -> ExecutionRecord | None:
        record = self.get_execution(execution_id)
        if record is None or record.status is not ExecutionStatus.IN_PROGRESS:
            logger.warning("%s: no active execution %s", caller, execution_id)
            return None
        return record

    def _require_active(self, execution_id: str) -> ExecutionRecord:
        record = self.get_execution(execution_id)
        if record is None or record.status is not ExecutionStatus.IN_PROGRESS:
            raise KeyError(f"No active execution with id {execution_id}")
        return record

    def _finalize(self, record: ExecutionRecord) -> ExecutionRecord:
        self._save(record)
        logger.info(
            "Execution %s finalized as %s",
            record.execution_id, record.status.value,
        )
        return record

    def _connect(self) -> sqlite3.Connection:
        # mode=rw: never create the database; the daemon owns it.
        conn = sqlite3.connect(
            f"{self._path.resolve().as_uri()}?mode=rw", uri=True, timeout=10,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _save(self, record: ExecutionRecord) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs (execution_id, procedure_slug, "
                "agent_id, status, started_at, completed_at, error, steps_json, "
                "deviations_json, outcomes_json, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.execution_id,
                    record.procedure_slug,
                    record.agent_id,
                    record.status.value,
                    record.started_at,
                    record.completed_at,
                    record.error,
                    json.dumps([_step_to_dict(s) for s in record.steps]),
                    json.dumps(record.deviations),
                    json.dumps(record.outcomes),
                    json.dumps(record.metadata) if record.metadata is not None else None,
                ),
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ExecutionRecord:
        return ExecutionRecord(
            execution_id=row["execution_id"],
            procedure_slug=row["procedure_slug"],
            agent_id=row["agent_id"],
            status=ExecutionStatus(row["status"]),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            steps=[_dict_to_step(s) for s in json.loads(row["steps_json"])],
            outcomes=json.loads(row["outcomes_json"]),
            deviations=json.loads(row["deviations_json"]),
            error=row["error"],
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else None,
        )

    def _import_old_runs_db(self) -> None:
        """Move runs from ``observations/runs.db`` (where development
        builds of v0.5 kept them) into the database, once: the file is
        renamed to ``runs.db.imported`` afterwards.  A run already in the
        table keeps its row."""
        old = Path(self._kb.root) / _OLD_RUNS_DB
        if not old.is_file():
            return
        try:
            with closing(self._connect()) as conn, conn:
                conn.execute("ATTACH DATABASE ? AS old", (f"{old.resolve().as_uri()}?mode=ro",))
                imported = conn.execute(
                    "INSERT OR IGNORE INTO runs SELECT execution_id, procedure_slug, "
                    "agent_id, status, started_at, completed_at, error, steps_json, "
                    "deviations_json, outcomes_json, metadata_json FROM old.runs"
                ).rowcount
            # The connection is closed (DETACH is implicit), so the file can move.
        except sqlite3.Error as exc:
            logger.warning("Could not import %s: %s", old, exc)
            return
        old.rename(old.with_name(old.name + ".imported"))
        logger.info("Imported %d run(s) from %s", imported, old)

    def _import_legacy_json(self) -> None:
        """Move runs from v0.4's ``executions.json`` into the table, once.

        The file is renamed to ``executions.json.imported`` afterwards so
        the import never repeats and nothing is deleted.
        """
        legacy = Path(self._kb.root) / "observations" / _LEGACY_JSON
        if not legacy.is_file():
            return
        try:
            records = json.loads(legacy.read_text()).get("records", [])
            for d in records:
                self._save(ExecutionRecord(
                    execution_id=d["execution_id"],
                    procedure_slug=d["procedure_slug"],
                    agent_id=d.get("agent_id", "unknown"),
                    status=ExecutionStatus(d["status"]),
                    started_at=d["started_at"],
                    completed_at=d.get("completed_at"),
                    steps=[_dict_to_step(s) for s in d.get("steps", [])],
                    outcomes=d.get("outcomes", []),
                    deviations=d.get("deviations", []),
                    error=d.get("error"),
                    metadata=d.get("metadata"),
                ))
        except (json.JSONDecodeError, OSError, KeyError, ValueError, AttributeError) as exc:
            logger.warning("Could not import %s: %s", legacy, exc)
            return
        legacy.rename(legacy.with_name(_LEGACY_JSON + ".imported"))
        logger.info("Imported %d run(s) from %s", len(records), legacy)
