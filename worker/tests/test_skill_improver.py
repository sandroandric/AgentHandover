"""Tests for the SkillImprover: a finished run written back onto its procedure."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenthandover_worker.execution_monitor import ExecutionMonitor
from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.ledger import unknown_slots
from agenthandover_worker.skill_improver import SkillImprover, execution_confidence

SLUG = "test-skill"


@pytest.fixture
def kb(tmp_path: Path) -> KnowledgeBase:
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    kb.save_procedure({
        "id": SLUG,
        "title": "Test Skill",
        "confidence_avg": 0.80,
        "steps": [
            {"step_id": "step_1", "action": "Open the dashboard"},
            {"step_id": "step_2", "action": "Export the report as CSV"},
            {"step_id": "step_3", "action": "Upload the CSV to the shared drive"},
        ],
        "staleness": {
            "last_confirmed": None,
            "confidence_trend": [0.80],
            "drift_signals": [],
        },
        "workflow_rhythm": {},
    })
    return kb


@pytest.fixture
def monitor(kb: KnowledgeBase) -> ExecutionMonitor:
    return ExecutionMonitor(kb)


@pytest.fixture
def improver(kb: KnowledgeBase, monitor: ExecutionMonitor) -> SkillImprover:
    return SkillImprover(kb, monitor)


def _completed(monitor: ExecutionMonitor):
    eid = monitor.start_execution(SLUG)
    for sid in ("step_1", "step_2", "step_3"):
        monitor.record_step(eid, sid)
    return monitor.complete_execution(eid)


def _deviated(monitor: ExecutionMonitor, step: str = "step_3", action: str = "Emailed the CSV"):
    eid = monitor.start_execution(SLUG)
    monitor.record_step(eid, step, action, deviated=True)
    return monitor.complete_execution(eid)


class TestSuccess:

    def test_confidence_avg_is_not_touched(self, improver, monitor, kb):
        improver.process_execution(_completed(monitor))
        proc = kb.get_procedure(SLUG)
        assert proc["confidence_avg"] == 0.80
        assert proc["staleness"]["confidence_trend"] == [0.80]
        assert proc["execution_confidence"] == pytest.approx(2 / 3, abs=1e-4)

    def test_confirms_freshness(self, improver, monitor, kb):
        record = _completed(monitor)
        result = improver.process_execution(record)
        assert kb.get_procedure(SLUG)["staleness"]["last_confirmed"] == record.completed_at
        assert "confirmed_fresh" in result

    def test_updates_timing_ema(self, improver, monitor, kb):
        proc = kb.get_procedure(SLUG)
        proc["workflow_rhythm"]["avg_duration_minutes"] = 10.0
        kb.save_procedure(proc)
        record = _completed(monitor)
        record.started_at = "2026-03-25T10:00:00+00:00"
        record.completed_at = "2026-03-25T10:15:00+00:00"
        improver.process_execution(record)
        # EMA: 10 * 0.7 + 15 * 0.3 = 11.5
        assert kb.get_procedure(SLUG)["workflow_rhythm"]["avg_duration_minutes"] == 11.5

    def test_execution_stats_come_from_the_runs_table(self, improver, monitor, kb):
        _completed(monitor)  # a run the improver never saw
        improver.process_execution(_completed(monitor))
        stats = kb.get_procedure(SLUG)["execution_stats"]
        assert stats["total"] == 2
        assert stats["completed"] == 2
        assert stats["success_rate"] == 1.0


class TestDeviation:

    def test_one_alternative_per_deviated_step(self, improver, monitor, kb):
        eid = monitor.start_execution(SLUG)
        monitor.record_step(eid, "step_1")
        monitor.record_step(eid, "2", "Copied the table by hand", deviated=True)
        monitor.record_step(eid, "step_3", "Emailed the CSV")
        record = monitor.complete_execution(eid)
        result = improver.process_execution(record)

        proc = kb.get_procedure(SLUG)
        alts = proc["observed_alternatives"]
        assert [(a["step_id"], a["observed_action"]) for a in alts] == [
            ("step_2", "Copied the table by hand"),
            ("step_3", "Emailed the CSV"),
        ]
        assert all(a["execution_id"] == eid for a in alts)
        assert proc["staleness"]["last_confirmed"] == record.completed_at
        assert proc["staleness"]["drift_signals"][0]["type"] == "execution_deviation"
        assert "added_drift_signal" in result
        assert proc["execution_stats"]["deviated"] == 1
        assert proc["execution_stats"]["deviations"] == 2

    def test_reprocessing_a_run_adds_nothing(self, improver, monitor, kb):
        record = _deviated(monitor)
        improver.process_execution(record)
        improver.process_execution(record)
        assert len(kb.get_procedure(SLUG)["observed_alternatives"]) == 1

    def test_suggests_branch_after_two(self, improver, monitor):
        improver.process_execution(_deviated(monitor))
        result = improver.process_execution(_deviated(monitor))
        assert "branch_suggested:step_3" in result


class TestFailure:

    def test_failure(self, improver, monitor, kb):
        eid = monitor.start_execution(SLUG)
        record = monitor.fail_execution(eid, "element not found")
        improver.process_execution(record)
        proc = kb.get_procedure(SLUG)
        assert proc["confidence_avg"] == 0.80
        assert proc["execution_confidence"] == pytest.approx(1 / 3, abs=1e-4)
        assert proc["staleness"]["last_confirmed"] is None
        signals = proc["staleness"]["drift_signals"]
        assert signals[0]["type"] == "execution_failure"
        assert "element not found" in signals[0]["detail"]
        assert proc["execution_stats"]["failed"] == 1
        assert proc["execution_stats"]["last_failure"] == record.completed_at


class TestEdgeCases:

    def test_missing_procedure(self, kb, monitor):
        eid = monitor.start_execution("no-such-skill")
        record = monitor.complete_execution(eid)
        assert SkillImprover(kb, monitor).process_execution(record) == []
        assert kb.get_procedure("no-such-skill") is None

    @pytest.mark.parametrize(("stats", "expected"), [
        ({}, 0.5),
        ({"completed": 8}, 0.9),
        ({"failed": 8}, 0.1),
        ({"deviated": 2}, 0.5),
        ({"aborted": 5}, 0.5),
    ])
    def test_execution_confidence(self, stats, expected):
        assert execution_confidence(stats) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Runs feed the learning ledger (WP-L5)
# ---------------------------------------------------------------------------


def _with(kb: KnowledgeBase, **fields) -> None:
    proc = kb.get_procedure(SLUG)
    proc.update(fields)
    kb.save_procedure(proc)


def _slot(kb: KnowledgeBase, slot: str) -> dict:
    return kb.get_procedure(SLUG)["knowledge"][slot]


def _runs(entry: dict) -> list[str]:
    return [e["id"] for e in entry["evidence"] if e["kind"] == "run"]


def _run_questions(kb: KnowledgeBase) -> list[dict]:
    return [
        c for c in kb.get_procedure(SLUG).get("agent_clarifications") or []
        if (c.get("origin") or {}).get("kind") == "agent_run"
    ]


# A decision the synthesiser wrote: it names no step, so it applies at every step.
UNANCHORED = [{"condition": "The export is empty", "action": "Stop and tell the owner"}]
# A decision divergence mining found before step 2.
AT_STEP_2 = [{"step_index": 1, "condition": "the screen shows \"Export failed\"",
              "action": "before step 2: retry", "source": "divergence",
              "evidence": [{"kind": "instance", "id": "obs-1"}]}]


class TestLedgerFromRuns:

    def test_the_fixture_starts_with_hypotheses(self, kb):
        _with(kb, branches=UNANCHORED)
        assert _slot(kb, "steps")["status"] == "hypothesis"
        assert _slot(kb, "decisions")["status"] == "hypothesis"

    def test_a_deviation_at_a_hypothesis_decision_demotes_it_and_asks(
        self, improver, monitor, kb,
    ):
        _with(kb, branches=UNANCHORED)
        record = _deviated(monitor, "step_3", "Emailed the CSV")
        result = improver.process_execution(record)

        decisions = _slot(kb, "decisions")
        assert decisions["status"] == "unknown"
        assert decisions["source"] == "agent"
        assert decisions["evidence"] == [{"kind": "run", "id": record.execution_id}]
        assert "ledger_unknown:decisions" in result and "question_queued" in result
        [question] = _run_questions(kb)
        assert question["status"] == "unanswered" and question["category"] == "decision"
        assert question["origin"] == {
            "kind": "agent_run", "id": record.execution_id, "step_id": "step_3",
        }
        assert 'step [step_3] ("Upload the CSV to the shared drive")' in question["question"]
        assert '"Emailed the CSV"' in question["question"]
        # The packet tells the next agent to ask.
        assert "decisions" in unknown_slots(kb.get_procedure(SLUG))

        # Processing the same run again queues nothing new.
        improver.process_execution(record)
        assert len(_run_questions(kb)) == 1

    def test_a_deviation_away_from_the_decision_step_keeps_it(self, improver, monitor, kb):
        _with(kb, branches=AT_STEP_2)
        improver.process_execution(_deviated(monitor, "step_3"))
        assert _slot(kb, "decisions")["status"] == "hypothesis"
        assert _run_questions(kb) == []

        improver.process_execution(_deviated(monitor, "step_2", "Retried twice"))
        assert _slot(kb, "decisions")["status"] == "unknown"
        [question] = _run_questions(kb)
        assert question["origin"]["step_id"] == "step_2"

    def test_a_decision_the_user_gave_is_not_demoted(self, improver, monitor, kb):
        _with(kb, branches=UNANCHORED, agent_clarifications=[{
            "category": "decision", "question": "When do you stop?",
            "answer": "When the export is empty", "status": "answered",
        }])
        assert _slot(kb, "decisions")["status"] == "known"
        improver.process_execution(_deviated(monitor))
        assert _slot(kb, "decisions")["status"] == "known"
        assert _run_questions(kb) == []

    def test_no_decision_no_question(self, improver, monitor, kb):
        assert _slot(kb, "decisions")["status"] == "unknown"
        improver.process_execution(_deviated(monitor))
        assert _run_questions(kb) == []
        assert _slot(kb, "decisions")["source"] is None

    def test_the_users_answer_makes_the_demoted_decision_known(self, improver, monitor, kb):
        _with(kb, branches=UNANCHORED)
        improver.process_execution(_deviated(monitor))
        proc = kb.get_procedure(SLUG)
        [question] = [c for c in proc["agent_clarifications"] if c.get("origin")]
        question.update(answer="Email it when the drive is full", status="answered")
        kb.save_procedure(proc)
        decisions = _slot(kb, "decisions")
        assert decisions["status"] == "known" and decisions["source"] == "user"

    def test_three_successful_runs_make_a_hypothesis_known(self, improver, monitor, kb):
        _with(kb, branches=UNANCHORED)
        ids = []
        for n in range(1, 4):
            record = _completed(monitor)
            ids.append(record.execution_id)
            result = improver.process_execution(record)
            steps, decisions = _slot(kb, "steps"), _slot(kb, "decisions")
            expected = "known" if n == 3 else "hypothesis"
            assert (steps["status"], decisions["status"]) == (expected, expected), n
            assert _runs(steps) == ids and _runs(decisions) == ids
            assert steps["source"] == decisions["source"] == "agent"
            assert f"ledger_{expected}:steps" in result
        # The observed value is still what the slot points at.
        assert _slot(kb, "decisions")["value_ref"] == "branches"
        # Counted once per run.
        improver.process_execution(record)
        assert _runs(_slot(kb, "steps")) == ids

    def test_a_run_that_skipped_reporting_steps_confirms_only_what_it_reported(
        self, improver, monitor, kb,
    ):
        _with(kb, branches=AT_STEP_2)
        eid = monitor.start_execution(SLUG)
        monitor.record_step(eid, "step_2")
        improver.process_execution(monitor.complete_execution(eid))
        # Step 2 (where the decision is) was reported; the other steps were not.
        assert _runs(_slot(kb, "decisions")) == [eid]
        assert _runs(_slot(kb, "steps")) == []

    def test_a_failed_run_confirms_nothing(self, improver, monitor, kb):
        _with(kb, branches=UNANCHORED)
        eid = monitor.start_execution(SLUG)
        for sid in ("step_1", "step_2", "step_3"):
            monitor.record_step(eid, sid)
        improver.process_execution(monitor.fail_execution(eid, "upload refused"))
        assert _runs(_slot(kb, "steps")) == [] and _runs(_slot(kb, "decisions")) == []

    def test_an_accepted_guess_does_not_erase_run_confirmations(self, improver, monitor, kb):
        _with(kb, agent_clarifications=[{
            "category": "decision", "question": "When do you stop?",
            "answer": "When the export is empty", "status": "answered",
            "source": "accepted_guess",
        }])
        assert _slot(kb, "decisions")["status"] == "hypothesis"
        record = _completed(monitor)
        improver.process_execution(record)
        decisions = _slot(kb, "decisions")
        assert decisions["source"] == "agent"
        assert decisions["evidence"] == [
            {"kind": "answer", "id": "agent_clarifications.0"},
            {"kind": "run", "id": record.execution_id},
        ]
