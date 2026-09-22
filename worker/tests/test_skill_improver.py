"""Tests for the SkillImprover: a finished run written back onto its procedure."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenthandover_worker.execution_monitor import ExecutionMonitor
from agenthandover_worker.knowledge_base import KnowledgeBase
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
