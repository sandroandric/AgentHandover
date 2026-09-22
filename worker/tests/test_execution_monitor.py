"""Tests for the execution monitor module."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.execution_monitor import (
    ExecutionMonitor,
    ExecutionStatus,
    actions_equivalent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_procedure(slug: str = "deploy-app", steps: list[dict] | None = None) -> dict:
    """Create a minimal procedure dict for testing."""
    if steps is None:
        steps = [
            {"action": "Open terminal", "step": "Open terminal"},
            {"action": "Run deploy command", "step": "Run deploy"},
            {"action": "Verify deployment", "step": "Verify"},
        ]
    return {
        "id": slug,
        "slug": slug,
        "title": f"Procedure: {slug}",
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    """Create a KnowledgeBase rooted in a temp directory."""
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    return kb


@pytest.fixture()
def kb_with_proc(kb: KnowledgeBase) -> KnowledgeBase:
    """KB with a sample procedure saved."""
    kb.save_procedure(_make_procedure("deploy-app"))
    return kb


@pytest.fixture()
def monitor(kb_with_proc: KnowledgeBase) -> ExecutionMonitor:
    """ExecutionMonitor with a KB that has a procedure."""
    return ExecutionMonitor(kb_with_proc)


# ---------------------------------------------------------------------------
# Start execution
# ---------------------------------------------------------------------------


class TestStartExecution:

    def test_returns_uuid(self, monitor: ExecutionMonitor) -> None:
        eid = monitor.start_execution("deploy-app", agent_id="test-agent")
        assert isinstance(eid, str)
        assert len(eid) == 36  # UUID format

    def test_creates_active_record(self, monitor: ExecutionMonitor) -> None:
        eid = monitor.start_execution("deploy-app")
        record = monitor.get_execution(eid)
        assert record is not None
        assert record.execution_id == eid
        assert record.procedure_slug == "deploy-app"
        assert record.status == ExecutionStatus.IN_PROGRESS

    def test_default_agent_id(self, monitor: ExecutionMonitor) -> None:
        eid = monitor.start_execution("deploy-app")
        record = monitor.get_execution(eid)
        assert record is not None
        assert record.agent_id == "unknown"

    def test_custom_agent_id(self, monitor: ExecutionMonitor) -> None:
        eid = monitor.start_execution("deploy-app", agent_id="claw-v2")
        record = monitor.get_execution(eid)
        assert record is not None
        assert record.agent_id == "claw-v2"

    def test_populates_expected_steps_from_procedure(
        self, monitor: ExecutionMonitor
    ) -> None:
        eid = monitor.start_execution("deploy-app")
        record = monitor.get_execution(eid)
        assert record is not None
        assert len(record.steps) == 3
        assert record.steps[0].expected_action == "Open terminal"
        assert record.steps[1].expected_action == "Run deploy command"
        assert record.steps[2].expected_action == "Verify deployment"

    def test_unknown_procedure_yields_no_steps(
        self, monitor: ExecutionMonitor
    ) -> None:
        eid = monitor.start_execution("nonexistent-proc")
        record = monitor.get_execution(eid)
        assert record is not None
        assert record.steps == []

    def test_started_at_is_set(self, monitor: ExecutionMonitor) -> None:
        eid = monitor.start_execution("deploy-app")
        record = monitor.get_execution(eid)
        assert record is not None
        assert record.started_at is not None


# ---------------------------------------------------------------------------
# Record step
# ---------------------------------------------------------------------------


class TestRecordStep:

    def test_record_matching_step(self, monitor: ExecutionMonitor) -> None:
        eid = monitor.start_execution("deploy-app")
        monitor.record_step(eid, "step_1", "Open terminal")
        record = monitor.get_execution(eid)
        assert record is not None
        step = record.steps[0]
        assert step.actual_action == "Open terminal"
        assert step.status == ExecutionStatus.COMPLETED
        assert step.completed_at is not None

    def test_record_step_detects_deviation(
        self, monitor: ExecutionMonitor
    ) -> None:
        eid = monitor.start_execution("deploy-app")
        monitor.record_step(eid, "step_1", "Open browser instead")
        record = monitor.get_execution(eid)
        assert record is not None
        step = record.steps[0]
        assert step.status == ExecutionStatus.DEVIATED
        assert step.deviation_detail is not None
        assert len(record.deviations) == 1

    def test_record_step_case_insensitive_match(
        self, monitor: ExecutionMonitor
    ) -> None:
        eid = monitor.start_execution("deploy-app")
        monitor.record_step(eid, "step_1", "open terminal")
        record = monitor.get_execution(eid)
        assert record is not None
        assert record.steps[0].status == ExecutionStatus.COMPLETED

    def test_record_step_dynamic(self, monitor: ExecutionMonitor) -> None:
        """Unknown step_id creates a dynamic step."""
        eid = monitor.start_execution("deploy-app")
        monitor.record_step(eid, "99", "Unexpected action")
        record = monitor.get_execution(eid)
        assert record is not None
        assert len(record.steps) == 4  # 3 original + 1 dynamic
        assert record.steps[3].step_id == "99"
        assert record.steps[3].expected_action == "(dynamic)"

    def test_record_step_unknown_execution(
        self, monitor: ExecutionMonitor
    ) -> None:
        """Recording a step for unknown execution_id does not raise."""
        assert monitor.record_step("nonexistent-id", "step_1", "whatever") is None


# ---------------------------------------------------------------------------
# Record deviation
# ---------------------------------------------------------------------------


class TestRecordDeviation:

    def test_explicit_deviation(self, monitor: ExecutionMonitor) -> None:
        eid = monitor.start_execution("deploy-app")
        monitor.record_deviation(eid, "step_2", "Skipped authentication check")
        record = monitor.get_execution(eid)
        assert record is not None
        assert len(record.deviations) == 1
        assert record.deviations[0]["detail"] == "Skipped authentication check"
        assert record.steps[1].status == ExecutionStatus.DEVIATED

    def test_deviation_unknown_step_id(
        self, monitor: ExecutionMonitor
    ) -> None:
        """Deviation for an unknown step_id still records in deviations list."""
        eid = monitor.start_execution("deploy-app")
        monitor.record_deviation(eid, "999", "Totally off script")
        record = monitor.get_execution(eid)
        assert record is not None
        assert len(record.deviations) == 1

    def test_deviation_unknown_execution(
        self, monitor: ExecutionMonitor
    ) -> None:
        """Deviation for unknown execution_id does not raise."""
        monitor.record_deviation("fake-id", "step_1", "whatever")


# ---------------------------------------------------------------------------
# Complete execution
# ---------------------------------------------------------------------------


class TestCompleteExecution:

    def test_complete_moves_to_history(
        self, monitor: ExecutionMonitor
    ) -> None:
        eid = monitor.start_execution("deploy-app")
        record = monitor.complete_execution(eid)
        assert record.status == ExecutionStatus.COMPLETED
        assert record.completed_at is not None
        # No longer active
        assert monitor.get_execution(eid).status is ExecutionStatus.COMPLETED
        # In history
        assert len(monitor.get_history()) == 1

    def test_complete_with_outcomes(self, monitor: ExecutionMonitor) -> None:
        eid = monitor.start_execution("deploy-app")
        outcomes = [{"type": "deployment", "url": "https://app.example.com"}]
        record = monitor.complete_execution(eid, outcomes=outcomes)
        assert record.outcomes == outcomes

    def test_complete_with_deviations_marks_deviated(
        self, monitor: ExecutionMonitor
    ) -> None:
        eid = monitor.start_execution("deploy-app")
        monitor.record_deviation(eid, "step_1", "Used different tool")
        record = monitor.complete_execution(eid)
        assert record.status == ExecutionStatus.DEVIATED

    def test_complete_unknown_raises(self, monitor: ExecutionMonitor) -> None:
        with pytest.raises(KeyError):
            monitor.complete_execution("fake-id")


# ---------------------------------------------------------------------------
# Fail execution
# ---------------------------------------------------------------------------


class TestFailExecution:

    def test_fail_records_error(self, monitor: ExecutionMonitor) -> None:
        eid = monitor.start_execution("deploy-app")
        record = monitor.fail_execution(eid, "Connection refused")
        assert record.status == ExecutionStatus.FAILED
        assert record.error == "Connection refused"
        assert record.completed_at is not None

    def test_fail_moves_to_history(self, monitor: ExecutionMonitor) -> None:
        eid = monitor.start_execution("deploy-app")
        monitor.fail_execution(eid, "Timeout")
        with pytest.raises(KeyError):
            monitor.complete_execution(eid)
        assert len(monitor.get_history()) == 1

    def test_fail_unknown_raises(self, monitor: ExecutionMonitor) -> None:
        with pytest.raises(KeyError):
            monitor.fail_execution("fake-id", "error")


# ---------------------------------------------------------------------------
# Abort execution
# ---------------------------------------------------------------------------


class TestAbortExecution:

    def test_abort_execution(self, monitor: ExecutionMonitor) -> None:
        eid = monitor.start_execution("deploy-app")
        record = monitor.abort_execution(eid)
        assert record.status == ExecutionStatus.ABORTED
        assert record.completed_at is not None

    def test_abort_moves_to_history(self, monitor: ExecutionMonitor) -> None:
        eid = monitor.start_execution("deploy-app")
        monitor.abort_execution(eid)
        assert monitor.record_step(eid, "step_1", "Open terminal") is None
        assert len(monitor.get_history()) == 1

    def test_abort_unknown_raises(self, monitor: ExecutionMonitor) -> None:
        with pytest.raises(KeyError):
            monitor.abort_execution("fake-id")


# ---------------------------------------------------------------------------
# Get execution
# ---------------------------------------------------------------------------


class TestGetExecution:

    def test_get_active_execution(self, monitor: ExecutionMonitor) -> None:
        eid = monitor.start_execution("deploy-app")
        record = monitor.get_execution(eid)
        assert record is not None
        assert record.execution_id == eid

    def test_get_historical_execution(
        self, monitor: ExecutionMonitor
    ) -> None:
        eid = monitor.start_execution("deploy-app")
        monitor.complete_execution(eid)
        record = monitor.get_execution(eid)
        assert record is not None
        assert record.execution_id == eid
        assert record.status == ExecutionStatus.COMPLETED

    def test_get_unknown_returns_none(
        self, monitor: ExecutionMonitor
    ) -> None:
        assert monitor.get_execution("nonexistent") is None


# ---------------------------------------------------------------------------
# Get history
# ---------------------------------------------------------------------------


class TestGetHistory:

    def test_empty_history(self, monitor: ExecutionMonitor) -> None:
        assert monitor.get_history() == []

    def test_history_includes_completed(
        self, monitor: ExecutionMonitor
    ) -> None:
        eid = monitor.start_execution("deploy-app")
        monitor.complete_execution(eid)
        history = monitor.get_history()
        assert len(history) == 1

    def test_history_filtered_by_slug(
        self, kb_with_proc: KnowledgeBase
    ) -> None:
        kb_with_proc.save_procedure(_make_procedure("other-proc"))
        monitor = ExecutionMonitor(kb_with_proc)

        eid1 = monitor.start_execution("deploy-app")
        monitor.complete_execution(eid1)
        eid2 = monitor.start_execution("other-proc")
        monitor.complete_execution(eid2)

        deploy_history = monitor.get_history(procedure_slug="deploy-app")
        assert len(deploy_history) == 1
        assert deploy_history[0].procedure_slug == "deploy-app"

    def test_history_with_limit(self, monitor: ExecutionMonitor) -> None:
        for _ in range(5):
            eid = monitor.start_execution("deploy-app")
            monitor.complete_execution(eid)

        assert len(monitor.get_history(limit=3)) == 3
        assert len(monitor.get_history(limit=10)) == 5

    def test_history_newest_first(self, monitor: ExecutionMonitor) -> None:
        eids = []
        for _ in range(3):
            eid = monitor.start_execution("deploy-app")
            monitor.complete_execution(eid)
            eids.append(eid)
        history = monitor.get_history()
        # The last completed should be first
        assert history[0].execution_id == eids[-1]


# ---------------------------------------------------------------------------
# Success rate
# ---------------------------------------------------------------------------


class TestSuccessRate:

    def test_no_history(self, monitor: ExecutionMonitor) -> None:
        stats = monitor.get_success_rate("deploy-app")
        assert stats["total"] == 0
        assert stats["success_rate"] == 0.0

    def test_all_completed(self, monitor: ExecutionMonitor) -> None:
        for _ in range(4):
            eid = monitor.start_execution("deploy-app")
            monitor.complete_execution(eid)
        stats = monitor.get_success_rate("deploy-app")
        assert stats["total"] == 4
        assert stats["completed"] == 4
        assert stats["success_rate"] == 1.0

    def test_mixed_results(self, monitor: ExecutionMonitor) -> None:
        # 2 completed, 1 failed, 1 aborted
        eid = monitor.start_execution("deploy-app")
        monitor.complete_execution(eid)
        eid = monitor.start_execution("deploy-app")
        monitor.complete_execution(eid)
        eid = monitor.start_execution("deploy-app")
        monitor.fail_execution(eid, "error")
        eid = monitor.start_execution("deploy-app")
        monitor.abort_execution(eid)

        stats = monitor.get_success_rate("deploy-app")
        assert stats["total"] == 4
        assert stats["completed"] == 2
        assert stats["failed"] == 1
        assert stats["aborted"] == 1
        assert stats["success_rate"] == 0.5

    def test_all_failed(self, monitor: ExecutionMonitor) -> None:
        for _ in range(3):
            eid = monitor.start_execution("deploy-app")
            monitor.fail_execution(eid, "boom")
        stats = monitor.get_success_rate("deploy-app")
        assert stats["total"] == 3
        assert stats["completed"] == 0
        assert stats["success_rate"] == 0.0

    def test_deviated_counted_separately(
        self, monitor: ExecutionMonitor
    ) -> None:
        eid = monitor.start_execution("deploy-app")
        monitor.record_deviation(eid, "step_1", "Went off script")
        monitor.complete_execution(eid)

        stats = monitor.get_success_rate("deploy-app")
        assert stats["deviated"] == 1
        assert stats["completed"] == 0  # deviated, not completed
        assert stats["success_rate"] == 0.0

    def test_filters_by_slug(
        self, kb_with_proc: KnowledgeBase
    ) -> None:
        kb_with_proc.save_procedure(_make_procedure("other-proc"))
        monitor = ExecutionMonitor(kb_with_proc)

        eid = monitor.start_execution("deploy-app")
        monitor.complete_execution(eid)
        eid = monitor.start_execution("other-proc")
        monitor.fail_execution(eid, "oops")

        deploy_stats = monitor.get_success_rate("deploy-app")
        assert deploy_stats["total"] == 1
        assert deploy_stats["completed"] == 1

        other_stats = monitor.get_success_rate("other-proc")
        assert other_stats["total"] == 1
        assert other_stats["failed"] == 1


# ---------------------------------------------------------------------------
# Multiple concurrent executions
# ---------------------------------------------------------------------------


class TestConcurrentExecutions:

    def test_multiple_active(self, monitor: ExecutionMonitor) -> None:
        eid1 = monitor.start_execution("deploy-app", agent_id="a1")
        eid2 = monitor.start_execution("deploy-app", agent_id="a2")
        assert eid1 != eid2

        r1 = monitor.get_execution(eid1)
        r2 = monitor.get_execution(eid2)
        assert r1 is not None
        assert r2 is not None
        assert r1.agent_id == "a1"
        assert r2.agent_id == "a2"

    def test_complete_one_keeps_other_active(
        self, monitor: ExecutionMonitor
    ) -> None:
        eid1 = monitor.start_execution("deploy-app")
        eid2 = monitor.start_execution("deploy-app")
        monitor.complete_execution(eid1)

        assert monitor.get_execution(eid1) is not None  # in history
        r2 = monitor.get_execution(eid2)
        assert r2 is not None
        assert r2.status == ExecutionStatus.IN_PROGRESS


# ---------------------------------------------------------------------------
# Persistence (save / load)
# ---------------------------------------------------------------------------


class TestPersistence:

    def test_save_and_load(self, kb_with_proc: KnowledgeBase) -> None:
        monitor1 = ExecutionMonitor(kb_with_proc)
        eid = monitor1.start_execution("deploy-app", agent_id="agent-x")
        monitor1.record_step(eid, "step_1", "Open terminal")
        monitor1.complete_execution(eid, outcomes=[{"ok": True}])

        # Create a new monitor that loads from the same KB
        monitor2 = ExecutionMonitor(kb_with_proc)
        history = monitor2.get_history()
        assert len(history) == 1
        rec = history[0]
        assert rec.execution_id == eid
        assert rec.procedure_slug == "deploy-app"
        assert rec.agent_id == "agent-x"
        assert rec.status == ExecutionStatus.COMPLETED
        assert rec.outcomes == [{"ok": True}]
        assert len(rec.steps) == 3

    def test_persistence_file_location(
        self, kb_with_proc: KnowledgeBase
    ) -> None:
        monitor = ExecutionMonitor(kb_with_proc)
        eid = monitor.start_execution("deploy-app")
        monitor.complete_execution(eid)

        path = kb_with_proc.root / "observations" / "runs.db"
        assert monitor.db_path == path
        conn = sqlite3.connect(str(path))
        try:
            rows = conn.execute("SELECT execution_id, status FROM runs").fetchall()
        finally:
            conn.close()
        assert rows == [(eid, "completed")]
        assert not (kb_with_proc.root / "observations" / "executions.json").exists()

    def test_load_empty_kb(self, kb: KnowledgeBase) -> None:
        """Monitor starts fine with no prior history file."""
        monitor = ExecutionMonitor(kb)
        assert monitor.get_history() == []

    def test_corrupt_legacy_file(
        self, kb_with_proc: KnowledgeBase
    ) -> None:
        """A corrupt v0.4 executions.json is left alone, not imported."""
        obs_dir = kb_with_proc.root / "observations"
        obs_dir.mkdir(parents=True, exist_ok=True)
        (obs_dir / "executions.json").write_text("{invalid json!!!")

        monitor = ExecutionMonitor(kb_with_proc)
        assert monitor.get_history() == []
        assert (obs_dir / "executions.json").is_file()

    def test_imports_legacy_json_once(
        self, kb_with_proc: KnowledgeBase
    ) -> None:
        """v0.4 runs move into the table; the file is renamed, not deleted."""
        obs_dir = kb_with_proc.root / "observations"
        obs_dir.mkdir(parents=True, exist_ok=True)
        legacy = {
            "records": [{
                "execution_id": "old-run",
                "procedure_slug": "deploy-app",
                "agent_id": "a",
                "status": "failed",
                "started_at": "2026-01-01T00:00:00+00:00",
                "completed_at": "2026-01-01T00:05:00+00:00",
                "steps": [{"step_id": "0", "expected_action": "Open terminal",
                           "status": "completed"}],
                "error": "boom",
            }],
        }
        (obs_dir / "executions.json").write_text(json.dumps(legacy))

        ExecutionMonitor(kb_with_proc)
        monitor = ExecutionMonitor(kb_with_proc)  # a second open does not re-import
        assert [r.execution_id for r in monitor.get_history()] == ["old-run"]
        assert monitor.get_success_rate("deploy-app")["last_failure"] == (
            "2026-01-01T00:05:00+00:00"
        )
        assert not (obs_dir / "executions.json").exists()
        assert (obs_dir / "executions.json.imported").is_file()

    def test_second_monitor_sees_active_run(
        self, kb_with_proc: KnowledgeBase
    ) -> None:
        """The MCP server and the worker are separate processes: a run one
        of them starts is visible to (and finishable by) the other."""
        eid = ExecutionMonitor(kb_with_proc).start_execution("deploy-app")
        other = ExecutionMonitor(kb_with_proc)
        other.record_step(eid, "step_1", "Open terminal")
        other.complete_execution(eid)
        assert ExecutionMonitor(kb_with_proc).get_success_rate("deploy-app")["completed"] == 1

    def test_persistence_of_failed_execution(
        self, kb_with_proc: KnowledgeBase
    ) -> None:
        monitor1 = ExecutionMonitor(kb_with_proc)
        eid = monitor1.start_execution("deploy-app")
        monitor1.fail_execution(eid, "Network error")

        monitor2 = ExecutionMonitor(kb_with_proc)
        rec = monitor2.get_execution(eid)
        assert rec is not None
        assert rec.status == ExecutionStatus.FAILED
        assert rec.error == "Network error"

    def test_persistence_of_deviations(
        self, kb_with_proc: KnowledgeBase
    ) -> None:
        monitor1 = ExecutionMonitor(kb_with_proc)
        eid = monitor1.start_execution("deploy-app")
        monitor1.record_step(eid, "step_1", "Used different terminal")
        monitor1.complete_execution(eid)

        monitor2 = ExecutionMonitor(kb_with_proc)
        rec = monitor2.get_execution(eid)
        assert rec is not None
        assert rec.status == ExecutionStatus.DEVIATED
        assert len(rec.deviations) == 1


# ---------------------------------------------------------------------------
# Step ids and deviations (WP-I3, coverage audit #28)
# ---------------------------------------------------------------------------


class TestStepMatching:

    def test_expected_step_ids_are_the_packet_ids(
        self, monitor: ExecutionMonitor
    ) -> None:
        record = monitor.get_execution(monitor.start_execution("deploy-app"))
        assert [s.step_id for s in record.steps] == ["step_1", "step_2", "step_3"]

    def test_procedure_step_ids_are_kept(self, kb: KnowledgeBase) -> None:
        kb.save_procedure(_make_procedure("custom", steps=[
            {"step_id": "login", "action": "Log in"},
            {"action": "Open report"},
        ]))
        monitor = ExecutionMonitor(kb)
        record = monitor.get_execution(monitor.start_execution("custom"))
        assert [s.step_id for s in record.steps] == ["login", "step_2"]
        # ``1`` is the first step by position, whatever its id.
        assert monitor.record_step(record.execution_id, "1").step_id == "login"

    @pytest.mark.parametrize("reported", ["step_2", "2", "Step 2", " step-2 "])
    def test_id_and_ordinal_land_on_the_same_step(
        self, monitor: ExecutionMonitor, reported: str
    ) -> None:
        eid = monitor.start_execution("deploy-app")
        step = monitor.record_step(eid, reported, "Pushed with git instead", deviated=True)
        assert step is not None and step.step_id == "step_2"
        record = monitor.get_execution(eid)
        assert len(record.steps) == 3
        assert [d["step_id"] for d in record.deviations] == ["step_2"]
        assert record.steps[1].status is ExecutionStatus.DEVIATED
        assert record.steps[0].status is ExecutionStatus.IN_PROGRESS

    def test_ordinal_out_of_range_is_dynamic(self, monitor: ExecutionMonitor) -> None:
        eid = monitor.start_execution("deploy-app")
        monitor.record_step(eid, "step_0", "Something")
        monitor.record_step(eid, "4", "Something else")
        record = monitor.get_execution(eid)
        assert [s.step_id for s in record.steps[3:]] == ["step_0", "4"]
        assert record.deviations == []

    def test_empty_actual_action_is_no_claim(self, monitor: ExecutionMonitor) -> None:
        eid = monitor.start_execution("deploy-app")
        step = monitor.record_step(eid, "step_1", "")
        assert step.status is ExecutionStatus.COMPLETED
        assert step.actual_action is None
        assert monitor.get_execution(eid).deviations == []

    def test_reported_deviation_without_action(self, monitor: ExecutionMonitor) -> None:
        eid = monitor.start_execution("deploy-app")
        monitor.record_step(eid, "step_3", deviated=True, notes="Checked the logs instead")
        record = monitor.get_execution(eid)
        assert record.deviations[0]["step_id"] == "step_3"
        assert record.deviations[0]["detail"] == "Checked the logs instead"

    def test_one_deviation_per_step(self, monitor: ExecutionMonitor) -> None:
        eid = monitor.start_execution("deploy-app")
        monitor.record_step(eid, "step_2", "Pushed with git", deviated=True, notes="first")
        monitor.record_deviation(eid, "2", "second")
        record = monitor.get_execution(eid)
        assert [(d["step_id"], d["detail"]) for d in record.deviations] == [("step_2", "second")]

    def test_later_matching_report_clears_the_deviation(
        self, monitor: ExecutionMonitor
    ) -> None:
        eid = monitor.start_execution("deploy-app")
        monitor.record_step(eid, "step_1", "Opened a browser")
        monitor.record_step(eid, "step_1", "Opened the terminal")
        record = monitor.get_execution(eid)
        assert record.deviations == []
        assert record.steps[0].status is ExecutionStatus.COMPLETED

    def test_paraphrase_is_not_a_deviation(self, monitor: ExecutionMonitor) -> None:
        eid = monitor.start_execution("deploy-app")
        monitor.record_step(eid, "step_2", "ran the deploy command")
        assert monitor.get_execution(eid).deviations == []


class TestActionsEquivalent:

    @pytest.mark.parametrize(("expected", "actual"), [
        ("Open terminal", "open terminal"),
        ("Open Chrome and go to example.com", "opened chrome, went to example.com"),
        ("Run deploy command", "Ran the deploy commands"),
    ])
    def test_equivalent(self, expected: str, actual: str) -> None:
        assert actions_equivalent(expected, actual)

    @pytest.mark.parametrize(("expected", "actual"), [
        ("Click Submit", "Press Enter in the search field"),
        ("Open terminal", "Open browser instead"),
        ("Verify deployment", "Skipped verification"),
    ])
    def test_different(self, expected: str, actual: str) -> None:
        assert not actions_equivalent(expected, actual)


class TestSuccessRateFields:

    def test_counts_deviated_steps_and_times(self, monitor: ExecutionMonitor) -> None:
        eid = monitor.start_execution("deploy-app")
        monitor.record_step(eid, "step_1", "Open browser instead")
        monitor.record_step(eid, "step_3", deviated=True)
        monitor.complete_execution(eid)
        eid2 = monitor.start_execution("deploy-app")
        failed = monitor.fail_execution(eid2, "boom")
        monitor.start_execution("deploy-app")  # still running: not counted

        stats = monitor.get_success_rate("deploy-app")
        assert stats["total"] == 2
        assert stats["deviated"] == 1 and stats["failed"] == 1
        assert stats["deviations"] == 2
        assert stats["last_failure"] == failed.completed_at
        assert stats["last_run"] == failed.completed_at
