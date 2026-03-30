"""Tests for the SkillImprover module."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from unittest.mock import MagicMock

import pytest

from agenthandover_worker.skill_improver import SkillImprover


class MockStatus(Enum):
    completed = "completed"
    deviated = "deviated"
    failed = "failed"


@dataclass
class MockRecord:
    execution_id: str = "exec-001"
    procedure_slug: str = "test-skill"
    agent_id: str = "claude-code"
    status: MockStatus = MockStatus.completed
    started_at: str = "2026-03-25T10:00:00+00:00"
    completed_at: str = "2026-03-25T10:15:00+00:00"
    steps: list = field(default_factory=list)
    outcomes: list = field(default_factory=list)
    deviations: list = field(default_factory=list)
    error: str | None = None


@pytest.fixture
def mock_kb():
    kb = MagicMock()
    kb.get_procedure.return_value = {
        "id": "test-skill",
        "title": "Test Skill",
        "confidence_avg": 0.80,
        "staleness": {
            "last_confirmed": None,
            "confidence_trend": [0.80],
            "drift_signals": [],
        },
        "workflow_rhythm": {},
        "evidence": {"total_observations": 3},
    }
    return kb


@pytest.fixture
def improver(mock_kb):
    return SkillImprover(mock_kb)


class TestHandleSuccess:

    def test_boosts_confidence(self, improver, mock_kb):
        record = MockRecord()
        result = improver.process_execution(record)
        proc = mock_kb.get_procedure.return_value
        assert proc["confidence_avg"] > 0.80
        assert "boosted_confidence" in result

    def test_confirms_freshness(self, improver, mock_kb):
        record = MockRecord()
        improver.process_execution(record)
        proc = mock_kb.get_procedure.return_value
        assert proc["staleness"]["last_confirmed"] is not None

    def test_updates_timing_ema(self, improver, mock_kb):
        proc = mock_kb.get_procedure.return_value
        proc["workflow_rhythm"]["avg_duration_minutes"] = 10.0
        record = MockRecord()  # 15 min duration
        improver.process_execution(record)
        # EMA: 10 * 0.7 + 15 * 0.3 = 11.5
        assert proc["workflow_rhythm"]["avg_duration_minutes"] == 11.5

    def test_tracks_execution_stats(self, improver, mock_kb):
        record = MockRecord()
        improver.process_execution(record)
        proc = mock_kb.get_procedure.return_value
        stats = proc["execution_stats"]
        assert stats["total"] == 1
        assert stats["success"] == 1
        assert stats["success_rate"] == 1.0

    def test_saves_procedure(self, improver, mock_kb):
        record = MockRecord()
        improver.process_execution(record)
        mock_kb.save_procedure.assert_called_once()


class TestHandleDeviation:

    def test_adds_drift_signal(self, improver, mock_kb):
        record = MockRecord(
            status=MockStatus.deviated,
            deviations=[{"step_id": "step_3", "detail": "Used API instead of UI"}],
        )
        result = improver.process_execution(record)
        proc = mock_kb.get_procedure.return_value
        assert len(proc["staleness"]["drift_signals"]) == 1
        assert "added_drift_signal" in result

    def test_tracks_alternatives(self, improver, mock_kb):
        record = MockRecord(
            status=MockStatus.deviated,
            deviations=[{"step_id": "step_3", "detail": "Used API"}],
        )
        improver.process_execution(record)
        proc = mock_kb.get_procedure.return_value
        assert len(proc["observed_alternatives"]) == 1

    def test_suggests_branch_after_two(self, improver, mock_kb):
        proc = mock_kb.get_procedure.return_value
        proc["observed_alternatives"] = [
            {"step_id": "step_3", "observed_action": "Used API", "execution_id": "e1", "timestamp": "t1"},
        ]
        record = MockRecord(
            status=MockStatus.deviated,
            deviations=[{"step_id": "step_3", "detail": "Used API again"}],
        )
        result = improver.process_execution(record)
        assert any("branch_suggested" in r for r in result)

    def test_updates_stats(self, improver, mock_kb):
        record = MockRecord(status=MockStatus.deviated, deviations=[])
        improver.process_execution(record)
        proc = mock_kb.get_procedure.return_value
        assert proc["execution_stats"]["deviated"] == 1


class TestHandleFailure:

    def test_reduces_confidence(self, improver, mock_kb):
        record = MockRecord(status=MockStatus.failed, error="element not found")
        improver.process_execution(record)
        proc = mock_kb.get_procedure.return_value
        assert proc["confidence_avg"] < 0.80

    def test_adds_failure_drift_signal(self, improver, mock_kb):
        record = MockRecord(status=MockStatus.failed, error="timeout")
        improver.process_execution(record)
        proc = mock_kb.get_procedure.return_value
        signals = proc["staleness"]["drift_signals"]
        assert len(signals) == 1
        assert "execution_failure" in signals[0]["type"]

    def test_updates_stats(self, improver, mock_kb):
        record = MockRecord(status=MockStatus.failed, error="crash")
        improver.process_execution(record)
        proc = mock_kb.get_procedure.return_value
        assert proc["execution_stats"]["failed"] == 1
        assert proc["execution_stats"]["success_rate"] == 0.0


class TestEdgeCases:

    def test_missing_procedure(self, mock_kb):
        mock_kb.get_procedure.return_value = None
        improver = SkillImprover(mock_kb)
        result = improver.process_execution(MockRecord())
        assert result == []
        mock_kb.save_procedure.assert_not_called()

    def test_confidence_capped_at_1(self, improver, mock_kb):
        proc = mock_kb.get_procedure.return_value
        proc["confidence_avg"] = 0.99
        record = MockRecord()
        improver.process_execution(record)
        assert proc["confidence_avg"] <= 1.0

    def test_confidence_floored_at_0(self, improver, mock_kb):
        proc = mock_kb.get_procedure.return_value
        proc["confidence_avg"] = 0.02
        record = MockRecord(status=MockStatus.failed, error="x")
        improver.process_execution(record)
        assert proc["confidence_avg"] >= 0.0
