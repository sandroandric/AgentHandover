"""Tests for the evidence tracker."""

from __future__ import annotations

from pathlib import Path

import pytest

from oc_apprentice_worker.evidence_tracker import (
    EvidenceTracker,
    ObservationEvidence,
    StepEvidence,
    _actions_match,
)
from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.procedure_schema import sop_to_procedure


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    return kb


@pytest.fixture()
def tracker(kb: KnowledgeBase) -> EvidenceTracker:
    return EvidenceTracker(knowledge_base=kb)


@pytest.fixture()
def sample_procedure() -> dict:
    return sop_to_procedure({
        "slug": "check-domains",
        "title": "Check Domains",
        "steps": [
            {"action": "Open browser", "confidence": 0.9},
            {"action": "Navigate to site", "confidence": 0.85},
            {"action": "Search for domains", "confidence": 0.8},
        ],
        "confidence_avg": 0.85,
        "apps_involved": ["Chrome"],
        "source": "passive",
    })


# ---------------------------------------------------------------------------
# build_evidence
# ---------------------------------------------------------------------------

class TestBuildEvidence:

    def test_build_evidence_no_procedure(self, tracker: EvidenceTracker) -> None:
        ev = tracker.build_evidence("nonexistent")
        assert ev["observations"] == []
        assert ev["total_observations"] == 0

    def test_build_evidence_with_procedure(
        self, kb: KnowledgeBase, tracker: EvidenceTracker, sample_procedure: dict
    ) -> None:
        kb.save_procedure(sample_procedure)
        ev = tracker.build_evidence("check-domains")
        assert isinstance(ev, dict)
        assert "observations" in ev

    def test_build_evidence_preserves_existing(
        self, kb: KnowledgeBase, tracker: EvidenceTracker, sample_procedure: dict
    ) -> None:
        sample_procedure["evidence"]["observations"] = [
            {"date": "2026-03-01", "type": "focus", "session_id": "s1",
             "duration_minutes": 5, "event_count": 10}
        ]
        sample_procedure["evidence"]["total_observations"] = 1
        kb.save_procedure(sample_procedure)

        ev = tracker.build_evidence("check-domains")
        assert ev["total_observations"] == 1
        assert len(ev["observations"]) == 1


# ---------------------------------------------------------------------------
# add_observation
# ---------------------------------------------------------------------------

class TestAddObservation:

    def test_add_observation(
        self, kb: KnowledgeBase, tracker: EvidenceTracker, sample_procedure: dict
    ) -> None:
        kb.save_procedure(sample_procedure)
        tracker.add_observation(
            "check-domains",
            ObservationEvidence(
                date="2026-03-10",
                type="focus",
                duration_minutes=5,
                session_id="session-1",
                event_count=15,
            ),
        )
        proc = kb.get_procedure("check-domains")
        assert proc is not None
        assert proc["evidence"]["total_observations"] == 1
        assert proc["evidence"]["observations"][0]["session_id"] == "session-1"

    def test_add_multiple_observations(
        self, kb: KnowledgeBase, tracker: EvidenceTracker, sample_procedure: dict
    ) -> None:
        kb.save_procedure(sample_procedure)
        for i in range(3):
            tracker.add_observation(
                "check-domains",
                ObservationEvidence(
                    date=f"2026-03-{10+i:02d}",
                    type="passive",
                    duration_minutes=3,
                    session_id=f"session-{i}",
                    event_count=10,
                ),
            )
        proc = kb.get_procedure("check-domains")
        assert proc is not None
        assert proc["evidence"]["total_observations"] == 3

    def test_add_observation_updates_staleness(
        self, kb: KnowledgeBase, tracker: EvidenceTracker, sample_procedure: dict
    ) -> None:
        kb.save_procedure(sample_procedure)
        tracker.add_observation(
            "check-domains",
            ObservationEvidence(
                date="2026-03-10",
                type="focus",
                duration_minutes=5,
                session_id="s1",
                event_count=15,
            ),
        )
        proc = kb.get_procedure("check-domains")
        assert proc is not None
        assert proc["staleness"]["last_observed"] is not None

    def test_add_observation_nonexistent_slug(
        self, tracker: EvidenceTracker
    ) -> None:
        # Should not raise, just log a warning
        tracker.add_observation(
            "nonexistent",
            ObservationEvidence(
                date="2026-03-10",
                type="focus",
                duration_minutes=5,
                session_id="s1",
                event_count=15,
            ),
        )


# ---------------------------------------------------------------------------
# compute_step_evidence
# ---------------------------------------------------------------------------

class TestComputeStepEvidence:

    def test_empty_demos(self, tracker: EvidenceTracker) -> None:
        result = tracker.compute_step_evidence("slug", [])
        assert result == []

    def test_single_demo(self, tracker: EvidenceTracker) -> None:
        demos = [[
            {"step_id": "step_1", "action": "Open browser"},
            {"step_id": "step_2", "action": "Navigate"},
        ]]
        result = tracker.compute_step_evidence("slug", demos)
        assert len(result) == 2
        assert result[0].step_id == "step_1"
        assert result[0].observed_count == 1
        assert result[0].consistent_count == 1

    def test_consistent_demos(self, tracker: EvidenceTracker) -> None:
        demos = [
            [{"step_id": "step_1", "action": "Open browser"}],
            [{"step_id": "step_1", "action": "Open browser"}],
            [{"step_id": "step_1", "action": "Open browser"}],
        ]
        result = tracker.compute_step_evidence("slug", demos)
        assert result[0].observed_count == 3
        assert result[0].consistent_count == 3
        assert result[0].consistency_ratio == 1.0

    def test_inconsistent_demos(self, tracker: EvidenceTracker) -> None:
        demos = [
            [{"step_id": "step_1", "action": "Open browser"}],
            [{"step_id": "step_1", "action": "Open browser"}],
            [{"step_id": "step_1", "action": "Open terminal"}],
        ]
        result = tracker.compute_step_evidence("slug", demos)
        assert result[0].observed_count == 3
        assert result[0].consistent_count == 2
        assert len(result[0].contradictions) == 1
        assert result[0].contradictions[0]["actual"] == "Open terminal"

    def test_case_insensitive_matching(self, tracker: EvidenceTracker) -> None:
        demos = [
            [{"step_id": "step_1", "action": "Open Browser"}],
            [{"step_id": "step_1", "action": "open browser"}],
        ]
        result = tracker.compute_step_evidence("slug", demos)
        assert result[0].consistent_count == 2

    def test_shorter_demo(self, tracker: EvidenceTracker) -> None:
        demos = [
            [
                {"step_id": "step_1", "action": "A"},
                {"step_id": "step_2", "action": "B"},
            ],
            [
                {"step_id": "step_1", "action": "A"},
                # step_2 missing from this demo
            ],
        ]
        result = tracker.compute_step_evidence("slug", demos)
        assert result[0].observed_count == 2  # step_1 in both
        assert result[1].observed_count == 1  # step_2 only in first

    def test_consistency_ratio_zero(self, tracker: EvidenceTracker) -> None:
        se = StepEvidence(step_id="s", observed_count=0, consistent_count=0)
        assert se.consistency_ratio == 0.0


# ---------------------------------------------------------------------------
# update_step_evidence
# ---------------------------------------------------------------------------

class TestUpdateStepEvidence:

    def test_update_persists(
        self, kb: KnowledgeBase, tracker: EvidenceTracker, sample_procedure: dict
    ) -> None:
        kb.save_procedure(sample_procedure)
        evidence = [
            StepEvidence(step_id="step_1", observed_count=3, consistent_count=3),
            StepEvidence(step_id="step_2", observed_count=3, consistent_count=2,
                         contradictions=[{"demo_index": 2, "expected": "A", "actual": "B", "step_id": "step_2"}]),
        ]
        tracker.update_step_evidence("check-domains", evidence)

        proc = kb.get_procedure("check-domains")
        assert proc is not None
        assert len(proc["evidence"]["step_evidence"]) == 2
        assert proc["evidence"]["step_evidence"][0]["consistent_count"] == 3
        assert len(proc["evidence"]["contradictions"]) == 1

    def test_update_nonexistent_does_nothing(
        self, tracker: EvidenceTracker
    ) -> None:
        tracker.update_step_evidence("nonexistent", [])


# ---------------------------------------------------------------------------
# _actions_match
# ---------------------------------------------------------------------------

class TestActionsMatch:

    def test_exact_match(self) -> None:
        assert _actions_match("Open browser", "Open browser") is True

    def test_case_insensitive(self) -> None:
        assert _actions_match("Open Browser", "open browser") is True

    def test_whitespace_stripped(self) -> None:
        assert _actions_match("  Open browser  ", "Open browser") is True

    def test_different_actions(self) -> None:
        assert _actions_match("Open browser", "Close browser") is False
