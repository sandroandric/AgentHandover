"""Tests for the procedure writer."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenthandover_worker.evidence_tracker import EvidenceTracker
from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.procedure_schema import validate_procedure
from agenthandover_worker.procedure_writer import ProcedureWriter


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    return kb


@pytest.fixture()
def tracker(kb: KnowledgeBase) -> EvidenceTracker:
    return EvidenceTracker(knowledge_base=kb)


@pytest.fixture()
def writer(kb: KnowledgeBase, tracker: EvidenceTracker) -> ProcedureWriter:
    return ProcedureWriter(kb=kb, evidence=tracker)


@pytest.fixture()
def sample_sop() -> dict:
    return {
        "slug": "check-domains",
        "title": "Check Expired Domains",
        "description": "Search for expired domains.",
        "tags": ["browsing"],
        "confidence_avg": 0.87,
        "episode_count": 2,
        "apps_involved": ["Chrome"],
        "steps": [
            {"action": "Open browser", "confidence": 0.9},
            {"action": "Navigate to auctions", "confidence": 0.85},
        ],
        "variables": [
            {"name": "query", "type": "string", "example": "ai"},
        ],
    }


# ---------------------------------------------------------------------------
# write_procedure
# ---------------------------------------------------------------------------

class TestWriteProcedure:

    def test_write_creates_file(
        self, writer: ProcedureWriter, kb: KnowledgeBase, sample_sop: dict
    ) -> None:
        path = writer.write_procedure(
            sample_sop, source="focus", source_id="session-1",
            event_count=15, duration_minutes=5,
        )
        assert path.is_file()
        proc = kb.get_procedure("check-domains")
        assert proc is not None
        assert proc["id"] == "check-domains"

    def test_write_validates_clean(
        self, writer: ProcedureWriter, kb: KnowledgeBase, sample_sop: dict
    ) -> None:
        writer.write_procedure(sample_sop, source="focus", source_id="s1")
        proc = kb.get_procedure("check-domains")
        assert proc is not None
        errors = validate_procedure(proc)
        assert errors == []

    def test_write_adds_evidence(
        self, writer: ProcedureWriter, kb: KnowledgeBase, sample_sop: dict
    ) -> None:
        writer.write_procedure(
            sample_sop, source="focus", source_id="session-1",
            event_count=15, duration_minutes=5,
        )
        proc = kb.get_procedure("check-domains")
        assert proc is not None
        assert proc["evidence"]["total_observations"] == 1
        obs = proc["evidence"]["observations"][0]
        assert obs["session_id"] == "session-1"
        assert obs["type"] == "focus"
        assert obs["event_count"] == 15

    def test_write_sets_source(
        self, writer: ProcedureWriter, kb: KnowledgeBase, sample_sop: dict
    ) -> None:
        writer.write_procedure(sample_sop, source="passive", source_id="s1")
        proc = kb.get_procedure("check-domains")
        assert proc is not None
        assert proc["source"] == "passive"

    def test_write_preserves_variables_as_inputs(
        self, writer: ProcedureWriter, kb: KnowledgeBase, sample_sop: dict
    ) -> None:
        writer.write_procedure(sample_sop, source="focus", source_id="s1")
        proc = kb.get_procedure("check-domains")
        assert proc is not None
        assert len(proc["inputs"]) == 1
        assert proc["inputs"][0]["name"] == "query"

    def test_write_minimal_sop(
        self, writer: ProcedureWriter, kb: KnowledgeBase
    ) -> None:
        sop = {
            "slug": "minimal",
            "title": "Minimal SOP",
            "steps": [{"action": "Do something"}],
        }
        path = writer.write_procedure(sop, source="focus", source_id="s1")
        assert path.is_file()


# ---------------------------------------------------------------------------
# update_procedure
# ---------------------------------------------------------------------------

class TestUpdateProcedure:

    def test_update_preserves_evidence(
        self, writer: ProcedureWriter, kb: KnowledgeBase, sample_sop: dict
    ) -> None:
        # Write initial
        writer.write_procedure(
            sample_sop, source="focus", source_id="s1",
            event_count=10, duration_minutes=3,
        )

        # Update
        sample_sop["confidence_avg"] = 0.90
        path = writer.update_procedure(
            "check-domains", sample_sop,
            source="passive", source_id="s2",
            event_count=20, duration_minutes=4,
        )

        proc = kb.get_procedure("check-domains")
        assert proc is not None
        # Evidence should have 2 observations (initial + update)
        assert proc["evidence"]["total_observations"] == 2

    def test_update_preserves_constraints(
        self, writer: ProcedureWriter, kb: KnowledgeBase, sample_sop: dict
    ) -> None:
        writer.write_procedure(sample_sop, source="focus", source_id="s1")

        # Manually set constraints
        proc = kb.get_procedure("check-domains")
        assert proc is not None
        proc["constraints"]["trust_level"] = "suggest"
        kb.save_procedure(proc)

        # Update
        writer.update_procedure("check-domains", sample_sop, source="passive", source_id="s2")

        proc = kb.get_procedure("check-domains")
        assert proc is not None
        assert proc["constraints"]["trust_level"] == "suggest"

    def test_update_appends_confidence_trend(
        self, writer: ProcedureWriter, kb: KnowledgeBase, sample_sop: dict
    ) -> None:
        writer.write_procedure(sample_sop, source="focus", source_id="s1")

        sample_sop["confidence_avg"] = 0.92
        writer.update_procedure("check-domains", sample_sop, source="passive", source_id="s2")

        proc = kb.get_procedure("check-domains")
        assert proc is not None
        trend = proc["staleness"]["confidence_trend"]
        assert len(trend) >= 2  # At least original + update

    def test_update_new_procedure(
        self, writer: ProcedureWriter, kb: KnowledgeBase, sample_sop: dict
    ) -> None:
        """Update on a slug that doesn't exist creates a new procedure."""
        path = writer.update_procedure(
            "new-slug",
            {**sample_sop, "slug": "new-slug"},
            source="passive", source_id="s1",
        )
        assert path.is_file()
        proc = kb.get_procedure("new-slug")
        assert proc is not None

    def test_update_without_source_id(
        self, writer: ProcedureWriter, kb: KnowledgeBase, sample_sop: dict
    ) -> None:
        writer.write_procedure(sample_sop, source="focus", source_id="s1")
        path = writer.update_procedure("check-domains", sample_sop)
        assert path.is_file()

    def test_update_preserves_branches(
        self, writer: ProcedureWriter, kb: KnowledgeBase, sample_sop: dict
    ) -> None:
        writer.write_procedure(sample_sop, source="focus", source_id="s1")

        # Add branches manually
        proc = kb.get_procedure("check-domains")
        assert proc is not None
        proc["branches"] = [{"step_id": "step_1", "condition": "if error"}]
        kb.save_procedure(proc)

        # Update should preserve
        writer.update_procedure("check-domains", sample_sop, source="passive", source_id="s2")
        proc = kb.get_procedure("check-domains")
        assert proc is not None
        assert len(proc["branches"]) == 1
