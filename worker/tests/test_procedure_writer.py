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

    def test_rewrite_upserts_and_keeps_evidence(
        self, writer: ProcedureWriter, kb: KnowledgeBase, sample_sop: dict
    ) -> None:
        """Three passive observations of one procedure count as three."""
        import copy

        for i in range(3):
            writer.write_procedure(
                copy.deepcopy(sample_sop), source="passive",
                source_id=f"cluster-{i}", event_count=10 + i,
            )
        proc = kb.get_procedure("check-domains")
        assert proc is not None
        assert proc["evidence"]["total_observations"] == 3
        assert [o["session_id"] for o in proc["evidence"]["observations"]] == [
            "cluster-0", "cluster-1", "cluster-2",
        ]

    def test_rewrite_keeps_lifecycle_and_extracted_evidence(
        self, writer: ProcedureWriter, kb: KnowledgeBase, sample_sop: dict
    ) -> None:
        import copy

        writer.write_procedure(copy.deepcopy(sample_sop), source="passive", source_id="c0")
        proc = kb.get_procedure("check-domains")
        proc["lifecycle_state"] = "reviewed"
        proc["evidence"]["extracted_evidence"] = {"domains": ["expireddomains.net"]}
        kb.save_procedure(proc)

        updated = copy.deepcopy(sample_sop)
        updated["steps"].append({"action": "Export the list", "confidence": 0.8})
        writer.write_procedure(updated, source="passive", source_id="c1")

        proc = kb.get_procedure("check-domains")
        assert proc["lifecycle_state"] == "reviewed"
        assert proc["evidence"]["extracted_evidence"] == {"domains": ["expireddomains.net"]}
        assert proc["evidence"]["total_observations"] == 2
        assert len(proc["steps"]) == 3

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

    def test_write_preserves_step_description(
        self, writer: ProcedureWriter, kb: KnowledgeBase
    ) -> None:
        """v0.3.0 fix: ``sop_template_to_procedure`` propagates step
        description into the saved procedure.

        Regression test for the parser-pipeline gap that caused all
        v0.2.x Skills to ship with empty descriptions even though the
        VLM was emitting them.  The diagnostic dump on the dailynews
        focus session confirmed Gemma 4 e4b returns 19/19 steps with
        rich descriptions; before this fix, ``sop_template_to_procedure``
        in ``procedure_schema.py`` constructed proc_step without
        copying ``step["description"]``, silently dropping it.
        """
        sop = {
            "slug": "with-description",
            "title": "Skill with descriptive steps",
            "steps": [
                {
                    "step": "Open the dashboard",
                    "description": "Initial entry point — opens the main "
                                   "dashboard so the agent can see the "
                                   "current state before making changes.",
                    "target": "https://example.com/dashboard",
                },
                {
                    "step": "Click Export",
                    "description": "Export current data so we have a "
                                   "snapshot to attach to the email.",
                },
            ],
        }
        writer.write_procedure(sop, source="focus", source_id="s1")
        proc = kb.get_procedure("with-description")
        assert proc is not None
        steps = proc.get("steps", [])
        assert len(steps) == 2
        assert "Initial entry point" in steps[0].get("description", "")
        assert "snapshot to attach" in steps[1].get("description", "")


# ---------------------------------------------------------------------------
# Regeneration: write_procedure over an existing procedure merges
# ---------------------------------------------------------------------------

class TestRegeneration:

    def test_regeneration_preserves_evidence(
        self, writer: ProcedureWriter, kb: KnowledgeBase, sample_sop: dict
    ) -> None:
        writer.write_procedure(
            sample_sop, source="focus", source_id="s1",
            event_count=10, duration_minutes=3,
        )
        sample_sop["confidence_avg"] = 0.90
        writer.write_procedure(
            sample_sop, source="passive", source_id="s2",
            event_count=20, duration_minutes=4,
        )
        proc = kb.get_procedure("check-domains")
        # Evidence has 2 observations (initial + regeneration)
        assert proc["evidence"]["total_observations"] == 2

    def test_regeneration_preserves_constraints(
        self, writer: ProcedureWriter, kb: KnowledgeBase, sample_sop: dict
    ) -> None:
        writer.write_procedure(sample_sop, source="focus", source_id="s1")
        proc = kb.get_procedure("check-domains")
        proc["constraints"]["trust_level"] = "suggest"
        kb.save_procedure(proc)

        writer.write_procedure(sample_sop, source="passive", source_id="s2")
        assert kb.get_procedure("check-domains")["constraints"]["trust_level"] == "suggest"

    def test_regeneration_appends_confidence_trend_once(
        self, writer: ProcedureWriter, kb: KnowledgeBase, sample_sop: dict
    ) -> None:
        writer.write_procedure(sample_sop, source="focus", source_id="s1")
        sample_sop["confidence_avg"] = 0.92
        writer.write_procedure(sample_sop, source="passive", source_id="s2")
        writer.write_procedure(sample_sop, source="passive", source_id="s3")
        trend = kb.get_procedure("check-domains")["staleness"]["confidence_trend"]
        assert trend == [0.87, 0.92]

    def test_regeneration_preserves_branches(
        self, writer: ProcedureWriter, kb: KnowledgeBase, sample_sop: dict
    ) -> None:
        writer.write_procedure(sample_sop, source="focus", source_id="s1")
        proc = kb.get_procedure("check-domains")
        proc["branches"] = [{"step_id": "step_1", "condition": "if error"}]
        kb.save_procedure(proc)

        writer.write_procedure(sample_sop, source="passive", source_id="s2")
        assert len(kb.get_procedure("check-domains")["branches"]) == 1

    def test_regeneration_keeps_the_agent_run_record(
        self, writer: ProcedureWriter, kb: KnowledgeBase, sample_sop: dict
    ) -> None:
        """What report_execution_complete wrote survives a regeneration."""
        writer.write_procedure(sample_sop, source="focus", source_id="s1")
        proc = kb.get_procedure("check-domains")
        proc["lifecycle_state"] = "agent_ready"
        proc["execution_stats"] = {"total": 3, "completed": 2, "deviated": 1}
        proc["execution_confidence"] = 0.7
        proc["observed_alternatives"] = [
            {"step_id": "step_2", "observed_action": "Used the API", "execution_id": "e1"},
        ]
        proc["staleness"]["last_confirmed"] = "2026-09-20T10:00:00+00:00"
        proc["staleness"]["drift_signals"] = [{"type": "execution_deviation"}]
        kb.save_procedure(proc)

        sample_sop["description"] = "Regenerated description."
        writer.write_procedure(sample_sop, source="passive", source_id="s2")

        after = kb.get_procedure("check-domains")
        assert after["description"] == "Regenerated description."
        assert after["lifecycle_state"] == "agent_ready"
        assert after["execution_stats"] == proc["execution_stats"]
        assert after["execution_confidence"] == 0.7
        assert after["observed_alternatives"] == proc["observed_alternatives"]
        assert after["staleness"]["last_confirmed"] == "2026-09-20T10:00:00+00:00"
        assert after["staleness"]["drift_signals"] == [{"type": "execution_deviation"}]

    def test_re_recording_is_a_new_draft_with_its_history(self) -> None:
        """main's focus path: keep_lifecycle=False resets the state only."""
        from agenthandover_worker.procedure_writer import merge_existing_procedure

        existing = {
            "lifecycle_state": "agent_ready",
            "lifecycle_history": [{"to": "agent_ready"}],
            "constraints": {"trust_level": "suggest"},
            "observed_alternatives": [{"step_id": "step_1"}],
            "staleness": {"last_confirmed": "2026-09-20T10:00:00+00:00"},
        }
        new = {"lifecycle_state": "draft", "confidence_avg": 0.8, "staleness": {}}
        merge_existing_procedure(new, existing, keep_lifecycle=False)
        assert new["lifecycle_state"] == "draft"
        assert new["lifecycle_history"] == [{"to": "agent_ready"}]
        assert new["constraints"] == {"trust_level": "suggest"}
        assert new["observed_alternatives"] == [{"step_id": "step_1"}]
        assert new["staleness"]["last_confirmed"] == "2026-09-20T10:00:00+00:00"

    def test_carry_run_history_keeps_new_answers(self) -> None:
        """main's Q&A path: runs carry over, the answered fields do not revert."""
        from agenthandover_worker.procedure_writer import carry_run_history

        existing = {
            "recurrence": "weekly",
            "execution_stats": {"total": 1},
            "staleness": {"last_confirmed": "t1", "drift_signals": [1]},
        }
        new = {"recurrence": "daily", "staleness": {}}
        carry_run_history(new, existing)
        assert new["recurrence"] == "daily"
        assert new["execution_stats"] == {"total": 1}
        assert new["staleness"]["last_confirmed"] == "t1"
        assert new["staleness"]["drift_signals"] == [1]
