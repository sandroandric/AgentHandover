"""End-to-end integration test for the full Phase 2+ knowledge pipeline.

Validates the complete flow:
  mock events → daily processing → procedures → evidence → profile →
  patterns → branches → decisions → outcomes → staleness → export
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from oc_apprentice_worker.account_detector import AccountDetector
from oc_apprentice_worker.branch_extractor import BranchExtractor
from oc_apprentice_worker.constraint_manager import ConstraintManager, TrustLevel
from oc_apprentice_worker.daily_processor import DailyBatchProcessor
from oc_apprentice_worker.decision_extractor import DecisionExtractor
from oc_apprentice_worker.evidence_tracker import EvidenceTracker, ObservationEvidence
from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.knowledge_export_adapter import KnowledgeBaseExportAdapter
from oc_apprentice_worker.outcome_tracker import OutcomeTracker
from oc_apprentice_worker.pattern_detector import PatternDetector
from oc_apprentice_worker.procedure_schema import sop_to_procedure, validate_procedure
from oc_apprentice_worker.procedure_writer import ProcedureWriter
from oc_apprentice_worker.profile_builder import ProfileBuilder
from oc_apprentice_worker.staleness_detector import StalenessDetector


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    return kb


def _make_event(
    event_id: str,
    timestamp: str,
    app: str,
    what_doing: str,
    location: str = "",
    kind: str = "FocusChange",
) -> dict:
    annotation = {
        "task_context": {"what_doing": what_doing, "is_workflow": True},
        "visual_context": {"active_app": app, "location": location},
    }
    return {
        "id": event_id,
        "timestamp": timestamp,
        "kind_json": json.dumps({kind: {}}),
        "window_json": json.dumps({"app": app, "title": f"{app} Window"}),
        "metadata_json": "{}",
        "scene_annotation_json": json.dumps(annotation),
    }


def _make_sop_template(slug: str, **overrides: object) -> dict:
    defaults = {
        "slug": slug,
        "title": f"Task: {slug}",
        "description": f"Description of {slug}",
        "tags": ["testing"],
        "confidence_avg": 0.85,
        "episode_count": 2,
        "apps_involved": ["Chrome"],
        "source": "passive",
        "steps": [
            {"action": "Open browser", "confidence": 0.9},
            {"action": "Navigate to site", "confidence": 0.85},
            {"action": "Perform action", "confidence": 0.80},
        ],
    }
    defaults.update(overrides)
    return defaults


class TestFullPipelineIntegration:
    """End-to-end tests that wire multiple modules together."""

    def test_sop_to_kb_round_trip(self, kb: KnowledgeBase) -> None:
        """SOP template → procedure → KB → validate."""
        sop = _make_sop_template("e2e-test")
        proc = sop_to_procedure(sop)
        errors = validate_procedure(proc)
        assert errors == []

        kb.save_procedure(proc)
        loaded = kb.get_procedure("e2e-test")
        assert loaded is not None
        assert loaded["schema_version"] == "3.0.0"
        assert len(loaded["steps"]) == 3

    def test_procedure_writer_with_evidence(self, kb: KnowledgeBase) -> None:
        """ProcedureWriter creates procedure + evidence in KB."""
        tracker = EvidenceTracker(knowledge_base=kb)
        writer = ProcedureWriter(kb=kb, evidence=tracker)

        sop = _make_sop_template("write-test")
        path = writer.write_procedure(
            sop, source="focus", source_id="session-1",
            event_count=20, duration_minutes=5,
        )
        assert path.is_file()

        proc = kb.get_procedure("write-test")
        assert proc is not None
        assert proc["evidence"]["total_observations"] == 1
        assert proc["evidence"]["observations"][0]["type"] == "focus"

    def test_daily_processor_creates_summary(self, kb: KnowledgeBase) -> None:
        """DailyBatchProcessor processes events into summary."""
        processor = DailyBatchProcessor(knowledge_base=kb)

        events = [
            _make_event("e1", "2026-03-10T09:00:00Z", "Chrome",
                        "Checking email", "https://mail.google.com"),
            _make_event("e2", "2026-03-10T09:02:00Z", "Chrome",
                        "Checking email", "https://mail.google.com"),
            _make_event("e3", "2026-03-10T09:10:00Z", "VS Code",
                        "Writing code", "/Users/test/project"),
            _make_event("e4", "2026-03-10T09:12:00Z", "VS Code",
                        "Writing code", "/Users/test/project"),
        ]

        summary = processor.process_day("2026-03-10", events)
        assert summary.date == "2026-03-10"
        assert summary.task_count >= 1
        assert len(summary.tasks) >= 1

        # Saved to KB
        loaded = kb.get_daily_summary("2026-03-10")
        assert loaded is not None

    def test_profile_builder_from_summaries(self, kb: KnowledgeBase) -> None:
        """ProfileBuilder infers profile from daily summaries."""
        # Create mock summaries
        for day in range(1, 6):
            kb.save_daily_summary(f"2026-03-{day:02d}", {
                "active_hours": 7.0,
                "task_count": 5,
                "tasks": [
                    {"intent": "Coding", "apps": ["VS Code"], "urls": ["https://github.com/repo"],
                     "start_time": "2026-03-01T09:00:00Z", "end_time": "2026-03-01T17:00:00Z",
                     "duration_minutes": 60, "matched_procedure": None},
                ],
                "top_apps": [
                    {"app": "Google Chrome", "minutes": 120},
                    {"app": "VS Code", "minutes": 180},
                    {"app": "Terminal", "minutes": 30},
                ],
                "procedures_observed": [],
                "new_workflows_detected": 0,
            })

        builder = ProfileBuilder(kb)
        profile = builder.update_profile()
        assert "tools" in profile
        assert "primary_apps" in profile["tools"]
        assert profile["updated_at"] is not None

    def test_pattern_detector_finds_daily(self, kb: KnowledgeBase) -> None:
        """PatternDetector finds daily patterns from summaries."""
        for day in range(1, 11):
            kb.save_daily_summary(f"2026-03-{day:02d}", {
                "active_hours": 7.0,
                "task_count": 3,
                "tasks": [
                    {"intent": "Check email", "apps": ["Chrome"],
                     "urls": [], "start_time": f"2026-03-{day:02d}T09:00:00Z",
                     "duration_minutes": 15, "matched_procedure": "check-email"},
                ],
                "top_apps": [{"app": "Chrome", "minutes": 60}],
                "procedures_observed": ["check-email"],
                "new_workflows_detected": 0,
            })

        detector = PatternDetector(kb, min_observations=3)
        patterns = detector.detect_recurrence()
        assert len(patterns) >= 1
        assert patterns[0].pattern == "daily"

    def test_branch_extraction(self, kb: KnowledgeBase) -> None:
        """BranchExtractor finds divergence across demos."""
        extractor = BranchExtractor(kb)

        demos = [
            [{"step_id": "s1", "action": "Open"}, {"step_id": "s2", "action": "Click A"}],
            [{"step_id": "s1", "action": "Open"}, {"step_id": "s2", "action": "Click A"}],
            [{"step_id": "s1", "action": "Open"}, {"step_id": "s2", "action": "Click B"}],
        ]

        branches = extractor.extract_branches("test", demos=demos)
        assert len(branches) == 1
        assert branches[0].step_id == "s2"
        assert branches[0].type == "data_dependent"

    def test_constraint_manager(self, kb: KnowledgeBase) -> None:
        """ConstraintManager controls execution permissions."""
        manager = ConstraintManager(kb)

        # Create a procedure
        proc = sop_to_procedure(_make_sop_template("constrained"))
        kb.save_procedure(proc)

        # Default: observe
        allowed, reason = manager.check_execution_allowed("constrained")
        assert allowed is False

        # Promote to autonomous
        manager.set_trust_level("constrained", TrustLevel.AUTONOMOUS)
        allowed, reason = manager.check_execution_allowed("constrained")
        assert allowed is True

    def test_decision_extraction(self, kb: KnowledgeBase) -> None:
        """DecisionExtractor finds rules from varying observations."""
        extractor = DecisionExtractor(kb)

        observations = [
            {"steps": [{"step_id": "s1", "action": "Open"}, {"step_id": "s2", "action": "Buy"}], "context": {}},
            {"steps": [{"step_id": "s1", "action": "Open"}, {"step_id": "s2", "action": "Buy"}], "context": {}},
            {"steps": [{"step_id": "s1", "action": "Open"}, {"step_id": "s2", "action": "Skip"}], "context": {}},
        ]

        decisions = extractor.extract_decisions("test", observations)
        assert len(decisions) >= 1
        assert decisions[0].applies_to_step == "s2"

    def test_outcome_tracking(self) -> None:
        """OutcomeTracker detects task outcomes."""
        tracker = OutcomeTracker()

        events = [
            _make_event("e1", "2026-03-10T09:00:00Z", "Chrome", "Copy data",
                        "https://site.com"),
            {
                "id": "e2",
                "timestamp": "2026-03-10T09:01:00Z",
                "kind_json": '{"ClipboardChange":{}}',
                "window_json": '{"app":"Chrome"}',
                "metadata_json": "{}",
                "scene_annotation_json": None,
            },
            _make_event("e3", "2026-03-10T09:02:00Z", "Google Sheets",
                        "Paste data"),
        ]

        outcomes = tracker.detect_outcomes(events)
        types = {o.type for o in outcomes}
        assert "data_transfer" in types

    def test_staleness_detection(self, kb: KnowledgeBase) -> None:
        """StalenessDetector flags old procedures."""
        detector = StalenessDetector(kb)

        from datetime import timedelta
        now = datetime.now(timezone.utc)

        # Current procedure
        proc_current = sop_to_procedure(_make_sop_template("current"))
        proc_current["staleness"]["last_observed"] = now.isoformat()
        kb.save_procedure(proc_current)

        # Old procedure
        proc_old = sop_to_procedure(_make_sop_template("old"))
        proc_old["staleness"]["last_observed"] = (now - timedelta(days=45)).isoformat()
        kb.save_procedure(proc_old)

        reports = detector.check_all()
        statuses = {r.slug: r.status for r in reports}
        assert statuses["current"] == "current"
        assert statuses["old"] == "needs_review"

    def test_account_detector(self) -> None:
        """AccountDetector identifies services from URLs."""
        detector = AccountDetector()

        result = detector.detect_from_url("https://dashboard.stripe.com/test/payments")
        assert result is not None
        assert result.service == "stripe"
        assert result.environment == "test"

    def test_export_adapter(self, kb: KnowledgeBase) -> None:
        """KnowledgeBaseExportAdapter writes SOPs as v3 procedures."""
        adapter = KnowledgeBaseExportAdapter(kb)

        sop = _make_sop_template("export-test")
        path = adapter.write_sop(sop)
        assert path.is_file()

        sops = adapter.list_sops()
        assert len(sops) == 1
        assert sops[0]["slug"] == "export-test"

    def test_full_pipeline_e2e(self, kb: KnowledgeBase) -> None:
        """Complete end-to-end: events → daily → procedure → evidence →
        profile → patterns → staleness.
        """
        # Step 1: Process daily events
        processor = DailyBatchProcessor(knowledge_base=kb)
        events = []
        for i in range(20):
            minute = i * 2
            events.append(_make_event(
                f"ev-{i}",
                f"2026-03-10T09:{minute:02d}:00Z",
                "Chrome" if i < 10 else "VS Code",
                "Checking domains" if i < 10 else "Writing code",
                "https://auctions.godaddy.com" if i < 10 else "/code",
            ))
        summary = processor.process_day("2026-03-10", events)
        assert summary.task_count >= 1

        # Step 2: Create a procedure from SOP
        tracker = EvidenceTracker(knowledge_base=kb)
        writer = ProcedureWriter(kb=kb, evidence=tracker)
        writer.write_procedure(
            _make_sop_template("check-domains"),
            source="passive", source_id="seg-1",
            event_count=10, duration_minutes=20,
        )

        # Step 3: Profile builder
        # Add more daily summaries for pattern detection
        for day in range(1, 8):
            kb.save_daily_summary(f"2026-03-{day:02d}", {
                "active_hours": 7.0,
                "task_count": 3,
                "tasks": [
                    {"intent": "Check domains", "apps": ["Chrome"],
                     "urls": ["https://auctions.godaddy.com"],
                     "start_time": f"2026-03-{day:02d}T09:00:00Z",
                     "duration_minutes": 20, "matched_procedure": "check-domains"},
                ],
                "top_apps": [{"app": "Chrome", "minutes": 60}],
                "procedures_observed": ["check-domains"],
                "new_workflows_detected": 0,
            })

        profile_builder = ProfileBuilder(kb)
        profile = profile_builder.update_profile()
        assert profile["updated_at"] is not None

        # Step 4: Pattern detection
        pattern_detector = PatternDetector(kb, min_observations=3)
        patterns = pattern_detector.detect_recurrence()
        if patterns:
            pattern_detector.update_triggers(patterns)

        # Step 5: Staleness check
        staleness = StalenessDetector(kb)
        reports = staleness.check_all()
        assert len(reports) >= 1

        # Step 6: Verify knowledge base structure
        assert (kb.root / "procedures" / "check-domains.json").is_file()
        assert kb.get_profile()["updated_at"] is not None

        proc = kb.get_procedure("check-domains")
        assert proc is not None
        assert proc["evidence"]["total_observations"] >= 1
        assert validate_procedure(proc) == []
