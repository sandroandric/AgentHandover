"""Agent handoff layer conformance tests.

Validates the contract between the worker's knowledge base and
AI agents that consume procedures for execution.

Tests cover:
- v3 procedure schema completeness (all agent-required fields present)
- Export adapter write_procedure() round-trip
- Freshness score computation
- Preflight verification
- Postcondition validation
- Bundle assembly (all fields present)
- Ready filtering
- Execution feedback loop
- Procedure composition (chain field)
- Environment enrichment
"""

from __future__ import annotations

import json
import math

import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock


# ---- Fixtures ----

@pytest.fixture
def kb(tmp_path):
    """Create a KnowledgeBase with a temp root."""
    from agenthandover_worker.knowledge_base import KnowledgeBase
    kb = KnowledgeBase(root=tmp_path)
    kb.ensure_structure()
    return kb


@pytest.fixture
def sample_sop_template():
    """A representative SOP template as produced by sop_generator."""
    return {
        "slug": "check-domain-availability",
        "title": "Check Domain Availability",
        "short_title": "Domain Check",
        "description": "Check if a domain name is available for purchase",
        "tags": ["domain", "web"],
        "steps": [
            {
                "step": "Navigate to registrar",
                "target": "https://namecheap.com",
                "app": "Chrome",
                "location": "https://namecheap.com",
                "input": "",
                "verify": "Namecheap homepage loads",
                "confidence": 0.92,
                "parameters": {
                    "app": "Chrome",
                    "location": "https://namecheap.com",
                    "verify": "Namecheap homepage loads",
                },
            },
            {
                "step": "Search for domain",
                "target": "search box",
                "app": "Chrome",
                "location": "",
                "input": "{{domain_name}}",
                "verify": "Search results appear",
                "confidence": 0.88,
                "parameters": {
                    "app": "Chrome",
                    "input": "{{domain_name}}",
                    "verify": "Search results appear",
                },
            },
            {
                "step": "Check price and availability",
                "target": "results panel",
                "app": "Chrome",
                "location": "",
                "input": "",
                "verify": "Price and status displayed",
                "confidence": 0.85,
                "parameters": {
                    "app": "Chrome",
                    "verify": "Price and status displayed",
                },
            },
        ],
        "variables": [
            {"name": "domain_name", "type": "string", "description": "Domain to check", "example": "example.com"},
        ],
        "confidence_avg": 0.88,
        "episode_count": 3,
        "apps_involved": ["Chrome"],
        "preconditions": ["Browser must be open"],
        "postconditions": ["Domain availability is known"],
        "exceptions_seen": [],
        "source": "v2_passive_discovery",
        "task_description": "Check if a domain name is available for registration",
        "execution_overview": {"when_to_use": "When exploring new domain names"},
    }


@pytest.fixture
def sample_procedure(sample_sop_template):
    """A v3 procedure converted from the sample SOP template."""
    from agenthandover_worker.procedure_schema import sop_to_procedure
    proc = sop_to_procedure(sample_sop_template)
    return proc


# ---- 1. v3 Schema Completeness ----

class TestSchemaCompleteness:
    """Every agent-required field must be present in a v3 procedure."""

    def test_required_top_level_fields(self, sample_procedure):
        """v3 procedure has all agent-critical fields."""
        required = [
            "schema_version", "id", "title", "steps",
            "inputs", "outputs", "environment", "branches",
            "expected_outcomes", "staleness", "evidence",
            "constraints", "recurrence", "chain",
        ]
        for field in required:
            assert field in sample_procedure, f"Missing field: {field}"

    def test_steps_have_on_failure(self, sample_procedure):
        """Every step must have an on_failure strategy."""
        for i, step in enumerate(sample_procedure["steps"]):
            assert "on_failure" in step, f"steps[{i}] missing on_failure"
            assert "strategy" in step["on_failure"]

    def test_environment_structure(self, sample_procedure):
        """Environment must have required_apps, accounts, setup_actions."""
        env = sample_procedure["environment"]
        assert "required_apps" in env
        assert "accounts" in env
        assert "setup_actions" in env

    def test_constraints_have_trust_level(self, sample_procedure):
        """Constraints must include trust_level."""
        assert "trust_level" in sample_procedure["constraints"]

    def test_chain_field_present(self, sample_procedure):
        """Chain field for procedure composition."""
        chain = sample_procedure["chain"]
        assert "depends_on" in chain
        assert "followed_by" in chain
        assert "can_compose" in chain

    def test_staleness_fields(self, sample_procedure):
        """Staleness must have last_observed and confidence_trend."""
        staleness = sample_procedure["staleness"]
        assert "last_observed" in staleness
        assert "confidence_trend" in staleness

    def test_evidence_fields(self, sample_procedure):
        """Evidence must have observations count and contradictions."""
        evidence = sample_procedure["evidence"]
        assert "total_observations" in evidence
        assert "contradictions" in evidence

    def test_validate_procedure_passes(self, sample_procedure):
        """validate_procedure() returns no errors for well-formed procedure."""
        from agenthandover_worker.procedure_schema import validate_procedure
        errors = validate_procedure(sample_procedure)
        assert errors == [], f"Validation errors: {errors}"


# ---- 2. Export Adapter write_procedure() ----

class TestExportWriteProcedure:
    """Export adapters can render from v3 procedure dicts."""

    def test_procedure_to_sop_template_roundtrip(self, sample_sop_template):
        """procedure_to_sop_template preserves critical fields."""
        from agenthandover_worker.procedure_schema import sop_to_procedure
        from agenthandover_worker.export_adapter import procedure_to_sop_template

        proc = sop_to_procedure(sample_sop_template)
        template = procedure_to_sop_template(proc)

        assert template["slug"] == sample_sop_template["slug"]
        assert template["title"] == sample_sop_template["title"]
        assert len(template["steps"]) == len(sample_sop_template["steps"])
        assert len(template["variables"]) == len(sample_sop_template["variables"])

    def test_openclaw_writer_write_procedure(self, tmp_path, sample_procedure):
        """OpenClawWriter.write_procedure() produces a file."""
        from agenthandover_worker.openclaw_writer import OpenClawWriter
        writer = OpenClawWriter(workspace_dir=tmp_path)
        path = writer.write_procedure(sample_procedure)
        assert path.exists()
        assert path.suffix == ".md"

    def test_skill_md_writer_write_procedure(self, tmp_path, sample_procedure):
        """SkillMdWriter.write_procedure() includes v3 sections."""
        from agenthandover_worker.skill_md_writer import SkillMdWriter
        writer = SkillMdWriter(workspace_dir=tmp_path)
        path = writer.write_procedure(sample_procedure)
        assert path.exists()
        content = path.read_text()
        assert "Check Domain Availability" in content

    def test_claude_skill_writer_write_procedure(self, tmp_path, sample_procedure):
        """ClaudeSkillWriter.write_procedure() produces SKILL.md."""
        from agenthandover_worker.claude_skill_writer import ClaudeSkillWriter
        writer = ClaudeSkillWriter(skills_dir=tmp_path)
        path = writer.write_procedure(sample_procedure)
        assert path.exists()
        assert path.name == "SKILL.md"

    def test_generic_writer_write_procedure(self, tmp_path, sample_procedure):
        """GenericWriter.write_procedure() produces .md file."""
        from agenthandover_worker.generic_writer import GenericWriter
        writer = GenericWriter(output_dir=tmp_path)
        path = writer.write_procedure(sample_procedure)
        assert path.exists()


# ---- 3. Freshness Score ----

class TestFreshnessScore:
    """Freshness score decays correctly over time."""

    def test_fresh_procedure_score_near_one(self):
        """Recently observed procedure has high freshness."""
        from agenthandover_worker.staleness_detector import procedure_freshness
        proc = {
            "staleness": {
                "last_observed": datetime.now(timezone.utc).isoformat(),
                "confidence_trend": [0.9, 0.9, 0.9],
            },
            "evidence": {"contradictions": []},
        }
        score = procedure_freshness(proc)
        assert score >= 0.9

    def test_old_procedure_score_decays(self):
        """Procedure not observed in 60 days has low freshness."""
        from agenthandover_worker.staleness_detector import procedure_freshness
        old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        proc = {
            "staleness": {
                "last_observed": old_date,
                "confidence_trend": [],
            },
            "evidence": {"contradictions": []},
        }
        score = procedure_freshness(proc)
        assert score < 0.3

    def test_confirmed_procedure_gets_bonus(self):
        """Recent confirmation boosts freshness."""
        from agenthandover_worker.staleness_detector import procedure_freshness
        old_date = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
        recent_confirm = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()

        proc_no_confirm = {
            "staleness": {
                "last_observed": old_date,
                "last_confirmed": None,
                "confidence_trend": [],
            },
            "evidence": {"contradictions": []},
        }
        proc_confirmed = {
            "staleness": {
                "last_observed": old_date,
                "last_confirmed": recent_confirm,
                "confidence_trend": [],
            },
            "evidence": {"contradictions": []},
        }

        score_no = procedure_freshness(proc_no_confirm)
        score_yes = procedure_freshness(proc_confirmed)
        assert score_yes > score_no

    def test_contradictions_reduce_score(self):
        """Contradictions in evidence reduce freshness."""
        from agenthandover_worker.staleness_detector import procedure_freshness
        proc_clean = {
            "staleness": {
                "last_observed": datetime.now(timezone.utc).isoformat(),
                "confidence_trend": [],
            },
            "evidence": {"contradictions": []},
        }
        proc_contradicted = {
            "staleness": {
                "last_observed": datetime.now(timezone.utc).isoformat(),
                "confidence_trend": [],
            },
            "evidence": {"contradictions": [
                {"step": "1", "detail": "mismatch"},
                {"step": "2", "detail": "mismatch"},
            ]},
        }

        score_clean = procedure_freshness(proc_clean)
        score_bad = procedure_freshness(proc_contradicted)
        assert score_bad < score_clean

    def test_declining_confidence_reduces_score(self):
        """Declining confidence trend reduces freshness."""
        from agenthandover_worker.staleness_detector import procedure_freshness
        proc_stable = {
            "staleness": {
                "last_observed": datetime.now(timezone.utc).isoformat(),
                "confidence_trend": [0.9, 0.9, 0.9],
            },
            "evidence": {"contradictions": []},
        }
        proc_declining = {
            "staleness": {
                "last_observed": datetime.now(timezone.utc).isoformat(),
                "confidence_trend": [0.9, 0.8, 0.7],
            },
            "evidence": {"contradictions": []},
        }

        score_stable = procedure_freshness(proc_stable)
        score_declining = procedure_freshness(proc_declining)
        assert score_declining < score_stable


# ---- 4. Preflight Verification ----

class TestPreflightVerification:
    """Preflight checks block or allow execution correctly."""

    def test_preflight_missing_procedure(self, kb):
        """Preflight fails for nonexistent procedure."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier
        verifier = ProcedureVerifier(kb)
        result = verifier.preflight("nonexistent-slug")
        assert not result.can_execute
        assert not result.can_draft

    def test_preflight_observe_trust_blocks_execution(self, kb, sample_procedure):
        """Procedure at 'observe' trust cannot execute."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier
        sample_procedure["constraints"]["trust_level"] = "observe"
        kb.save_procedure(sample_procedure)

        verifier = ProcedureVerifier(kb)
        result = verifier.preflight(sample_procedure["id"])
        assert not result.can_execute

    def test_preflight_autonomous_trust_allows_execution(self, kb, sample_procedure):
        """Procedure at 'autonomous' trust can execute."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier
        sample_procedure["constraints"]["trust_level"] = "autonomous"
        sample_procedure["evidence"]["total_observations"] = 5
        sample_procedure["lifecycle_state"] = "agent_ready"
        kb.save_procedure(sample_procedure)

        verifier = ProcedureVerifier(kb)
        result = verifier.preflight(sample_procedure["id"])
        assert result.can_execute

    def test_preflight_draft_trust_allows_drafting(self, kb, sample_procedure):
        """Procedure at 'draft' trust can draft but not execute."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier
        sample_procedure["constraints"]["trust_level"] = "draft"
        sample_procedure["evidence"]["total_observations"] = 3
        sample_procedure["lifecycle_state"] = "draft"
        kb.save_procedure(sample_procedure)

        verifier = ProcedureVerifier(kb)
        result = verifier.preflight(sample_procedure["id"])
        assert result.can_draft
        assert not result.can_execute

    def test_preflight_blocked_domain(self, kb, sample_procedure):
        """Blocked domain in constraints prevents execution."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier
        sample_procedure["constraints"]["trust_level"] = "autonomous"
        sample_procedure["evidence"]["total_observations"] = 5
        # Add a URL to a step that matches a blocked domain
        sample_procedure["steps"][0]["location"] = "https://evil.blocked.com/page"
        kb.save_procedure(sample_procedure)
        kb.update_constraints({"blocked_domains": ["blocked.com"]})

        verifier = ProcedureVerifier(kb)
        result = verifier.preflight(sample_procedure["id"])
        assert not result.can_execute


# ---- 5. Postcondition Validation ----

class TestPostconditionValidation:
    """Postcondition checks validate execution outcomes."""

    def test_no_expected_outcomes_passes(self, kb, sample_procedure):
        """If no expected outcomes defined, postcondition passes."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier
        sample_procedure["expected_outcomes"] = []
        kb.save_procedure(sample_procedure)

        verifier = ProcedureVerifier(kb)
        result = verifier.validate_postconditions(
            sample_procedure["id"], "exec-1", []
        )
        assert result.all_passed

    def test_matching_outcomes_pass(self, kb, sample_procedure):
        """Matching outcome types pass validation."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier
        sample_procedure["expected_outcomes"] = [
            {"type": "data_transfer", "description": "Domain info copied"},
        ]
        kb.save_procedure(sample_procedure)

        verifier = ProcedureVerifier(kb)
        result = verifier.validate_postconditions(
            sample_procedure["id"], "exec-1",
            [{"type": "data_transfer", "description": "Copied domain info to clipboard"}],
        )
        assert result.all_passed

    def test_missing_outcome_fails(self, kb, sample_procedure):
        """Missing expected outcome type fails validation."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier
        sample_procedure["expected_outcomes"] = [
            {"type": "file_created", "description": "Report saved"},
        ]
        kb.save_procedure(sample_procedure)

        verifier = ProcedureVerifier(kb)
        result = verifier.validate_postconditions(
            sample_procedure["id"], "exec-1", [],
        )
        assert not result.all_passed


# ---- 6. Execution Feedback Loop ----

class TestExecutionFeedback:
    """ExecutionMonitor tracks agent execution correctly."""

    def test_start_and_complete(self, kb, sample_procedure):
        """Full execution lifecycle: start -> steps -> complete."""
        from agenthandover_worker.execution_monitor import ExecutionMonitor, ExecutionStatus
        kb.save_procedure(sample_procedure)
        monitor = ExecutionMonitor(kb)

        exec_id = monitor.start_execution(sample_procedure["id"], "test-agent")
        assert exec_id

        monitor.record_step(exec_id, "0", "Navigate to registrar")
        monitor.record_step(exec_id, "1", "Search for domain")

        record = monitor.complete_execution(exec_id)
        assert record.status == ExecutionStatus.COMPLETED

    def test_failed_execution(self, kb, sample_procedure):
        """Failed execution records error."""
        from agenthandover_worker.execution_monitor import ExecutionMonitor, ExecutionStatus
        kb.save_procedure(sample_procedure)
        monitor = ExecutionMonitor(kb)

        exec_id = monitor.start_execution(sample_procedure["id"], "test-agent")
        record = monitor.fail_execution(exec_id, "Timeout waiting for page load")
        assert record.status == ExecutionStatus.FAILED
        assert record.error == "Timeout waiting for page load"

    def test_deviation_detected(self, kb, sample_procedure):
        """Deviation recorded when actual differs from expected."""
        from agenthandover_worker.execution_monitor import ExecutionMonitor, ExecutionStatus
        kb.save_procedure(sample_procedure)
        monitor = ExecutionMonitor(kb)

        exec_id = monitor.start_execution(sample_procedure["id"], "test-agent")
        # Record a different action than expected
        monitor.record_step(exec_id, "0", "Completely different action")
        record = monitor.complete_execution(exec_id)

        assert record.status == ExecutionStatus.DEVIATED
        assert len(record.deviations) > 0

    def test_success_rate_computation(self, kb, sample_procedure):
        """Success rate computed from execution history."""
        from agenthandover_worker.execution_monitor import ExecutionMonitor
        kb.save_procedure(sample_procedure)
        monitor = ExecutionMonitor(kb)

        slug = sample_procedure["id"]

        # 2 successes
        for _ in range(2):
            eid = monitor.start_execution(slug, "agent")
            monitor.complete_execution(eid)

        # 1 failure
        eid = monitor.start_execution(slug, "agent")
        monitor.fail_execution(eid, "error")

        stats = monitor.get_success_rate(slug)
        assert stats["total"] == 3
        assert stats["completed"] == 2
        assert stats["failed"] == 1
        assert abs(stats["success_rate"] - 2/3) < 0.01


# ---- 7. Environment Enrichment ----

class TestEnvironmentEnrichment:
    """ProcedureWriter enriches environment from observation data."""

    def test_apps_extracted_from_steps(self, kb, sample_sop_template):
        """Required apps populated from step app fields."""
        from agenthandover_worker.evidence_tracker import EvidenceTracker
        from agenthandover_worker.procedure_writer import ProcedureWriter

        et = EvidenceTracker(knowledge_base=kb)
        pw = ProcedureWriter(kb=kb, evidence=et)
        path = pw.write_procedure(sample_sop_template, "test", "test-session")

        proc = kb.get_procedure(sample_sop_template["slug"])
        assert proc is not None
        env = proc["environment"]
        assert "Chrome" in env["required_apps"]

    def test_accounts_detected_from_urls(self, kb):
        """Account hints extracted from step URLs."""
        from agenthandover_worker.evidence_tracker import EvidenceTracker
        from agenthandover_worker.procedure_writer import ProcedureWriter

        et = EvidenceTracker(knowledge_base=kb)
        pw = ProcedureWriter(kb=kb, evidence=et)

        template = {
            "slug": "github-pr-review",
            "title": "Review PR on GitHub",
            "steps": [
                {
                    "step": "Open PR",
                    "target": "https://github.com/org/repo/pull/123",
                    "location": "https://github.com/org/repo/pull/123",
                    "app": "Chrome",
                    "confidence": 0.9,
                },
            ],
            "variables": [],
            "confidence_avg": 0.9,
            "apps_involved": ["Chrome"],
            "source": "passive",
        }
        pw.write_procedure(template, "test", "test-session")

        proc = kb.get_procedure("github-pr-review")
        assert proc is not None
        accounts = proc["environment"].get("accounts", [])
        services = [a["service"] for a in accounts]
        assert "github" in services


# ---- 8. Procedure Composition (Chain) ----

class TestProcedureComposition:
    """Chain field tracks procedure relationships."""

    def test_chain_field_initialized(self, sample_procedure):
        """New procedure has chain field with correct structure."""
        chain = sample_procedure["chain"]
        assert isinstance(chain["depends_on"], list)
        assert isinstance(chain["followed_by"], list)
        assert isinstance(chain["can_compose"], bool)

    def test_chain_enrichment_from_summaries(self, kb, sample_procedure):
        """Chain.followed_by populated from daily summaries."""
        from agenthandover_worker.evidence_tracker import EvidenceTracker
        from agenthandover_worker.procedure_writer import ProcedureWriter

        slug = sample_procedure["id"]
        kb.save_procedure(sample_procedure)

        # Create daily summaries with procedure co-occurrence
        for i in range(3):
            date = f"2026-03-{10+i:02d}"
            kb.save_daily_summary(date, {
                "procedures_observed": [slug, "next-procedure"],
                "active_hours": 8.0,
                "task_count": 5,
            })

        et = EvidenceTracker(knowledge_base=kb)
        pw = ProcedureWriter(kb=kb, evidence=et)
        pw.enrich_chains(slug)

        proc = kb.get_procedure(slug)
        assert "next-procedure" in proc["chain"]["followed_by"]
        assert proc["chain"]["can_compose"] is True


# ---- 9. Query API Integration ----

class TestQueryAPIContract:
    """API endpoints return expected structure for agents."""

    def test_bundle_contains_required_fields(self, kb, sample_procedure):
        """Bundle response has all agent-required top-level keys."""
        sample_procedure["constraints"]["trust_level"] = "autonomous"
        kb.save_procedure(sample_procedure)

        # Test the handler logic directly
        from agenthandover_worker.staleness_detector import procedure_freshness
        freshness = procedure_freshness(sample_procedure)

        # Verify the fields that the bundle endpoint would include
        assert sample_procedure.get("constraints", {}).get("trust_level")
        assert freshness >= 0.0
        assert "chain" in sample_procedure
        assert "recurrence" in sample_procedure
        assert "environment" in sample_procedure

    def test_ready_filter_excludes_observe_trust(self, kb, sample_procedure):
        """Procedures at 'observe' trust are not ready for agents."""
        from agenthandover_worker.staleness_detector import procedure_freshness

        sample_procedure["constraints"]["trust_level"] = "observe"
        kb.save_procedure(sample_procedure)

        # Simulate the /ready filter logic
        trust = sample_procedure["constraints"]["trust_level"]
        ready_levels = {"draft", "execute_with_approval", "autonomous"}
        assert trust not in ready_levels

    def test_ready_filter_includes_autonomous(self, kb, sample_procedure):
        """Procedures at 'autonomous' trust with good freshness are ready."""
        from agenthandover_worker.staleness_detector import procedure_freshness

        sample_procedure["constraints"]["trust_level"] = "autonomous"
        kb.save_procedure(sample_procedure)

        trust = sample_procedure["constraints"]["trust_level"]
        freshness = procedure_freshness(sample_procedure)

        ready_levels = {"draft", "execute_with_approval", "autonomous"}
        assert trust in ready_levels
        assert freshness >= 0.3


# ---- 10. Contract Consistency ----

class TestContractConsistency:
    """API readiness, preflight readiness, and export readiness must agree.

    For any given procedure, ``/ready``, ``preflight()``, and export must
    yield the same truthful answer about whether it can be executed.
    """

    def _simulate_ready_filter(self, proc):
        """Simulate the /ready endpoint filter logic."""
        from agenthandover_worker.staleness_detector import procedure_freshness

        _EXECUTABLE_TRUST_LEVELS = frozenset({
            "execute_with_approval", "autonomous",
        })
        _DRAFTABLE_TRUST_LEVELS = frozenset({
            "draft", "execute_with_approval", "autonomous",
        })
        _MIN_FRESHNESS = 0.3

        trust_level = proc.get("constraints", {}).get("trust_level", "observe")
        if trust_level not in _DRAFTABLE_TRUST_LEVELS:
            return None  # excluded from /ready

        freshness = procedure_freshness(proc)
        if freshness < _MIN_FRESHNESS:
            return None  # excluded from /ready

        can_execute = trust_level in _EXECUTABLE_TRUST_LEVELS
        return {
            "can_execute": can_execute,
            "can_draft": True,
            "trust_level": trust_level,
            "freshness_score": freshness,
        }

    def test_draft_procedure_consistent(self, kb, sample_procedure):
        """Draft procedure: can_draft=True, can_execute=False in both /ready and preflight."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier

        sample_procedure["constraints"]["trust_level"] = "draft"
        sample_procedure["evidence"]["total_observations"] = 3
        sample_procedure["lifecycle_state"] = "draft"
        kb.save_procedure(sample_procedure)

        # /ready filter
        ready_entry = self._simulate_ready_filter(sample_procedure)
        assert ready_entry is not None, "Draft should appear in /ready"
        assert ready_entry["can_draft"] is True
        assert ready_entry["can_execute"] is False

        # Preflight
        verifier = ProcedureVerifier(kb)
        preflight = verifier.preflight(sample_procedure["id"])
        assert preflight.can_draft is True
        assert preflight.can_execute is False

    def test_observe_procedure_consistent(self, kb, sample_procedure):
        """Observe-only procedure: excluded from /ready, can_execute=False, can_draft=False in preflight."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier

        sample_procedure["constraints"]["trust_level"] = "observe"
        kb.save_procedure(sample_procedure)

        # /ready filter — should not appear
        ready_entry = self._simulate_ready_filter(sample_procedure)
        assert ready_entry is None, "Observe-only must not appear in /ready"

        # Preflight
        verifier = ProcedureVerifier(kb)
        preflight = verifier.preflight(sample_procedure["id"])
        assert preflight.can_execute is False
        assert preflight.can_draft is False

    def test_autonomous_fresh_consistent(self, kb, sample_procedure):
        """Autonomous + fresh procedure: can_execute=True everywhere."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier

        sample_procedure["constraints"]["trust_level"] = "autonomous"
        sample_procedure["evidence"]["total_observations"] = 5
        sample_procedure["lifecycle_state"] = "agent_ready"
        kb.save_procedure(sample_procedure)

        # /ready filter
        ready_entry = self._simulate_ready_filter(sample_procedure)
        assert ready_entry is not None
        assert ready_entry["can_execute"] is True

        # Preflight
        verifier = ProcedureVerifier(kb)
        preflight = verifier.preflight(sample_procedure["id"])
        assert preflight.can_execute is True

    def test_stale_procedure_consistent(self, kb, sample_procedure):
        """Stale procedure: excluded from /ready, freshness blocks preflight."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier

        sample_procedure["constraints"]["trust_level"] = "autonomous"
        sample_procedure["staleness"]["last_observed"] = "2024-01-01T00:00:00+00:00"
        sample_procedure["staleness"]["confidence_trend"] = []
        sample_procedure["evidence"]["total_observations"] = 5
        kb.save_procedure(sample_procedure)

        # /ready filter — should be excluded due to low freshness
        ready_entry = self._simulate_ready_filter(sample_procedure)
        assert ready_entry is None, "Stale procedure must not appear in /ready"

        # Preflight
        verifier = ProcedureVerifier(kb)
        preflight = verifier.preflight(sample_procedure["id"])
        assert preflight.can_execute is False
        assert preflight.can_draft is False

    def test_suggest_trust_excluded(self, kb, sample_procedure):
        """Suggest-level trust: excluded from /ready, no drafting allowed."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier

        sample_procedure["constraints"]["trust_level"] = "suggest"
        kb.save_procedure(sample_procedure)

        ready_entry = self._simulate_ready_filter(sample_procedure)
        assert ready_entry is None

        verifier = ProcedureVerifier(kb)
        preflight = verifier.preflight(sample_procedure["id"])
        assert preflight.can_execute is False
        assert preflight.can_draft is False


# ---- 11. No False-Ready ----

class TestNoFalseReady:
    """No combination of parameters can produce a false-ready state."""

    def test_draft_never_executable(self, kb, sample_procedure):
        """Draft trust + any freshness must never yield can_execute=True."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier

        sample_procedure["constraints"]["trust_level"] = "draft"
        sample_procedure["evidence"]["total_observations"] = 100
        kb.save_procedure(sample_procedure)

        verifier = ProcedureVerifier(kb)
        preflight = verifier.preflight(sample_procedure["id"])
        assert preflight.can_execute is False

    def test_stale_autonomous_never_executable(self, kb, sample_procedure):
        """Autonomous trust but stale freshness must not yield can_execute=True."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier

        sample_procedure["constraints"]["trust_level"] = "autonomous"
        sample_procedure["staleness"]["last_observed"] = "2020-01-01T00:00:00+00:00"
        sample_procedure["staleness"]["confidence_trend"] = []
        sample_procedure["evidence"]["total_observations"] = 50
        kb.save_procedure(sample_procedure)

        verifier = ProcedureVerifier(kb)
        preflight = verifier.preflight(sample_procedure["id"])
        assert preflight.can_execute is False

    def test_no_steps_blocks_execution(self, kb, sample_procedure):
        """Procedure with no steps must not be executable."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier

        sample_procedure["constraints"]["trust_level"] = "autonomous"
        sample_procedure["steps"] = []
        sample_procedure["evidence"]["total_observations"] = 5
        kb.save_procedure(sample_procedure)

        verifier = ProcedureVerifier(kb)
        preflight = verifier.preflight(sample_procedure["id"])
        assert preflight.can_execute is False

    def test_blocked_domain_blocks_execution(self, kb, sample_procedure):
        """Even autonomous + fresh, a blocked domain prevents execution."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier

        sample_procedure["constraints"]["trust_level"] = "autonomous"
        sample_procedure["evidence"]["total_observations"] = 5
        sample_procedure["steps"][0]["location"] = "https://malicious.example.com/attack"
        kb.save_procedure(sample_procedure)
        kb.update_constraints({"blocked_domains": ["malicious.example.com"]})

        verifier = ProcedureVerifier(kb)
        preflight = verifier.preflight(sample_procedure["id"])
        assert preflight.can_execute is False


# ---- 12. Preflight Advisory Labeling ----

class TestPreflightAdvisoryLabeling:
    """Advisory checks are labeled correctly and don't gate execution."""

    def test_required_apps_is_advisory(self, kb, sample_procedure):
        """required_apps check has severity='advisory'."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier

        sample_procedure["constraints"]["trust_level"] = "autonomous"
        sample_procedure["evidence"]["total_observations"] = 5
        kb.save_procedure(sample_procedure)

        verifier = ProcedureVerifier(kb)
        result = verifier.preflight(sample_procedure["id"])

        app_checks = [c for c in result.checks if c.name == "required_apps"]
        assert len(app_checks) == 1
        assert app_checks[0].severity == "advisory"

    def test_observations_is_advisory(self, kb, sample_procedure):
        """observations check has severity='advisory'."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier

        sample_procedure["constraints"]["trust_level"] = "autonomous"
        sample_procedure["evidence"]["total_observations"] = 5
        kb.save_procedure(sample_procedure)

        verifier = ProcedureVerifier(kb)
        result = verifier.preflight(sample_procedure["id"])

        obs_checks = [c for c in result.checks if c.name == "observations"]
        assert len(obs_checks) == 1
        assert obs_checks[0].severity == "advisory"

    def test_advisories_dont_gate_execution(self, kb, sample_procedure):
        """Procedure with advisory-only issues can still execute."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier

        sample_procedure["constraints"]["trust_level"] = "autonomous"
        sample_procedure["evidence"]["total_observations"] = 0  # advisory: no observations
        sample_procedure["lifecycle_state"] = "agent_ready"
        kb.save_procedure(sample_procedure)

        verifier = ProcedureVerifier(kb)
        result = verifier.preflight(sample_procedure["id"])

        # Observations check is advisory, doesn't block
        obs_check = [c for c in result.checks if c.name == "observations"][0]
        assert obs_check.severity == "advisory"
        assert obs_check.passed is False  # hasn't been observed

        # But execution is still allowed (not gated by advisory)
        assert result.can_execute is True

    def test_advisory_property_returns_advisory_checks(self, kb, sample_procedure):
        """PreflightResult.advisories returns advisory checks."""
        from agenthandover_worker.procedure_verifier import ProcedureVerifier

        sample_procedure["constraints"]["trust_level"] = "autonomous"
        sample_procedure["evidence"]["total_observations"] = 5
        kb.save_procedure(sample_procedure)

        verifier = ProcedureVerifier(kb)
        result = verifier.preflight(sample_procedure["id"])

        # Should have at least required_apps and observations
        advisories = result.advisories
        advisory_names = {a.name for a in advisories}
        assert "required_apps" in advisory_names
        assert "observations" in advisory_names


# ---- 13. Export Parity ----

class TestExportParity:
    """All export adapters render the same procedure with consistent required fields."""

    def test_all_adapters_produce_files(self, tmp_path, sample_procedure):
        """All 4 adapters successfully write_procedure()."""
        from agenthandover_worker.openclaw_writer import OpenClawWriter
        from agenthandover_worker.skill_md_writer import SkillMdWriter
        from agenthandover_worker.claude_skill_writer import ClaudeSkillWriter
        from agenthandover_worker.generic_writer import GenericWriter

        results = {}

        oc = OpenClawWriter(workspace_dir=tmp_path / "openclaw")
        results["openclaw"] = oc.write_procedure(sample_procedure)

        sm = SkillMdWriter(workspace_dir=tmp_path / "skillmd")
        results["skill_md"] = sm.write_procedure(sample_procedure)

        cs = ClaudeSkillWriter(skills_dir=tmp_path / "claude")
        results["claude"] = cs.write_procedure(sample_procedure)

        gw = GenericWriter(output_dir=tmp_path / "generic")
        results["generic"] = gw.write_procedure(sample_procedure)

        for name, path in results.items():
            assert path.exists(), f"{name} adapter did not produce a file"
            content = path.read_text()
            assert len(content) > 0, f"{name} adapter produced empty file"

    def test_v3_aware_adapters_include_environment(self, tmp_path, sample_procedure):
        """v3-aware adapters include environment/constraints in output."""
        from agenthandover_worker.openclaw_writer import OpenClawWriter
        from agenthandover_worker.skill_md_writer import SkillMdWriter
        from agenthandover_worker.claude_skill_writer import ClaudeSkillWriter
        from agenthandover_worker.generic_writer import GenericWriter

        # Enrich procedure with non-empty v3 sections
        sample_procedure["environment"]["required_apps"] = ["Chrome", "Slack"]
        sample_procedure["constraints"]["trust_level"] = "autonomous"
        sample_procedure["constraints"]["guardrails"] = ["Do not modify production data"]

        adapters = {
            "openclaw": OpenClawWriter(workspace_dir=tmp_path / "oc"),
            "skill_md": SkillMdWriter(workspace_dir=tmp_path / "sm"),
            "claude": ClaudeSkillWriter(skills_dir=tmp_path / "cs"),
            "generic": GenericWriter(output_dir=tmp_path / "gw"),
        }

        for name, adapter in adapters.items():
            path = adapter.write_procedure(sample_procedure)
            content = path.read_text()
            assert "Chrome" in content, f"{name}: missing required app in output"
            assert "autonomous" in content or "Trust level" in content, (
                f"{name}: missing trust level in output"
            )

    def test_openclaw_v3_json_sidecar(self, tmp_path, sample_procedure):
        """OpenClawWriter produces a v3 JSON sidecar alongside the markdown."""
        from agenthandover_worker.openclaw_writer import OpenClawWriter

        writer = OpenClawWriter(workspace_dir=tmp_path)
        writer.write_procedure(sample_procedure)

        slug = sample_procedure["id"]
        json_path = tmp_path / "memory" / "apprentice" / "sops" / f"sop.{slug}.v3.json"
        assert json_path.exists(), "v3 JSON sidecar not written"

        data = json.loads(json_path.read_text())
        assert data["schema_version"] in ("3.0.0", "3.1.0", "3.2.0")
        assert data["id"] == slug
        assert "environment" in data
        assert "constraints" in data
        assert "evidence" in data

    def test_generic_v3_json(self, tmp_path, sample_procedure):
        """GenericWriter produces v3 JSON alongside the markdown."""
        from agenthandover_worker.generic_writer import GenericWriter

        writer = GenericWriter(output_dir=tmp_path, json_export=True)
        writer.write_procedure(sample_procedure)

        slug = sample_procedure["id"]
        json_path = tmp_path / "sops" / f"sop.{slug}.v3.json"
        assert json_path.exists(), "v3 JSON not written"

        data = json.loads(json_path.read_text())
        assert data["schema_version"] in ("3.0.0", "3.1.0", "3.2.0")
        assert data["id"] == slug

    def test_export_via_adapter_uses_v3_when_available(self, kb, tmp_path, sample_procedure):
        """_export_via_adapter() calls write_procedure() when v3 exists in KB."""
        from agenthandover_worker.evidence_tracker import EvidenceTracker
        from agenthandover_worker.procedure_writer import ProcedureWriter
        from agenthandover_worker.generic_writer import GenericWriter
        from agenthandover_worker.main import _export_via_adapter
        from agenthandover_worker.export_adapter import procedure_to_sop_template

        et = EvidenceTracker(knowledge_base=kb)
        pw = ProcedureWriter(kb=kb, evidence=et)

        # Save the procedure to KB
        kb.save_procedure(sample_procedure)

        # Create an SOP template that matches the procedure slug
        sop_template = procedure_to_sop_template(sample_procedure)

        adapter = GenericWriter(output_dir=tmp_path)
        paths = _export_via_adapter(adapter, [sop_template], pw)

        assert len(paths) == 1
        assert paths[0].exists()

    def test_export_via_adapter_falls_back_to_sop(self, tmp_path, sample_sop_template):
        """_export_via_adapter() falls back to write_sop() when no v3 in KB."""
        from agenthandover_worker.generic_writer import GenericWriter
        from agenthandover_worker.main import _export_via_adapter

        adapter = GenericWriter(output_dir=tmp_path)
        # No procedure_writer → falls back to write_sop
        paths = _export_via_adapter(adapter, [sample_sop_template], None)

        assert len(paths) == 1
        assert paths[0].exists()
