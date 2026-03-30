"""Tests for the QualityJudge module and its integration with ProcedureCurator."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from agenthandover_worker.llm_reasoning import LLMReasoner, ReasoningConfig
from agenthandover_worker.quality_judge import QualityAssessment, QualityJudge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_procedure(
    slug: str = "test-proc",
    title: str = "Test Procedure",
    steps: list[dict] | None = None,
    confidence: float = 0.85,
    episodes: int = 5,
) -> dict:
    """Build a minimal procedure dict for testing."""
    if steps is None:
        steps = [
            {"step": "Open Chrome", "app": "Chrome", "confidence": 0.9},
            {"step": "Navigate to site", "app": "Chrome", "confidence": 0.8},
            {"step": "Submit form", "app": "Chrome", "confidence": 0.85},
        ]
    return {
        "id": slug,
        "title": title,
        "steps": steps,
        "confidence_avg": confidence,
        "episode_count": episodes,
        "evidence": {
            "observations": [{"date": "2025-03-01"}, {"date": "2025-03-02"}],
            "contradictions": [],
        },
        "staleness": {},
    }


# ---------------------------------------------------------------------------
# Tests: QualityJudge
# ---------------------------------------------------------------------------


class TestQualityJudge:
    def test_assess_passes_good_procedure(self) -> None:
        """LLM returns a passing assessment for a good procedure."""
        reasoner = LLMReasoner()
        judge = QualityJudge(reasoner)

        response_json = json.dumps({
            "passed": True,
            "score": 0.92,
            "reasons": [
                "All steps are documented",
                "Steps match observed evidence",
                "Agent can execute this workflow",
            ],
            "gaps": [],
        })

        def mock_ollama(prompt, system, num_predict=None, **kwargs):
            return (response_json, 1.2)

        proc = _make_procedure()

        with patch.object(LLMReasoner, "_call_ollama", side_effect=mock_ollama):
            result = judge.assess(proc, "reviewed")

        assert result.passed is True
        assert result.score == pytest.approx(0.92)
        assert len(result.reasons) == 3
        assert result.gaps == []
        assert result.abstained is False
        assert "model" in result.provenance

    def test_assess_fails_with_gaps(self) -> None:
        """LLM returns a failing assessment with identified gaps."""
        reasoner = LLMReasoner()
        judge = QualityJudge(reasoner)

        response_json = json.dumps({
            "passed": False,
            "score": 0.35,
            "reasons": ["Steps are incomplete"],
            "gaps": [
                "Missing login step before form submission",
                "No error handling for failed submission",
            ],
        })

        def mock_ollama(prompt, system, num_predict=None, **kwargs):
            return (response_json, 0.8)

        proc = _make_procedure()

        with patch.object(LLMReasoner, "_call_ollama", side_effect=mock_ollama):
            result = judge.assess(proc, "reviewed")

        assert result.passed is False
        assert result.score == pytest.approx(0.35)
        assert len(result.gaps) == 2
        assert "Missing login step" in result.gaps[0]
        assert result.abstained is False

    def test_assess_abstains_proceeds(self) -> None:
        """When LLM abstains, assessment returns passed=True with abstained=True."""
        reasoner = LLMReasoner()
        judge = QualityJudge(reasoner)

        # Abstention via INSUFFICIENT_EVIDENCE marker in raw response
        def mock_ollama(prompt, system, num_predict=None, **kwargs):
            return ("INSUFFICIENT_EVIDENCE: not enough data to assess", 0.5)

        proc = _make_procedure()

        with patch.object(LLMReasoner, "_call_ollama", side_effect=mock_ollama):
            result = judge.assess(proc, "reviewed")

        assert result.passed is True
        assert result.abstained is True
        assert result.gaps == []

    def test_over_budget_defaults_to_pass(self) -> None:
        """When LLM is over budget, returns a default-pass assessment."""
        from unittest.mock import MagicMock

        # Create a mock VLM queue that says "over budget"
        mock_queue = MagicMock()
        mock_queue.can_dispatch.return_value = False

        reasoner = LLMReasoner(vlm_queue=mock_queue)
        judge = QualityJudge(reasoner)

        proc = _make_procedure()
        result = judge.assess(proc, "reviewed")

        assert result.passed is True
        assert "Default pass" in result.reasons[0]
        # _call_ollama should NOT have been called
        assert result.provenance == {}

    def test_agent_ready_blocked_when_over_budget(self) -> None:
        """Promotion to agent_ready is BLOCKED when LLM is over budget."""
        from unittest.mock import MagicMock

        mock_queue = MagicMock()
        mock_queue.can_dispatch.return_value = False

        reasoner = LLMReasoner(vlm_queue=mock_queue)
        judge = QualityJudge(reasoner)

        proc = _make_procedure()
        result = judge.assess(proc, "agent_ready")

        assert result.passed is False
        assert "agent_ready" in result.reasons[0].lower()

    def test_agent_ready_blocked_on_structured_abstention(self) -> None:
        """Promotion to agent_ready is BLOCKED when LLM abstains via JSON."""
        reasoner = LLMReasoner()
        judge = QualityJudge(reasoner)

        # LLM returns structured JSON abstention (not the raw marker)
        response_json = json.dumps({
            "abstained": True,
            "reason": "Not enough evidence to evaluate",
        })

        def mock_ollama(prompt, system, num_predict=None, **kwargs):
            return (response_json, 1.0)

        proc = _make_procedure()

        with patch.object(LLMReasoner, "_call_ollama", side_effect=mock_ollama):
            result = judge.assess(proc, "agent_ready")

        assert result.passed is False
        assert result.abstained is True
        assert "agent_ready" in result.reasons[0].lower()

    def test_agent_ready_blocked_on_raw_abstention(self) -> None:
        """Promotion to agent_ready is BLOCKED when LLM returns INSUFFICIENT_EVIDENCE."""
        reasoner = LLMReasoner()
        judge = QualityJudge(reasoner)

        def mock_ollama(prompt, system, num_predict=None, **kwargs):
            return ("INSUFFICIENT_EVIDENCE: not enough data", 1.0)

        proc = _make_procedure()

        with patch.object(LLMReasoner, "_call_ollama", side_effect=mock_ollama):
            result = judge.assess(proc, "agent_ready")

        assert result.passed is False
        assert result.abstained is True

    def test_low_stakes_passes_when_over_budget(self) -> None:
        """Promotion to 'reviewed' still passes when LLM is over budget."""
        from unittest.mock import MagicMock

        mock_queue = MagicMock()
        mock_queue.can_dispatch.return_value = False

        reasoner = LLMReasoner(vlm_queue=mock_queue)
        judge = QualityJudge(reasoner)

        proc = _make_procedure()
        result = judge.assess(proc, "reviewed")

        assert result.passed is True


# ---------------------------------------------------------------------------
# Integration: QualityJudge + ProcedureCurator
# ---------------------------------------------------------------------------


class TestCuratorQualityGate:
    def test_curator_blocks_promotion_on_failure(self, tmp_path) -> None:
        """ProcedureCurator.execute_promote blocks when QualityJudge fails."""
        from agenthandover_worker.knowledge_base import KnowledgeBase
        from agenthandover_worker.lifecycle_manager import LifecycleManager
        from agenthandover_worker.procedure_curator import ProcedureCurator
        from agenthandover_worker.staleness_detector import StalenessDetector
        from agenthandover_worker.trust_advisor import TrustAdvisor
        from agenthandover_worker.procedure_schema import sop_to_procedure

        kb = KnowledgeBase(root=tmp_path)
        kb.ensure_structure()

        # Create and save a procedure in "observed" state
        template = {
            "slug": "block-me",
            "title": "Block Me Proc",
            "steps": [
                {"step": "Open site", "app": "Chrome", "confidence": 0.9, "location": ""},
            ],
            "confidence_avg": 0.80,
            "episode_count": 5,
            "apps_involved": ["Chrome"],
            "source": "test",
        }
        proc = sop_to_procedure(template)
        proc["lifecycle_state"] = "observed"
        kb.save_procedure(proc)

        # Create a QualityJudge that will fail
        reasoner = LLMReasoner()
        judge = QualityJudge(reasoner)

        response_json = json.dumps({
            "passed": False,
            "score": 0.25,
            "reasons": ["Procedure is incomplete"],
            "gaps": ["Missing authentication step", "No error handling"],
        })

        def mock_ollama(prompt, system, num_predict=None, **kwargs):
            return (response_json, 0.5)

        sd = StalenessDetector(kb)
        ta = TrustAdvisor(kb)
        lm = LifecycleManager(kb)
        curator = ProcedureCurator(kb, sd, ta, lm, quality_judge=judge)

        with patch.object(LLMReasoner, "_call_ollama", side_effect=mock_ollama):
            result = curator.execute_promote("block-me", "draft")

        assert result["success"] is False
        assert "Quality assessment failed" in result["error"]
        assert len(result["gaps"]) == 2
        assert "Missing authentication step" in result["gaps"][0]

        # Verify the procedure was NOT promoted
        proc_after = kb.get_procedure("block-me")
        assert proc_after["lifecycle_state"] == "observed"

    def test_curator_proceeds_without_judge(self, tmp_path) -> None:
        """Without a quality_judge, execute_promote works as before."""
        from agenthandover_worker.knowledge_base import KnowledgeBase
        from agenthandover_worker.lifecycle_manager import LifecycleManager
        from agenthandover_worker.procedure_curator import ProcedureCurator
        from agenthandover_worker.staleness_detector import StalenessDetector
        from agenthandover_worker.trust_advisor import TrustAdvisor
        from agenthandover_worker.procedure_schema import sop_to_procedure

        kb = KnowledgeBase(root=tmp_path)
        kb.ensure_structure()

        template = {
            "slug": "pass-me",
            "title": "Pass Me Proc",
            "steps": [
                {"step": "Open site", "app": "Chrome", "confidence": 0.9, "location": ""},
            ],
            "confidence_avg": 0.80,
            "episode_count": 5,
            "apps_involved": ["Chrome"],
            "source": "test",
        }
        proc = sop_to_procedure(template)
        proc["lifecycle_state"] = "observed"
        kb.save_procedure(proc)

        sd = StalenessDetector(kb)
        ta = TrustAdvisor(kb)
        lm = LifecycleManager(kb)
        curator = ProcedureCurator(kb, sd, ta, lm)  # No quality_judge

        result = curator.execute_promote("pass-me", "draft")
        assert result["success"] is True
        assert result["new_state"] == "draft"
