"""Tests for the TrustAdvisor module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.trust_advisor import (
    TRUST_LEVELS,
    TrustAdvisor,
    TrustSuggestion,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    """Create a KnowledgeBase in a temp directory."""
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    return kb


def _save_procedure(kb: KnowledgeBase, slug: str, trust_level: str = "observe") -> None:
    """Helper: save a minimal procedure with a given trust level."""
    kb.save_procedure({
        "id": slug,
        "constraints": {"trust_level": trust_level},
        "steps": [],
    })


def _save_executions(kb: KnowledgeBase, stats: dict) -> None:
    """Helper: write execution stats to observations/executions.json."""
    path = kb.root / "observations" / "executions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"procedures": stats}, f)


# ---------------------------------------------------------------------------
# Tests: _next_level
# ---------------------------------------------------------------------------


class TestNextLevel:
    def test_observe_to_suggest(self, kb: KnowledgeBase) -> None:
        advisor = TrustAdvisor(kb)
        assert advisor._next_level("observe") == "suggest"

    def test_suggest_to_draft(self, kb: KnowledgeBase) -> None:
        advisor = TrustAdvisor(kb)
        assert advisor._next_level("suggest") == "draft"

    def test_draft_to_execute_with_approval(self, kb: KnowledgeBase) -> None:
        advisor = TrustAdvisor(kb)
        assert advisor._next_level("draft") == "execute_with_approval"

    def test_execute_with_approval_to_autonomous(self, kb: KnowledgeBase) -> None:
        advisor = TrustAdvisor(kb)
        assert advisor._next_level("execute_with_approval") == "autonomous"

    def test_autonomous_returns_none(self, kb: KnowledgeBase) -> None:
        advisor = TrustAdvisor(kb)
        assert advisor._next_level("autonomous") is None

    def test_unknown_level_treated_as_observe(self, kb: KnowledgeBase) -> None:
        advisor = TrustAdvisor(kb)
        assert advisor._next_level("bogus") == "suggest"


# ---------------------------------------------------------------------------
# Tests: evaluate_procedure
# ---------------------------------------------------------------------------


class TestEvaluateProcedure:
    def test_sufficient_success_creates_suggestion(self, kb: KnowledgeBase) -> None:
        _save_procedure(kb, "deploy-api", "observe")
        _save_executions(kb, {
            "deploy-api": {"total": 5, "successes": 5, "failures": 0, "last_failure": None},
        })
        advisor = TrustAdvisor(kb, min_observations=3, min_success_rate=0.90)
        suggestion = advisor.evaluate_procedure("deploy-api")
        assert suggestion is not None
        assert suggestion.procedure_slug == "deploy-api"
        assert suggestion.current_level == "observe"
        assert suggestion.suggested_level == "suggest"
        assert suggestion.evidence["observations"] == 5
        assert suggestion.evidence["success_rate"] == 1.0

    def test_insufficient_observations_no_suggestion(self, kb: KnowledgeBase) -> None:
        _save_procedure(kb, "deploy-api", "observe")
        _save_executions(kb, {
            "deploy-api": {"total": 2, "successes": 2, "failures": 0},
        })
        advisor = TrustAdvisor(kb, min_observations=3)
        assert advisor.evaluate_procedure("deploy-api") is None

    def test_low_success_rate_no_suggestion(self, kb: KnowledgeBase) -> None:
        _save_procedure(kb, "deploy-api", "observe")
        _save_executions(kb, {
            "deploy-api": {"total": 10, "successes": 5, "failures": 5},
        })
        advisor = TrustAdvisor(kb, min_observations=3, min_success_rate=0.90)
        assert advisor.evaluate_procedure("deploy-api") is None

    def test_already_at_max_level_no_suggestion(self, kb: KnowledgeBase) -> None:
        _save_procedure(kb, "deploy-api", "autonomous")
        _save_executions(kb, {
            "deploy-api": {"total": 10, "successes": 10, "failures": 0},
        })
        advisor = TrustAdvisor(kb)
        assert advisor.evaluate_procedure("deploy-api") is None

    def test_no_execution_stats_no_suggestion(self, kb: KnowledgeBase) -> None:
        _save_procedure(kb, "deploy-api", "observe")
        # No executions.json at all
        advisor = TrustAdvisor(kb)
        assert advisor.evaluate_procedure("deploy-api") is None

    def test_missing_procedure_no_crash(self, kb: KnowledgeBase) -> None:
        advisor = TrustAdvisor(kb)
        # No procedure saved — should not crash, just no suggestion
        assert advisor.evaluate_procedure("nonexistent") is None

    def test_suggest_level_promotes_to_draft(self, kb: KnowledgeBase) -> None:
        _save_procedure(kb, "email-send", "suggest")
        _save_executions(kb, {
            "email-send": {"total": 5, "successes": 5, "failures": 0},
        })
        advisor = TrustAdvisor(kb, min_observations=3, min_success_rate=0.90)
        suggestion = advisor.evaluate_procedure("email-send")
        assert suggestion is not None
        assert suggestion.suggested_level == "draft"


# ---------------------------------------------------------------------------
# Tests: no duplicate suggestions
# ---------------------------------------------------------------------------


class TestNoDuplicates:
    def test_pending_suggestion_prevents_new_one(self, kb: KnowledgeBase) -> None:
        _save_procedure(kb, "deploy-api", "observe")
        _save_executions(kb, {
            "deploy-api": {"total": 5, "successes": 5, "failures": 0},
        })
        advisor = TrustAdvisor(kb, min_observations=3)
        s1 = advisor.evaluate_procedure("deploy-api")
        assert s1 is not None
        # Second evaluation should return None — pending suggestion exists
        s2 = advisor.evaluate_procedure("deploy-api")
        assert s2 is None

    def test_dismissed_suggestion_allows_new_one(self, kb: KnowledgeBase) -> None:
        _save_procedure(kb, "deploy-api", "observe")
        _save_executions(kb, {
            "deploy-api": {"total": 5, "successes": 5, "failures": 0},
        })
        advisor = TrustAdvisor(kb, min_observations=3)
        advisor.evaluate_procedure("deploy-api")
        advisor.dismiss_suggestion("deploy-api")
        # Now a new suggestion should be possible
        s2 = advisor.evaluate_procedure("deploy-api")
        assert s2 is not None


# ---------------------------------------------------------------------------
# Tests: accept / dismiss
# ---------------------------------------------------------------------------


class TestAcceptDismiss:
    def test_accept_applies_trust_level(self, kb: KnowledgeBase) -> None:
        _save_procedure(kb, "deploy-api", "observe")
        _save_executions(kb, {
            "deploy-api": {"total": 5, "successes": 5, "failures": 0},
        })
        advisor = TrustAdvisor(kb, min_observations=3)
        advisor.evaluate_procedure("deploy-api")
        result = advisor.accept_suggestion("deploy-api")
        assert result is True

        # Verify the trust level was updated in the KB
        proc = kb.get_procedure("deploy-api")
        assert proc is not None
        assert proc["constraints"]["trust_level"] == "suggest"

    def test_accept_nonexistent_returns_false(self, kb: KnowledgeBase) -> None:
        advisor = TrustAdvisor(kb)
        assert advisor.accept_suggestion("nonexistent") is False

    def test_dismiss_marks_as_dismissed(self, kb: KnowledgeBase) -> None:
        _save_procedure(kb, "deploy-api", "observe")
        _save_executions(kb, {
            "deploy-api": {"total": 5, "successes": 5, "failures": 0},
        })
        advisor = TrustAdvisor(kb, min_observations=3)
        advisor.evaluate_procedure("deploy-api")
        result = advisor.dismiss_suggestion("deploy-api")
        assert result is True

        # Should not appear in non-dismissed list
        active = advisor.get_suggestions(include_dismissed=False)
        assert len(active) == 0

        # But should appear in full list
        all_s = advisor.get_suggestions(include_dismissed=True)
        assert len(all_s) == 1
        assert all_s[0].dismissed is True

    def test_dismiss_nonexistent_returns_false(self, kb: KnowledgeBase) -> None:
        advisor = TrustAdvisor(kb)
        assert advisor.dismiss_suggestion("nonexistent") is False

    def test_cannot_accept_dismissed_suggestion(self, kb: KnowledgeBase) -> None:
        _save_procedure(kb, "deploy-api", "observe")
        _save_executions(kb, {
            "deploy-api": {"total": 5, "successes": 5, "failures": 0},
        })
        advisor = TrustAdvisor(kb, min_observations=3)
        advisor.evaluate_procedure("deploy-api")
        advisor.dismiss_suggestion("deploy-api")
        assert advisor.accept_suggestion("deploy-api") is False


# ---------------------------------------------------------------------------
# Tests: evaluate_all
# ---------------------------------------------------------------------------


class TestEvaluateAll:
    def test_evaluates_multiple_procedures(self, kb: KnowledgeBase) -> None:
        _save_procedure(kb, "deploy-api", "observe")
        _save_procedure(kb, "send-report", "observe")
        _save_executions(kb, {
            "deploy-api": {"total": 5, "successes": 5, "failures": 0},
            "send-report": {"total": 4, "successes": 4, "failures": 0},
        })
        advisor = TrustAdvisor(kb, min_observations=3)
        results = advisor.evaluate_all()
        assert len(results) == 2
        slugs = {r.procedure_slug for r in results}
        assert slugs == {"deploy-api", "send-report"}

    def test_evaluate_all_skips_ineligible(self, kb: KnowledgeBase) -> None:
        _save_procedure(kb, "deploy-api", "observe")
        _save_procedure(kb, "not-enough", "observe")
        _save_executions(kb, {
            "deploy-api": {"total": 5, "successes": 5, "failures": 0},
            "not-enough": {"total": 1, "successes": 1, "failures": 0},
        })
        advisor = TrustAdvisor(kb, min_observations=3)
        results = advisor.evaluate_all()
        assert len(results) == 1
        assert results[0].procedure_slug == "deploy-api"


# ---------------------------------------------------------------------------
# Tests: get_suggestions
# ---------------------------------------------------------------------------


class TestGetSuggestions:
    def test_get_pending_only(self, kb: KnowledgeBase) -> None:
        _save_procedure(kb, "a", "observe")
        _save_procedure(kb, "b", "observe")
        _save_executions(kb, {
            "a": {"total": 5, "successes": 5, "failures": 0},
            "b": {"total": 5, "successes": 5, "failures": 0},
        })
        advisor = TrustAdvisor(kb, min_observations=3)
        advisor.evaluate_all()
        advisor.dismiss_suggestion("a")
        pending = advisor.get_suggestions(include_dismissed=False)
        assert len(pending) == 1
        assert pending[0].procedure_slug == "b"

    def test_get_all_including_dismissed(self, kb: KnowledgeBase) -> None:
        _save_procedure(kb, "a", "observe")
        _save_executions(kb, {
            "a": {"total": 5, "successes": 5, "failures": 0},
        })
        advisor = TrustAdvisor(kb, min_observations=3)
        advisor.evaluate_all()
        advisor.dismiss_suggestion("a")
        all_s = advisor.get_suggestions(include_dismissed=True)
        assert len(all_s) == 1


# ---------------------------------------------------------------------------
# Tests: persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_suggestions_survive_reload(self, kb: KnowledgeBase) -> None:
        _save_procedure(kb, "deploy-api", "observe")
        _save_executions(kb, {
            "deploy-api": {"total": 5, "successes": 5, "failures": 0},
        })
        advisor1 = TrustAdvisor(kb, min_observations=3)
        advisor1.evaluate_procedure("deploy-api")
        assert len(advisor1.get_suggestions()) == 1

        # Create a new TrustAdvisor pointing to the same KB
        advisor2 = TrustAdvisor(kb, min_observations=3)
        loaded = advisor2.get_suggestions()
        assert len(loaded) == 1
        assert loaded[0].procedure_slug == "deploy-api"
        assert loaded[0].suggested_level == "suggest"

    def test_accepted_state_persists(self, kb: KnowledgeBase) -> None:
        _save_procedure(kb, "deploy-api", "observe")
        _save_executions(kb, {
            "deploy-api": {"total": 5, "successes": 5, "failures": 0},
        })
        advisor1 = TrustAdvisor(kb, min_observations=3)
        advisor1.evaluate_procedure("deploy-api")
        advisor1.accept_suggestion("deploy-api")

        advisor2 = TrustAdvisor(kb, min_observations=3)
        all_s = advisor2.get_suggestions(include_dismissed=True)
        assert len(all_s) == 1
        assert all_s[0].accepted is True

    def test_empty_kb_loads_no_suggestions(self, kb: KnowledgeBase) -> None:
        advisor = TrustAdvisor(kb)
        assert advisor.get_suggestions() == []


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_evidence_includes_last_failure(self, kb: KnowledgeBase) -> None:
        _save_procedure(kb, "deploy-api", "observe")
        _save_executions(kb, {
            "deploy-api": {
                "total": 10,
                "successes": 9,
                "failures": 1,
                "last_failure": "2025-01-15T10:00:00Z",
            },
        })
        advisor = TrustAdvisor(kb, min_observations=3, min_success_rate=0.90)
        suggestion = advisor.evaluate_procedure("deploy-api")
        assert suggestion is not None
        assert suggestion.evidence["last_failure"] == "2025-01-15T10:00:00Z"

    def test_exact_threshold_success_rate(self, kb: KnowledgeBase) -> None:
        _save_procedure(kb, "deploy-api", "observe")
        _save_executions(kb, {
            "deploy-api": {"total": 10, "successes": 9, "failures": 1},
        })
        advisor = TrustAdvisor(kb, min_observations=3, min_success_rate=0.90)
        suggestion = advisor.evaluate_procedure("deploy-api")
        assert suggestion is not None  # 0.90 == 0.90, should pass

    def test_just_below_threshold_no_suggestion(self, kb: KnowledgeBase) -> None:
        _save_procedure(kb, "deploy-api", "observe")
        _save_executions(kb, {
            "deploy-api": {"total": 10, "successes": 8, "failures": 2},
        })
        advisor = TrustAdvisor(kb, min_observations=3, min_success_rate=0.90)
        assert advisor.evaluate_procedure("deploy-api") is None
