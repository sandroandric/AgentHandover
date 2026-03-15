"""Tests for the LifecycleManager — procedure lifecycle state machine.

Covers:
- Valid transitions (7)
- Invalid transitions (5)
- Auto-transitions / demotion (3)
- No auto-promotion (3)
- History recording (3)
- Backward compatibility (2)
- Persistence (2)
- Edge cases (5)
"""

from __future__ import annotations

import pytest

from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.lifecycle_manager import (
    InvalidTransitionError,
    LifecycleManager,
    LifecycleTransition,
    ProcedureLifecycle,
)
from oc_apprentice_worker.procedure_schema import sop_to_procedure


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kb(tmp_path):
    kb = KnowledgeBase(root=tmp_path)
    kb.ensure_structure()
    return kb


@pytest.fixture
def lm(kb):
    return LifecycleManager(kb)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_procedure(kb, slug, lifecycle_state="observed", trust_level="observe", **overrides):
    """Build a minimal v3 procedure, save it, and return the dict."""
    template = {
        "slug": slug,
        "title": f"Test {slug}",
        "steps": [{"step": "Do thing", "app": "Chrome", "confidence": 0.9}],
        "confidence_avg": 0.9,
        "apps_involved": ["Chrome"],
        "source": "test",
    }
    proc = sop_to_procedure(template)
    proc["constraints"]["trust_level"] = trust_level
    proc["lifecycle_state"] = lifecycle_state
    for k, v in overrides.items():
        proc[k] = v
    kb.save_procedure(proc)
    return proc


# ===================================================================
# 1) Valid transitions (7)
# ===================================================================

class TestValidTransitions:
    """Each of the 7 valid lifecycle transitions should succeed."""

    def test_observed_to_draft(self, kb, lm):
        _save_procedure(kb, "t1", lifecycle_state="observed")
        assert lm.transition("t1", ProcedureLifecycle.DRAFT, trigger="promote", actor="user")
        assert lm.get_state("t1") == ProcedureLifecycle.DRAFT

    def test_draft_to_reviewed(self, kb, lm):
        _save_procedure(kb, "t2", lifecycle_state="draft")
        assert lm.transition("t2", ProcedureLifecycle.REVIEWED, trigger="review", actor="user")
        assert lm.get_state("t2") == ProcedureLifecycle.REVIEWED

    def test_reviewed_to_verified(self, kb, lm):
        _save_procedure(kb, "t3", lifecycle_state="reviewed")
        assert lm.transition("t3", ProcedureLifecycle.VERIFIED, trigger="verify", actor="user")
        assert lm.get_state("t3") == ProcedureLifecycle.VERIFIED

    def test_verified_to_agent_ready(self, kb, lm):
        _save_procedure(kb, "t4", lifecycle_state="verified")
        assert lm.transition("t4", ProcedureLifecycle.AGENT_READY, trigger="approve", actor="user")
        assert lm.get_state("t4") == ProcedureLifecycle.AGENT_READY

    def test_agent_ready_to_stale(self, kb, lm):
        _save_procedure(kb, "t5", lifecycle_state="agent_ready")
        assert lm.transition("t5", ProcedureLifecycle.STALE, trigger="decay", actor="system")
        assert lm.get_state("t5") == ProcedureLifecycle.STALE

    def test_stale_to_draft(self, kb, lm):
        _save_procedure(kb, "t6", lifecycle_state="stale")
        assert lm.transition("t6", ProcedureLifecycle.DRAFT, trigger="refresh", actor="user")
        assert lm.get_state("t6") == ProcedureLifecycle.DRAFT

    def test_archived_to_draft(self, kb, lm):
        _save_procedure(kb, "t7", lifecycle_state="archived")
        assert lm.transition("t7", ProcedureLifecycle.DRAFT, trigger="reopen", actor="user")
        assert lm.get_state("t7") == ProcedureLifecycle.DRAFT


# ===================================================================
# 2) Invalid transitions (5)
# ===================================================================

class TestInvalidTransitions:
    """Invalid transitions should raise InvalidTransitionError."""

    def test_observed_to_agent_ready(self, kb, lm):
        _save_procedure(kb, "bad1", lifecycle_state="observed")
        with pytest.raises(InvalidTransitionError):
            lm.transition("bad1", ProcedureLifecycle.AGENT_READY, trigger="skip")

    def test_draft_to_verified(self, kb, lm):
        _save_procedure(kb, "bad2", lifecycle_state="draft")
        with pytest.raises(InvalidTransitionError):
            lm.transition("bad2", ProcedureLifecycle.VERIFIED, trigger="skip")

    def test_observed_to_reviewed(self, kb, lm):
        _save_procedure(kb, "bad3", lifecycle_state="observed")
        with pytest.raises(InvalidTransitionError):
            lm.transition("bad3", ProcedureLifecycle.REVIEWED, trigger="skip")

    def test_archived_to_agent_ready(self, kb, lm):
        _save_procedure(kb, "bad4", lifecycle_state="archived")
        with pytest.raises(InvalidTransitionError):
            lm.transition("bad4", ProcedureLifecycle.AGENT_READY, trigger="skip")

    def test_agent_ready_to_observed(self, kb, lm):
        _save_procedure(kb, "bad5", lifecycle_state="agent_ready")
        with pytest.raises(InvalidTransitionError):
            lm.transition("bad5", ProcedureLifecycle.OBSERVED, trigger="skip")


# ===================================================================
# 3) Auto-transitions (3)
# ===================================================================

class TestAutoTransitions:
    """Auto-demotion to stale when freshness drops below threshold."""

    def test_draft_auto_demoted_when_stale(self, kb, lm):
        _save_procedure(
            kb, "auto1", lifecycle_state="draft",
            staleness={
                "last_observed": "2020-01-01T00:00:00Z",
                "last_confirmed": None,
                "drift_signals": [],
                "confidence_trend": [],
            },
        )
        applied = lm.apply_auto_transitions()
        assert len(applied) == 1
        assert applied[0][0] == "auto1"
        assert applied[0][2] == "stale"

    def test_agent_ready_auto_demoted_when_stale(self, kb, lm):
        _save_procedure(
            kb, "auto2", lifecycle_state="agent_ready",
            staleness={
                "last_observed": "2020-01-01T00:00:00Z",
                "last_confirmed": None,
                "drift_signals": [],
                "confidence_trend": [],
            },
        )
        applied = lm.apply_auto_transitions()
        assert len(applied) == 1
        assert applied[0][0] == "auto2"
        assert applied[0][2] == "stale"

    def test_reviewed_not_auto_demoted(self, kb, lm):
        """reviewed is NOT in _AUTO_STALE_STATES, so no demotion."""
        _save_procedure(
            kb, "auto3", lifecycle_state="reviewed",
            staleness={
                "last_observed": "2020-01-01T00:00:00Z",
                "last_confirmed": None,
                "drift_signals": [],
                "confidence_trend": [],
            },
        )
        applied = lm.apply_auto_transitions()
        assert len(applied) == 0
        assert lm.get_state("auto3") == ProcedureLifecycle.REVIEWED


# ===================================================================
# 4) No auto-promotion (3)
# ===================================================================

class TestNoAutoPromotion:
    """Auto-transitions should never promote, only demote to stale."""

    def test_draft_not_auto_promoted_to_reviewed(self, kb, lm):
        _save_procedure(kb, "nopro1", lifecycle_state="draft")
        proposals = lm.check_auto_transitions()
        promoted = [p for p in proposals if p[1] != ProcedureLifecycle.STALE]
        assert len(promoted) == 0

    def test_reviewed_not_auto_promoted_to_verified(self, kb, lm):
        _save_procedure(kb, "nopro2", lifecycle_state="reviewed")
        proposals = lm.check_auto_transitions()
        promoted = [p for p in proposals if p[1] != ProcedureLifecycle.STALE]
        assert len(promoted) == 0

    def test_verified_not_auto_promoted_to_agent_ready(self, kb, lm):
        _save_procedure(kb, "nopro3", lifecycle_state="verified")
        proposals = lm.check_auto_transitions()
        promoted = [p for p in proposals if p[1] != ProcedureLifecycle.STALE]
        assert len(promoted) == 0


# ===================================================================
# 5) History recording (3)
# ===================================================================

class TestHistoryRecording:
    """Transitions should record history entries."""

    def test_single_transition_records_history(self, kb, lm):
        _save_procedure(kb, "hist1", lifecycle_state="observed")
        lm.transition("hist1", ProcedureLifecycle.DRAFT, trigger="promote", actor="tester", reason="test run")
        history = lm.get_transition_history("hist1")
        assert len(history) == 1
        assert history[0].from_state == "observed"
        assert history[0].to_state == "draft"
        assert history[0].trigger == "promote"
        assert history[0].actor == "tester"
        assert history[0].reason == "test run"

    def test_multiple_transitions_append_history(self, kb, lm):
        _save_procedure(kb, "hist2", lifecycle_state="observed")
        lm.transition("hist2", ProcedureLifecycle.DRAFT, trigger="promote", actor="u1")
        lm.transition("hist2", ProcedureLifecycle.REVIEWED, trigger="review", actor="u2")
        history = lm.get_transition_history("hist2")
        assert len(history) == 2
        assert history[0].to_state == "draft"
        assert history[1].to_state == "reviewed"

    def test_history_has_timestamp(self, kb, lm):
        _save_procedure(kb, "hist3", lifecycle_state="observed")
        lm.transition("hist3", ProcedureLifecycle.DRAFT, trigger="promote")
        history = lm.get_transition_history("hist3")
        assert len(history) == 1
        assert history[0].timestamp  # non-empty
        # Should be ISO format with T separator
        assert "T" in history[0].timestamp


# ===================================================================
# 6) Backward compatibility (2)
# ===================================================================

class TestBackwardCompat:
    """Procedures without lifecycle_state should default to 'observed'."""

    def test_legacy_procedure_defaults_to_observed(self, kb, lm):
        """A procedure without lifecycle_state should be treated as 'observed'."""
        proc = sop_to_procedure({
            "slug": "legacy1",
            "title": "Legacy",
            "steps": [{"step": "Do thing", "app": "Chrome", "confidence": 0.9}],
            "confidence_avg": 0.9,
            "apps_involved": ["Chrome"],
            "source": "test",
        })
        # Remove the lifecycle_state key to simulate a legacy procedure
        proc.pop("lifecycle_state", None)
        kb.save_procedure(proc)
        assert lm.get_state("legacy1") == ProcedureLifecycle.OBSERVED

    def test_transition_from_default_works(self, kb, lm):
        """Transitioning from the default state (observed) should work."""
        proc = sop_to_procedure({
            "slug": "legacy2",
            "title": "Legacy 2",
            "steps": [{"step": "Do thing", "app": "Chrome", "confidence": 0.9}],
            "confidence_avg": 0.9,
            "apps_involved": ["Chrome"],
            "source": "test",
        })
        proc.pop("lifecycle_state", None)
        kb.save_procedure(proc)
        assert lm.transition("legacy2", ProcedureLifecycle.DRAFT, trigger="upgrade")
        assert lm.get_state("legacy2") == ProcedureLifecycle.DRAFT


# ===================================================================
# 7) Persistence (2)
# ===================================================================

class TestPersistence:
    """Lifecycle state must be persisted through save/load cycles."""

    def test_state_persisted_after_transition(self, kb, lm):
        _save_procedure(kb, "persist1", lifecycle_state="observed")
        lm.transition("persist1", ProcedureLifecycle.DRAFT, trigger="promote")
        # Re-read from disk
        proc = kb.get_procedure("persist1")
        assert proc["lifecycle_state"] == "draft"

    def test_history_persisted_after_transition(self, kb, lm):
        _save_procedure(kb, "persist2", lifecycle_state="observed")
        lm.transition("persist2", ProcedureLifecycle.DRAFT, trigger="promote")
        # Re-read from disk
        proc = kb.get_procedure("persist2")
        assert len(proc.get("lifecycle_history", [])) == 1
        assert proc["lifecycle_history"][0]["to_state"] == "draft"


# ===================================================================
# 8) Edge cases (5)
# ===================================================================

class TestEdgeCases:
    """Miscellaneous edge-case coverage."""

    def test_transition_nonexistent_procedure_returns_false(self, kb, lm):
        result = lm.transition("ghost", ProcedureLifecycle.DRAFT, trigger="test")
        assert result is False

    def test_get_state_nonexistent_returns_observed(self, kb, lm):
        assert lm.get_state("no-such-slug") == ProcedureLifecycle.OBSERVED

    def test_can_transition_checks_without_mutation(self, kb, lm):
        _save_procedure(kb, "check1", lifecycle_state="observed")
        assert lm.can_transition("check1", ProcedureLifecycle.DRAFT) is True
        assert lm.can_transition("check1", ProcedureLifecycle.AGENT_READY) is False
        # State unchanged
        assert lm.get_state("check1") == ProcedureLifecycle.OBSERVED

    def test_get_transition_history_empty_for_new_procedure(self, kb, lm):
        _save_procedure(kb, "newhist", lifecycle_state="observed")
        history = lm.get_transition_history("newhist")
        assert history == []

    def test_get_transition_history_nonexistent_returns_empty(self, kb, lm):
        history = lm.get_transition_history("no-exist")
        assert history == []
