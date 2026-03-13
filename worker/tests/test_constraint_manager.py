"""Tests for the constraint manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from oc_apprentice_worker.constraint_manager import (
    ConstraintManager,
    TrustLevel,
)
from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.procedure_schema import sop_to_procedure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_procedure(slug: str = "test-proc") -> dict:
    return sop_to_procedure({
        "slug": slug,
        "title": "Test Procedure",
        "steps": [
            {"action": "Open browser", "confidence": 0.9},
            {"action": "Click button", "confidence": 0.85},
        ],
        "confidence_avg": 0.87,
        "apps_involved": ["Chrome"],
        "source": "passive",
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    return kb


@pytest.fixture()
def cm(kb: KnowledgeBase) -> ConstraintManager:
    return ConstraintManager(kb)


# ---------------------------------------------------------------------------
# TrustLevel enum
# ---------------------------------------------------------------------------

class TestTrustLevel:

    def test_all_values(self) -> None:
        assert TrustLevel.OBSERVE.value == "observe"
        assert TrustLevel.SUGGEST.value == "suggest"
        assert TrustLevel.DRAFT.value == "draft"
        assert TrustLevel.EXECUTE_WITH_APPROVAL.value == "execute_with_approval"
        assert TrustLevel.AUTONOMOUS.value == "autonomous"

    def test_from_string_valid(self) -> None:
        assert TrustLevel.from_string("observe") == TrustLevel.OBSERVE
        assert TrustLevel.from_string("suggest") == TrustLevel.SUGGEST
        assert TrustLevel.from_string("draft") == TrustLevel.DRAFT
        assert TrustLevel.from_string("execute_with_approval") == TrustLevel.EXECUTE_WITH_APPROVAL
        assert TrustLevel.from_string("autonomous") == TrustLevel.AUTONOMOUS

    def test_from_string_unknown_returns_observe(self) -> None:
        assert TrustLevel.from_string("superuser") == TrustLevel.OBSERVE
        assert TrustLevel.from_string("") == TrustLevel.OBSERVE
        assert TrustLevel.from_string("OBSERVE") == TrustLevel.OBSERVE  # case-sensitive


# ---------------------------------------------------------------------------
# Trust level — get / set
# ---------------------------------------------------------------------------

class TestGetSetTrustLevel:

    def test_default_trust_level_is_observe(
        self, kb: KnowledgeBase, cm: ConstraintManager
    ) -> None:
        proc = _make_procedure()
        kb.save_procedure(proc)
        assert cm.get_trust_level("test-proc") == TrustLevel.OBSERVE

    def test_nonexistent_procedure_returns_observe(
        self, cm: ConstraintManager
    ) -> None:
        assert cm.get_trust_level("missing") == TrustLevel.OBSERVE

    def test_set_and_get_trust_level(
        self, kb: KnowledgeBase, cm: ConstraintManager
    ) -> None:
        proc = _make_procedure()
        kb.save_procedure(proc)

        cm.set_trust_level("test-proc", TrustLevel.DRAFT)
        assert cm.get_trust_level("test-proc") == TrustLevel.DRAFT

    def test_set_trust_level_autonomous(
        self, kb: KnowledgeBase, cm: ConstraintManager
    ) -> None:
        proc = _make_procedure()
        kb.save_procedure(proc)

        cm.set_trust_level("test-proc", TrustLevel.AUTONOMOUS)
        assert cm.get_trust_level("test-proc") == TrustLevel.AUTONOMOUS

    def test_set_trust_level_overwrites_previous(
        self, kb: KnowledgeBase, cm: ConstraintManager
    ) -> None:
        proc = _make_procedure()
        kb.save_procedure(proc)

        cm.set_trust_level("test-proc", TrustLevel.SUGGEST)
        cm.set_trust_level("test-proc", TrustLevel.EXECUTE_WITH_APPROVAL)
        assert cm.get_trust_level("test-proc") == TrustLevel.EXECUTE_WITH_APPROVAL

    def test_set_trust_nonexistent_is_noop(
        self, cm: ConstraintManager
    ) -> None:
        # Should not raise.
        cm.set_trust_level("nonexistent", TrustLevel.AUTONOMOUS)

    def test_trust_level_persists_in_procedure(
        self, kb: KnowledgeBase, cm: ConstraintManager
    ) -> None:
        proc = _make_procedure()
        kb.save_procedure(proc)

        cm.set_trust_level("test-proc", TrustLevel.DRAFT)
        loaded = kb.get_procedure("test-proc")
        assert loaded["constraints"]["trust_level"] == "draft"


# ---------------------------------------------------------------------------
# check_execution_allowed
# ---------------------------------------------------------------------------

class TestCheckExecutionAllowed:

    def test_autonomous_allowed(
        self, kb: KnowledgeBase, cm: ConstraintManager
    ) -> None:
        proc = _make_procedure()
        kb.save_procedure(proc)
        cm.set_trust_level("test-proc", TrustLevel.AUTONOMOUS)

        allowed, reason = cm.check_execution_allowed("test-proc")
        assert allowed is True
        assert "autonomous" in reason

    def test_execute_with_approval_not_allowed(
        self, kb: KnowledgeBase, cm: ConstraintManager
    ) -> None:
        proc = _make_procedure()
        kb.save_procedure(proc)
        cm.set_trust_level("test-proc", TrustLevel.EXECUTE_WITH_APPROVAL)

        allowed, reason = cm.check_execution_allowed("test-proc")
        assert allowed is False
        assert "approval" in reason

    def test_draft_not_allowed(
        self, kb: KnowledgeBase, cm: ConstraintManager
    ) -> None:
        proc = _make_procedure()
        kb.save_procedure(proc)
        cm.set_trust_level("test-proc", TrustLevel.DRAFT)

        allowed, reason = cm.check_execution_allowed("test-proc")
        assert allowed is False
        assert "draft" in reason

    def test_suggest_not_allowed(
        self, kb: KnowledgeBase, cm: ConstraintManager
    ) -> None:
        proc = _make_procedure()
        kb.save_procedure(proc)
        cm.set_trust_level("test-proc", TrustLevel.SUGGEST)

        allowed, reason = cm.check_execution_allowed("test-proc")
        assert allowed is False
        assert "suggest" in reason

    def test_observe_not_allowed(
        self, kb: KnowledgeBase, cm: ConstraintManager
    ) -> None:
        proc = _make_procedure()
        kb.save_procedure(proc)

        allowed, reason = cm.check_execution_allowed("test-proc")
        assert allowed is False
        assert "observe" in reason

    def test_nonexistent_procedure_not_allowed(
        self, cm: ConstraintManager
    ) -> None:
        allowed, reason = cm.check_execution_allowed("missing")
        assert allowed is False
        assert "observe" in reason


# ---------------------------------------------------------------------------
# Global constraints
# ---------------------------------------------------------------------------

class TestGlobalConstraints:

    def test_get_empty_global_constraints(
        self, cm: ConstraintManager
    ) -> None:
        result = cm.get_constraints(slug=None)
        assert result == {}

    def test_set_and_get_global_constraint(
        self, cm: ConstraintManager
    ) -> None:
        cm.set_constraint(None, "max_spend_usd", 100)
        result = cm.get_constraints(slug=None)
        assert result["max_spend_usd"] == 100

    def test_multiple_global_constraints(
        self, cm: ConstraintManager
    ) -> None:
        cm.set_constraint(None, "max_spend_usd", 100)
        cm.set_constraint(None, "max_daily_actions", 50)
        result = cm.get_constraints(slug=None)
        assert result["max_spend_usd"] == 100
        assert result["max_daily_actions"] == 50

    def test_global_constraint_overwrite(
        self, cm: ConstraintManager
    ) -> None:
        cm.set_constraint(None, "max_spend_usd", 100)
        cm.set_constraint(None, "max_spend_usd", 200)
        result = cm.get_constraints(slug=None)
        assert result["max_spend_usd"] == 200


# ---------------------------------------------------------------------------
# Per-procedure constraints
# ---------------------------------------------------------------------------

class TestPerProcedureConstraints:

    def test_set_and_get_per_procedure_constraint(
        self, cm: ConstraintManager
    ) -> None:
        cm.set_constraint("my-proc", "timeout_seconds", 30)
        # Direct global check — per-procedure should not pollute global.
        global_c = cm.get_constraints(slug=None)
        assert "timeout_seconds" not in global_c

    def test_per_procedure_constraint_available_via_slug(
        self, cm: ConstraintManager
    ) -> None:
        cm.set_constraint("my-proc", "timeout_seconds", 30)
        result = cm.get_constraints(slug="my-proc")
        assert result["timeout_seconds"] == 30

    def test_per_procedure_does_not_leak_to_other_procedures(
        self, cm: ConstraintManager
    ) -> None:
        cm.set_constraint("proc-a", "foo", "bar")
        result = cm.get_constraints(slug="proc-b")
        assert "foo" not in result


# ---------------------------------------------------------------------------
# Merged constraints (per-proc overrides global)
# ---------------------------------------------------------------------------

class TestMergedConstraints:

    def test_global_visible_through_slug(
        self, cm: ConstraintManager
    ) -> None:
        cm.set_constraint(None, "max_spend_usd", 100)
        result = cm.get_constraints(slug="any-proc")
        assert result["max_spend_usd"] == 100

    def test_per_proc_overrides_global(
        self, cm: ConstraintManager
    ) -> None:
        cm.set_constraint(None, "max_spend_usd", 100)
        cm.set_constraint("special-proc", "max_spend_usd", 500)
        result = cm.get_constraints(slug="special-proc")
        assert result["max_spend_usd"] == 500

    def test_merge_keeps_non_overridden_global(
        self, cm: ConstraintManager
    ) -> None:
        cm.set_constraint(None, "max_spend_usd", 100)
        cm.set_constraint(None, "max_daily_actions", 50)
        cm.set_constraint("proc", "max_spend_usd", 500)

        result = cm.get_constraints(slug="proc")
        assert result["max_spend_usd"] == 500  # overridden
        assert result["max_daily_actions"] == 50  # inherited from global


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

class TestGuardrails:

    def test_get_guardrails_empty(
        self, kb: KnowledgeBase, cm: ConstraintManager
    ) -> None:
        proc = _make_procedure()
        kb.save_procedure(proc)
        assert cm.get_guardrails("test-proc") == []

    def test_get_guardrails_nonexistent_procedure(
        self, cm: ConstraintManager
    ) -> None:
        assert cm.get_guardrails("missing") == []

    def test_add_guardrail(
        self, kb: KnowledgeBase, cm: ConstraintManager
    ) -> None:
        proc = _make_procedure()
        kb.save_procedure(proc)

        cm.add_guardrail("test-proc", {
            "type": "spending_limit",
            "max_usd": 50,
            "description": "Do not spend more than $50",
        })

        rails = cm.get_guardrails("test-proc")
        assert len(rails) == 1
        assert rails[0]["type"] == "spending_limit"
        assert rails[0]["max_usd"] == 50

    def test_add_multiple_guardrails(
        self, kb: KnowledgeBase, cm: ConstraintManager
    ) -> None:
        proc = _make_procedure()
        kb.save_procedure(proc)

        cm.add_guardrail("test-proc", {"type": "rate_limit", "max_per_hour": 10})
        cm.add_guardrail("test-proc", {"type": "no_delete", "description": "Never delete"})

        rails = cm.get_guardrails("test-proc")
        assert len(rails) == 2
        types = {r["type"] for r in rails}
        assert types == {"rate_limit", "no_delete"}

    def test_add_guardrail_nonexistent_procedure_is_noop(
        self, cm: ConstraintManager
    ) -> None:
        # Should not raise.
        cm.add_guardrail("missing", {"type": "anything"})

    def test_guardrails_persist_across_reads(
        self, kb: KnowledgeBase, cm: ConstraintManager
    ) -> None:
        proc = _make_procedure()
        kb.save_procedure(proc)

        cm.add_guardrail("test-proc", {"type": "test_guardrail"})

        # Create a new ConstraintManager to verify persistence.
        cm2 = ConstraintManager(kb)
        rails = cm2.get_guardrails("test-proc")
        assert len(rails) == 1
        assert rails[0]["type"] == "test_guardrail"

    def test_guardrails_stored_in_procedure_constraints(
        self, kb: KnowledgeBase, cm: ConstraintManager
    ) -> None:
        proc = _make_procedure()
        kb.save_procedure(proc)

        cm.add_guardrail("test-proc", {"type": "safety"})

        loaded = kb.get_procedure("test-proc")
        assert "guardrails" in loaded["constraints"]
        assert len(loaded["constraints"]["guardrails"]) == 1
