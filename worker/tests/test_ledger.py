"""Regression pins for the learning ledger (WP-L1), next to J-focus's
ledger assertions and J-mcp's hand-off gate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.ledger import (
    REQUIRED_SLOTS,
    SLOTS,
    TRACKED_SLOTS,
    confidence,
    derive_knowledge,
    handoff_gate,
    readiness,
    set_slot,
    unknown_slots,
    validate_knowledge,
)
from agenthandover_worker.lifecycle_manager import LifecycleManager, ProcedureLifecycle
from daemon_db import kb_db
from agenthandover_worker.procedure_schema import (
    backfill_procedure_fields,
    sop_to_procedure,
    validate_procedure,
)

NOW = datetime(2026, 9, 15, 9, 0, tzinfo=timezone.utc)


def _recorded(**overrides) -> dict:
    """A procedure as one focus recording leaves it."""
    proc = sop_to_procedure({
        "slug": "daily-digest", "title": "Daily digest",
        "task_description": "Post a digest of three stories.",
        "outcome": "The digest is posted in #daily-news.",
        "steps": [
            {"step": "Open Hacker News", "evidence": {"frames": ["f1", "f2"]}},
            {"step": "Post the digest", "evidence": {"frames": ["f2", "f3"]}},
        ],
        "variables": [{"name": "story", "type": "text"}],
    })
    proc["evidence"]["observations"] = [{
        "date": "2026-09-15", "type": "focus", "duration_minutes": 0,
        "session_id": "sess-1", "event_count": 40,
    }]
    proc["staleness"]["last_observed"] = NOW.isoformat()
    proc.update(overrides)
    return proc


class TestDerivation:
    def test_one_recording(self) -> None:
        ledger = derive_knowledge(_recorded())
        assert list(ledger) == list(SLOTS)
        instance = {"kind": "instance", "id": "sess-1"}

        assert ledger["steps"]["status"] == "known"
        assert ledger["steps"]["value_ref"] == "steps"
        assert ledger["steps"]["evidence"] == [
            instance,
            {"kind": "frame", "id": "f1"}, {"kind": "frame", "id": "f2"},
            {"kind": "frame", "id": "f3"},
        ]
        assert ledger["goal"] == {
            "status": "hypothesis", "value_ref": "task_description",
            "evidence": [instance], "source": "observation",
            "updated_at": ledger["goal"]["updated_at"],
        }
        assert ledger["done_criteria"]["value_ref"] == "outcome"
        # One run cannot show which values vary.
        assert ledger["inputs"]["status"] == "hypothesis"
        assert ledger["context"]["status"] == "unknown"  # no app recorded
        for slot in ("trigger", "decisions", "exceptions", "guardrails", "verification", "voice"):
            assert ledger[slot] == {
                "status": "unknown", "value_ref": None, "evidence": [],
                "source": None, "updated_at": ledger[slot]["updated_at"],
            }, slot
        assert validate_knowledge(ledger) == []

    def test_known_needs_evidence(self) -> None:
        proc = _recorded()
        for step in proc["steps"]:
            step["evidence"] = {"frames": [], "dom_snapshot_ids": []}
        proc["evidence"]["observations"] = []
        ledger = derive_knowledge(proc)
        assert ledger["steps"]["status"] == "hypothesis"
        assert ledger["steps"]["evidence"] == []

    def test_two_instances_ground_the_inputs(self) -> None:
        proc = _recorded()
        proc["evidence"]["observations"].append(
            {**proc["evidence"]["observations"][0], "session_id": "sess-2"},
        )
        assert derive_knowledge(proc)["inputs"]["status"] == "known"

    def test_passively_mined_observations_are_instances(self) -> None:
        """A passive draft carries pointers into workflow_observations
        (``workflow_mining.observation_pointers``: ``obs_id``, no
        ``session_id``); each one is an observed instance."""
        proc = sop_to_procedure({
            "slug": "payout-recon", "title": "Payout reconciliation",
            "task_description": "Reconcile the Stripe payouts.",
            "steps": [{"step": "Open the payouts"}],
            "variables": [{"name": "payout_date", "type": "date"}],
            "observations": [
                {"obs_id": f"obs-{i}", "timestamp": "2026-09-1%dT09:00:00Z" % i,
                 "date": "2026-09-1%d" % i, "duration_s": 600, "source": "passive"}
                for i in (1, 2, 3)
            ],
        })
        ledger = derive_knowledge(proc)
        assert ledger["goal"]["evidence"] == [
            {"kind": "instance", "id": f"obs-{i}"} for i in (1, 2, 3)
        ]
        assert ledger["inputs"]["status"] == "known"
        assert validate_knowledge(ledger) == []

    def test_trigger_from_an_observed_recurrence(self) -> None:
        proc = _recorded()
        proc["recurrence"]["pattern"] = "weekly"
        trigger = derive_knowledge(proc)["trigger"]
        assert trigger["status"] == "hypothesis"
        assert trigger["value_ref"] == "recurrence.pattern"

    def test_an_unclear_goal_is_unknown(self) -> None:
        proc = _recorded(goal="Unclear: the recipient is not visible")
        assert derive_knowledge(proc)["goal"]["status"] == "unknown"

    def test_a_users_answer_makes_the_slot_known(self) -> None:
        proc = _recorded(agent_clarifications=[
            {"category": "scheduling", "question": "When?", "answer": "Every weekday at 9",
             "status": "answered"},
            {"category": "verification", "question": "How do you check?", "answer": "",
             "status": "unanswered"},
        ])
        ledger = derive_knowledge(proc)
        assert ledger["trigger"] == {
            "status": "known", "value_ref": "agent_clarifications.0",
            "evidence": [{"kind": "answer", "id": "agent_clarifications.0"}],
            "source": "user", "updated_at": ledger["trigger"]["updated_at"],
        }
        assert ledger["verification"]["status"] == "unknown"  # unanswered stays unknown

    def test_an_unchanged_entry_keeps_its_timestamp(self) -> None:
        proc = _recorded()
        proc["knowledge"] = derive_knowledge(proc)
        proc["knowledge"]["steps"]["updated_at"] = "2026-01-01T00:00:00+00:00"
        again = derive_knowledge(proc)
        assert again["steps"]["updated_at"] == "2026-01-01T00:00:00+00:00"
        proc["steps"].append({"action": "Check the channel", "evidence": {"frames": ["f9"]}})
        assert derive_knowledge(proc)["steps"]["updated_at"] != "2026-01-01T00:00:00+00:00"


class TestLearnersWrite:
    def test_a_learners_entry_survives_derivation(self) -> None:
        proc = _recorded()
        set_slot(proc, "decisions", status="hypothesis", source="contrast",
                 evidence=[{"kind": "instance", "id": "sess-1"}], value_ref="selection_criteria")
        ledger = derive_knowledge(proc)
        assert ledger["decisions"]["source"] == "contrast"
        assert ledger["decisions"]["status"] == "hypothesis"

    def test_a_learner_can_demote_to_unknown(self) -> None:
        proc = _recorded()
        set_slot(proc, "steps", status="unknown", source="agent",
                 evidence=[{"kind": "run", "id": "run-7"}])
        assert derive_knowledge(proc)["steps"]["status"] == "unknown"

    def test_a_users_answer_wins_over_a_learner(self) -> None:
        proc = _recorded(agent_clarifications=[
            {"category": "decision", "question": "Which?", "answer": "Newest", "status": "answered"},
        ])
        set_slot(proc, "decisions", status="hypothesis", source="contrast",
                 evidence=[{"kind": "instance", "id": "sess-1"}])
        assert derive_knowledge(proc)["decisions"]["source"] == "user"

    @pytest.mark.parametrize("kwargs", [
        {"slot": "mood", "status": "known", "source": "user", "evidence": [{"kind": "answer", "id": "a"}]},
        {"slot": "goal", "status": "sure", "source": "user", "evidence": [{"kind": "answer", "id": "a"}]},
        {"slot": "goal", "status": "known", "source": "observation", "evidence": [{"kind": "answer", "id": "a"}]},
        {"slot": "goal", "status": "known", "source": "user", "evidence": []},
        {"slot": "goal", "status": "known", "source": "user", "evidence": [{"kind": "guess", "id": "a"}]},
    ])
    def test_set_slot_validates(self, kwargs) -> None:
        slot = kwargs.pop("slot")
        with pytest.raises(ValueError):
            set_slot(_recorded(), slot, **kwargs)


class TestReadinessAndGate:
    def test_required_slots(self) -> None:
        assert REQUIRED_SLOTS == ("goal", "steps", "done_criteria")
        assert readiness(_recorded()).ready
        not_ready = readiness(_recorded(task_description="", outcome=""))
        assert not not_ready.ready
        assert not_ready.unknown_required == ["goal", "done_criteria"]

    def test_confidence_is_the_grounded_fraction(self) -> None:
        ledger = derive_knowledge(_recorded())
        assert confidence(ledger) == round(1 / len(TRACKED_SLOTS), 2)  # only steps known

    def test_unknown_slots_in_order(self) -> None:
        assert unknown_slots(_recorded()) == [
            "trigger", "decisions", "exceptions", "guardrails", "verification",
        ]

    def test_the_gate_names_every_failing_check(self) -> None:
        stale = _recorded(
            lifecycle_state="draft", task_description="",
            staleness={"last_observed": (NOW - timedelta(days=150)).isoformat()},
        )
        gate = handoff_gate(stale, now=NOW)
        assert not gate.ok
        assert gate.reasons == [
            "lifecycle is 'draft', not 'agent_ready' (not approved for agents)",
            "not yet known: goal",
            "trust level is 'observe' (must be above 'observe')",
            "freshness 0.03 is below 0.3",
        ]

    def test_an_approved_fresh_procedure_passes(self) -> None:
        proc = _recorded(lifecycle_state="agent_ready")
        proc["constraints"]["trust_level"] = "suggest"
        assert handoff_gate(proc, now=NOW).ok


class TestStorage:
    def test_every_save_stores_the_ledger(self, tmp_path) -> None:
        kb = KnowledgeBase(root=tmp_path, db_path=kb_db(tmp_path))
        kb.ensure_structure()
        kb.save_procedure(_recorded())
        stored = kb.get_procedure("daily-digest")
        assert stored["knowledge"]["steps"]["status"] == "known"
        assert validate_procedure(stored) == []

    def test_validate_procedure_rejects_a_malformed_ledger(self) -> None:
        proc = _recorded(knowledge={
            "goal": {"status": "known", "evidence": [], "source": "user"},
            "mood": {"status": "unknown", "evidence": []},
        })
        errors = validate_procedure(proc)
        assert "knowledge.goal: known without evidence" in errors
        assert "knowledge.mood: unknown slot" in errors


class TestApprovalTrust:
    def test_approval_grants_suggest_and_never_lowers(self, tmp_path) -> None:
        kb = KnowledgeBase(root=tmp_path, db_path=kb_db(tmp_path))
        kb.ensure_structure()
        kb.save_procedure(_recorded(lifecycle_state="draft"))
        lm = LifecycleManager(kb)
        lm.promote("daily-digest", ProcedureLifecycle.AGENT_READY, trigger="human_promote")
        assert kb.get_procedure("daily-digest")["constraints"]["trust_level"] == "suggest"

        proc = _recorded(lifecycle_state="verified")
        proc["constraints"]["trust_level"] = "autonomous"
        kb.save_procedure(proc)
        lm.promote("daily-digest", ProcedureLifecycle.AGENT_READY, trigger="human_promote")
        assert kb.get_procedure("daily-digest")["constraints"]["trust_level"] == "autonomous"

    def test_a_system_transition_grants_nothing(self, tmp_path) -> None:
        kb = KnowledgeBase(root=tmp_path, db_path=kb_db(tmp_path))
        kb.ensure_structure()
        kb.save_procedure(_recorded(lifecycle_state="verified"))
        LifecycleManager(kb).transition(
            "daily-digest", ProcedureLifecycle.AGENT_READY, trigger="t", actor="system",
        )
        assert kb.get_procedure("daily-digest")["constraints"]["trust_level"] == "observe"

    def test_backfill_grants_an_earlier_human_approval(self) -> None:
        approved = _recorded(lifecycle_state="agent_ready", lifecycle_history=[
            {"from_state": "verified", "to_state": "agent_ready", "actor": "human"},
        ])
        assert backfill_procedure_fields(approved) is True
        assert approved["constraints"]["trust_level"] == "suggest"

        by_system = _recorded(lifecycle_state="agent_ready", lifecycle_history=[
            {"from_state": "verified", "to_state": "agent_ready", "actor": "system"},
        ])
        backfill_procedure_fields(by_system)
        assert by_system["constraints"]["trust_level"] == "observe"
