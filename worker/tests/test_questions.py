"""WP-L4 pins: micro-questions from the ledger, their budget and timing,
and what an answer (or a skip) writes.

The journeys: J-ledger's skipped question (``journeys/test_j_ledger.py``,
waiting on its fixtures) and J-passive's labelling card, pinned model-free
on the ``passive_3weeks`` fixture in
``journeys/test_workflow_observations.py``.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agenthandover_worker import questions as qs
from agenthandover_worker import workflow_mining as wm
from agenthandover_worker.db import WorkerDB
from agenthandover_worker.focus_questioner import (
    FocusQAResult,
    merge_answers,
    parse_qa_result_from_file,
    write_focus_questions,
)
from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.ledger import derive_knowledge
from daemon_db import create_daemon_db

NOW = datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc)


def _proc(**overrides) -> dict:
    proc = {
        "id": "weekly-report",
        "slug": "weekly-report",
        "title": "Weekly report",
        "goal": "Send the weekly numbers to the team",
        "steps": [
            {"action": "Open the dashboard", "app": "Safari",
             "evidence": {"frames": ["e1"]}},
            {"action": "Post the numbers", "app": "Slack",
             "evidence": {"frames": ["e2"]}},
        ],
        "evidence": {"observations": [{"session_id": "s1"}]},
    }
    proc.update(overrides)
    return proc


def _ids(questions) -> list[str]:
    return [q.id for q in questions]


# ---------------------------------------------------------------------------
# Slot questions
# ---------------------------------------------------------------------------


def test_one_question_per_slot_not_known_unknown_required_first() -> None:
    proc = _proc(goal="")  # no goal, no done criteria: both required
    found = {q.slot: q for q in qs.slot_questions(proc)}
    assert "steps" not in found  # known from the frames
    assert set(found) == set(qs.ASKED_SLOTS)
    ranked = qs.next_questions([proc], 3)
    assert [q.slot for q in ranked] == ["goal", "done_criteria", "trigger"]
    # Unknown slots offer no guess.
    assert all(q.default == "" for q in ranked)
    assert all(q.id == f"slot:weekly-report:{q.slot}" for q in ranked)


def test_a_hypothesis_is_offered_as_the_guess_and_a_choice() -> None:
    [goal] = [q for q in qs.slot_questions(_proc()) if q.slot == "goal"]
    assert goal.default == "Send the weekly numbers to the team"
    assert goal.choices == ["Send the weekly numbers to the team"]


def test_choices_come_from_what_the_runs_showed() -> None:
    proc = _proc(
        learned={
            "trigger": {"description": "Most Mondays around 09:00",
                        "calendar_event": "Weekly sync", "instances": ["s1", "s2"]},
            "done": {"description": "Done when \"Send\" is clicked in Slack",
                     "instances": ["s1"]},
            "verification": {"checks": [{"description": "Within a minute of finishing, "
                                         "checks #team in Slack", "instances": ["s1"]}],
                             "instances": ["s1"]},
        },
        selection_criteria=[
            {"criterion": "are at most 2 hours old", "attribute": "age_minutes",
             "op": "<=", "value": 120.0},
            {"criterion": 'mention "rust"', "attribute": "keyword", "op": "contains",
             "value": "rust"},
            {"criterion": "have at least 100 points", "attribute": "point",
             "op": ">=", "value": 100.0},
        ],
    )
    found = {q.slot: q for q in qs.slot_questions(proc)}
    assert found["trigger"].choices == [
        'The calendar event "Weekly sync"', "Most Mondays around 09:00",
        "Nothing fixed: I start it when I decide to",
    ]
    assert found["done_criteria"].choices == ['Done when "Send" is clicked in Slack']
    assert found["verification"].choices == [
        "Within a minute of finishing, checks #team in Slack", "I don't check it",
    ]
    decisions = found["decisions"]
    assert decisions.question == "Why do you skip the entries you don't pick?"
    assert decisions.choices == [
        "Too old (the ones you pick are at most 2 hours old)",
        'Off-topic (the ones you pick mention "rust")',
        "Too few points (the ones you pick have at least 100 points)",
    ]


def test_an_answered_slot_is_not_asked_again() -> None:
    proc = _proc(agent_clarifications=[{
        "category": "trigger", "question": "What starts this task?",
        "answer": "Monday 9am", "status": "answered", "source": "typed",
    }])
    assert "trigger" not in {q.slot for q in qs.slot_questions(proc)}


def test_next_questions_asks_labels_first_within_budget_and_never_twice() -> None:
    label = qs.Question(
        id="label:k1", kind=qs.KIND_LABEL, subject="k1", title="Payouts",
        question="Was this a routine you would hand off?", context="",
        choices=[qs.LABEL_YES, qs.LABEL_NO, qs.LABEL_NOT_SURE], value=4_000_000,
    )
    first = qs.next_questions([_proc(goal="")], 2, workflows=[label])
    assert _ids(first) == ["label:k1", "slot:weekly-report:goal"]
    again = qs.next_questions([_proc(goal="")], 2, workflows=[label], asked=set(_ids(first)))
    assert _ids(again) == ["slot:weekly-report:done_criteria", "slot:weekly-report:trigger"]
    assert qs.next_questions([_proc()], 0) == []


# ---------------------------------------------------------------------------
# Budget, focus recordings, answers
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path):
    with WorkerDB(create_daemon_db(tmp_path / "events.db")) as worker_db:
        yield worker_db


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    base = KnowledgeBase(root=tmp_path / "knowledge")
    base.ensure_structure()
    return base


def test_the_weekly_budget_caps_what_focus_recordings_ask(db) -> None:
    first = qs.ask_after_focus(db, _proc(id="a", goal=""), now=NOW)
    assert len(first) == qs.FOCUS_MAX_QUESTIONS
    assert [q.category for q in first] == ["goal", "done_criteria", "trigger"]
    assert all(q.id and q.choices is not None for q in first)
    second = qs.ask_after_focus(db, _proc(id="b", goal=""), now=NOW + timedelta(hours=1))
    assert len(second) == qs.WEEKLY_BUDGET - qs.FOCUS_MAX_QUESTIONS
    assert qs.ask_after_focus(db, _proc(id="c", goal=""), now=NOW + timedelta(days=2)) == []
    # Once the first three are more than seven days old, three may be
    # asked again (the other two still count).
    later = qs.ask_after_focus(db, _proc(id="c", goal=""), now=NOW + timedelta(days=7, minutes=30))
    assert len(later) == 3
    rows = db.get_micro_questions(channel=qs.CHANNEL_FOCUS)
    assert len(rows) == 8 and all(r["status"] == "open" for r in rows)
    # A budget of 0 asks nothing.
    assert qs.ask_after_focus(db, _proc(id="d"), now=NOW + timedelta(days=30),
                              weekly_budget=0) == []


def _qa_file(tmp_path: Path, questions) -> dict:
    write_focus_questions(tmp_path, "focus-1", "a", questions, title="A")
    return json.loads((tmp_path / "focus-questions.json").read_text(encoding="utf-8"))


def test_a_skipped_question_leaves_its_slot_as_it_was(db, tmp_path: Path) -> None:
    """J-ledger's skipped micro-question, at unit level: no slot claims a
    user answer, and the question is closed as skipped."""
    proc = _proc(id="a")
    asked = qs.ask_after_focus(db, proc, now=NOW)
    data = _qa_file(tmp_path, asked)
    assert [q["choices"] for q in data["questions"]] == [q.choices for q in asked]
    data["status"] = "skipped"
    merged = merge_answers(proc, parse_qa_result_from_file(data))
    ledger = derive_knowledge(merged)
    assert not any(e["kind"] == "answer" for entry in ledger.values() for e in entry["evidence"])
    assert {s: ledger[s]["status"] for s in qs.ASKED_SLOTS} == {
        s: derive_knowledge(proc)[s]["status"] for s in qs.ASKED_SLOTS
    }
    qs.close_focus_questions(db, data, now=NOW)
    assert {r["status"] for r in db.get_micro_questions()} == {"skipped"}


def test_a_typed_answer_makes_its_slot_known_with_answer_evidence(db, tmp_path: Path) -> None:
    proc = _proc(id="a")
    asked = qs.ask_after_focus(db, proc, now=NOW)
    trigger = next(i for i, q in enumerate(asked) if q.category == "trigger")
    data = _qa_file(tmp_path, asked)
    data.update(status="answered", answers={str(trigger): "Every Monday at 9"},
                answer_sources={str(trigger): "typed"})
    merged = merge_answers(proc, parse_qa_result_from_file(data))
    entry = derive_knowledge(merged)["trigger"]
    assert entry["status"] == "known" and entry["source"] == "user"
    assert entry["evidence"] == [
        {"kind": "answer", "id": f"agent_clarifications.{trigger}"},
    ]
    assert merged["agent_clarifications"][trigger]["question_id"] == "slot:a:trigger"
    qs.close_focus_questions(db, data, now=NOW)
    statuses = {r["slot"]: r["status"] for r in db.get_micro_questions()}
    assert statuses.pop("trigger") == "answered"
    assert set(statuses.values()) == {"skipped"}


# ---------------------------------------------------------------------------
# Review cards
# ---------------------------------------------------------------------------


def test_a_card_answer_is_a_typed_clarification(db, kb, tmp_path: Path) -> None:
    kb.save_procedure(_proc(id="a", slug="a"))
    new = qs.refresh(db, kb, tmp_path, now=NOW)
    assert new and all(q.kind == qs.KIND_SLOT for q in new)
    cards = json.loads((tmp_path / qs.MICRO_QUESTIONS_FILE).read_text())["questions"]
    assert [c["id"] for c in cards] == _ids(new)
    card = next(c for c in cards if c["slot"] == "verification")

    assert qs.apply_answer(
        {"question_id": card["id"], "answer": "I don't check it"}, db, kb, tmp_path, now=NOW,
    )
    entry = kb.get_procedure("a")["knowledge"]["verification"]
    assert entry["status"] == "known"
    assert entry["evidence"][0]["kind"] == "answer"
    remaining = json.loads((tmp_path / qs.MICRO_QUESTIONS_FILE).read_text())["questions"]
    assert card["id"] not in [c["id"] for c in remaining]
    # Answered once: the card is closed and is never asked again.
    assert not qs.apply_answer(
        {"question_id": card["id"], "answer": "again"}, db, kb, tmp_path, now=NOW,
    )
    assert card["id"] not in _ids(qs.refresh(db, kb, tmp_path, now=NOW + timedelta(days=8)))


def test_a_skipped_card_changes_nothing(db, kb, tmp_path: Path) -> None:
    kb.save_procedure(_proc(id="a", slug="a"))
    [card, *_] = qs.refresh(db, kb, tmp_path, now=NOW)
    before = kb.get_procedure("a")["knowledge"]
    assert qs.apply_answer({"question_id": card.id, "answer": ""}, db, kb, tmp_path, now=NOW)
    after = kb.get_procedure("a")
    assert after["knowledge"] == before
    assert not after.get("agent_clarifications")
    assert [r["status"] for r in db.get_micro_questions() if r["question_id"] == card.id] == [
        "skipped"
    ]


def _observation(obs_id: str, key: str, started: datetime, **extra) -> dict:
    local = started.astimezone()
    return {
        "obs_id": obs_id, "workflow_key": key,
        "started_utc": started.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "started_local_date": local.date().isoformat(),
        "started_local_weekday": local.weekday(),
        "started_local_minute": local.hour * 60 + local.minute,
        "duration_s": 600.0, "apps_json": json.dumps(["Safari", "Numbers"]),
        "hosts_json": json.dumps(["dashboard.stripe.com"]),
        "step_sig_json": json.dumps([["Safari", "dashboard.stripe.com", "open"],
                                     ["Numbers", "payouts.numbers", "type"]]),
        "evidence_json": "{}", "source": "passive", "task_label": "Stripe payouts",
        "event_ids_json": json.dumps([f"{obs_id}-e1", f"{obs_id}-e2"]),
        **extra,
    }


def _three_runs(db) -> None:
    for n in range(3):
        db.save_workflow_observation(_observation(f"o{n}", "k1", NOW - timedelta(days=n + 1)))


def test_a_labelling_card_per_new_workflow(db, kb, tmp_path: Path) -> None:
    db.save_workflow_observation(_observation("o0", "k1", NOW - timedelta(days=1)))
    assert qs.label_questions(db, now=NOW) == []  # seen once: not discovered yet
    _three_runs(db)
    [card] = qs.label_questions(db, now=NOW)
    assert card.id == "label:k1" and card.asks_name
    assert card.choices == ["Yes", "No", "Not sure"]
    assert card.workflow_keys == ["k1"] and sorted(card.obs_ids) == ["o0", "o1", "o2"]
    assert "Seen 3 times" in card.context and "dashboard.stripe.com" in card.context
    assert _ids(qs.refresh(db, kb, tmp_path, now=NOW)) == ["label:k1"]
    # Asked once, never again.
    assert qs.refresh(db, kb, tmp_path, now=NOW + timedelta(days=8)) == []


def test_a_workflow_recorded_in_focus_gets_no_labelling_card(db) -> None:
    _three_runs(db)
    db.save_workflow_observation(
        _observation("f0", "k1", NOW - timedelta(days=5), source="focus"),
    )
    assert qs.label_questions(db, now=NOW) == []


def test_a_yes_label_names_the_workflow(db, kb, tmp_path: Path) -> None:
    _three_runs(db)
    qs.refresh(db, kb, tmp_path, now=NOW)
    assert qs.apply_answer(
        {"question_id": "label:k1", "answer": "Yes", "name": "Payout reconciliation"},
        db, kb, tmp_path, now=NOW,
    )
    rows = db.get_workflow_observations()
    assert {(r["user_label"], r["user_is_routine"]) for r in rows} == {
        ("Payout reconciliation", 1),
    }
    # A routine stays a review candidate.
    assert len(wm.review_candidates(db, now=NOW)) == 1


def test_a_no_label_rejects_the_draft_and_suppresses_the_workflow_for_42_days(
    db, kb, tmp_path: Path,
) -> None:
    _three_runs(db)
    assert len(wm.review_candidates(db, now=NOW)) == 1
    sop_id = db.save_generated_sop(
        slug="stripe-payouts", title="Stripe payouts", source="passive",
        sop_template={}, source_id="k1", auto_approve=False,
    )
    db.set_observations_promoted(["o0"], "stripe-payouts", promoted_at=NOW)
    qs.refresh(db, kb, tmp_path, now=NOW)
    assert qs.apply_answer(
        {"question_id": "label:k1", "answer": "No", "name": ""}, db, kb, tmp_path, now=NOW,
    )
    assert db.get_generated_sop(sop_id)["status"] == "rejected"
    assert {r["user_is_routine"] for r in db.get_workflow_observations()} == {0}
    # An unpromoted "no" workflow is kept out of the queue for 42 days.
    db.save_workflow_observation(_observation("p0", "k2", NOW - timedelta(days=1)))
    db.save_workflow_observation(_observation("p1", "k2", NOW - timedelta(days=2)))
    db.save_workflow_observation(_observation("p2", "k2", NOW - timedelta(days=3)))
    db.label_workflow(["k2"], label="", is_routine=False, labelled_at=NOW)
    assert wm.suppressed(db.get_workflow_observations(obs_ids=["p0"]), NOW)
    assert not wm.suppressed(
        db.get_workflow_observations(obs_ids=["p0"]), NOW + timedelta(days=wm.SUPPRESS_DAYS + 1),
    )


def test_not_sure_labels_nothing_as_a_routine(db, kb, tmp_path: Path) -> None:
    _three_runs(db)
    qs.refresh(db, kb, tmp_path, now=NOW)
    qs.apply_answer({"question_id": "label:k1", "answer": "Not sure"}, db, kb, tmp_path, now=NOW)
    rows = db.get_workflow_observations()
    assert {r["user_is_routine"] for r in rows} == {None}
    assert all(r["labelled_at"] for r in rows)
    assert len(wm.review_candidates(db, now=NOW)) == 1


def test_the_digest_lists_open_cards_and_the_days_sessions(db, kb, tmp_path: Path) -> None:
    _three_runs(db)
    qs.refresh(db, kb, tmp_path, now=NOW)
    date = db.get_workflow_observations(obs_ids=["o0"])[0]["started_local_date"]
    questions, sessions = qs.digest_items(db, date)
    assert questions == [{
        "question": "Was this a routine you would hand off?", "about": "Stripe payouts",
        "choices": "Yes / No / Not sure",
    }]
    assert [(s["session"], s["minutes"], s["routine"]) for s in sessions] == [
        ("Stripe payouts", "10", "not tagged"),
    ]
    qs.apply_answer({"question_id": "label:k1", "answer": "Yes", "name": "Payouts"},
                    db, kb, tmp_path, now=NOW)
    questions, sessions = qs.digest_items(db, date)
    assert questions == []
    assert [(s["name"], s["routine"]) for s in sessions] == [("Payouts", "yes")]


def test_old_observation_tables_gain_the_label_columns(tmp_path: Path) -> None:
    path = create_daemon_db(tmp_path / "events.db")
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE workflow_observations (obs_id TEXT PRIMARY KEY, workflow_key TEXT NOT NULL,"
        " started_utc TEXT NOT NULL, started_local_date TEXT NOT NULL,"
        " started_local_weekday INTEGER NOT NULL, started_local_minute INTEGER NOT NULL,"
        " duration_s REAL NOT NULL, apps_json TEXT NOT NULL DEFAULT '[]',"
        " hosts_json TEXT NOT NULL DEFAULT '[]', step_sig_json TEXT NOT NULL DEFAULT '[]',"
        " evidence_json TEXT NOT NULL DEFAULT '{}', source TEXT NOT NULL DEFAULT 'passive',"
        " task_label TEXT NOT NULL DEFAULT '', event_ids_json TEXT NOT NULL DEFAULT '[]',"
        " promoted_slug TEXT, promoted_at TEXT)"
    )
    conn.commit()
    conn.close()
    with WorkerDB(path) as db:
        assert db.save_workflow_observation(_observation("o0", "k1", NOW))
        assert db.label_workflow(["k1"], label="X", is_routine=True, labelled_at=NOW) == 1
        [row] = db.get_workflow_observations()
        assert (row["user_label"], row["user_is_routine"]) == ("X", 1)
        # Re-recording the session keeps the answer.
        assert db.save_workflow_observation(_observation("o0", "k1", NOW))
        assert db.get_workflow_observations()[0]["user_label"] == "X"


def test_an_unknown_question_id_is_refused(db, kb, tmp_path: Path) -> None:
    assert not qs.apply_answer({"question_id": "slot:x:goal", "answer": "y"},
                               db, kb, tmp_path, now=NOW)


def test_focus_qa_result_keeps_ids_and_choices(tmp_path: Path) -> None:
    from agenthandover_worker.focus_questioner import FocusQuestion

    q = FocusQuestion(question="What starts this task?", category="trigger", context="",
                      default="", id="slot:a:trigger", choices=["Mondays"])
    data = _qa_file(tmp_path, [q])
    data["status"] = "skipped"
    result = parse_qa_result_from_file(data)
    assert isinstance(result, FocusQAResult)
    assert (result.questions[0].id, result.questions[0].choices) == ("slot:a:trigger", ["Mondays"])
