"""J-focus, Q&A leg: the worker has paused a focus session's export on
clarifying questions; the person skips them (or answers one) in the app,
the next tick merges that into the draft, and the person approves it.

The model-driven half of J-focus (annotation, SOP generation, synthesis,
question generation) needs the WP-T4 cassette; this journey starts from
what that half leaves behind (the session's ``focus-pending/<id>.json``
and ``focus-questions.json``, written with the worker's own functions, and a
draft review record) and drives the rest through the worker's tick and
trigger spool, the way the app does it.
"""

from __future__ import annotations

import json

import pytest
import yaml

from agenthandover_worker import trigger_spool
from agenthandover_worker.focus_questioner import (
    FOCUS_PENDING_DIR,
    FOCUS_QUESTIONS_FILE,
    FocusQuestion,
    ask_focus_questions,
    read_focus_pending,
    write_focus_pending,
    write_focus_questions,
)
from agenthandover_worker.procedure_schema import sop_to_procedure
from fixtures.vlm_answers import generated_sop_template

SLUG = "check-domain-availability"
TITLE = "Check Domain Availability"
GOAL = "Find out whether a domain is free to register and what it costs."
GUESSES = (
    "Assume credentials are provided via a logged in browser",
    "Assume the first result is the exact match",
)
OPEN_LOGIN_QUESTION = (
    "- Open question the user did not answer: "
    "\"Is the user already logged in to Namecheap?\""
)


def _paused_session(state_dir, db) -> dict:
    """What ``_process_focus_sessions_v2`` leaves behind when it pauses:
    the two Q&A files and a draft review record."""
    template = generated_sop_template("check_domain_availability")
    proc = sop_to_procedure(template)
    proc.update({
        "id": SLUG, "slug": SLUG, "title": TITLE,
        "source": "focus", "lifecycle_state": "draft", "goal": GOAL,
    })
    proc["inputs"].append({
        "name": "namecheap_password", "type": "text", "required": True,
        "default": None, "credential": True,
    })
    questions = [
        FocusQuestion(question="Is the user already logged in to Namecheap?",
                      category="access", context="A password field was seen",
                      default=GUESSES[0]),
        FocusQuestion(question="Which search result is the exact match?",
                      category="decision_logic", context="Several rows were listed",
                      default=GUESSES[1], step_indexes=[3]),
    ]
    write_focus_questions(state_dir, "focus-7f3a", SLUG, questions)
    write_focus_pending(state_dir, "focus-7f3a", SLUG, sop_template=template, procedure=proc)
    db.save_generated_sop(
        slug=SLUG, title=TITLE, source="focus", sop_template=proc,
        confidence=proc["confidence_avg"], source_id="focus-7f3a", auto_approve=False,
    )
    return proc


def _app_writes(
    state_dir, *, status: str, answers: dict | None = None, sources: dict | None = None,
) -> None:
    """The app's FocusQAView write-back: status, plus the non-blank answers
    and where each came from (``FocusQAView.answeredFile``)."""
    path = state_dir / FOCUS_QUESTIONS_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = status
    if answers is not None:
        data["answers"] = answers
    if sources is not None:
        data["answer_sources"] = sources
    path.write_text(json.dumps(data), encoding="utf-8")


def _merge_then_approve(journey, state_dir) -> None:
    """Tick (the answers merge into the draft), then make it agent-ready in
    the app, which approves the draft and exports it."""
    assert journey.run_tick()["focus_sops"] == 0  # auto_approve is off: still a draft
    assert not (state_dir / FOCUS_QUESTIONS_FILE).exists()
    trigger_spool.write_trigger(
        state_dir, "lifecycle-promote", {"slug": SLUG, "to_state": "agent_ready"},
    )
    assert journey.run_tick()["triggers_consumed"] == 1
    assert journey.lifecycle_manager.get_state(SLUG).value == "agent_ready"


def _packet(journey) -> str:
    if journey.adapter == "claude-skill":
        path = journey.cfg.export.skills_dir / SLUG / "SKILL.md"
    else:
        path = journey.sop_writer.get_sops_dir() / f"sop.{SLUG}.md"
    return path.read_text(encoding="utf-8")


def _frontmatter(packet: str) -> dict:
    return yaml.safe_load(packet[4:packet.index("\n---\n", 3)])


@pytest.mark.parametrize("adapter", ["claude-skill", "openclaw"])
def test_skipped_questions_stay_open_and_credentials_stay_by_reference(
    events_db, worker, clock, temp_home, kb_root, adapter,
) -> None:
    fixture = events_db("focus_dailynews", rebase_to=clock.now())
    journey = worker(fixture, adapter=adapter)
    _paused_session(temp_home.data_dir, journey.db)
    _app_writes(temp_home.data_dir, status="skipped")
    _merge_then_approve(journey, temp_home.data_dir)

    packet = _packet(journey)
    # No guess is presented as the user's answer.
    assert "Assume" not in packet
    assert "The user answered" not in packet
    assert OPEN_LOGIN_QUESTION in packet
    # The credential is named by reference and never taken as an argument.
    assert "`namecheap_password` (credential) by reference (never in this file)" in packet
    fm = _frontmatter(packet)
    assert "namecheap_password" not in str(fm.get("argument-hint", ""))
    # The goal is carried to the packet and describes the Skill.
    assert f"\n## Goal\n\n{GOAL}\n" in packet
    assert fm["description"] == GOAL
    assert fm["description"] != f"{TITLE}."

    stored = json.loads((kb_root / "procedures" / f"{SLUG}.json").read_text(encoding="utf-8"))
    assert [c["status"] for c in stored["agent_clarifications"]] == ["unanswered", "unanswered"]
    assert all(c["answer"] == "" for c in stored["agent_clarifications"])
    assert stored["goal"] == GOAL
    assert not any(g in json.dumps(stored) for g in GUESSES)


def test_a_blank_field_is_unanswered_and_a_typed_answer_is_kept(
    events_db, worker, clock, temp_home,
) -> None:
    fixture = events_db("focus_dailynews", rebase_to=clock.now())
    journey = worker(fixture, adapter="claude-skill")
    _paused_session(temp_home.data_dir, journey.db)
    # The user cleared the first field: the app leaves it out
    # (FocusQAView.submission); an older app wrote no answer_sources.
    _app_writes(temp_home.data_dir, status="answered", answers={"1": "The first row"})
    _merge_then_approve(journey, temp_home.data_dir)

    packet = _packet(journey)
    assert (
        "- The user answered \"Which search result is the exact match?\": The first row"
    ) in packet
    assert OPEN_LOGIN_QUESTION in packet
    assert "Assume" not in packet


def test_a_guess_submitted_as_is_stays_a_guess(
    events_db, worker, clock, temp_home, kb_root,
) -> None:
    """The app pre-fills each field with the guess (WP-Q1): Submit with the
    first untouched and the second typed over records an accepted guess and
    an answer; the ledger keeps the guess a hypothesis and the packet tells
    the agent to confirm it."""
    fixture = events_db("focus_dailynews", rebase_to=clock.now())
    journey = worker(fixture, adapter="claude-skill")
    _paused_session(temp_home.data_dir, journey.db)
    _app_writes(
        temp_home.data_dir, status="answered",
        answers={"0": GUESSES[0], "1": "The first row"},
        sources={"0": "accepted_guess", "1": "typed"},
    )
    _merge_then_approve(journey, temp_home.data_dir)

    stored = json.loads((kb_root / "procedures" / f"{SLUG}.json").read_text(encoding="utf-8"))
    assert [(c["status"], c["source"]) for c in stored["agent_clarifications"]] == [
        ("answered", "accepted_guess"), ("answered", "typed"),
    ]
    knowledge = stored["knowledge"]
    assert knowledge["inputs"]["status"] == "hypothesis"  # access: the kept guess
    assert knowledge["inputs"]["evidence"] == [{"kind": "answer", "id": "agent_clarifications.0"}]
    assert knowledge["decisions"]["status"] == "known"  # decision_logic: typed

    packet = _packet(journey)
    assert (
        "- Unconfirmed: the user let this guess stand for "
        "\"Is the user already logged in to Namecheap?\": " + GUESSES[0]
    ) in packet
    assert "- The user answered \"Which search result is the exact match?\": The first row" in packet


def test_a_review_card_answered_in_the_app_reaches_the_ledger(
    events_db, worker, clock, temp_home, kb_root,
) -> None:
    """WP-L4: a slot the Skill does not know is asked as a review card
    (``micro-questions.json``); the app's tap on a choice is a
    ``micro-answer`` trigger, and the next tick makes the slot known with
    the answer as its evidence and takes the card off the file."""
    from agenthandover_worker import questions as qs

    fixture = events_db("focus_dailynews", rebase_to=clock.now())
    journey = worker(fixture, adapter="claude-skill")
    proc = sop_to_procedure(generated_sop_template("check_domain_availability"))
    proc.update({"id": SLUG, "slug": SLUG, "title": TITLE, "source": "focus", "goal": GOAL})
    journey.knowledge_base.save_procedure(proc)
    assert journey.knowledge_base.get_procedure(SLUG)["knowledge"]["guardrails"]["status"] != "known"

    qs.refresh(journey.db, journey.knowledge_base, temp_home.data_dir, now=clock.now())
    cards = json.loads((temp_home.data_dir / qs.MICRO_QUESTIONS_FILE).read_text(encoding="utf-8"))
    card = next(c for c in cards["questions"] if c["slot"] == "guardrails")
    assert card["choices"] == ["No, nothing special"]

    trigger_spool.write_trigger(
        temp_home.data_dir, "micro-answer",
        {"question_id": card["id"], "answer": card["choices"][0]},
    )
    assert journey.run_tick()["triggers_consumed"] == 1

    stored = json.loads((kb_root / "procedures" / f"{SLUG}.json").read_text(encoding="utf-8"))
    entry = stored["knowledge"]["guardrails"]
    assert entry["status"] == "known" and entry["source"] == "user"
    assert [e["kind"] for e in entry["evidence"]] == ["answer"]
    left = json.loads((temp_home.data_dir / qs.MICRO_QUESTIONS_FILE).read_text(encoding="utf-8"))
    assert card["id"] not in [c["id"] for c in left["questions"]]


def _asks(journey, state_dir, clock, session_id: str, slug: str, title: str) -> list:
    """What ``_process_focus_sessions_v2`` does when a session pauses on
    the ledger's questions: ask them (recorded as open micro-questions),
    queue them with the session's own pending state, keep the draft.
    Returns the asked questions."""
    from agenthandover_worker.questions import ask_after_focus

    proc = sop_to_procedure(generated_sop_template("check_domain_availability"))
    proc.update({
        "id": slug, "slug": slug, "title": title,
        "source": "focus", "lifecycle_state": "draft", "goal": "",
    })
    questions = ask_after_focus(journey.db, proc, now=clock.now())
    assert questions
    ask_focus_questions(
        state_dir, session_id, slug, questions, title=title,
        sop_template=proc, procedure=proc,
    )
    journey.db.save_generated_sop(
        slug=slug, title=title, source="focus", sop_template=proc,
        confidence=proc["confidence_avg"], source_id=session_id, auto_approve=False,
    )
    return questions


def _micro_status(db, ids: list[str]) -> set[str]:
    return {r["status"] for r in db.get_micro_questions() if r["question_id"] in ids}


def test_a_second_sessions_questions_wait_for_the_first(
    events_db, worker, clock, temp_home, kb_root,
) -> None:
    """WP-F15, the live check's sequence: session A asks, session B finishes
    and asks before A is answered.  The app keeps showing A (its answered
    file is never overwritten); A's answers merge into A's own procedure,
    then the app shows B, whose answers merge into B's."""
    state_dir = temp_home.data_dir
    fixture = events_db("focus_dailynews", rebase_to=clock.now())
    journey = worker(fixture, adapter="claude-skill")
    a_ids = [q.id for q in _asks(journey, state_dir, clock, "57352dfb", "check-domain-a", "Domain A")]
    b_questions = _asks(journey, state_dir, clock, "62406a55", "check-domain-b", "Domain B")
    b_ids = [q.id for q in b_questions]
    # The live check's counts: three for A, then the week's last two for B.
    assert (len(a_ids), len(b_ids)) == (3, 2)
    shown = json.loads((state_dir / FOCUS_QUESTIONS_FILE).read_text(encoding="utf-8"))
    assert shown["session_id"] == "57352dfb" and shown["status"] == "pending"
    assert [q["id"] for q in shown["questions"]] == a_ids

    # A tick with nothing answered changes nothing.
    journey.run_tick()
    assert json.loads((state_dir / FOCUS_QUESTIONS_FILE).read_text(encoding="utf-8")) == shown

    _app_writes(state_dir, status="answered", answers={"0": "Answer for A"})
    answered_a = (state_dir / FOCUS_QUESTIONS_FILE).read_bytes()
    # B asks again (its session re-processed) while A's answers wait: the
    # answered file stays A's.
    b_proc = read_focus_pending(state_dir, "62406a55")["procedure"]
    assert ask_focus_questions(
        state_dir, "62406a55", "check-domain-b", b_questions, title="Domain B",
        sop_template=b_proc, procedure=b_proc,
    ) is False
    assert (state_dir / FOCUS_QUESTIONS_FILE).read_bytes() == answered_a

    assert journey.run_tick()["focus_sops"] == 0  # auto_approve is off: a draft
    stored_a = json.loads((kb_root / "procedures" / "check-domain-a.json").read_text(encoding="utf-8"))
    assert [c.get("question_id") for c in stored_a["agent_clarifications"]] == a_ids
    assert stored_a["agent_clarifications"][0]["answer"] == "Answer for A"
    assert not (kb_root / "procedures" / "check-domain-b.json").exists()
    assert _micro_status(journey.db, a_ids) == {"answered", "skipped"}
    assert _micro_status(journey.db, b_ids) == {"open"}

    # The app now shows B's questions, with B's title.
    shown = json.loads((state_dir / FOCUS_QUESTIONS_FILE).read_text(encoding="utf-8"))
    assert (shown["session_id"], shown["title"], shown["status"]) == (
        "62406a55", "Domain B", "pending",
    )
    assert [q["id"] for q in shown["questions"]] == b_ids

    _app_writes(state_dir, status="answered", answers={"1": "Answer for B"})
    assert journey.run_tick()["focus_sops"] == 0
    stored_b = json.loads((kb_root / "procedures" / "check-domain-b.json").read_text(encoding="utf-8"))
    assert [c.get("question_id") for c in stored_b["agent_clarifications"]] == b_ids
    assert [c["answer"] for c in stored_b["agent_clarifications"]] == ["", "Answer for B"]
    assert stored_b["title"] == "Domain B"
    assert _micro_status(journey.db, b_ids) == {"answered", "skipped"}
    assert not (state_dir / FOCUS_QUESTIONS_FILE).exists()
    assert not list((state_dir / FOCUS_PENDING_DIR).iterdir())
