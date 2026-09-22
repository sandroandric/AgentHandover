"""J-focus, Q&A leg: the worker has paused a focus session's export on
clarifying questions; the person skips them (or answers one) in the app,
the next tick merges that into the draft, and the person approves it.

The model-driven half of J-focus (annotation, SOP generation, synthesis,
question generation) needs the WP-T4 cassette; this journey starts from
what that half leaves behind (``focus-pending.json`` and
``focus-questions.json``, written with the worker's own functions, and a
draft review record) and drives the rest through the worker's tick and
trigger spool, the way the app does it.
"""

from __future__ import annotations

import json

import pytest
import yaml

from agenthandover_worker import trigger_spool
from agenthandover_worker.focus_questioner import (
    FOCUS_QUESTIONS_FILE,
    FocusQuestion,
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


def _app_writes(state_dir, *, status: str, answers: dict | None = None) -> None:
    """The app's FocusQAView write-back: status, plus the non-blank answers."""
    path = state_dir / FOCUS_QUESTIONS_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = status
    if answers is not None:
        data["answers"] = answers
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
    # The app omits the untouched first field (FocusQAView.submittedAnswers).
    _app_writes(temp_home.data_dir, status="answered", answers={"1": "The first row"})
    _merge_then_approve(journey, temp_home.data_dir)

    packet = _packet(journey)
    assert (
        "- The user answered \"Which search result is the exact match?\": The first row"
    ) in packet
    assert OPEN_LOGIN_QUESTION in packet
    assert "Assume" not in packet
