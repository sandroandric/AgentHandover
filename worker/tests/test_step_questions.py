"""WP-F10 pins: step questions from what step grounding flagged.

Step grounding (``step_grounding``, ``clipboard_grounding``) knows which
steps of a focus Skill are uncertain; ``questions.step_questions`` asks
about each, with concrete choices, and ``questions.apply_step_answer``
applies a choice in code.  Any other answer goes to the focus answers' one
revision call.  The journey: ``journeys/test_j_focus_typed.py`` (the
typed-line question on ``focus_iana_typed``).
"""

from __future__ import annotations

import copy
import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agenthandover_worker import questions as qs
from agenthandover_worker.clipboard_grounding import (
    attach_clipboard_events,
    ground_clipboard_claims,
)
from agenthandover_worker.db import WorkerDB
from agenthandover_worker.focus_questioner import (
    FocusQuestion,
    answer_revision_prompt,
    apply_answer_revisions,
    merge_answers,
    parse_qa_result_from_file,
    write_focus_questions,
)
from agenthandover_worker.inference_runner import InferenceRunner, model_wait
from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.ledger import derive_knowledge
from agenthandover_worker.procedure_schema import sop_to_procedure, validate_procedure
from agenthandover_worker.skill_render import render_packet
from agenthandover_worker.sop_generator import SOPGenerator
from agenthandover_worker.step_grounding import ground_steps
from daemon_db import create_daemon_db, kb_db
from fixtures import iana_typing_session as iana
from fixtures.vlm_answers import load_answer

NOW = datetime(2026, 9, 24, 9, 0, tzinfo=timezone.utc)
SCROLL_CLICK = "Scroll down the page and click the 'Example Domain' link."
SELECT = "Select the typed text in the TextEdit document."


def _live_run_procedure() -> dict:
    """The Skill of the live run of 2026-09-24 (session 46b767ef), grounded
    as the worker grounds it: two steps dropped, the first line's input
    finished, the second line added as a step."""
    generator = SOPGenerator()
    answer = load_answer("iana_typing_46b767ef")
    generator._call = lambda prompt, **_kw: (json.dumps(answer), 1.0)
    result = generator.generate_from_focus(iana.timeline(), iana.TITLE)
    assert result.success, result.error
    return sop_to_procedure(result.sop)


def _by_kind(proc: dict) -> dict[str, list[qs.Question]]:
    out: dict[str, list[qs.Question]] = {}
    for q in qs.step_questions(proc):
        out.setdefault(q.id.split(":")[2], []).append(q)
    return out


def _answer(proc: dict, question: qs.Question, answer: str) -> bool:
    """Record *answer* as a clarification (as the Q&A window and the review
    cards do) and apply it."""
    clarifications = proc.setdefault("agent_clarifications", [])
    clarifications.append({"category": "steps", "question": question.question,
                           "answer": answer, "status": "answered", "source": "typed",
                           "question_id": question.id})
    return qs.apply_step_answer(proc, question.id, answer,
                                clarification=len(clarifications) - 1,
                                question=question.question)


def _answer_pointer(proc: dict, question: qs.Question) -> dict:
    index = next(i for i, c in enumerate(proc["agent_clarifications"])
                 if c["question_id"] == question.id)
    return {"kind": "answer", "id": f"agent_clarifications.{index}"}


# ---------------------------------------------------------------------------
# The questions
# ---------------------------------------------------------------------------


def test_the_live_runs_flags_become_one_question_each() -> None:
    proc = _live_run_procedure()
    found = _by_kind(proc)
    assert {k: len(v) for k, v in found.items()} == {"dropped": 2, "input": 1, "added": 1}
    dropped = {q.question: q for q in found["dropped"]}
    scroll = dropped[
        f'The model thought you also did "{SCROLL_CLICK}", but the recording did not show '
        "it. Did you?"
    ]
    assert scroll.choices == [qs.NO_I_DID_NOT, qs.YES_I_DID]
    assert "'Example Domain' is not in the frames" in scroll.context
    [typed_line] = found["added"]
    assert typed_line.question == (
        f'The recording shows you typed "{iana.LINE_2}" (step 4), which the model had '
        "left out. Did you type it?"
    )
    assert typed_line.choices == [qs.TYPED_THAT, qs.LEAVE_IT_OUT]
    assert typed_line.step_indexes == [3]
    [finished] = found["input"]
    assert finished.question.startswith(f'The recording shows you typed "{iana.LINE_1}" at step 3')
    assert finished.choices == [qs.YES, qs.NO]
    for q in qs.step_questions(proc):
        assert q.kind == qs.KIND_STEP and q.slot == "steps" and q.default == ""
        assert q.id.startswith(f"step:{proc['id']}:")
    # Deterministic: the same procedure asks the same questions.
    assert [q.to_json() for q in qs.step_questions(copy.deepcopy(proc))] == [
        q.to_json() for q in qs.step_questions(proc)
    ]


def test_step_questions_outrank_every_slot_but_an_unknown_required_one() -> None:
    """A concrete step question fixes what the agent will execute; a
    generic optional slot does not.  So every step question, the typed
    line included, comes after the unknown required slots and before the
    unknown optional slots and the hypotheses."""
    proc = _live_run_procedure()
    ranked = qs.next_questions([proc], 20)
    kinds = [q.id.split(":")[2] if q.kind == qs.KIND_STEP else q.slot for q in ranked]
    ledger = derive_knowledge(proc)
    unknown_required = [s for s in qs.REQUIRED_SLOTS if ledger[s]["status"] == "unknown"]
    assert kinds[:len(unknown_required)] == unknown_required
    rest = kinds[len(unknown_required):]
    assert rest[:4] == ["dropped", "dropped", "input", "added"]
    slots = [k for k in rest if k in qs.ASKED_SLOTS]
    unknown_optional = [s for s in slots if ledger[s]["status"] == "unknown"]
    hypotheses = [s for s in slots if ledger[s]["status"] == "hypothesis"]
    assert unknown_optional and hypotheses
    assert rest[4:] == unknown_optional + hypotheses
    # Each question once.
    assert len(set(q.id for q in ranked)) == len(ranked)


# ---------------------------------------------------------------------------
# Choices apply in code
# ---------------------------------------------------------------------------


def test_yes_restores_a_dropped_step_where_it_stood() -> None:
    proc = _live_run_procedure()
    [select] = [q for q in _by_kind(proc)["dropped"] if SELECT in q.question]
    before = [s["action"] for s in proc["steps"]]
    assert _answer(proc, select, qs.YES_I_DID)
    actions = [s["action"] for s in proc["steps"]]
    assert len(actions) == len(before) + 1
    restored = next(s for s in proc["steps"] if s["action"] == SELECT)
    # Where its frame stands in the recording: the last frame, after the
    # second typed line.
    assert actions == [*before, SELECT]
    assert restored["evidence"]["frames"] == ["e5"]
    assert restored["evidence"]["source"] == "user_answer"
    assert restored["evidence"]["user_answer"]["answer"] == qs.YES_I_DID
    assert [s["step_id"] for s in proc["steps"]] == [
        f"step_{i + 1}" for i in range(len(actions))
    ]
    assert SELECT not in [d["action"] for d in proc["evidence"]["dropped_steps"]]
    assert _answer_pointer(proc, select) in derive_knowledge(proc)["steps"]["evidence"]
    assert validate_procedure(proc) == []
    assert select.id not in [q.id for q in qs.step_questions(proc)]
    # The other dropped step cited Safari's frame: it goes back after the
    # Safari step, although a step was added since it was asked.
    [scroll] = _by_kind(proc)["dropped"]
    assert scroll.step_indexes == [0, 1]
    assert _answer(proc, scroll, qs.YES_I_DID)
    actions = [s["action"] for s in proc["steps"]]
    assert actions == [before[0], SCROLL_CLICK, *before[1:], SELECT]


def test_no_keeps_a_dropped_step_out_with_the_answer() -> None:
    proc = _live_run_procedure()
    [select] = [q for q in _by_kind(proc)["dropped"] if SELECT in q.question]
    steps = copy.deepcopy(proc["steps"])
    assert _answer(proc, select, "no, i didn't")  # spacing and case aside
    assert proc["steps"] == steps
    entry = next(d for d in proc["evidence"]["dropped_steps"] if d["action"] == SELECT)
    assert entry["user_answer"]["answer"] == "no, i didn't"
    steps_slot = derive_knowledge(proc)["steps"]
    assert steps_slot["status"] == "known"
    assert _answer_pointer(proc, select) in steps_slot["evidence"]
    assert f'- Not a step: "{SELECT}"' in render_packet(proc, target="claude-skill")
    assert "the user answered: no, i didn't" in render_packet(proc, target="claude-skill")
    assert select.id not in [q.id for q in qs.step_questions(proc)]


def test_leave_it_out_removes_an_added_typed_line() -> None:
    proc = _live_run_procedure()
    [typed_line] = _by_kind(proc)["added"]
    assert _answer(proc, typed_line, qs.LEAVE_IT_OUT)
    assert iana.LINE_2 not in json.dumps(proc["steps"])
    entry = proc["evidence"]["dropped_steps"][-1]
    assert entry["reason"] == "the user said they did not type it"
    assert _answer_pointer(proc, typed_line) in derive_knowledge(proc)["steps"]["evidence"]


def test_typed_it_keeps_an_added_typed_line_and_cites_the_answer() -> None:
    proc = _live_run_procedure()
    [typed_line] = _by_kind(proc)["added"]
    assert _answer(proc, typed_line, qs.TYPED_THAT)
    step = proc["steps"][3]
    assert step["input"] == iana.LINE_2
    assert "source" not in step["evidence"]
    assert step["evidence"]["user_answer"]["answer"] == qs.TYPED_THAT
    assert _answer_pointer(proc, typed_line) in derive_knowledge(proc)["steps"]["evidence"]


def test_a_removed_or_kept_step_leaves_a_skills_branches_on_their_steps() -> None:
    """A Skill another run was matched to carries the run's hypotheses
    (``branch_mining``): "No, leave it out" moves those after the removed
    step back by one; a choice that keeps the step keeps the step ids."""
    proc = _live_run_procedure()
    for n, step in enumerate(proc["steps"]):
        step["step_id"] = f"own-{n}"
    last = len(proc["steps"])
    detour = {"kind": "detour", "step_index": last, "rejoin_index": last,
              "alternative_steps": [], "condition_state": {}, "evidence": []}
    proc["branch_hypotheses"] = [dict(detour)]

    kept = copy.deepcopy(proc)
    [typed_line] = _by_kind(kept)["added"]
    assert _answer(kept, typed_line, qs.TYPED_THAT)
    assert [s["step_id"] for s in kept["steps"]] == [f"own-{n}" for n in range(last)]
    assert kept["branch_hypotheses"] == [detour]

    [typed_line] = _by_kind(proc)["added"]
    assert _answer(proc, typed_line, qs.LEAVE_IT_OUT)
    assert len(proc["steps"]) == last - 1
    assert [s["step_id"] for s in proc["steps"]] == [f"step_{n + 1}" for n in range(last - 1)]
    [hyp] = proc["branch_hypotheses"]
    assert (hyp["step_index"], hyp["rejoin_index"]) == (last - 1, last - 1)


def _scroll_procedure() -> dict:
    timeline = [
        {"event_id": "a", "timestamp": "2026-09-24T09:00:00Z",
         "annotation": {"app": "Safari", "location": "https://example.com"},
         "content_text": ["Example Domain"], "click_targets": []},
        {"event_id": "b", "timestamp": "2026-09-24T09:00:10Z",
         "annotation": {"app": "Safari", "location": "https://example.com"},
         "content_text": ["Example Domain"], "click_targets": []},
    ]
    steps = [
        {"step": "Open Safari and scroll down", "description": "", "parameters": {},
         "evidence": {"frames": ["a"], "dom_snapshot_ids": []}},
        {"step": "Read the page", "description": "", "parameters": {},
         "evidence": {"frames": ["b"], "dom_snapshot_ids": []}},
    ]
    report = ground_steps(steps, timeline)
    assert report["rewritten"], report
    return sop_to_procedure({"slug": "look", "title": "Look", "steps": steps})


def test_a_cut_scroll_is_asked_about_and_yes_restores_it() -> None:
    proc = _scroll_procedure()
    [scroll] = qs.step_questions(proc)
    assert scroll.question == (
        'The recording did not show a scroll at step 1 ("Open Safari"). Did you scroll?'
    )
    assert scroll.context == 'The model wrote: "Open Safari and scroll down".'
    assert scroll.choices == [qs.NO_I_DID_NOT, qs.YES_I_DID]
    assert "Unconfirmed: cut from the step" in render_packet(proc, target="claude-skill")

    kept = copy.deepcopy(proc)
    assert _answer(kept, scroll, qs.NO_I_DID_NOT)
    assert kept["steps"][0]["action"] == "Open Safari"
    assert "source" not in kept["steps"][0]["evidence"]
    packet = render_packet(kept, target="claude-skill")
    assert "Unconfirmed" not in packet
    assert f"The user answered: {qs.NO_I_DID_NOT}" in packet

    assert _answer(proc, scroll, qs.YES_I_DID)
    step = proc["steps"][0]
    assert step["action"] == "Open Safari and scroll down"
    assert step["evidence"]["source"] == "user_answer"
    assert step["evidence"]["grounding"]["user_answer"]["answer"] == qs.YES_I_DID
    assert _answer_pointer(proc, scroll) in derive_knowledge(proc)["steps"]["evidence"]
    assert qs.step_questions(proc) == []


def _clipboard_procedure() -> dict:
    frames = [
        {"event_id": f"f{i}", "timestamp": f"2026-09-23T10:00:{i * 10:02d}Z",
         "annotation": {"app": app}, "diff": None}
        for i, app in enumerate(["Safari", "Safari", "TextEdit", "TextEdit"])
    ]
    attach_clipboard_events(frames, [])  # the session recorded no copy or paste
    steps = [
        {"step": "Copy the text regarding IANA reserved domains", "description": "",
         "parameters": {"verify": "The text is copied to the clipboard"},
         "evidence": {"frames": ["f0", "f1"]}},
        {"step": "Paste the copied text", "description": "Insert the previously copied text.",
         "parameters": {"verify": "The text appears in the document"},
         "evidence": {"frames": ["f2"]}},
    ]
    assert len(ground_clipboard_claims(steps, frames)) == 2
    return sop_to_procedure({"slug": "note", "title": "Note", "steps": steps})


def test_an_unrecorded_paste_asks_what_the_user_did() -> None:
    proc = _clipboard_procedure()
    copy_q, paste_q = qs.step_questions(proc)
    assert paste_q.question == (
        'Step 2 says "Paste the copied text", but nothing was copied before it in the '
        "recording. What did you do?"
    )
    assert paste_q.choices == [qs.TYPED_IT, qs.PASTED_IT]
    assert copy_q.question == (
        'Step 1 says "Copy the text regarding IANA reserved domains", but the recording '
        "shows no copy. Did you copy it?"
    )
    assert copy_q.choices == [qs.READ_IT, qs.COPIED_IT]

    typed = copy.deepcopy(proc)
    assert _answer(typed, paste_q, qs.TYPED_IT)
    assert typed["steps"][1]["action"] == "Type the text"

    assert _answer(proc, paste_q, qs.PASTED_IT)
    step = proc["steps"][1]
    assert step["action"] == "Paste the copied text"
    assert step["description"] == "Insert the previously copied text."
    assert step["verify"] == step["parameters"]["verify"] == "The text appears in the document"
    assert step["evidence"]["source"] == "user_answer"
    assert "Unconfirmed: no copy or paste" not in render_packet(proc, target="claude-skill")


def test_only_the_start_puts_back_the_models_input() -> None:
    step = {"action": "Type the domains", "input": "Reserved for documentation: example.com, "
            "example.net", "parameters": {"input": "Reserved for documentation: example.com, "
                                          "example.net"},
            "evidence": {"frames": ["e3"], "grounding": {
                "status": "input_added", "reason": "the frames show the finished line",
                "original": "Reserved for documentation: example.com"}}}
    proc = {"id": "p", "title": "P", "steps": [step]}
    [q] = qs.step_questions(proc)
    assert q.choices == [qs.WHOLE_LINE, 'Only "Reserved for documentation: example.com"']
    assert _answer(proc, q, q.choices[1])
    assert proc["steps"][0]["input"] == "Reserved for documentation: example.com"
    assert proc["steps"][0]["parameters"]["input"] == "Reserved for documentation: example.com"


# ---------------------------------------------------------------------------
# Other answers: the revision call, never a step changed in code
# ---------------------------------------------------------------------------


def test_a_free_text_answer_goes_to_the_revision_with_its_steps_as_they_stand(
    tmp_path: Path,
) -> None:
    proc = _live_run_procedure()
    found = _by_kind(proc)
    [select] = [q for q in found["dropped"] if SELECT in q.question]
    [typed_line] = found["added"]
    asked = [select, typed_line]
    write_focus_questions(tmp_path, "s", proc["id"], [
        FocusQuestion(question=q.question, category=q.slot, context=q.context, default="",
                         id=q.id, choices=q.choices, step_indexes=q.step_indexes)
        for q in asked
    ])
    data = json.loads((tmp_path / "focus-questions.json").read_text(encoding="utf-8"))
    free = "I typed it, then fixed a typo in example.net"
    data.update(status="answered", answers={"0": qs.YES_I_DID, "1": free},
                answer_sources={"0": "typed", "1": "typed"})
    qa_result = parse_qa_result_from_file(data)
    merged = merge_answers(proc, qa_result)
    # The restored step moved the added line down by one.
    line_2 = next(i for i, s in enumerate(merged["steps"]) if s.get("input") == iana.LINE_2)
    assert qa_result.questions[1].step_indexes == [line_2]
    prompt = answer_revision_prompt(merged, qa_result)
    assert prompt is not None
    assert f"1. Question (about steps {line_2 + 1}): {typed_line.question}" in prompt
    assert qs.YES_I_DID not in prompt.split("ANSWERS:")[1]
    # The free answer changed no step in code; the step cites it.
    assert merged["steps"][line_2]["input"] == iana.LINE_2
    assert merged["steps"][line_2]["evidence"]["user_answer"]["answer"] == free
    revised = apply_answer_revisions(merged, qa_result, {"revisions": [{
        "answer": 1, "replaces": [line_2 + 1],
        "steps": [{"action": "Type the source line", "input": iana.LINE_2, "verify": ""}],
    }]})
    assert revised and merged["steps"][line_2]["evidence"]["source"] == "user_answer"


def test_a_stale_question_changes_no_step() -> None:
    proc = _scroll_procedure()
    [scroll] = qs.step_questions(proc)
    proc["steps"][0]["evidence"].pop("grounding")
    before = copy.deepcopy(proc["steps"])
    assert not _answer(proc, scroll, qs.YES_I_DID)
    assert proc["steps"] == before
    assert proc["agent_clarifications"][-1]["answer"] == qs.YES_I_DID


# ---------------------------------------------------------------------------
# Budget, focus window, review cards
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path):
    with WorkerDB(create_daemon_db(tmp_path / "events.db")) as worker_db:
        yield worker_db


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    base = KnowledgeBase(root=tmp_path / "knowledge", db_path=kb_db(tmp_path / "knowledge"))
    base.ensure_structure()
    return base


def test_step_questions_share_the_focus_window_and_the_weekly_budget(db) -> None:
    proc = _live_run_procedure()
    first = qs.ask_after_focus(db, proc, now=NOW)
    assert len(first) == qs.FOCUS_MAX_QUESTIONS
    ids = [q.id for q in first]
    assert [q.id for q in qs.next_questions([proc], qs.FOCUS_MAX_QUESTIONS)] == ids
    steps = [q for q in first if q.category == "steps"]
    assert steps and all(q.choices and q.step_indexes is not None for q in steps)
    # What is left of the week's budget: two more, then none.
    second = qs.ask_after_focus(db, proc, now=NOW + timedelta(hours=1))
    assert len(second) == qs.WEEKLY_BUDGET - qs.FOCUS_MAX_QUESTIONS
    assert not set(ids) & {q.id for q in second}
    assert qs.ask_after_focus(db, proc, now=NOW + timedelta(days=2)) == []


def test_a_step_card_answer_applies_the_choice_in_the_knowledge_base(
    db, kb, tmp_path: Path,
) -> None:
    proc = _scroll_procedure()
    kb.save_procedure(proc)
    new = qs.refresh(db, kb, tmp_path, now=NOW)
    [card] = [q for q in new if q.kind == qs.KIND_STEP]
    cards = json.loads((tmp_path / qs.MICRO_QUESTIONS_FILE).read_text())["questions"]
    written = next(c for c in cards if c["id"] == card.id)
    assert written["choices"] == [qs.NO_I_DID_NOT, qs.YES_I_DID]
    assert written["kind"] == "step" and written["slot"] == "steps"

    assert qs.apply_answer({"question_id": card.id, "answer": qs.YES_I_DID},
                           db, kb, tmp_path, now=NOW)
    saved = kb.get_procedure("look")
    assert saved["steps"][0]["action"] == "Open Safari and scroll down"
    [clarification] = [c for c in saved["agent_clarifications"] if c["question_id"] == card.id]
    assert clarification["category"] == "steps"
    assert saved["knowledge"]["steps"]["status"] == "known"
    assert {"kind": "answer", "id": "agent_clarifications.0"} in (
        saved["knowledge"]["steps"]["evidence"]
    )
    assert card.id not in [q.id for q in qs.step_questions(saved)]


def test_a_free_text_card_answer_is_kept_on_its_step(db, kb, tmp_path: Path) -> None:
    proc = _scroll_procedure()
    kb.save_procedure(proc)
    [card] = [q for q in qs.refresh(db, kb, tmp_path, now=NOW) if q.kind == qs.KIND_STEP]
    answer = "I used the arrow keys, not the mouse"
    assert qs.apply_answer({"question_id": card.id, "answer": answer},
                           db, kb, tmp_path, now=NOW)
    saved = kb.get_procedure("look")
    assert saved["steps"][0]["action"] == "Open Safari"
    assert saved["steps"][0]["evidence"]["user_answer"]["answer"] == answer
    assert f"The user answered: {answer}" in render_packet(saved, target="claude-skill")


# ---------------------------------------------------------------------------
# A review card's free-text answer revises its step on the inference thread
# ---------------------------------------------------------------------------

CARD_ANSWER = "I typed the domain list again under it, then the source line"
REVISION_KEY = "step-answer-revision:"


class _Reviser:
    """``SOPGenerator.revise`` as the worker calls it: waits for the model
    (``model_wait`` lets go of the state lock), on whatever thread it is
    called from, which it records.  It answers with *response* once *go*
    is set, or raises *error*."""

    def __init__(self, response: dict | None = None, error: Exception | None = None,
                 during=None) -> None:
        self.response = response or {}
        self.error = error
        self.during = during
        self.go = threading.Event()
        self.go.set()
        self.calls: list[tuple[str, int]] = []

    def revise(self, prompt: str) -> dict:
        self.calls.append((prompt, threading.get_ident()))
        with model_wait("fake revise"):
            assert self.go.wait(10)
            if self.during is not None:
                self.during()
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture()
def runner():
    """The worker's two threads: this test's thread is the control thread."""
    started = InferenceRunner().start()
    yield started
    started.stop()


def _typed_line_card(db, kb, tmp_path: Path) -> qs.Question:
    """The live run's Skill in the knowledge base, with its typed-line
    question (step 4, which the model had left out) open as a review card."""
    proc = _live_run_procedure()
    kb.save_procedure(proc)
    [card] = _by_kind(proc)["added"]
    qs.record_asked(db, [card], channel=qs.CHANNEL_CARD, now=NOW)
    qs.write_cards(db, tmp_path)
    return card


def test_a_free_text_card_answer_queues_one_revision_job_that_revises_the_step(
    db, kb, tmp_path: Path, runner: InferenceRunner,
) -> None:
    card = _typed_line_card(db, kb, tmp_path)
    slug = card.subject
    before = kb.get_procedure(slug)
    assert before["steps"][2]["input"] == iana.LINE_1
    # The answer's revision replaces step 4 by two steps: the domain list
    # typed again (the untouched step 3 before it types the same, so it is
    # dropped), and the source line cut short (it keeps the typed line).
    reviser = _Reviser({"revisions": [{
        "answer": 1, "replaces": [4],
        "steps": [
            {"action": "Type the domain list", "input": iana.LINE_1, "verify": ""},
            {"action": "Type the source line", "input": "Source: IANA", "verify": ""},
        ],
    }]})
    reviser.go.clear()

    assert qs.apply_answer({"question_id": card.id, "answer": CARD_ANSWER},
                           db, kb, tmp_path, now=NOW, reviser=reviser)
    # Recorded at once, on the control thread; the model call is one P0 job.
    [row] = db.get_micro_questions(status="answered")
    assert row["question_id"] == card.id
    status = runner.status()
    assert status["depth"] == 1
    assert (status["running"] or status["queued"][0]) == f"P0 {REVISION_KEY}{card.id}"
    recorded = kb.get_procedure(slug)
    assert recorded["steps"][3]["evidence"]["user_answer"]["answer"] == CARD_ANSWER

    reviser.go.set()
    assert runner.wait_idle(10)
    [(prompt, thread)] = reviser.calls
    assert thread != threading.get_ident()
    assert f"1. Question (about steps 4): {card.question}" in prompt
    assert f"Answer: {CARD_ANSWER}" in prompt
    assert runner.status()["completed"] == 1 and runner.status()["failed"] == 0

    saved = kb.get_procedure(slug)
    assert [s.get("input") for s in saved["steps"]] == ["", "", iana.LINE_1, iana.LINE_2]
    assert saved["steps"][2]["action"] == "Type the domain list"
    for step in saved["steps"][2:]:
        assert step["evidence"]["source"] == "user_answer"
        assert step["evidence"]["user_answer"]["answer"] == CARD_ANSWER
    [clarification] = [c for c in saved["agent_clarifications"] if c.get("question_id") == card.id]
    assert saved["steps"][3]["evidence"]["user_answer"]["clarification"] == (
        saved["agent_clarifications"].index(clarification)
    )


def test_a_choice_card_answer_makes_no_model_call(
    db, kb, tmp_path: Path, runner: InferenceRunner,
) -> None:
    card = _typed_line_card(db, kb, tmp_path)
    reviser = _Reviser()
    assert qs.apply_answer({"question_id": card.id, "answer": qs.LEAVE_IT_OUT},
                           db, kb, tmp_path, now=NOW, reviser=reviser)
    assert runner.wait_idle(10)
    assert reviser.calls == []
    assert runner.status()["completed"] == 0 and runner.status()["depth"] == 0
    assert iana.LINE_2 not in json.dumps(kb.get_procedure(card.subject)["steps"])


def test_the_model_unavailable_leaves_the_card_answer_recorded(
    db, kb, tmp_path: Path, runner: InferenceRunner,
) -> None:
    from agenthandover_worker.ollama import OllamaUnavailable

    card = _typed_line_card(db, kb, tmp_path)
    before = kb.get_procedure(card.subject)["steps"]
    reviser = _Reviser(error=OllamaUnavailable("the generation model could not be loaded"))
    assert qs.apply_answer({"question_id": card.id, "answer": CARD_ANSWER},
                           db, kb, tmp_path, now=NOW, reviser=reviser)
    assert runner.wait_idle(10)
    assert len(reviser.calls) == 1
    # Handled in the job: no failure, no backoff, nothing lost.
    assert runner.status()["failed"] == 0
    saved = kb.get_procedure(card.subject)
    assert [s["action"] for s in saved["steps"]] == [s["action"] for s in before]
    assert saved["steps"][3]["evidence"]["user_answer"]["answer"] == CARD_ANSWER
    assert any(c.get("question_id") == card.id and c["answer"] == CARD_ANSWER
               for c in saved["agent_clarifications"])
    assert [r["question_id"] for r in db.get_micro_questions(status="answered")] == [card.id]
    assert f"The user answered: {CARD_ANSWER}" in render_packet(saved, target="claude-skill")


def test_steps_changed_during_the_revision_are_not_revised(
    db, kb, tmp_path: Path, runner: InferenceRunner,
) -> None:
    """The model call lets go of the state lock: an edit saved meanwhile
    (an approval, another answer) wins, and the revision numbered on the
    old steps is not applied."""
    card = _typed_line_card(db, kb, tmp_path)

    def edit() -> None:
        proc = kb.get_procedure(card.subject)
        proc["steps"][0]["action"] = "Open Safari"
        kb.save_procedure(proc)

    reviser = _Reviser({"revisions": [{
        "answer": 1, "replaces": [4],
        "steps": [{"action": "Type the source line", "input": iana.LINE_2, "verify": ""}],
    }]}, during=edit)
    assert qs.apply_answer({"question_id": card.id, "answer": CARD_ANSWER},
                           db, kb, tmp_path, now=NOW, reviser=reviser)
    assert runner.wait_idle(10)
    saved = kb.get_procedure(card.subject)
    assert saved["steps"][0]["action"] == "Open Safari"
    assert "source" not in saved["steps"][3]["evidence"]
    assert saved["steps"][3]["evidence"]["user_answer"]["answer"] == CARD_ANSWER


def test_the_control_thread_passes_the_sop_generator_to_the_micro_answer(
    db, kb, tmp_path: Path,
) -> None:
    """The trigger handler (``stages.control``) hands the worker's SOP
    generator to ``questions.apply_answer`` as its reviser."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from agenthandover_worker.stages import control

    reviser = object()
    ctx = SimpleNamespace(
        db=db, knowledge_base=kb, now=lambda: NOW, sop_generator=reviser,
        export_kwargs=lambda: {},
    )
    with patch("agenthandover_worker.questions.apply_answer") as apply, \
            patch("agenthandover_worker.main._status_dir", return_value=tmp_path):
        control.trigger_handlers(ctx)["micro-answer"]({"question_id": "q", "answer": "a"})
    assert apply.call_args.kwargs["reviser"] is reviser



# ---------------------------------------------------------------------------
# A worker restart queues again the revisions it had not run
# ---------------------------------------------------------------------------

_REVISION = {"revisions": [{
    "answer": 1, "replaces": [4],
    "steps": [{"action": "Type the source line", "input": iana.LINE_2, "verify": ""}],
}]}


def _clarification(proc: dict, card: qs.Question) -> dict:
    [entry] = [c for c in proc["agent_clarifications"] if c.get("question_id") == card.id]
    return entry


def test_a_restart_queues_the_revision_a_stop_cut_short_once(
    db, kb, tmp_path: Path, runner: InferenceRunner,
) -> None:
    """The answer was recorded and its job queued, then the worker stopped
    (the queue lives in memory): at the next start the revision is queued
    again, runs on the inference thread, records its result, and is never
    queued a second time."""
    card = _typed_line_card(db, kb, tmp_path)
    # The answer as the stopped worker left it: recorded, its job lost.
    assert qs.apply_answer({"question_id": card.id, "answer": CARD_ANSWER},
                           db, kb, tmp_path, now=NOW, reviser=None)
    assert "revision" not in _clarification(kb.get_procedure(card.subject), card)
    assert [q.id for q, _ in qs.pending_step_answers(db, kb)] == [card.id]

    reviser = _Reviser(_REVISION)
    reviser.go.clear()
    assert qs.requeue_step_answers(db, kb, reviser) == 1
    # Queued, not run on this (the control) thread, and one job per key.
    assert qs.requeue_step_answers(db, kb, reviser) == 1
    status = runner.status()
    assert status["depth"] == 1
    assert (status["running"] or status["queued"][0]) == f"P0 {REVISION_KEY}{card.id}"
    reviser.go.set()
    assert runner.wait_idle(10)
    [(prompt, thread)] = reviser.calls
    assert thread != threading.get_ident()
    assert f"Answer: {CARD_ANSWER}" in prompt

    saved = kb.get_procedure(card.subject)
    assert saved["steps"][3]["evidence"]["source"] == "user_answer"
    assert saved["steps"][3]["input"] == iana.LINE_2
    assert _clarification(saved, card)["revision"] == qs.REVISION_REVISED
    # The next start queues nothing.
    assert qs.pending_step_answers(db, kb) == []
    assert qs.requeue_step_answers(db, kb, _Reviser(_REVISION)) == 0
    assert runner.wait_idle(10) and runner.status()["completed"] == 1


def test_a_model_that_is_down_or_times_out_leaves_the_revision_for_the_next_start(
    db, kb, tmp_path: Path, runner: InferenceRunner,
) -> None:
    from agenthandover_worker.ollama import OllamaUnavailable

    card = _typed_line_card(db, kb, tmp_path)
    down = _Reviser(error=OllamaUnavailable("the generation model could not be loaded"))
    assert qs.apply_answer({"question_id": card.id, "answer": CARD_ANSWER},
                           db, kb, tmp_path, now=NOW, reviser=down)
    assert runner.wait_idle(10) and len(down.calls) == 1
    assert "revision" not in _clarification(kb.get_procedure(card.subject), card)

    slow = _Reviser(error=TimeoutError("the model did not answer in time"))
    assert qs.requeue_step_answers(db, kb, slow) == 1
    assert runner.wait_idle(10) and len(slow.calls) == 1
    assert "revision" not in _clarification(kb.get_procedure(card.subject), card)
    assert runner.status()["failed"] == 0

    reviser = _Reviser(_REVISION)
    assert qs.requeue_step_answers(db, kb, reviser) == 1
    assert runner.wait_idle(10) and len(reviser.calls) == 1
    assert _clarification(kb.get_procedure(card.subject), card)["revision"] == (
        qs.REVISION_REVISED
    )


def test_an_unusable_or_rejected_revision_is_settled_and_not_queued_again(
    db, kb, tmp_path: Path, runner: InferenceRunner,
) -> None:
    card = _typed_line_card(db, kb, tmp_path)
    before = [s["action"] for s in kb.get_procedure(card.subject)["steps"]]
    unusable = _Reviser(error=ValueError("no JSON object in the answer"))
    assert qs.apply_answer({"question_id": card.id, "answer": CARD_ANSWER},
                           db, kb, tmp_path, now=NOW, reviser=unusable)
    assert runner.wait_idle(10)
    saved = kb.get_procedure(card.subject)
    assert _clarification(saved, card)["revision"] == qs.REVISION_UNCHANGED
    assert [s["action"] for s in saved["steps"]] == before
    assert qs.requeue_step_answers(db, kb, _Reviser(_REVISION)) == 0

    # A revision the rules reject (it replaces a step that does not exist)
    # settles as unchanged too.
    del _clarification(saved, card)["revision"]
    kb.save_procedure(saved)
    rejected = _Reviser({"revisions": [{
        "answer": 1, "replaces": [99], "steps": [{"action": "Type it"}],
    }]})
    assert qs.requeue_step_answers(db, kb, rejected) == 1
    assert runner.wait_idle(10) and len(rejected.calls) == 1
    saved = kb.get_procedure(card.subject)
    assert _clarification(saved, card)["revision"] == qs.REVISION_UNCHANGED
    assert [s["action"] for s in saved["steps"]] == before


def test_only_free_text_card_answers_are_queued_again(
    db, kb, tmp_path: Path, runner: InferenceRunner,
) -> None:
    """A choice answer is applied in code, and a focus-window answer is
    revised by its focus run: neither is queued again at startup."""
    card = _typed_line_card(db, kb, tmp_path)
    assert qs.apply_answer({"question_id": card.id, "answer": qs.LEAVE_IT_OUT},
                           db, kb, tmp_path, now=NOW, reviser=None)
    focus = qs.Question.from_json({**card.to_json(), "id": f"{card.id}-focus"})
    qs.record_asked(db, [focus], channel=qs.CHANNEL_FOCUS, now=NOW)
    db.close_micro_question(focus.id, status="answered", answer=CARD_ANSWER, at=NOW)
    reviser = _Reviser(_REVISION)
    assert qs.pending_step_answers(db, kb) == []
    assert qs.requeue_step_answers(db, kb, reviser) == 0
    assert runner.wait_idle(10) and reviser.calls == []
    # Without a reviser (the v2 pipeline off) nothing is queued either.
    assert qs.requeue_step_answers(db, kb, None) == 0
