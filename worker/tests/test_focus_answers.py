"""WP-Q2b pins: the questions after a focus recording, and the answers.

Live on the test Mac (2026-09-24, session 46b767ef): the typing-only
session's Skill was grounded (no copy step), but its questions still said
"Should I copy the exact text ...", and the typed answer "Type just that
one line, then a Source line naming the IANA page." was stored without
changing a step.
"""

from __future__ import annotations

import json

from agenthandover_worker import main as wm
from agenthandover_worker.focus_questioner import (
    ANSWER_ACCEPTED_GUESS,
    ANSWER_TYPED,
    FocusQAResult,
    FocusQuestion,
    answer_revision_prompt,
    apply_answer_revisions,
    merge_answers,
)
from agenthandover_worker.ledger import derive_knowledge, validate_knowledge
from agenthandover_worker.procedure_schema import sop_to_procedure, validate_procedure
from agenthandover_worker.skill_render import render_packet
from agenthandover_worker.sop_generator import SOPGenerator
from fixtures import iana_typing_session as iana
from fixtures.vlm_answers import load_answer

def _iana_sop() -> dict:
    """The IANA session's Skill as the generator leaves it (grounded)."""
    generator = SOPGenerator()
    answer = load_answer("iana_typing_46b767ef")
    generator._call = lambda prompt, **_kw: (json.dumps(answer), 1.0)
    result = generator.generate_from_focus(iana.timeline(), iana.TITLE)
    assert result.success, result.error
    return result.sop


def _iana_procedure() -> dict:
    sop = _iana_sop()
    proc = sop_to_procedure(sop)
    proc.update({"id": "note-iana", "slug": "note-iana", "title": iana.TITLE, "source": "focus"})
    return proc


# ---------------------------------------------------------------------------
# Questions grounded like the Skill
# ---------------------------------------------------------------------------


def test_a_session_that_copied_nothing_gets_no_copy_question() -> None:
    """Live, the model's questions asked "Should I copy the exact text
    ..." of a session that only typed.  The ledger's questions are written
    from the slots, never from the screen's or the synthesis's wording."""
    from agenthandover_worker.clipboard_grounding import claims
    from agenthandover_worker.questions import slot_questions

    proc = _iana_procedure()
    proc["strategy"] = "The user copies the reserved domains from the IANA page into TextEdit."
    questions = slot_questions(proc)
    assert questions
    for q in questions:
        assert not claims([q.question, q.context, q.default, *q.choices]), q


# ---------------------------------------------------------------------------
# Typed answers revise the steps
# ---------------------------------------------------------------------------

LIVE_ANSWER = "Type just that one line, then a Source line naming the IANA page."
# What the revision call returns for the live answer (the model's shape).
REVISION = {"revisions": [{
    "answer": 1, "replaces": [3, 4],
    "steps": [
        {"action": "Type the reserved domains line", "input": iana.LINE_1,
         "verify": "The line appears in the document."},
        {"action": "Type the source line on a new line", "input": iana.LINE_2,
         "verify": "The source line appears under it."},
    ],
}]}


def _qa(answer: str = LIVE_ANSWER, source: str = ANSWER_TYPED) -> FocusQAResult:
    return FocusQAResult(
        questions=[
            FocusQuestion(question="Should I type the whole list or one line?",
                          category="output_format", context="Step 3 types the list",
                          default="The whole list", step_indexes=[2, 3]),
            FocusQuestion(question="Why do you keep this note?", category="strategy",
                          context="", default="Reference", step_indexes=[]),
        ],
        answers={0: answer, 1: "To have the reserved names at hand"},
        sources={0: source, 1: ANSWER_TYPED},
    )


def _merged(qa: FocusQAResult) -> dict:
    return merge_answers(_iana_procedure(), qa)


def test_the_revision_prompt_carries_the_typed_step_answers_only() -> None:
    qa = _qa()
    proc = _merged(qa)
    prompt = answer_revision_prompt(proc, qa)
    assert f"Answer: {LIVE_ANSWER}" in prompt
    assert "Question (about steps 3, 4): Should I type the whole list or one line?" in prompt
    assert f"3. Switch back to TextEdit and type the reserved domain list. (in TextEdit) [input: {iana.LINE_1}]" in prompt
    # The strategy answer fills a field, not a step: not asked about.
    assert "reserved names at hand" not in prompt
    # A kept guess changes nothing: no call at all.
    assert answer_revision_prompt(proc, _qa(source=ANSWER_ACCEPTED_GUESS)) is None


def test_a_typed_answer_revises_its_steps_and_the_ledger_knows() -> None:
    qa = _qa()
    proc = _merged(qa)
    frames_before = [s["evidence"]["frames"] for s in proc["steps"][2:4]]
    applied = apply_answer_revisions(proc, qa, REVISION)
    assert len(applied) == 1
    steps = proc["steps"]
    assert [s["action"] for s in steps] == [
        "Open Safari and navigate to the IANA reserved domains page.",
        "Open TextEdit and create a new document.",
        "Type the reserved domains line",
        "Type the source line on a new line",
    ]
    assert [s["step_id"] for s in steps] == ["step_1", "step_2", "step_3", "step_4"]
    assert [s["input"] for s in steps[2:]] == [iana.LINE_1, iana.LINE_2]
    for step in steps[2:]:
        evidence = step["evidence"]
        assert evidence["source"] == "user_answer"
        assert evidence["user_answer"]["answer"] == LIVE_ANSWER
        assert evidence["frames"] == [f for fs in frames_before for f in fs]
        assert step["app"] == "TextEdit"
    assert validate_procedure(proc) == []

    # The ledger: the steps' evidence names the answer, which is known.
    knowledge = derive_knowledge(proc)
    assert validate_knowledge(knowledge) == []
    clarification = steps[2]["evidence"]["user_answer"]["clarification"]
    assert proc["agent_clarifications"][clarification]["answer"] == LIVE_ANSWER
    answer_ref = {"kind": "answer", "id": f"agent_clarifications.{clarification}"}
    assert knowledge["steps"]["status"] == "known"
    assert answer_ref in knowledge["steps"]["evidence"]
    assert knowledge["done_criteria"]["status"] == "known"  # output_format, typed

    # The packet says both lines.
    packet = render_packet(proc, target="claude-skill")
    assert f"Input: {iana.LINE_1}" in packet and f"Input: {iana.LINE_2}" in packet


def test_revisions_outside_the_answers_steps_are_not_applied() -> None:
    qa = _qa()
    for bad in (
        {"answer": 1, "replaces": [1], "steps": [{"action": "Open Chrome"}]},  # not its steps
        {"answer": 1, "replaces": [9], "steps": [{"action": "x"}]},  # no such step
        {"answer": 2, "replaces": [3], "steps": [{"action": "x"}]},  # not a revisable answer
        {"answer": 0, "replaces": [3], "steps": [{"action": "x"}]},
        {"answer": 1, "replaces": [3], "steps": [{"action": ""}]},  # no action
        {"answer": 1, "replaces": [3], "steps": [{"action": f"s{i}"} for i in range(4)]},
    ):
        proc = _merged(qa)
        before = [s["action"] for s in proc["steps"]]
        assert apply_answer_revisions(proc, qa, {"revisions": [bad]}) == [], bad
        assert [s["action"] for s in proc["steps"]] == before


def test_a_revision_may_also_take_the_step_next_to_its_own() -> None:
    """Seen live: the question named steps 4-7 and the typed answer ("...
    then a Source line") rightly rewrote step 8 too; the whole revision was
    refused. The steps right beside the question's own are allowed."""
    qa = _qa()
    proc = _merged(qa)
    if len(proc["steps"]) < 5:
        proc["steps"].append(dict(proc["steps"][-1], action="Type the Source line"))
    applied = apply_answer_revisions(
        proc, qa,
        {"revisions": [{"answer": 1, "replaces": [3, 4, 5],
                        "steps": [{"action": "Type the one line"}, {"action": "Type the Source line"}]}]},
    )
    assert len(applied) == 1
    assert [s["action"] for s in proc["steps"]][2:4] == ["Type the one line", "Type the Source line"]


def test_a_revision_keeps_typed_text_and_drops_the_step_it_repeats() -> None:
    """Seen live (focus_iana_typed): the revision wrote the typing step with
    only "... example.com" (grounding had put the whole line) and a new
    Source step beside the old one."""
    from agenthandover_worker.focus_questioner import (
        _drop_duplicated_neighbours, _keep_typed_text,
    )
    full = "Reserved for documentation: example.com, example.net, example.org"
    proc = {"variables": [{"name": "source", "example": "Source: IANA reserved domains page"}]}
    old = [{"action": "Type the line", "input": full}]
    built = [{"action": "Type the line", "input": "Reserved for documentation: example.com"},
             {"action": "Type the Source line", "input": "Source: IANA reserved domains page"}]
    _keep_typed_text(built, old, proc)
    assert built[0]["input"] == full
    steps = [{"action": "Open TextEdit", "input": ""}, *built,
             {"action": "Type the source", "input": "{{source}}"}]
    _drop_duplicated_neighbours(steps, 1, 2, proc)
    assert [s["action"] for s in steps] == ["Open TextEdit", "Type the line", "Type the Source line"]


def test_an_unavailable_model_leaves_the_steps_as_recorded() -> None:
    qa = _qa()
    proc = _merged(qa)
    before = json.dumps(proc["steps"])

    class Down:
        def revise(self, prompt: str) -> dict:
            raise ConnectionError("Ollama unreachable")

    assert wm._revise_steps_from_answers(proc, qa, Down(), iana.TITLE) == []
    assert json.dumps(proc["steps"]) == before


def test_the_revision_is_one_small_call_on_the_sop_model(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return json.dumps(REVISION), 1.0

    monkeypatch.setattr("agenthandover_worker.sop_generator._call_ollama", fake_call)
    generator = SOPGenerator()
    qa = _qa()
    proc = _merged(qa)
    applied = wm._revise_steps_from_answers(proc, qa, generator, iana.TITLE)
    assert len(applied) == 1 and len(calls) == 1
    assert calls[0]["model"] == generator.config.model
    assert calls[0]["think"] is False and calls[0]["format_json"] is True
    assert calls[0]["num_predict"] <= 1500
    # No typed answer that may change a step: no call.
    calls.clear()
    assert wm._revise_steps_from_answers(
        _merged(_qa(source=ANSWER_ACCEPTED_GUESS)), _qa(source=ANSWER_ACCEPTED_GUESS),
        generator, iana.TITLE,
    ) == []
    assert calls == []
