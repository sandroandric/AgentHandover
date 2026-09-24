"""J-focus, typing-only leg: the session of 2026-09-24 (46b767ef).

The user had Safari on the IANA reserved domains page, switched to
TextEdit, pressed Cmd-N and typed two lines; nothing was copied
(``fixtures/focus_iana_typed``, ``record/generate_focus_iana_typed.py``).
Live, the Skill put the new document before Safari, clicked an "Example
Domain" link seen only in the tab strip, selected text no frame showed
selected and lost the second line; its questions asked about copying; the
typed answer changed no step.  Here the recording becomes steps its frames
show, in their order, with both lines; no question assumes a copy; a
typed answer revises the typing steps.
"""

from __future__ import annotations

import json
import os
import re

import pytest

from agenthandover_worker.focus_questioner import FOCUS_QUESTIONS_FILE
from agenthandover_worker.ledger import derive_knowledge
from agenthandover_worker.skill_render import render_packet
from journeys.cassette import CASSETTE_DIR, load_entries
from journeys.focus_run import TICK_SECONDS, start_focus_run
from journeys.packet import parse_packet

JOURNEY = "focus_iana_typed"
LINE_1 = "Reserved for documentation: example.com, example.net, example.org"
LINE_2 = "Source: IANA reserved domains page"
ANSWER = "Type just that one line, then a Source line naming the IANA page."
_CLAIM = re.compile(r"\b(copy|copies|copied|copying|paste|pastes|pasted|pasting)\b", re.I)

# The run's first requests (Ollama's model list, the embedding and the
# annotation model's preload): the cassette does not exist yet, and every
# later call (annotations, diffs, SOP, synthesis, questions, the answer
# revision) follows from their answers.
MISSING_KEYS: tuple[str, ...] = (
    "d0da2494ea7fb173d0d0c9a4067b288be6e5842e8e67fb217ec9906e2a410d46",  # /api/tags
    "68fd4c0c191e9b6e34802659ef608b3bb0390b7cfe1081ccdd20eafb3f344f05",  # /api/embed
    "f9c3abdbec72602f4d6d2f5f86c1c6443be24e8a63606acfeb1bc898d9d31579",  # /api/generate
)
_PATH = CASSETTE_DIR / f"{JOURNEY}.jsonl"
_ENTRIES = load_entries(_PATH) if _PATH.exists() else {}
needs_recording = pytest.mark.skipif(
    any(key not in _ENTRIES for key in MISSING_KEYS) and os.environ.get("AH_RECORD") != "1",
    reason=(
        f"cassettes/{JOURNEY}.jsonl lacks the typing session's requests (WP-Q2b): "
        + ", ".join(k for k in MISSING_KEYS if k not in _ENTRIES)
        + " (and the calls they unblock); record with AH_RECORD=1 on the test Mac"
    ),
)
# The questions are the ledger's since WP-L4 (no model call), so the answer
# revision's prompt, which quotes the question, and the embedding of the
# answered procedure are new requests.
REVISION_MISSING_KEYS: tuple[str, ...] = (
    "b8ac16d6e56932ea4ab826d4b3316b925337fa874bf71efbf08a96084706167a",  # /api/generate
    "9bd6ccc1158d3dbfd6bf8ca504c2a69a648fe9e7a6b1d7322f2a57026b5053e8",  # /api/embed
)
needs_revision_recording = pytest.mark.skipif(
    any(key not in _ENTRIES for key in REVISION_MISSING_KEYS)
    and os.environ.get("AH_RECORD") != "1",
    reason=(
        f"cassettes/{JOURNEY}.jsonl lacks the answer revision after the ledger's "
        "questions (WP-L4): "
        + ", ".join(k for k in REVISION_MISSING_KEYS if k not in _ENTRIES)
        + "; record with AH_RECORD=1 on the test Mac"
    ),
)


def _start(events_db, worker, clock, temp_home, cassette):
    return start_focus_run(
        events_db=events_db, worker=worker, clock=clock, temp_home=temp_home,
        cassette=cassette, journey=JOURNEY, fixture=JOURNEY,
    )


def _tick_until(run, clock, done, *, max_ticks: int = 6):
    for _ in range(max_ticks):
        run.ticks.append(run.worker.run_tick())
        result = done()
        if result:
            return result
        clock.advance(seconds=TICK_SECONDS)
    raise AssertionError(f"not reached after {max_ticks} ticks: {run.ticks!r}")


def _pending_questions(state_dir) -> dict | None:
    path = state_dir / FOCUS_QUESTIONS_FILE
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if data.get("status") == "pending" else None


def _first_frame_times(run, proc: dict) -> list[str]:
    import sqlite3

    with sqlite3.connect(run.fixture.db_path) as conn:
        times = dict(conn.execute("SELECT id, timestamp FROM events"))
    return [min(times[f] for f in s["evidence"]["frames"]) for s in proc["steps"]]


@needs_recording
def test_the_typing_session_becomes_the_steps_its_frames_show(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    run = _start(events_db, worker, clock, temp_home, cassette)
    state_dir = run.worker.state_dir
    questions = _tick_until(
        run, clock, lambda: _pending_questions(state_dir) or run.procedures(),
    )
    procs = run.procedures()
    # Waiting for answers, the Skill is the review queue's draft.
    proc = procs[0] if procs else (
        run.worker.db.get_generated_sop_by_slug(questions["slug"])["sop_json"]
    )

    # Every step cites its frames, and the steps follow them.
    assert all(s["evidence"]["frames"] for s in proc["steps"]), proc["steps"]
    times = _first_frame_times(run, proc)
    assert times == sorted(times), [(s["action"], t) for s, t in zip(proc["steps"], times)]
    packet = parse_packet(render_packet(proc, target="claude-skill"))
    steps_text = packet.section("Steps")
    # Both typed lines; no copy or paste; no tab-title click; no selection.
    assert LINE_1 in steps_text and LINE_2 in steps_text, steps_text
    for step in packet.steps():
        assert not _CLAIM.search(step.text.splitlines()[0]), step.text
        assert not step.text.splitlines()[0].split("] ", 1)[-1].lower().startswith("select")
    assert "Example Domain" not in steps_text

    # No question assumes a copy that was not recorded.
    for q in (questions or {}).get("questions") or []:
        assert not _CLAIM.search(" ".join([q["question"], q["context"], q["default"]])), q


@needs_recording
@needs_revision_recording
def test_a_typed_answer_revises_the_typing_steps(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    run = _start(events_db, worker, clock, temp_home, cassette)
    state_dir = run.worker.state_dir
    questions = _tick_until(run, clock, lambda: _pending_questions(state_dir))
    assert questions["questions"], "the recorded run asked no question"
    # The app's Submit: the first answer typed over the guess, the rest left
    # as the guesses (accepted, they change nothing).
    questions["status"] = "answered"
    questions["answers"] = {
        str(q["index"]): ANSWER if q["index"] == 0 else q["default"]
        for q in questions["questions"]
    }
    questions["answer_sources"] = {
        str(q["index"]): "typed" if q["index"] == 0 else "accepted_guess"
        for q in questions["questions"]
    }
    (state_dir / FOCUS_QUESTIONS_FILE).write_text(json.dumps(questions), encoding="utf-8")
    clock.advance(seconds=TICK_SECONDS)

    proc = _tick_until(run, clock, lambda: (run.procedures() or [None])[0])
    revised = [s for s in proc["steps"] if s["evidence"].get("source") == "user_answer"]
    assert revised, [s["action"] for s in proc["steps"]]
    assert all(s["evidence"]["user_answer"]["answer"] == ANSWER for s in revised)
    typed = "\n".join(str(s.get("input") or "") + " " + s["action"] for s in proc["steps"])
    assert LINE_1 in typed and LINE_2 in typed, typed
    clarification = revised[0]["evidence"]["user_answer"]["clarification"]
    assert {"kind": "answer", "id": f"agent_clarifications.{clarification}"} in (
        derive_knowledge(proc)["steps"]["evidence"]
    )
