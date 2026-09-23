"""J-focus: a stopped focus recording becomes a Skill an agent can follow.

Load ``focus_dailynews`` (read Hacker News in Chrome, copy three stories
into Notes, post the digest in Slack), write the app's "stopped" signal
and tick the worker on the recorded cassette until the procedure exists;
then read the packet an agent receives.
"""

from __future__ import annotations

import re
import statistics

from agenthandover_worker.frame_differ import DIFF_SYSTEM_PROMPT
from agenthandover_worker.model_profiles import get_profile
from agenthandover_worker.scene_annotator import annotation_system_prompt
from agenthandover_worker.skill_render import render_packet
from journeys.focus_run import start_focus_run, tick_until_skill
from journeys.packet import assert_matches_golden, parse_packet

# WP-C5: a focus run annotates at most this many keyframes before the Skill.
KEYFRAMES = 20


def _run(events_db, worker, clock, temp_home, cassette):
    run = start_focus_run(
        events_db=events_db, worker=worker, clock=clock,
        temp_home=temp_home, cassette=cassette,
    )
    proc = tick_until_skill(run, clock)
    return run, proc, parse_packet(render_packet(proc, target="claude-skill"))


def test_focus_recording_becomes_a_complete_skill(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    run, proc, packet = _run(events_db, worker, clock, temp_home, cassette)

    # The frontmatter parsed as YAML and names the Skill.
    assert packet.frontmatter["name"]
    # The description is a sentence of its own, not the title.
    description = packet.frontmatter["description"]
    title = proc["title"]
    assert title == run.fixture.manifest["focus_session"]["title"]
    assert description.rstrip(".").strip().lower() != title.lower()
    assert description.endswith((".", "!", "?"))
    assert len(description.split()) >= 5, description

    # Every step tells the agent where to act, on what, and how to check it.
    steps = packet.steps()
    assert len(steps) >= 3, packet.section("Steps")
    for step in steps:
        for field in ("App", "Target", "Verify"):
            assert step.fields.get(field), f"step {step.number} has no {field}:\n{step.text}"


def test_focus_packet_matches_the_reviewed_golden(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    """Replaying the cassette gives the packet reviewed once in
    ``golden/``; J-ollama-down compares against the same file."""
    _run_, proc, _packet = _run(events_db, worker, clock, temp_home, cassette)
    assert_matches_golden("focus_dailynews.claude-skill.md", render_packet(proc, target="claude-skill"))


def test_focus_run_stays_within_its_model_budget(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    """Time to Skill, in cassette model-seconds, is at most 20 keyframes
    (each annotated and diffed) plus the synthesis calls (WP-C5's bound).
    The keyframes the SOP did not need are annotated after the Skill."""
    run, _proc, _packet = _run(events_db, worker, clock, temp_home, cassette)
    assert run.served_at_skill is not None, "the focus processor never reached the Skill"
    to_skill = run.tape.served[:run.served_at_skill]
    annotations = [e for e in to_skill if _kind(e) == "annotation"]
    assert len(annotations) <= KEYFRAMES, len(annotations)
    spent = sum(_seconds(e) for e in to_skill)
    budget = focus_budget_seconds(to_skill)
    assert spent <= budget, (spent, budget)


def _seconds(entry: dict) -> float:
    return entry["response"].get("total_duration", 0) / 1e9


def _kind(entry: dict) -> str:
    """``annotation``, ``diff`` or ``other``, told apart by the system prompt."""
    request = entry["request"]
    model = request.get("model")
    if not model:
        return "other"
    profile = get_profile(model)
    system = request.get("system") or ""
    if system == annotation_system_prompt(profile.ann_system):
        return "annotation"
    if system == (profile.diff_system or DIFF_SYSTEM_PROMPT):
        return "diff"
    return "other"


def focus_budget_seconds(served: list[dict]) -> float:
    """``KEYFRAMES`` x (median annotation + median frame diff) + every other
    model call (SOP generation, synthesis, questions, embeddings)."""
    kinds: dict[str, list[float]] = {"annotation": [], "diff": [], "other": []}
    for entry in served:
        kinds[_kind(entry)].append(_seconds(entry))
    assert kinds["annotation"] and kinds["other"], "no annotation or no synthesis call"
    per_keyframe = statistics.median(kinds["annotation"]) + (
        statistics.median(kinds["diff"]) if kinds["diff"] else 0.0
    )
    return KEYFRAMES * per_keyframe + sum(kinds["other"])


def test_focus_packet_has_no_assumed_text(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    _run_, _proc, packet = _run(events_db, worker, clock, temp_home, cassette)
    assert not re.search(r"\bAssum(e|ed|ing)\b", packet.body), packet.body


def test_focus_paste_step_carries_the_clipboard_text(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    """The digest was copied from Notes and pasted into Slack: the step
    that pastes it quotes what was on the clipboard."""
    run, _proc, packet = _run(events_db, worker, clock, temp_home, cassette)
    story_titles = [
        line.split("\n")[0] for line in _clipboard_previews(run.fixture)
    ]
    paste_steps = [s for s in packet.steps() if re.search(r"\bpast(e|es|ed|ing)\b", s.text, re.I)]
    assert paste_steps, packet.section("Steps")
    assert any(
        title in step.text for step in paste_steps for title in story_titles
    ), (story_titles, [s.text for s in paste_steps])


def test_focus_browser_step_carries_a_dom_anchor(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    """At least one browser step names the element by role and accessible
    name (the Slack composer is ``role=textbox``, ``Message #daily-news``)."""
    _run_, proc, _packet = _run(events_db, worker, clock, temp_home, cassette)
    anchors = [
        s.get("anchor") for s in proc.get("steps") or []
        if isinstance(s, dict) and isinstance(s.get("anchor"), dict)
    ]
    assert any(a.get("role") and a.get("name") for a in anchors), [
        (s.get("step_id"), s.get("selector"), s.get("anchor")) for s in proc.get("steps") or []
    ]


def _clipboard_previews(fixture) -> list[str]:
    import json
    import sqlite3

    conn = sqlite3.connect(str(fixture.db_path))
    try:
        kinds = [json.loads(k) for (k,) in conn.execute("SELECT kind_json FROM events")]
    finally:
        conn.close()
    return [k["content_preview"] for k in kinds if k.get("type") == "ClipboardChange"]
