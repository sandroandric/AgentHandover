"""J-focus: a stopped focus recording becomes a Skill an agent can follow.

Load ``focus_dailynews`` (read Hacker News in Chrome, copy three stories
into Notes, post the digest in Slack), write the app's "stopped" signal
and tick the worker on the recorded cassette until the procedure exists;
then read the packet an agent receives.
"""

from __future__ import annotations

import re
import statistics

import pytest

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
    (each annotated and diffed) plus the synthesis calls (WP-C5's bound)."""
    run, _proc, _packet = _run(events_db, worker, clock, temp_home, cassette)
    budget = focus_budget_seconds(run.tape)
    assert run.tape.model_seconds() <= budget, (run.tape.model_seconds(), budget)


def _seconds(entry: dict) -> float:
    return entry["response"].get("total_duration", 0) / 1e9


def focus_budget_seconds(tape) -> float:
    """``KEYFRAMES`` x (median annotation + median frame diff) + every other
    model call of the run (SOP generation, questions, synthesis, embeddings).
    Annotation and diff calls are told apart by their system prompt."""
    annotation, diff, other = [], [], []
    for entry in tape.served:
        request = entry["request"]
        model = request.get("model")
        if not model:
            continue
        profile = get_profile(model)
        system = request.get("system") or ""
        if system == annotation_system_prompt(profile.ann_system):
            annotation.append(_seconds(entry))
        elif system == (profile.diff_system or DIFF_SYSTEM_PROMPT):
            diff.append(_seconds(entry))
        else:
            other.append(_seconds(entry))
    assert annotation and other, "the run made no annotation or no synthesis call"
    per_keyframe = statistics.median(annotation) + (statistics.median(diff) if diff else 0.0)
    return KEYFRAMES * per_keyframe + sum(other)


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


@pytest.mark.xfail(strict=True, reason=(
    "no browser step gets an anchor: a DwellSnapshot frame carries no URL "
    "(the daemon attaches app_state only on an app switch, "
    "crates/daemon/src/observer/event_loop.rs:268-276), so the WINDOW header "
    "built by SceneAnnotator.annotate_event (scene_annotator.py:908-910) has "
    "none and the model invents the Slack location (https://example.com/slack); "
    "_attach_clicks (focus_processor.py:688-722) then matches no frame to the "
    "app.slack.com clicks, and the Hacker News click lands on the frame before "
    "it while the step cites the frame after it, which is all "
    "_anchor_for_step (sop_generator.py:1183-1209) looks at"
))
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
