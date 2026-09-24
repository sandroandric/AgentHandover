"""WP-Q2b pins: a focus Skill's steps are the ones its frames show.

The live run of 2026-09-24 (session 46b767ef, qwen3.5:4b): the new TextEdit
document came first although Safari was on screen before it, "click the
'Example Domain' link" named a tab title, "Select the typed text" happened
in no frame, and the second typed line was lost
(``fixtures/iana_typing_session.py``).
"""

from __future__ import annotations

import copy
import json

from agenthandover_worker.clipboard_grounding import attach_clipboard_events
from agenthandover_worker.procedure_schema import sop_to_procedure, validate_procedure
from agenthandover_worker.skill_render import render_packet
from agenthandover_worker.sop_generator import SOPGenerator
from agenthandover_worker.sop_linter import lint_sop
from agenthandover_worker.step_grounding import ground_steps, typed_lines
from fixtures import iana_typing_session as iana
from fixtures.vlm_answers import load_answer


def _generate(timeline: list[dict], answer: dict) -> dict:
    generator = SOPGenerator()
    generator._call = lambda prompt, **_kw: (json.dumps(answer), 1.0)
    result = generator.generate_from_focus(timeline, iana.TITLE)
    assert result.success, result.error
    return result.sop


def _step(action: str, frames: list[str], **params) -> dict:
    return {"step": action, "description": "", "parameters": dict(params),
            "evidence": {"frames": frames, "dom_snapshot_ids": []}}


def _frame(event_id: str, second: int, app: str, content: list[str], **extra) -> dict:
    return {"event_id": event_id, "timestamp": f"2026-09-24T09:00:{second:02d}Z",
            "annotation": {"app": app, "location": extra.pop("location", "")},
            "diff": extra.pop("diff", None), "content_text": content,
            "click_targets": extra.pop("click_targets", []), **extra}


# ---------------------------------------------------------------------------
# The live run
# ---------------------------------------------------------------------------


def test_the_live_runs_steps_become_what_the_frames_show() -> None:
    sop = _generate(iana.timeline(), load_answer("iana_typing_46b767ef"))
    steps = sop["steps"]
    assert [s["step"] for s in steps] == [
        # Safari was on screen first: its step comes first.
        "Open Safari and navigate to the IANA reserved domains page.",
        "Open TextEdit and create a new document.",
        "Switch back to TextEdit and type the reserved domain list.",
        f'Type "{iana.LINE_2}"',
    ]
    # The first line fills the typing step's empty input; the second, in no
    # step, is a step of its own on the frame that shows it.
    assert steps[2]["parameters"]["input"] == iana.LINE_1
    assert steps[2]["evidence"]["grounding"]["status"] == "input_added"
    assert steps[3]["parameters"]["input"] == iana.LINE_2
    assert steps[3]["parameters"]["app"] == "TextEdit"
    assert steps[3]["evidence"]["frames"] == ["e5"]
    assert steps[3]["evidence"]["grounding"]["status"] == "added"
    # What no frame shows is dropped, with the reason kept.
    dropped = {d["action"]: d["reason"] for d in sop["dropped_steps"]}
    assert set(dropped) == {
        "Scroll down the page and click the 'Example Domain' link.",
        "Select the typed text in the TextEdit document.",
    }
    assert "'Example Domain' is not in the frames" in dropped[
        "Scroll down the page and click the 'Example Domain' link."]
    assert "selection" in dropped["Select the typed text in the TextEdit document."]
    assert sop["session_clipboard"] == []
    assert lint_sop(sop, source="focus_v2").errors == []

    # The procedure keeps the dropped steps as evidence; the packet names
    # them and carries every typed line.
    proc = sop_to_procedure(sop)
    assert validate_procedure(proc) == []
    assert proc["evidence"]["dropped_steps"] == sop["dropped_steps"]
    packet = render_packet(proc, target="claude-skill")
    assert iana.LINE_1 in packet and iana.LINE_2 in packet
    assert "- Not a step: \"Select the typed text in the TextEdit document.\"" in packet


def test_the_tab_title_is_chrome_the_page_text_is_content() -> None:
    timeline = iana.timeline()
    assert "Example Domain" not in timeline[0]["content_text"]
    assert "Example domains" in timeline[0]["content_text"]
    assert timeline[1]["content_text"] == []  # a blank document: only chrome


def test_typed_lines_are_the_ones_the_document_gained() -> None:
    # The partial line (typing in progress) is part of the full one.
    assert typed_lines(iana.timeline()) == [(3, iana.LINE_1), (4, iana.LINE_2)]


def test_a_line_caught_mid_typing_stands_for_its_finished_line() -> None:
    """Seen live (focus_iana_typed): a frame taken while the user typed
    showed "... example.or"; the finished line was already on screen by
    its next frame, so only the partial counted and became a junk
    'Type "example.or"' step. A line a later frame of the same app
    extends is typing in progress: it stands for the finished line."""
    partial = "Reserved for documentation: example.com, example.net, example.or"
    full = "Reserved for documentation: example.com, example.net, example.org"
    timeline = [
        _frame("a", 0, "TextEdit", []),
        _frame("b", 2, "TextEdit", [partial]),
        _frame("c", 4, "TextEdit", [full]),
        _frame("d", 6, "TextEdit", [full]),
    ]
    timeline[1]["annotation"]["key_text"] = {"typed_text": partial}
    timeline[2]["annotation"]["key_text"] = {"typed_text": ""}
    assert typed_lines(timeline) == [(1, full)]
    # Counted as it is when nothing extends it.
    timeline[2]["content_text"] = [partial]
    timeline[3]["content_text"] = [partial]
    assert typed_lines(timeline) == [(1, partial)]


def test_a_typing_step_with_the_start_of_a_line_gets_the_whole_line() -> None:
    """Seen in focus_iana_typed: "Type the phrase 'Reserved for
    documentation:' followed by the first domain name" had input
    "Reserved for documentation: example.com"; the line the user finished
    also listed example.net and example.org."""
    partial = "Reserved for documentation: example.com"
    full = "Reserved for documentation: example.com, example.net, example.org"
    timeline = [
        _frame("a", 0, "TextEdit", []),
        _frame("b", 2, "TextEdit", [partial]),
        _frame("c", 4, "TextEdit", [full]),
    ]
    timeline[1]["annotation"]["key_text"] = {"typed_text": partial}
    steps = [_step("Type the phrase and the first domain", ["b"], input=partial)]
    report = ground_steps(steps, timeline)
    assert steps[0]["parameters"]["input"] == full
    assert report["added"][0]["original"] == partial
    assert len(steps) == 1


def test_a_line_only_a_verify_mentions_is_not_typed_by_any_step() -> None:
    full = "Reserved for documentation: example.com, example.net, example.org"
    timeline = [_frame("a", 0, "TextEdit", []), _frame("b", 2, "TextEdit", [full])]
    timeline[1]["annotation"]["key_text"] = {"typed_text": full}
    steps = [_step("Type the source line", ["b"], input="Source: IANA",
                   verify="The full text 'Reserved for documentation: {{domains}}' is visible")]
    variables = [{"name": "domains", "example": "example.com, example.net, example.org"}]
    report = ground_steps(steps, timeline, variables)
    assert [a["input"] for a in report["added"]] == [full]


def test_a_loading_page_is_not_a_scroll_but_shifted_content_is() -> None:
    """Seen live: the diff model called a loading page "scrolled down"; the
    check trusted those words. Now only the content says: lines that left
    while others arrived, on the same page."""
    loaded = ["IANA-managed Reserved Domains", "Example Domains", "Test Domains"]
    timeline = [
        _frame("a", 0, "Safari", loaded[:1], location="iana.org"),
        _frame("b", 2, "Safari", loaded, location="iana.org",
               diff={"diff_type": "action", "step_description": "The user scrolled down"}),
    ]
    steps = [_step("Scroll down the page to the example domains", ["b"])]
    assert ground_steps(steps, timeline)["dropped"], steps
    timeline[1]["content_text"] = ["Example Domains", "Test Domains", "Second-level domains"]
    timeline[0]["content_text"] = ["IANA-managed Reserved Domains", "Example Domains", "Test Domains"]
    steps = [_step("Scroll down the page to the example domains", ["b"])]
    assert not ground_steps(steps, timeline)["dropped"]


def test_a_new_untitled_window_is_not_a_save() -> None:
    """Seen live: "Save the document and rename it to 'Untitled 4'" for a
    new TextEdit window that was never saved."""
    timeline = [
        _frame("a", 0, "TextEdit", [], location="Untitled 3"),
        _frame("b", 2, "TextEdit", ["Reserved for documentation"], location="Untitled 4"),
    ]
    steps = [_step("Type the line", ["b"], input="Reserved for documentation"),
             _step("Save the document and rename it to 'Untitled 4'", ["b"])]
    report = ground_steps(steps, timeline)
    assert [s["step"] for s in steps] == ["Type the line"], report
    timeline[1]["annotation"]["location"] = "IANA notes"
    steps = [_step("Save the document as 'IANA notes'", ["b"])]
    assert not ground_steps(steps, timeline)["dropped"]


def test_one_stray_citation_does_not_move_a_step() -> None:
    """Seen in the focus_dailynews recording: "Navigate to the GitHub
    repository page" cited frames 11, 13, 14, 15; frame 11 was the paste
    of the previous article. Ordered by the first cited frame it jumped
    ahead of that paste; by its longest run of frames (13-15) it stays
    after it."""
    timeline = [_frame(f"f{i}", i, "Safari" if i in (13, 14, 15) else "Notes", [])
                for i in range(10, 17)]
    steps = [
        _step("Navigate to the GitHub repository page", ["f11", "f13", "f14", "f15"]),
        _step("Paste the second article details into the note", ["f11"]),
        _step("Type a summary sentence", ["f12"]),
        _step("Copy the project title and URL", ["f15", "f16"]),
    ]
    assert ground_steps(steps, timeline)["reordered"]
    assert [s["step"] for s in steps] == [
        "Paste the second article details into the note",
        "Type a summary sentence",
        "Navigate to the GitHub repository page",
        "Copy the project title and URL",
    ]


# ---------------------------------------------------------------------------
# The checks, one at a time
# ---------------------------------------------------------------------------


def test_a_click_its_frames_show_is_kept() -> None:
    send = {"target": {"ariaLabel": "Send now", "tagName": "button"}}
    timeline = [
        _frame("a", 0, "Google Chrome", ["# daily-news", "Daily news - 15 Sep"]),
        _frame("b", 10, "Google Chrome", ["# daily-news"], click_targets=[send]),
    ]
    steps = [
        _step("Click on the story titled 'Daily news - 15 Sep'", ["a"]),
        _step("Click the Send button to post the message", ["b"]),
    ]
    before = copy.deepcopy(steps)
    report = ground_steps(steps, timeline)
    assert steps == before
    assert report == {"dropped": [], "rewritten": [], "added": [], "reordered": False}


def test_a_click_on_a_control_in_a_dom_snapshot_is_kept() -> None:
    nodes = [{"tag": "a", "text": "Example Domain", "selector": "a#ex", "name": "Example Domain"}]
    timeline = [_frame("a", 0, "Google Chrome", ["Reserved Domains"], dom_nodes=nodes)]
    steps = [_step("Click the 'Example Domain' link", ["a"])]
    assert ground_steps(steps, timeline)["dropped"] == []


def test_an_unshown_click_is_cut_from_a_step_that_does_more() -> None:
    timeline = [_frame("a", 0, "Safari", ["IANA-managed Reserved Domains"],
                       location="https://www.iana.org/domains/reserved")]
    steps = [_step("Open Safari and click the 'Example Domain' link", ["a"])]
    steps[0]["description"] = "Go to the Example Domain page."
    report = ground_steps(steps, timeline)
    assert steps[0]["step"] == "Open Safari"
    assert steps[0]["description"] == "Open Safari."
    record = steps[0]["evidence"]["grounding"]
    assert record["status"] == "rewritten"
    assert record["original"]["step"] == "Open Safari and click the 'Example Domain' link"
    assert [r["action"] for r in report["rewritten"]] == ["Open Safari"]
    proc = sop_to_procedure({"slug": "s", "title": "S", "steps": steps})
    assert proc["steps"][0]["evidence"]["grounding"] == record
    packet = render_packet(proc, target="claude-skill")
    assert "Unconfirmed: cut from the step: 'Example Domain' is not in the frames" in packet


def test_a_quoted_name_is_not_split_at_its_and() -> None:
    timeline = [_frame("a", 0, "Google Chrome", ["Terms and Conditions"])]
    steps = [_step("Click the 'Terms and Conditions' link", ["a"])]
    assert ground_steps(steps, timeline)["rewritten"] == []
    assert steps[0]["step"] == "Click the 'Terms and Conditions' link"


def test_pressing_a_key_is_not_a_control() -> None:
    timeline = [_frame("a", 0, "TextEdit", ["Reserved for documentation"])]
    steps = [_step("Press 'Return' to start a new line", ["a"])]
    assert ground_steps(steps, timeline)["dropped"] == []


def test_a_selection_needs_a_frame_that_shows_one() -> None:
    timeline = [_frame("a", 0, "TextEdit", ["Reserved"]), _frame("b", 10, "TextEdit", ["Reserved"])]
    attach_clipboard_events(timeline, [])
    steps = [_step("Select the typed text", ["a"])]
    assert len(ground_steps(steps, timeline)["dropped"]) == 1
    # A copy right after the step's frame: the text was selected.
    attach_clipboard_events(timeline, [{"kind": "copy", "timestamp": "2026-09-24T09:00:05Z"}])
    steps = [_step("Select the typed text", ["a"])]
    assert ground_steps(steps, timeline)["dropped"] == []
    # A diff that says so.
    timeline[0]["diff"] = {"diff_type": "action", "actions": ["The user selected all the text"]}
    attach_clipboard_events(timeline, [])
    steps = [_step("Select the typed text", ["a"])]
    assert ground_steps(steps, timeline)["dropped"] == []


def test_a_step_that_cites_no_frame_is_dropped() -> None:
    timeline = [_frame("a", 0, "Notes", ["Daily news"])]
    steps = [_step("Open Notes", ["a"]), _step("Save the note", [])]
    report = ground_steps(steps, timeline)
    assert [s["step"] for s in steps] == ["Open Notes"]
    assert report["dropped"] == [
        {"action": "Save the note", "frames": [], "reason": "the step cites no recorded frame"},
    ]


def test_steps_without_any_citation_are_left_alone() -> None:
    timeline = [_frame("a", 0, "Notes", ["Daily news"])]
    steps = [_step("Open Notes", []), _step("Click the 'Share' button", [])]
    before = copy.deepcopy(steps)
    assert ground_steps(steps, timeline)["dropped"] == []
    assert steps == before


def test_steps_follow_their_first_frame_and_keep_their_order_on_a_tie() -> None:
    timeline = [_frame(e, i * 10, "Notes", []) for i, e in enumerate("abc")]
    steps = [_step("Paste the title", ["b"]), _step("Type the summary", ["b", "c"]),
             _step("Open Notes", ["a", "c"])]
    report = ground_steps(steps, timeline)
    assert report["reordered"] is True
    assert [s["step"] for s in steps] == ["Open Notes", "Paste the title", "Type the summary"]


def test_a_typed_line_in_a_variable_example_is_covered() -> None:
    timeline = [
        _frame("a", 0, "Notes", []),
        _frame("b", 10, "Notes", ["Readers keep going while one writer appends."],
               diff={"diff_type": "action", "actions": ["typed"],
                     "inputs": [{"field": "note", "value": "Readers keep going while one writer appends."}]}),
    ]
    steps = [_step("Type the summary", ["b"], input="{{summary_text}}")]
    variables = [{"name": "summary_text", "example": "Readers keep going while one writer appends."}]
    assert ground_steps(steps, timeline, variables)["added"] == []


def test_a_fields_placeholder_and_text_already_there_are_not_typed() -> None:
    composer = {"target": {"ariaLabel": "Message #daily-news", "role": "textbox", "tagName": "div"}}
    timeline = [
        _frame("a", 0, "Google Chrome", ["Daily news - 14 Sep (3 links)"], click_targets=[composer]),
        _frame("b", 10, "Google Chrome", ["Daily news - 14 Sep (3 links)", "Message #daily-news"]),
    ]
    timeline[1]["annotation"]["key_text"] = {
        "typed_text": "Message #daily-news\nDaily news - 14 Sep (3 links)",
    }
    assert typed_lines(timeline) == []
