"""WP-Q1 pins: what a Skill says is grounded in what was recorded.

From the live run of 2026-09-23 (qwen3.5:4b for every role): a typing
session in TextEdit became "Copy ... / Paste ..." steps with no clipboard
event recorded, a frame diff read TextEdit's font toolbar ("Halvetica
Ragular") as an edit, and the Skill used placeholders it never declared.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from agenthandover_worker.clipboard_grounding import (
    attach_clipboard_events,
    clipboard_note,
    ground_clipboard_claims,
)
from agenthandover_worker.focus_processor import FocusProcessor
from agenthandover_worker.frame_differ import (
    FrameDiffer,
    content_lines,
    ground_text_changes,
    is_chrome_line,
    is_numeric_line,
    is_ruler_line,
)
from agenthandover_worker.focus_questioner import (
    ANSWER_ACCEPTED_GUESS,
    ANSWER_TYPED,
    FocusQuestion,
    merge_answers,
    parse_qa_result_from_file,
    write_focus_questions,
)
from agenthandover_worker.ledger import derive_knowledge, validate_knowledge
from agenthandover_worker.procedure_schema import sop_to_procedure, validate_procedure
from agenthandover_worker.skill_render import render_packet
from agenthandover_worker.sop_generator import SOPGenerator, _build_focus_prompt, _vlm_sop_to_template
from agenthandover_worker.sop_linter import lint_sop


# ---------------------------------------------------------------------------
# Variables: every placeholder is declared, every declared variable is used
# ---------------------------------------------------------------------------


def _sop(steps: list[dict], variables: list[dict]) -> dict:
    return {"title": "Look up the IANA reserved domains page and note it",
            "steps": steps, "variables": variables}


def test_a_variable_wired_into_a_description_is_kept() -> None:
    """The run logged "Wired 4 variable substitution(s)" and shipped
    ``variables: []``: the wiring wrote into descriptions, which the
    unused-variable pass did not read, so it dropped every variable."""
    template = _vlm_sop_to_template(_sop(
        steps=[{"action": "Open TextEdit", "app": "TextEdit",
                "description": "Write the note about Reserved domains per RFC 2606.",
                "verify": "The note shows the text"}],
        variables=[{"name": "source_note", "example": "Reserved domains per RFC 2606",
                    "description": "What the note says"}],
    ))
    assert "{{source_note}}" in template["steps"][0]["description"]
    assert [v["name"] for v in template["variables"]] == ["source_note"]
    assert template["variables"][0]["example"] == "Reserved domains per RFC 2606"
    assert not [i for i in lint_sop(template).issues if "source_note" in i.message]
    assert validate_procedure(sop_to_procedure(template)) == []


def test_a_placeholder_the_model_did_not_declare_is_declared() -> None:
    template = _vlm_sop_to_template(_sop(
        steps=[{"action": "Type {{iana_reserved_domains_text}} into the document",
                "app": "TextEdit", "description": "The note body.", "verify": "Text shows"}],
        variables=[{"name": "unused_thing", "example": "never used here"}],
    ))
    assert [v["name"] for v in template["variables"]] == ["iana_reserved_domains_text"]
    declared = template["variables"][0]
    assert declared["example"] == ""  # not seen in the recording: unknown
    assert "iana reserved domains text" in declared["description"]
    assert lint_sop(template).valid


def test_lint_and_validation_fail_on_an_undeclared_placeholder() -> None:
    proc = {
        "schema_version": "3.2.0", "id": "p", "title": "P",
        "steps": [
            {"action": "Paste", "verify": "The note reads {{source_note}}"},
            {"action": "Save", "description": "Keep {{ file_name }}", "verify": "Saved"},
        ],
        "inputs": [],
    }
    errors = validate_procedure(proc)
    assert any("{{source_note}}" in e and "{{file_name}}" in e for e in errors), errors
    assert {i.field for i in lint_sop(proc).errors} >= {
        "variables.source_note", "variables.file_name",
    }
    proc["inputs"] = [{"name": "source_note", "type": "text"},
                      {"name": "file_name", "type": "text"}]
    assert validate_procedure(proc) == []
    assert not [i for i in lint_sop(proc).errors if i.field.startswith("variables.")]


# ---------------------------------------------------------------------------
# Q&A: a submitted guess is an accepted guess, a cleared field is unanswered
# ---------------------------------------------------------------------------

# What the app's Submit writes (``FocusQATests.testSubmitWritesTheSharedFixture``
# encodes the same file from the app's side): the first field left with its
# pre-filled guess, the second typed over, the third cleared.
ANSWERED_FILE = Path(__file__).parent / "fixtures" / "focus_questions_answered.json"


def _answered() -> dict:
    return json.loads(ANSWERED_FILE.read_text(encoding="utf-8"))


def test_the_apps_answer_file_parses_with_its_sources() -> None:
    result = parse_qa_result_from_file(_answered())
    assert result is not None
    assert result.answers == {0: "Assume the page is public", 1: "Once a week, on Mondays"}
    assert [result.source(i) for i in range(3)] == [
        ANSWER_ACCEPTED_GUESS, ANSWER_TYPED, ANSWER_TYPED,
    ]
    assert [q.step_indexes for q in result.questions] == [[], [0], [2]]
    assert result.counts() == {"typed": 1, "accepted_guess": 1, "unanswered": 1}
    assert result.status() == "partial"


def test_an_older_apps_answers_were_typed() -> None:
    data = _answered()
    del data["answer_sources"]
    result = parse_qa_result_from_file(data)
    assert result is not None
    assert result.counts() == {"typed": 2, "accepted_guess": 0, "unanswered": 1}


def test_the_status_counts_what_the_user_did() -> None:
    """The run's worker logged "status=answered" for three questions it
    stored as unanswered: Submit with every field cleared."""
    data = _answered()
    data["answers"], data["answer_sources"] = {}, {}
    result = parse_qa_result_from_file(data)
    assert result is not None and result.status() == "unanswered"
    data = _answered()
    data["answers"]["2"] = "The note lists example.com"
    assert parse_qa_result_from_file(data).status() == "answered"


def test_merged_answers_keep_their_source_and_the_ledger_tells_them_apart() -> None:
    result = parse_qa_result_from_file(_answered())
    proc = {"id": "look-up", "title": "Look up", "steps": [
        {"action": "Open Safari", "target": "Safari", "parameters": {}},
    ]}
    merged = merge_answers(proc, result)
    clarifications = merged["agent_clarifications"]
    assert [(c["status"], c.get("source")) for c in clarifications] == [
        ("answered", ANSWER_ACCEPTED_GUESS),
        ("answered", ANSWER_TYPED),
        ("unanswered", None),
    ]
    # The accepted guess fills nothing; the typed scope answer sets the recurrence.
    assert merged.get("recurrence") == "Once a week, on Mondays"

    ledger = derive_knowledge(merged)
    assert ledger["inputs"]["status"] == "hypothesis"  # access: the accepted guess
    assert ledger["inputs"]["source"] == "user"
    assert ledger["inputs"]["evidence"] == [{"kind": "answer", "id": "agent_clarifications.0"}]
    assert ledger["trigger"]["status"] == "known"  # scope: typed
    assert ledger["trigger"]["value_ref"] == "agent_clarifications.1"
    assert ledger["verification"]["source"] != "user"  # cleared: nothing learned
    assert validate_knowledge(ledger) == []


def test_an_accepted_guess_does_not_lower_a_known_slot() -> None:
    proc = {
        "id": "p", "title": "P", "steps": [],
        "agent_clarifications": [
            {"category": "scope", "question": "How often?", "answer": "Every day",
             "status": "answered", "source": ANSWER_ACCEPTED_GUESS},
        ],
        "knowledge": {"trigger": {
            "status": "known", "value_ref": "recurrence", "source": "agent",
            "evidence": [{"kind": "run", "id": "run-1"}], "updated_at": "2026-09-01T00:00:00+00:00",
        }},
    }
    assert derive_knowledge(proc)["trigger"]["status"] == "known"
    assert derive_knowledge(proc)["trigger"]["source"] == "agent"
    proc["knowledge"] = {}
    assert derive_knowledge(proc)["trigger"]["status"] == "hypothesis"
    proc["agent_clarifications"].append(
        {"category": "timing", "question": "When?", "answer": "At 9",
         "status": "answered", "source": ANSWER_TYPED},
    )
    trigger = derive_knowledge(proc)["trigger"]
    assert (trigger["status"], trigger["value_ref"]) == ("known", "agent_clarifications.1")


def test_the_questions_file_carries_the_title_as_typed(tmp_path: Path) -> None:
    questions = [FocusQuestion(question="How often?", category="scope", context="",
                               default="Every day", step_indexes=[1])]
    path = write_focus_questions(
        tmp_path, "83f40b9b", "look-up-the-iana-reserved-domains-page-and-note-it",
        questions, title="Look up the IANA reserved domains page and note it",
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["title"] == "Look up the IANA reserved domains page and note it"
    assert data["questions"][0]["step_indexes"] == [1]
    assert data["status"] == "pending"


# ---------------------------------------------------------------------------
# Clipboard grounding: copy/paste only where the clipboard recorded one
# ---------------------------------------------------------------------------


def _ts(second: int) -> str:
    return f"2026-09-23T10:00:{second:02d}Z"


def _timeline(clipboard: list[tuple[str, int]] | None) -> list[dict]:
    """Four frames 10 s apart (Safari, Safari, TextEdit, TextEdit) and the
    session's clipboard events ``(kind, second)``; None: no clipboard record
    at all (a passive timeline)."""
    frames = [
        {"event_id": f"f{i}", "timestamp": _ts(i * 10),
         "annotation": {"app": app, "task_context": {"what_doing": what}}, "diff": None}
        for i, (app, what) in enumerate([
            ("Safari", "Reading the IANA reserved domains page"),
            ("Safari", "Reading the IANA reserved domains page"),
            ("TextEdit", "copying text from the Example Domain page into TextEdit"),
            ("TextEdit", "Typing in TextEdit"),
        ])
    ]
    if clipboard is not None:
        attach_clipboard_events(frames, [
            {"kind": kind, "timestamp": _ts(second), "event_id": f"c{second}"}
            for kind, second in clipboard
        ])
    return frames


def _live_run_steps() -> list[dict]:
    """The Skill's steps from the live run, citing the frames they came from."""
    return [
        {"step": "Copy the text regarding IANA reserved domains", "description":
         "Select and copy the paragraph.", "parameters": {"verify": "The text is copied to the clipboard"},
         "evidence": {"frames": ["f0", "f1"]}},
        {"step": "Paste the copied text", "description": "Insert the previously copied text.",
         "parameters": {"verify": "The text appears in the document"},
         "evidence": {"frames": ["f2"]}},
        {"step": "Save the document", "description": "Keep the note.",
         "parameters": {"verify": "The title bar shows no edits"}, "evidence": {"frames": ["f3"]}},
    ]


def test_without_clipboard_events_copy_and_paste_are_rewritten() -> None:
    steps = _live_run_steps()
    rewritten = ground_clipboard_claims(steps, _timeline([]))
    assert [r["step_index"] for r in rewritten] == [0, 1]
    assert steps[0]["step"] == "Read the text regarding IANA reserved domains"
    assert steps[0]["description"] == "Select and read the paragraph."
    assert steps[0]["parameters"]["verify"] == "The text is read"
    assert steps[1]["step"] == "Type the text"
    assert steps[1]["description"] == "Insert the text."
    assert steps[2]["step"] == "Save the document"
    record = steps[1]["evidence"]["clipboard"]
    assert record["recorded"] is False and record["claimed"] == ["copy", "paste"]
    assert record["original"]["step"] == "Paste the copied text"
    assert "clipboard" in record["reason"]
    # The record survives into the procedure and the packet tells the agent.
    proc = sop_to_procedure({"slug": "look-up", "title": "Look up", "steps": steps})
    assert proc["steps"][1]["evidence"]["clipboard"] == record
    packet = render_packet(proc, target="claude-skill")
    assert packet.count("Unconfirmed: no ") == 2


def test_a_recorded_copy_and_paste_keep_their_steps() -> None:
    steps = _live_run_steps()
    # Copied while f1 was on screen (recorded before f2), pasted before f3.
    assert ground_clipboard_claims(steps, _timeline([("copy", 15), ("paste", 25)])) == []
    assert steps[0]["step"].startswith("Copy") and steps[1]["step"].startswith("Paste")
    # A paste outside the browser is not recorded; an earlier copy supports it.
    steps = _live_run_steps()
    assert ground_clipboard_claims(steps, _timeline([("copy", 15)])) == []


def test_a_copy_outside_the_steps_window_does_not_count() -> None:
    steps = _live_run_steps()
    # The only copy was recorded with the last frame (f3): not while the
    # Copy step (frames f0-f1, recorded up to f2) happened.  The Paste step
    # (f2) is checked up to the frame after it, where the copy is.
    rewritten = ground_clipboard_claims(steps, _timeline([("copy", 35)]))
    assert [(r["step_index"], r["claimed"]) for r in rewritten] == [(0, ["copy"])]
    assert steps[1]["step"] == "Paste the copied text"


def test_a_paste_alone_leaves_the_copy_step_unsupported() -> None:
    steps = _live_run_steps()
    rewritten = ground_clipboard_claims(steps, _timeline([("paste", 25)]))
    assert [(r["step_index"], r["claimed"]) for r in rewritten] == [(0, ["copy"]), (1, ["copy"])]
    # The paste itself was recorded: "Paste" stays, "the copied" goes.
    assert steps[1]["step"] == "Paste the text"


def test_a_passive_timeline_is_left_alone() -> None:
    steps = _live_run_steps()
    assert ground_clipboard_claims(steps, _timeline(None)) == []
    assert steps[0]["step"].startswith("Copy")


def test_the_prompt_says_so_only_when_nothing_was_copied() -> None:
    assert "no copy, cut or paste was recorded" in clipboard_note(_timeline([]))
    assert clipboard_note(_timeline([("copy", 15)])) == ""
    assert clipboard_note(_timeline(None)) == ""
    prompt = _build_focus_prompt("Look up", _timeline([("paste", 25)]))
    assert ">>> PASTE ACTION" in prompt
    assert "CLIPBOARD:" not in prompt


_LIVE_RUN_ANSWER = json.dumps({
    "title": "Look up the IANA reserved domains page and note it",
    "steps": [
        {"action": "Copy the text regarding IANA reserved domains", "app": "Safari",
         "description": "Select the paragraph.", "verify": "Copied", "frames": [1, 2]},
        {"action": "Paste the copied text", "app": "TextEdit",
         "description": "Into the note.", "verify": "The text shows", "frames": [3]},
    ],
    "variables": [],
})


def _generate(timeline: list[dict]) -> tuple[dict, list[str]]:
    generator = SOPGenerator()
    prompts: list[str] = []

    def call(prompt: str, *, retry: bool = False, system: str | None = None):
        prompts.append(prompt)
        return _LIVE_RUN_ANSWER, 1.0

    generator._call = call  # the model: the live run's answer, whatever the prompt
    result = generator.generate_from_focus(timeline, "Look up the IANA reserved domains page and note it")
    assert result.success, result.error
    return result.sop, prompts


def test_the_generator_grounds_the_models_copy_and_paste() -> None:
    sop, prompts = _generate(_timeline([]))
    assert "CLIPBOARD: no copy, cut or paste was recorded" in prompts[0]
    # The frame's own "copying" is not handed to the model as a fact.
    assert "copying text" not in prompts[0]
    assert "Activity: reading text from the Example Domain page into TextEdit" in prompts[0]
    assert [s["step"] for s in sop["steps"]] == [
        "Read the text regarding IANA reserved domains", "Type the text",
    ]
    sop, prompts = _generate(_timeline([("copy", 15)]))
    assert "CLIPBOARD:" not in prompts[0]
    assert "Activity: copying text from the Example Domain page" in prompts[0]
    assert [s["step"] for s in sop["steps"]] == [
        "Copy the text regarding IANA reserved domains", "Paste the copied text",
    ]


def test_the_focus_processor_keeps_every_clipboard_event() -> None:
    processor = FocusProcessor(annotator=None, differ=None, sop_generator=None)
    events = [
        {"id": "f0", "timestamp": _ts(0), "kind_json": '{"type":"DwellSnapshot"}',
         "window_json": '{"app_id":"com.apple.Safari","title":"IANA"}'},
        {"id": "c1", "timestamp": _ts(5), "kind_json": json.dumps({
            "type": "ClipboardChange", "content_types": ["public.utf8-plain-text"],
            "byte_size": 12, "high_entropy": False, "content_hash": "ab"})},
        {"id": "f1", "timestamp": _ts(10), "kind_json": '{"type":"DwellSnapshot"}',
         "window_json": '{"app_id":"com.apple.TextEdit","title":"Note"}'},
        {"id": "p1", "timestamp": _ts(12),
         "kind_json": '{"type":"PasteDetected","matched_copy_hash":"ab"}'},
    ]
    kept = processor._attach_clipboard_context(events)
    assert [e["id"] for e in kept] == ["f0", "f1"]
    assert [(e["kind"], e["event_id"]) for e in processor._clipboard_events] == [
        ("copy", "c1"), ("paste", "p1"),
    ]
    # Without any copy the paste is still kept (and dropped from the frames).
    kept = processor._attach_clipboard_context([events[0], events[3]])
    assert [e["id"] for e in kept] == ["f0"]
    assert [e["kind"] for e in processor._clipboard_events] == ["paste"]


def test_the_session_query_returns_the_browsers_pastes(tmp_path: Path) -> None:
    """Neither kind of clipboard event carries the focus session id."""
    import sqlite3

    from agenthandover_worker.db import WorkerDB
    from test_focus_processor import _create_test_db

    db_path = _create_test_db(tmp_path)
    rows = [
        ("f0", _ts(0), '{"type":"DwellSnapshot"}', '{"focus_session_id": "s1"}'),
        ("c1", _ts(5), '{"type":"ClipboardChange","content_types":[],"byte_size":3}', "{}"),
        ("p1", _ts(7), '{"type":"PasteDetected","matched_copy_hash":null}', "{}"),
        ("f1", _ts(10), '{"type":"DwellSnapshot"}', '{"focus_session_id": "s1"}'),
        ("p2", _ts(40), '{"type":"PasteDetected","matched_copy_hash":null}', "{}"),
    ]
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO events (id, timestamp, kind_json, window_json, metadata_json, "
            "display_topology_json, primary_display_id) "
            "VALUES (?, ?, ?, '{}', ?, '[]', 'main')", rows,
        )
    with WorkerDB(db_path) as db:
        ids = [e["id"] for e in db.get_focus_session_events("s1")]
    assert ids == ["f0", "c1", "p1", "f1"]  # p2 is after the session


# ---------------------------------------------------------------------------
# Window chrome: the font toolbar is not an edit
# ---------------------------------------------------------------------------

# A TextEdit window at (100, 100), 800 x 600 points, on a 1512 x 982 display.
_DISPLAY = [{"display_id": "1", "bounds_global_px": [0, 0, 1512, 982],
             "scale_factor": 2.0, "orientation": 0}]
_DOC_LINES = ["IANA-managed Reserved Domains", "example.com is reserved per RFC 2606"]


def _textedit_frame(
    event_id: str, second: int, font_popup: str, extra: str = "",
    toolbar: tuple[tuple[float, float, str], ...] = (),
) -> dict:
    """A TextEdit frame: the menu bar above the window, the format bar's
    font popup (OCR'd as *font_popup*) inside its toolbar band, more
    *toolbar* text as ``(y, x, text)`` in points (the format bar's size and
    spacing popups, the ruler's ticks), the document below."""
    def box(y_points: float, text: str, x: float = 120, w: float = 300) -> dict:
        return {"text": text, "confidence": 1.0,
                "bbox_normalized": [x / 1512, y_points / 982, w / 1512, 14 / 982]}

    doc = _DOC_LINES + ([extra] if extra else [])
    elements = [
        box(4, "TextEdit File Edit Format View Window Help", x=30, w=500),
        box(135, font_popup, x=160, w=120),
        *(box(y, text, x=x, w=10 * len(text)) for y, x, text in toolbar),
        *(box(250 + 30 * i, line) for i, line in enumerate(doc)),
    ]
    annotation = {
        "app": "TextEdit", "location": "Untitled.rtf",
        "task_context": {"what_doing": "Typing in TextEdit"},
        "_screen_text": "\n".join(e["text"] for e in elements),
    }
    return {
        "id": event_id, "timestamp": _ts(second), "annotation_status": "completed",
        "scene_annotation_json": json.dumps(annotation),
        "window_json": json.dumps({"window_id": "7", "app_id": "pid:900:TextEdit",
                                   "title": "Untitled.rtf",
                                   "bounds_global_px": [100, 100, 800, 600]}),
        "display_topology_json": json.dumps(_DISPLAY), "primary_display_id": "1",
        "metadata_json": json.dumps({"ocr": {"elements": elements}}),
    }


def test_toolbar_menu_and_font_text_is_chrome() -> None:
    frame = _textedit_frame("t1", 0, "Halvetica Ragular")
    annotation = json.loads(frame["scene_annotation_json"])
    content, chrome = content_lines(frame, annotation)
    assert content == _DOC_LINES
    assert chrome == ["TextEdit File Edit Format View Window Help", "Halvetica Ragular"]
    # Without geometry the words alone still mark the font popup and menu bar.
    assert content_lines(None, annotation)[0] == _DOC_LINES
    assert is_chrome_line("Helvetica Regular 12") and is_chrome_line("Halvetica Ragular")
    assert not is_chrome_line("IANA") and not is_chrome_line("Edit the file")
    assert not is_chrome_line("Notes", "Notes") and not is_chrome_line("12")
    # A short label in the toolbar band is chrome only with the geometry.
    frame = _textedit_frame("t2", 0, "IANA Reserved")
    annotation = json.loads(frame["scene_annotation_json"])
    assert "IANA Reserved" in content_lines(frame, annotation)[1]
    assert "IANA Reserved" in content_lines(frame, annotation, geometry=False)[0]


def test_an_ocr_flicker_in_the_font_popup_is_no_change() -> None:
    """Frames that differ only in how OCR read the toolbar are not sent to
    the model: nothing the user did changed."""
    prev = _textedit_frame("t1", 0, "Helvetica Regular")
    curr = _textedit_frame("t2", 10, "Halvetica Ragular")
    with patch("agenthandover_worker.scene_annotator._call_ollama_vlm") as call:
        result = FrameDiffer().diff_pair(prev, curr)
    assert result.diff["diff_type"] == "no_change"
    call.assert_not_called()


def test_a_font_popup_edit_is_not_kept() -> None:
    """The live run's diff: "The user corrected the text ... changing
    'Halvetica Ragular' to 'IANA'" (TextEdit's font popup, misread)."""
    prev = _textedit_frame("t1", 0, "Helvetica Regular")
    curr = _textedit_frame("t2", 10, "Halvetica Ragular", extra="Reserved for documentation")
    prompts: list[str] = []
    answer = {
        "diff_type": "action",
        "actions": [
            "The user corrected the text, changing 'Halvetica Ragular' to 'IANA'",
            "The user typed 'Reserved for documentation' below the heading",
        ],
        "inputs": [{"field": "Font", "value": "Halvetica Ragular"},
                   {"field": "Document", "value": "Reserved for documentation"}],
        "navigation": "none",
        "step_description": "Changed 'Halvetica Ragular' to 'IANA'",
    }

    def model(**kwargs):
        prompts.append(kwargs["prompt"])
        return json.dumps(answer), 1.0

    with patch("agenthandover_worker.scene_annotator._call_ollama_vlm", side_effect=model):
        diff = FrameDiffer().diff_pair(prev, curr).diff
    # The font popup and menu bar never reached the model.
    assert "alvetica" not in prompts[0] and "Helvetica" not in prompts[0]
    assert "File Edit Format" not in prompts[0]
    assert "Reserved for documentation" in prompts[0]
    # The toolbar "edit" is dropped, the real one kept.
    assert diff["actions"] == ["The user typed 'Reserved for documentation' below the heading"]
    assert diff["inputs"] == [{"field": "Document", "value": "Reserved for documentation"}]
    assert diff["step_description"] == diff["actions"][0]
    assert any("Halvetica Ragular" in d for d in diff["ungrounded"])


# The live run of session ec05f823 (qwen3.5:4b): "The user edited the
# metadata fields in a TextEdit document, changing the font size from 12 to
# 10 and the line ...": the format bar's size and spacing popups and the
# ruler's ticks, OCR'd differently in each frame.  The window's toolbar
# band ends at y = 160 points, the ruler below it; the document starts at
# y = 250.
_TOOLBAR_A = ((135, 300, "12"), (135, 360, "1.0"), (168, 110, "0 2 4 6 8 10 12 14"))
_TOOLBAR_B = ((135, 300, "10"), (135, 360, "1.5"), (168, 110, "2 4 6 8 10 12 14 16"),
              (168, 800, "18"))


def test_toolbar_and_ruler_numbers_are_chrome() -> None:
    for toolbar in (_TOOLBAR_A, _TOOLBAR_B):
        frame = _textedit_frame("t1", 0, "Helvetica Regular", toolbar=toolbar)
        annotation = json.loads(frame["scene_annotation_json"])
        for geometry in (True, False):
            content, chrome = content_lines(frame, annotation, geometry=geometry)
            assert content == _DOC_LINES
            assert all(text in chrome for _, _, text in toolbar)
        # Without the geometry at all (a row without OCR boxes).
        assert content_lines(None, annotation)[0] == _DOC_LINES
    assert is_numeric_line("12") and is_numeric_line("1.0") and is_numeric_line("100%")
    assert is_numeric_line("0 2 4 6 8 10 12 14")
    assert not is_numeric_line("Invoice 4471") and not is_numeric_line("3 items")
    assert not is_numeric_line("") and not is_numeric_line("-")
    assert is_ruler_line("0 2 4 6 8 10 12 14") and is_ruler_line("2 4 6 8 1 12 14")
    assert not is_ruler_line("555 123 4567") and not is_ruler_line("2026 09 24")
    assert not is_ruler_line("12") and not is_ruler_line("Page 1 2 3")


def test_toolbar_numbers_read_differently_are_no_change() -> None:
    """Frames that differ only in how OCR read the size popup, the
    spacing popup and the ruler are not sent to the model."""
    prev = _textedit_frame("t1", 0, "Helvetica Regular", toolbar=_TOOLBAR_A)
    curr = _textedit_frame("t2", 10, "Helvetica Regular", toolbar=_TOOLBAR_B)
    with patch("agenthandover_worker.scene_annotator._call_ollama_vlm") as call:
        result = FrameDiffer().diff_pair(prev, curr)
    assert result.diff["diff_type"] == "no_change"
    call.assert_not_called()


def test_toolbar_numbers_never_reach_the_diff_and_a_size_edit_is_not_kept() -> None:
    """A real edit beside the flickering toolbar: the model sees the
    document's new line and no toolbar number, and an input read off the
    size popup is dropped."""
    prev = _textedit_frame("t1", 0, "Helvetica Regular", toolbar=_TOOLBAR_A)
    curr = _textedit_frame("t2", 10, "Helvetica Regular", extra="Reserved for documentation",
                           toolbar=_TOOLBAR_B)
    prompts: list[str] = []
    answer = {
        "diff_type": "action",
        "actions": ["The user typed 'Reserved for documentation' below the heading"],
        "inputs": [{"field": "Font size", "value": "10"},
                   {"field": "Document", "value": "Reserved for documentation"}],
        "navigation": "none",
        "step_description": "Typed a line below the heading",
    }

    def model(**kwargs):
        prompts.append(kwargs["prompt"])
        return json.dumps(answer), 1.0

    with patch("agenthandover_worker.scene_annotator._call_ollama_vlm", side_effect=model):
        diff = FrameDiffer().diff_pair(prev, curr).diff
    screen_lines = [line for line in prompts[0].splitlines() if line.startswith("Screen text:")]
    assert len(screen_lines) == 2
    for line in screen_lines:
        parts = line.removeprefix("Screen text: ").split(" | ")
        assert not any(is_numeric_line(part) for part in parts), line
    assert "Reserved for documentation" in prompts[0]
    assert diff["inputs"] == [{"field": "Document", "value": "Reserved for documentation"}]
    assert diff["ungrounded"] == ["input Font size: 10"]


def test_a_typed_number_line_in_the_document_is_content() -> None:
    """A number the user typed on its own line (placed in the document by
    the geometry) is content for the diff, the no-change check and the
    step checks; without the geometry it cannot be told from a toolbar's."""
    prev = _textedit_frame("t1", 0, "Helvetica Regular", toolbar=_TOOLBAR_A)
    curr = _textedit_frame("t2", 10, "Helvetica Regular", extra="4471", toolbar=_TOOLBAR_A)
    annotation = json.loads(curr["scene_annotation_json"])
    for geometry in (True, False):
        assert content_lines(curr, annotation, geometry=geometry)[0] == [*_DOC_LINES, "4471"]
    assert content_lines(None, annotation)[0] == _DOC_LINES
    prompts: list[str] = []

    def model(**kwargs):
        prompts.append(kwargs["prompt"])
        return json.dumps({"diff_type": "action", "actions": ["The user typed '4471'"],
                           "inputs": [{"field": "Document", "value": "4471"}],
                           "step_description": "Typed '4471'"}), 1.0

    with patch("agenthandover_worker.scene_annotator._call_ollama_vlm", side_effect=model):
        diff = FrameDiffer().diff_pair(prev, curr).diff
    assert len(prompts) == 1 and "| 4471" in prompts[0]
    assert diff["actions"] == ["The user typed '4471'"] and "ungrounded" not in diff
    # A number within a line stays content without any geometry.
    annotation = {"app": "TextEdit", "_screen_text": "Invoice 4471\n3 items\n12"}
    assert content_lines(None, annotation)[0] == ["Invoice 4471", "3 items"]


def test_grounding_keeps_edits_the_document_shows() -> None:
    diff = {"actions": ["Typed 'hello world' into the note"], "inputs": [],
            "step_description": "Typed a greeting"}
    assert ground_text_changes(diff, "Hello  world\nother") is diff
    # Quoted text without an edit verb is a reference, not a claimed edit.
    diff = {"actions": ["Opened the 'Settings' pane"], "inputs": []}
    assert ground_text_changes(diff, "") is diff
