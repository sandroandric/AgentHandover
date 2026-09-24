"""Typed text needs keyboard evidence (WP-F14).

Live on the test Mac (2026-09-24, Comet + TextEdit, qwen3.5:4b):

- Session 57352dfb "Note the IANA reserved example domains": the user
  opened a new, empty TextEdit document and typed nothing.  The daemon's
  OCR read one low-confidence element in the window's format bar (the
  font-size and line-spacing popups): "SEEN 10", then "# 10".  The vision
  call answered ``typed_text = "# 10"``, the frame diff reported the input
  "# 10", and the Skill got "Type the text '# 10' into the document".
- Session 62406a55: Cmd-N opened "Untitled 2", and the next frame's answer
  was ``typed_text = "# 10"``: four characters after one key press.

The daemon now records on every kept frame how many keys were pressed
since the previous one (``keys_since_prev_frame``).  New typed text needs
a key press per character (up to three: autocomplete and autofill insert
more), or a paste of a recorded copy (``scene_annotator.settle_typed_text``,
applied before the annotation is stored).  These pins run the focus
processor's own annotation and diff steps, and the passive annotation
loop, on a real daemon database.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

from agenthandover_worker.db import WorkerDB
from agenthandover_worker.focus_processor import FocusProcessor
from agenthandover_worker.frame_differ import FrameDiffer
from agenthandover_worker.main import _process_annotations
from agenthandover_worker.scene_annotator import (
    AnnotationConfig,
    AnnotationResult,
    SceneAnnotator,
)
from agenthandover_worker.sop_generator import SOPGenerator, _build_focus_prompt
from agenthandover_worker.step_grounding import typed_lines
from daemon_db import create_daemon_db

SESSION = "57352dfb-0000-4000-8000-000000000001"
_DISPLAY = [{"display_id": "1", "bounds_global_px": [0, 0, 1512, 982],
             "scale_factor": 2.0, "orientation": 0}]
_COMET = {"app_id": "pid:700:Comet", "title": "IANA-managed Reserved Domains",
          "bounds_global_px": [0, 25, 1512, 957]}
# The new document's window: its format bar (the font-size popup the OCR
# read as "# 10") is inside the window's toolbar band.
_TEXTEDIT = {"app_id": "pid:902:TextEdit", "title": "Untitled",
             "bounds_global_px": [250, 60, 1000, 700]}
LINE = "Reserved for documentation: example.com, example.net, example.org"


def _element(text: str, x: float, y: float, w: float, confidence: float = 1.0) -> dict:
    return {"text": text, "confidence": confidence, "bbox_normalized": [x, y, w, 0.015]}


def _toolbar(text: str) -> list[dict]:
    """The live frames' only OCR element: the format bar, y 0.09, x 0.67-0.78."""
    return [_element(text, 0.67, 0.09, 0.11, confidence=0.5)]


def _document(*lines: str) -> list[dict]:
    return [_element(line, 0.2, 0.3 + 0.03 * i, 0.4) for i, line in enumerate(lines)]


def _insert(
    conn: sqlite3.Connection, event_id: str, timestamp: str, window: dict,
    elements: list[dict], keys: int | None, *, status: str = "pending",
    session: str | None = SESSION,
) -> None:
    metadata: dict = {"ocr": {"elements": elements}, "capture_trigger": "focus_interval"}
    if session is not None:
        metadata["focus_session_id"] = session
    if keys is not None:
        metadata["keys_since_prev_frame"] = keys
    conn.execute(
        "INSERT INTO events (id, timestamp, kind_json, window_json, display_topology_json, "
        " primary_display_id, metadata_json, annotation_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (event_id, timestamp, '{"type":"DwellSnapshot"}', json.dumps(window),
         json.dumps(_DISPLAY), "1", json.dumps(metadata), status),
    )


def _insert_copy(conn: sqlite3.Connection, timestamp: str, text: str, *,
                 preview: bool = False) -> None:
    """A ``ClipboardChange`` as the daemon writes it (no session tag)."""
    kind = {"type": "ClipboardChange", "content_types": ["public.utf8-plain-text"],
            "byte_size": len(text.encode()), "high_entropy": False,
            "content_hash": hashlib.sha256(text.encode()).hexdigest()}
    if preview:
        kind["content_preview"] = text
    conn.execute(
        "INSERT INTO events (id, timestamp, kind_json, display_topology_json, "
        " primary_display_id, metadata_json, annotation_status) "
        "VALUES (?, ?, ?, '[]', '1', '{}', 'skipped')",
        (f"copy-{timestamp}", timestamp, json.dumps(kind)),
    )


def _annotator(typed: dict[str, str]) -> MagicMock:
    """The model's answers: each frame's ``typed_text`` from *typed*, its
    screen text the frame's OCR (as the annotator stores it)."""
    annotator = MagicMock(spec=SceneAnnotator)
    annotator.config = AnnotationConfig()
    annotator.ready.return_value = True

    def annotate(event, *, recent_annotations=None, artifact_dir=None, page=None):
        metadata = json.loads(event["metadata_json"])
        window = json.loads(event["window_json"])
        app = window["app_id"].split(":")[-1]
        return AnnotationResult(
            event_id=event["id"], status="completed",
            annotation={
                "app": app, "location": window["title"],
                "task_context": {"what_doing": f"Working in {app}", "is_workflow": True},
                "ui_state": {"active_element": "", "modal": ""},
                "key_text": {"typed_text": typed.get(event["id"], ""), "urls": [], "emails": []},
                "visible_items": [], "needs_vision": False,
                "_screen_text": "\n".join(e["text"] for e in metadata["ocr"]["elements"]),
            },
        )

    annotator.annotate_event.side_effect = annotate
    return annotator


def _processor(typed: dict[str, str]) -> FocusProcessor:
    return FocusProcessor(_annotator(typed), FrameDiffer(), MagicMock(spec=SOPGenerator))


def _annotate_session(db_path: Path, tmp_path: Path, typed: dict[str, str]) -> dict[str, str]:
    """Run the focus processor's annotation step on the session's pending
    frames; returns each annotated frame's stored typed text."""
    with WorkerDB(db_path) as db:
        events = [e for e in db.get_focus_session_events(SESSION)
                  if e["annotation_status"] == "pending"]
        _processor(typed)._ensure_annotations(db, events, tmp_path)
        return {
            e["id"]: json.loads(db.get_event_by_id(e["id"])["scene_annotation_json"])
            ["key_text"]["typed_text"]
            for e in events
        }


# The live diff model's answer for the empty document.
_LIVE_DIFF = {
    "diff_type": "action",
    "actions": ["The user typed the text '# 10'"],
    "inputs": [{"field": "typed_text", "value": "# 10"}],
    "navigation": "none",
    "step_description": "The user typed the text '# 10'",
}


def _run_live_session(tmp_path: Path, keys: int) -> tuple[dict[str, dict], list[dict], str]:
    """Annotate and diff session 57352dfb's frames, each counting *keys*
    key presses; returns the stored annotations and diffs by frame, the
    timeline and the focus SOP prompt."""
    db_path = create_daemon_db(tmp_path / "events.db")
    conn = sqlite3.connect(db_path)
    _insert(conn, "comet-01", "2026-09-24T17:28:50.000Z", _COMET,
            _document("IANA-managed Reserved Domains", "Example domains"), keys)
    _insert(conn, "abe30abd", "2026-09-24T17:29:09.000Z", _TEXTEDIT, _toolbar("SEEN 10"), keys)
    _insert(conn, "c51c1bda", "2026-09-24T17:31:08.000Z", _TEXTEDIT, _toolbar("# 10"), keys)
    _insert(conn, "15db24e7", "2026-09-24T17:31:10.000Z", _TEXTEDIT, _toolbar("# 10"), keys)
    conn.commit()
    conn.close()
    processor = _processor({"c51c1bda": "# 10", "15db24e7": "# 10"})
    with WorkerDB(db_path) as db:
        events = db.get_focus_session_events(SESSION)
        processor._ensure_annotations(db, events, tmp_path)
        with patch("agenthandover_worker.scene_annotator._call_ollama_vlm",
                   return_value=(json.dumps(_LIVE_DIFF), 1.0)):
            processor._ensure_diffs(db, events)
        rows = {e["id"]: db.get_event_by_id(e["id"]) for e in events}
        timeline = processor._collect_timeline(db, events)
    stored = {
        eid: {"annotation": json.loads(row["scene_annotation_json"]),
              "diff": json.loads(row["frame_diff_json"] or "null")}
        for eid, row in rows.items()
    }
    return stored, timeline, _build_focus_prompt("Note the IANA reserved example domains", timeline)


def test_toolbar_ocr_without_a_key_press_is_not_typing(tmp_path: Path) -> None:
    stored, timeline, prompt = _run_live_session(tmp_path, keys=0)
    for eid in ("c51c1bda", "15db24e7"):
        annotation = stored[eid]["annotation"]
        assert annotation["key_text"]["typed_text"] == ""
        # The model's answer is kept for the record, not as typing.
        assert annotation["_typed_text_without_keys"] == "# 10"
    # The frame diff reports no typed input: "# 10" is the format bar's.
    diff = stored["c51c1bda"]["diff"]
    assert diff["inputs"] == []
    assert not any("# 10" in a for a in diff["actions"])
    assert any("# 10" in d for d in diff["ungrounded"])
    # Nothing typed reaches step grounding or the SOP prompt.
    assert typed_lines(timeline) == []
    assert 'Typed: "# 10"' not in prompt


def test_with_enough_key_presses_the_models_typed_text_stands(tmp_path: Path) -> None:
    """The same frames, each with four key presses: today's behaviour."""
    stored, _timeline, prompt = _run_live_session(tmp_path, keys=4)
    for eid in ("c51c1bda", "15db24e7"):
        annotation = stored[eid]["annotation"]
        assert annotation["key_text"]["typed_text"] == "# 10"
        assert "_typed_text_without_keys" not in annotation
    assert stored["c51c1bda"]["diff"]["inputs"] == [{"field": "typed_text", "value": "# 10"}]
    assert 'Typed: "# 10"' in prompt


def test_one_key_press_for_a_new_document_does_not_type_four_characters(
    tmp_path: Path,
) -> None:
    """Session 62406a55: Cmd-N (one key press) opened "Untitled 2" at
    17:38:19.8; the frame at 17:38:21.880 answered "# 10".  The same noise
    at 17:38:34.876 and 17:38:41.908 came with no key press at all."""
    untitled_2 = {**_TEXTEDIT, "title": "Untitled 2"}
    db_path = create_daemon_db(tmp_path / "events.db")
    conn = sqlite3.connect(db_path)
    _insert(conn, "comet", "2026-09-24T17:38:15.000Z", _COMET, _document("Example domains"), 0)
    _insert(conn, "n1", "2026-09-24T17:38:21.880Z", untitled_2, _toolbar("# 10"), 1)
    _insert(conn, "n2", "2026-09-24T17:38:34.876Z", untitled_2, _toolbar("# 10"), 0)
    _insert(conn, "n3", "2026-09-24T17:38:41.908Z", untitled_2, _toolbar("# 10"), 0)
    conn.commit()
    conn.close()
    typed = _annotate_session(db_path, tmp_path, {"n1": "# 10", "n2": "# 10", "n3": "# 10"})
    assert typed == {"comet": "", "n1": "", "n2": "", "n3": ""}


def test_a_typing_run_keeps_its_typed_lines(tmp_path: Path) -> None:
    """Key presses on some frames only: those on a skipped frame count, a
    frame repeating the previous typed text needs none, and text that grew
    with too few key presses falls back to the previous frame's."""
    first = "Reserved for documentation"
    db_path = create_daemon_db(tmp_path / "events.db")
    conn = sqlite3.connect(db_path)
    _insert(conn, "t1", "2026-09-24T09:12:01.000Z", _TEXTEDIT, _document("Reserved"), 8)
    # A frame the keyframe selection skipped: its key presses count.
    _insert(conn, "t2", "2026-09-24T09:12:03.000Z", _TEXTEDIT, _document("Reserved for"), 14,
            status="skipped")
    _insert(conn, "t3", "2026-09-24T09:12:05.000Z", _TEXTEDIT, _document(first), 4)
    _insert(conn, "t4", "2026-09-24T09:12:07.000Z", _TEXTEDIT, _document(first), 0)
    # Two key presses cannot have typed ": example.com, ..." (39 characters).
    _insert(conn, "t5", "2026-09-24T09:12:09.000Z", _TEXTEDIT, _document(first), 2)
    _insert(conn, "t6", "2026-09-24T09:12:20.000Z", _TEXTEDIT, _document(LINE), 40)
    conn.commit()
    conn.close()
    typed = _annotate_session(db_path, tmp_path, {
        "t1": "Reserved", "t3": first, "t4": first, "t5": LINE, "t6": LINE,
    })
    assert typed == {"t1": "Reserved", "t3": first, "t4": first, "t5": first, "t6": LINE}
    with WorkerDB(db_path) as db:
        events = [e for e in db.get_focus_session_events(SESSION)
                  if e["annotation_status"] == "completed"]
        timeline = _processor({})._collect_timeline(db, events)
    # The half-typed line stands for its finished form (``typed_lines``).
    assert [text for _i, text in typed_lines(timeline)] == [LINE]


def test_autocomplete_inserts_more_than_the_keys_pressed(tmp_path: Path) -> None:
    """Three key presses ("exa", then Tab or Enter on a suggestion) can
    insert "example.com" (11 characters); two cannot."""
    db_path = create_daemon_db(tmp_path / "events.db")
    conn = sqlite3.connect(db_path)
    _insert(conn, "a1", "2026-09-24T09:00:00.000Z", _TEXTEDIT, _document("To:"), 0)
    _insert(conn, "a2", "2026-09-24T09:00:02.000Z", _TEXTEDIT, _document("example.com"), 3)
    _insert(conn, "a3", "2026-09-24T09:00:04.000Z", _TEXTEDIT, _document("example.com"), 0)
    _insert(conn, "a4", "2026-09-24T09:00:06.000Z", _TEXTEDIT,
            _document("example.com, example.net"), 2)
    conn.commit()
    conn.close()
    typed = _annotate_session(db_path, tmp_path, {
        "a2": "example.com", "a3": "example.com", "a4": "example.com, example.net",
    })
    assert typed == {"a1": "", "a2": "example.com", "a3": "example.com", "a4": "example.com"}


def test_a_paste_of_a_recorded_copy_needs_one_key_press(tmp_path: Path) -> None:
    """Cmd-V is one key press: pasted text is kept when a copy before it
    holds it (by the copy's hash, or its preview), not otherwise."""
    db_path = create_daemon_db(tmp_path / "events.db")
    conn = sqlite3.connect(db_path)
    _insert(conn, "c1", "2026-09-24T09:30:00.000Z", _COMET, _document("Example domains"), 0)
    _insert_copy(conn, "2026-09-24T09:30:01.000Z", LINE)
    _insert(conn, "c2", "2026-09-24T09:30:05.000Z", _TEXTEDIT, _document(LINE), 1)
    # A copy the user turned previews on for, holding more than was pasted.
    _insert_copy(conn, "2026-09-24T09:30:06.000Z", "Source: IANA reserved domains page, 2026",
                 preview=True)
    _insert(conn, "c3", "2026-09-24T09:30:10.000Z", _TEXTEDIT,
            _document(LINE, "Source: IANA reserved domains page"), 1)
    # Nothing copied holds this, and one key press cannot type it.
    _insert(conn, "c4", "2026-09-24T09:30:15.000Z", _TEXTEDIT,
            _document(LINE, "Source: IANA reserved domains page", "Checked"), 1)
    # A copy with no key press and no paste event since the previous frame.
    _insert_copy(conn, "2026-09-24T09:30:16.000Z", "Reviewed")
    _insert(conn, "c5", "2026-09-24T09:30:20.000Z", _TEXTEDIT,
            _document(LINE, "Source: IANA reserved domains page", "Reviewed"), 0)
    conn.commit()
    conn.close()
    two = f"{LINE}\nSource: IANA reserved domains page"
    typed = _annotate_session(db_path, tmp_path, {
        "c2": LINE, "c3": two, "c4": f"{two}\nChecked", "c5": f"{two}\nReviewed",
    })
    assert typed == {"c1": "", "c2": LINE, "c3": two, "c4": two, "c5": two}


def test_frames_without_the_count_are_left_alone(tmp_path: Path) -> None:
    """Rows from before the daemon recorded the count (and the journeys'
    cassettes) keep the model's typed text."""
    db_path = create_daemon_db(tmp_path / "events.db")
    conn = sqlite3.connect(db_path)
    _insert(conn, "abe30abd", "2026-09-24T17:29:09.000Z", _TEXTEDIT, _toolbar("SEEN 10"), None)
    _insert(conn, "c51c1bda", "2026-09-24T17:31:08.000Z", _TEXTEDIT, _toolbar("# 10"), None)
    conn.commit()
    conn.close()
    typed = _annotate_session(db_path, tmp_path, {"c51c1bda": "# 10"})
    assert typed["c51c1bda"] == "# 10"


def test_passive_frames_are_settled_too(tmp_path: Path) -> None:
    """The passive annotation loop applies the same rule; keys pressed
    during a focus session between two passive frames count."""
    db_path = create_daemon_db(tmp_path / "events.db")
    conn = sqlite3.connect(db_path)
    _insert(conn, "p1", "2026-09-24T10:00:00.000Z", _TEXTEDIT, _toolbar("SEEN 10"), 0,
            session=None)
    _insert(conn, "p2", "2026-09-24T10:01:00.000Z", _TEXTEDIT, _toolbar("# 10"), 1,
            session=None)
    _insert(conn, "f1", "2026-09-24T10:02:00.000Z", _TEXTEDIT, _document("Notes"), 5,
            status="skipped")
    _insert(conn, "p3", "2026-09-24T10:03:00.000Z", _TEXTEDIT, _document("Notes"), 0,
            session=None)
    conn.commit()
    conn.close()
    annotator = _annotator({"p2": "# 10", "p3": "Notes"})
    screenshots = tmp_path / "shots"
    screenshots.mkdir()
    with WorkerDB(db_path) as db:
        stats = _process_annotations(db, annotator, screenshots)
        typed = {
            eid: json.loads(db.get_event_by_id(eid)["scene_annotation_json"])
            ["key_text"]["typed_text"]
            for eid in ("p1", "p2", "p3")
        }
    assert stats["annotated"] == 3
    assert typed == {"p1": "", "p2": "", "p3": "Notes"}


def test_a_paste_is_known_by_the_copys_size_when_previews_are_off(tmp_path: Path) -> None:
    """Previews are off by default, and the model's reading of pasted text
    rarely hashes like the copied bytes (here the copy ends in a newline
    and a period the reading lacks).  A copy of about its size stands for
    it; "# 10" after one key press is not a paste of a 60-byte copy."""
    db_path = create_daemon_db(tmp_path / "events.db")
    conn = sqlite3.connect(db_path)
    _insert(conn, "s1", "2026-09-24T11:00:00.000Z", _COMET, _document("Example domains"), 0)
    _insert_copy(conn, "2026-09-24T11:00:01.000Z", LINE + ".\n")
    _insert(conn, "s2", "2026-09-24T11:00:05.000Z", _TEXTEDIT, _document(LINE), 1)
    _insert(conn, "s3", "2026-09-24T11:00:09.000Z", {**_TEXTEDIT, "title": "Untitled 2"},
            _toolbar("# 10"), 1)
    conn.commit()
    conn.close()
    typed = _annotate_session(db_path, tmp_path, {"s2": LINE, "s3": "# 10"})
    assert typed == {"s1": "", "s2": LINE, "s3": ""}


def test_a_browser_paste_event_is_evidence_on_its_own(tmp_path: Path) -> None:
    """The extension's paste event records no content: it proves a paste
    whatever the copy's size or preview."""
    db_path = create_daemon_db(tmp_path / "events.db")
    conn = sqlite3.connect(db_path)
    _insert(conn, "b1", "2026-09-24T12:00:00.000Z", _COMET, _document("Compose"), 0)
    conn.execute(
        "INSERT INTO events (id, timestamp, kind_json, display_topology_json, "
        " primary_display_id, metadata_json, annotation_status) "
        "VALUES ('paste-1', '2026-09-24T12:00:02.000Z', ?, '[]', '1', '{}', 'skipped')",
        (json.dumps({"type": "PasteDetected", "matched_copy_hash": None}),),
    )
    _insert(conn, "b2", "2026-09-24T12:00:04.000Z", _COMET, _document(LINE), 0)
    conn.commit()
    conn.close()
    typed = _annotate_session(db_path, tmp_path, {"b2": LINE})
    assert typed["b2"] == LINE
