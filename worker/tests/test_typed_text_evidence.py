"""Typed text needs keyboard evidence (WP-F14).

Live on the test Mac (2026-09-24, focus session 57352dfb "Note the IANA
reserved example domains", Comet + TextEdit): the user opened a new, empty
TextEdit document and typed nothing.  The daemon's OCR read one
low-confidence element in the window's format bar (the font-size and
line-spacing popups): "SEEN 10", then "# 10".  The vision call answered
``typed_text = "# 10"``, the frame diff reported the input "# 10", and the
Skill got the step "Type the text '# 10' into the document".

The daemon now marks every kept frame with ``keys_since_prev_frame``; a
frame's typed text cannot change without a key press since the previous
annotated frame (``scene_annotator.settle_typed_text``, applied before the
annotation is stored).  These pins run the focus processor's own
annotation and diff steps on a real daemon database.
"""

from __future__ import annotations

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


def _element(text: str, x: float, y: float, w: float, confidence: float = 1.0) -> dict:
    return {"text": text, "confidence": confidence, "bbox_normalized": [x, y, w, 0.015]}


def _toolbar(text: str) -> list[dict]:
    """The live frames' only OCR element: the format bar, y 0.09, x 0.67-0.78."""
    return [_element(text, 0.67, 0.09, 0.11, confidence=0.5)]


def _document(*lines: str) -> list[dict]:
    return [_element(line, 0.2, 0.3 + 0.03 * i, 0.4) for i, line in enumerate(lines)]


def _insert(
    conn: sqlite3.Connection, event_id: str, timestamp: str, window: dict,
    elements: list[dict], keys: bool | None, *, status: str = "pending",
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
        (event_id, timestamp, '{"DwellSnapshot":{}}', json.dumps(window),
         json.dumps(_DISPLAY), "1", json.dumps(metadata), status),
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


# The live diff model's answer for the empty document.
_LIVE_DIFF = {
    "diff_type": "action",
    "actions": ["The user typed the text '# 10'"],
    "inputs": [{"field": "typed_text", "value": "# 10"}],
    "navigation": "none",
    "step_description": "The user typed the text '# 10'",
}


def _diff_model(**kwargs) -> tuple[str, float]:
    """The live answer for the pair it was given (the empty document,
    "SEEN 10" then "# 10"); the switch from Comet into TextEdit is a switch."""
    frame_a = kwargs["prompt"].split("FRAME B")[0]
    if "SEEN 10" in frame_a:
        return json.dumps(_LIVE_DIFF), 1.0
    return json.dumps({
        "diff_type": "action", "actions": ["The user switched to TextEdit"], "inputs": [],
        "navigation": "none", "step_description": "The user switched to TextEdit",
    }), 1.0


def _run_live_session(tmp_path: Path, keys: bool) -> tuple[dict[str, dict], list[dict], str]:
    """Annotate and diff the live session's frames, every frame flagged
    *keys*; returns the stored annotations and diffs by frame, the
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
        with patch("agenthandover_worker.scene_annotator._call_ollama_vlm", side_effect=_diff_model):
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
    stored, timeline, prompt = _run_live_session(tmp_path, keys=False)
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
    assert "typed_text: # 10" not in prompt


def test_with_a_key_press_the_models_typed_text_stands(tmp_path: Path) -> None:
    """The same frames with keyboard evidence: today's behaviour."""
    stored, _timeline, prompt = _run_live_session(tmp_path, keys=True)
    for eid in ("c51c1bda", "15db24e7"):
        annotation = stored[eid]["annotation"]
        assert annotation["key_text"]["typed_text"] == "# 10"
        assert "_typed_text_without_keys" not in annotation
    assert stored["c51c1bda"]["diff"]["inputs"] == [{"field": "typed_text", "value": "# 10"}]
    assert 'Typed: "# 10"' in prompt


def test_a_typing_run_keeps_its_typed_lines(tmp_path: Path) -> None:
    """Keys on some frames only: a key pressed on a skipped frame counts,
    a frame repeating the previous typed text keeps it, and text that grew
    with no key press since the previous frame falls back to it."""
    line = "Reserved for documentation"
    db_path = create_daemon_db(tmp_path / "events.db")
    conn = sqlite3.connect(db_path)
    _insert(conn, "t1", "2026-09-24T09:12:01.000Z", _TEXTEDIT, _document("Reserved"), True)
    # A frame the keyframe selection skipped: the key presses on it count.
    _insert(conn, "t2", "2026-09-24T09:12:03.000Z", _TEXTEDIT, _document("Reserved for"), True,
            status="skipped")
    _insert(conn, "t3", "2026-09-24T09:12:05.000Z", _TEXTEDIT, _document(line), False)
    _insert(conn, "t4", "2026-09-24T09:12:07.000Z", _TEXTEDIT, _document(line), False)
    _insert(conn, "t5", "2026-09-24T09:12:09.000Z", _TEXTEDIT, _document(line), False)
    conn.commit()
    conn.close()
    processor = _processor({
        "t1": "Reserved", "t3": line, "t4": line, "t5": f"{line}, example.com",
    })
    with WorkerDB(db_path) as db:
        events = [e for e in db.get_focus_session_events(SESSION)
                  if e["annotation_status"] == "pending"]
        processor._ensure_annotations(db, events, tmp_path)
        typed = {
            e["id"]: json.loads(db.get_event_by_id(e["id"])["scene_annotation_json"])
            ["key_text"]["typed_text"]
            for e in events
        }
        timeline = processor._collect_timeline(db, events)
    assert typed == {"t1": "Reserved", "t3": line, "t4": line, "t5": line}
    assert [text for _i, text in typed_lines(timeline)] == [line]


def test_frames_without_the_flag_are_left_alone(tmp_path: Path) -> None:
    """Rows from before the daemon recorded the flag (and the journeys'
    cassettes) keep the model's typed text."""
    db_path = create_daemon_db(tmp_path / "events.db")
    conn = sqlite3.connect(db_path)
    _insert(conn, "abe30abd", "2026-09-24T17:29:09.000Z", _TEXTEDIT, _toolbar("SEEN 10"), None)
    _insert(conn, "c51c1bda", "2026-09-24T17:31:08.000Z", _TEXTEDIT, _toolbar("# 10"), None)
    conn.commit()
    conn.close()
    processor = _processor({"c51c1bda": "# 10"})
    with WorkerDB(db_path) as db:
        events = db.get_focus_session_events(SESSION)
        processor._ensure_annotations(db, events, tmp_path)
        annotation = json.loads(db.get_event_by_id("c51c1bda")["scene_annotation_json"])
    assert annotation["key_text"]["typed_text"] == "# 10"
    assert "_typed_text_without_keys" not in annotation


def test_passive_frames_are_settled_too(tmp_path: Path) -> None:
    """The passive annotation loop applies the same rule; a key pressed
    during a focus session between two passive frames counts."""
    db_path = create_daemon_db(tmp_path / "events.db")
    conn = sqlite3.connect(db_path)
    _insert(conn, "p1", "2026-09-24T10:00:00.000Z", _TEXTEDIT, _toolbar("SEEN 10"), False,
            session=None)
    _insert(conn, "p2", "2026-09-24T10:01:00.000Z", _TEXTEDIT, _toolbar("# 10"), False,
            session=None)
    _insert(conn, "f1", "2026-09-24T10:02:00.000Z", _TEXTEDIT, _document("Notes"), True,
            status="skipped")
    _insert(conn, "p3", "2026-09-24T10:03:00.000Z", _TEXTEDIT, _document("Notes"), False,
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
