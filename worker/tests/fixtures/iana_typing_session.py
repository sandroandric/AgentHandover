"""The typing-only IANA session of 2026-09-24 (46b767ef) as a focus timeline.

On the test Mac (qwen3.5:4b for every role) the user had Safari on
iana.org/domains/reserved, opened TextEdit, pressed Cmd-N and typed
"Reserved for documentation: example.com, example.net, example.org",
Enter, "Source: IANA reserved domains page".  Nothing was copied.  The
Skill's five steps are in ``vlm_sop_answers/iana_typing_46b767ef.json``
(their wording as the live run gave them; the other fields of that answer
and the frames each step cites are reconstructed): the new document came
first, "click the 'Example Domain' link" named a title visible only in
Safari's tab strip, "Select the typed text" happened in no frame, and the
second line was in no step.

:func:`timeline` builds the five frames the way ``FocusProcessor.
_collect_timeline`` does: each from a daemon row with the window's bounds
and the OCR boxes, its ``content_text`` from ``frame_differ.content_lines``
(so the tab strip and TextEdit's menu and format bar are chrome), and the
session's clipboard record (empty).
"""

from __future__ import annotations

import json

from agenthandover_worker.clipboard_grounding import attach_clipboard_events
from agenthandover_worker.frame_differ import content_lines

SESSION_ID = "46b767ef-5c1e-4b8e-9d7a-2f0c1a9e3b51"
TITLE = "Note the IANA reserved domains"
LINE_1 = "Reserved for documentation: example.com, example.net, example.org"
LINE_2 = "Source: IANA reserved domains page"
IANA_URL = "https://www.iana.org/domains/reserved"

_DISPLAY = [{"display_id": "1", "bounds_global_px": [0, 0, 1512, 982],
             "scale_factor": 2.0, "orientation": 0}]
# Safari fills the screen below the menu bar; TextEdit's window is smaller.
_SAFARI = {"app_id": "pid:611:Safari", "bounds": [0, 25, 1512, 957]}
_TEXTEDIT = {"app_id": "pid:902:TextEdit", "bounds": [300, 120, 800, 600]}

_SAFARI_PAGE = [
    "IANA-managed Reserved Domains",
    "Certain domains are set aside, and nominally registered to IANA, for specific "
    "policy or technical purposes.",
    "Example domains",
    "As described in RFC 2606 and RFC 6761, a number of domains such as example.com "
    "and example.org are maintained for documentation purposes.",
    "Test IDN top-level domains",
]


def _ts(second: int) -> str:
    return f"2026-09-24T09:12:{second:02d}+00:00"


def _box(x: float, y: float, w: float, text: str) -> dict:
    """An OCR element at (*x*, *y*) points, *w* wide, one line high."""
    return {"text": text, "confidence": 1.0,
            "bbox_normalized": [x / 1512, y / 982, w / 1512, 14 / 982]}


def _row(event_id: str, second: int, window: dict, title: str, elements: list[dict],
         annotation: dict) -> dict:
    annotation = {**annotation, "_screen_text": "\n".join(e["text"] for e in elements)}
    return {
        "id": event_id, "timestamp": _ts(second), "annotation_status": "completed",
        "scene_annotation_json": json.dumps(annotation),
        "window_json": json.dumps({"window_id": event_id, "app_id": window["app_id"],
                                   "title": title, "bounds_global_px": window["bounds"]}),
        "display_topology_json": json.dumps(_DISPLAY), "primary_display_id": "1",
        "metadata_json": json.dumps({"ocr": {"elements": elements}}),
    }


def _safari(event_id: str, second: int, what: str) -> dict:
    elements = [
        _box(10, 4, 400, "Safari File Edit View History Bookmarks Window Help"),
        # The tab strip and the address bar, inside the window's toolbar band.
        _box(200, 40, 260, "IANA-managed Reserved Domains"),
        _box(480, 40, 160, "Example Domain"),
        _box(600, 66, 220, "iana.org/domains/reserved"),
        *(_box(120, 160 + 40 * i, 1000, line) for i, line in enumerate(_SAFARI_PAGE)),
    ]
    return _row(event_id, second, _SAFARI, "IANA-managed Reserved Domains", elements, {
        "app": "Safari", "location": IANA_URL,
        "task_context": {"what_doing": what},
        "key_text": {"typed_text": "", "urls": [IANA_URL]},
    })


def _textedit(event_id: str, second: int, lines: list[str], typed: str, what: str) -> dict:
    elements = [
        _box(10, 4, 420, "TextEdit File Edit Format View Window Help"),
        _box(330, 150, 140, "Helvetica Regular"),
        *(_box(330, 220 + 24 * i, 700, line) for i, line in enumerate(lines)),
    ]
    return _row(event_id, second, _TEXTEDIT, "Untitled", elements, {
        "app": "TextEdit", "location": "Untitled",
        "task_context": {"what_doing": what},
        "key_text": {"typed_text": typed, "urls": []},
    })


def _diff(actions: list[str], value: str = "") -> dict:
    return {"diff_type": "action", "actions": actions,
            "inputs": [{"field": "document", "value": value}] if value else [],
            "navigation": "none", "step_description": actions[0]}


def rows() -> list[tuple[dict, dict | None]]:
    """The session's five annotated frames as daemon rows, each with the
    frame diff the worker stored for it."""
    partial = "Reserved for documentation: example.com, exa"
    return [
        (_safari("e1", 0, "Reading the IANA reserved domains page"), None),
        (_textedit("e2", 10, [], "", "Creating a new TextEdit document"),
         {"diff_type": "app_switch", "from_app": "Safari", "to_app": "TextEdit"}),
        (_textedit("e3", 20, [partial], partial, "Typing the reserved domains"),
         _diff([f"The user typed '{partial}'"], partial)),
        (_textedit("e4", 30, [LINE_1], LINE_1, "Typing the reserved domains"),
         _diff([f"The user typed '{LINE_1}'"], LINE_1)),
        (_textedit("e5", 40, [LINE_1, LINE_2], f"{LINE_1}\n{LINE_2}",
                   "Typing the source of the list"),
         _diff([f"The user typed '{LINE_2}' on a new line"], LINE_2)),
    ]


def timeline() -> list[dict]:
    """The frames as ``FocusProcessor._collect_timeline`` returns them."""
    frames = []
    for row, diff in rows():
        annotation = json.loads(row["scene_annotation_json"])
        frames.append({
            "annotation": annotation, "diff": diff, "dom_nodes": None,
            "dom_snapshot_id": None, "host": "", "click_targets": [],
            "timestamp": row["timestamp"], "app": annotation["app"],
            "event_id": row["id"], "clipboard_context": None,
            "focused_field": None, "shell_commands": [],
            "content_text": content_lines(row, annotation)[0],
            "kind": "DwellSnapshot",
        })
    frames[0]["diff"] = None
    attach_clipboard_events(frames, [])  # nothing was copied or pasted
    return frames
