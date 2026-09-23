"""Generate the ``focus_iana_typed`` fixture: the typing-only focus session
of 2026-09-24 (46b767ef on the test Mac), in the row format the daemon
writes.

The session: Safari already on iana.org/domains/reserved (its tab strip
also shows an "Example Domain" tab), the user switches to TextEdit, presses
Cmd-N and types "Reserved for documentation: example.com, example.net,
example.org", Enter, "Source: IANA reserved domains page".  Nothing is
copied or pasted, and Safari has no extension (no DOM snapshots, no
clicks).

Rows follow ``generate_focus_dailynews.py`` (the same row builders): the
native observer's ``AppSwitch`` / ``WindowTitleChange`` and the
focus-capture ``DwellSnapshot`` frames every 2 s with ``window_json``
(bounds on the primary display), the display topology, the OCR result
(``elements`` with ``bbox_normalized`` in the screenshot's 0..1 space,
``full_text``) and ``focus_session_id``.  The OCR boxes are laid out in
points on a 1512 x 982 display: the menu bar above the windows, Safari's
tab strip and address bar and TextEdit's format bar inside their
windows' toolbar bands, the page and the document below them.

Usage (from the repository root)::

    .venv/bin/python worker/tests/journeys/record/generate_focus_iana_typed.py

The output is deterministic.
"""

from __future__ import annotations

import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

JOURNEYS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JOURNEYS_DIR.parent))  # worker/tests, for daemon_db
sys.path.insert(0, str(JOURNEYS_DIR / "record"))

from daemon_db import schema_version  # noqa: E402
from generate_focus_dailynews import compact, metadata_text, rfc3339  # noqa: E402
from journeys.fixture_loader import (  # noqa: E402
    FIXTURES_DIR,
    SCREENSHOTS_TOKEN,
    row_counts,
    write_manifest,
    write_rows,
)

NAME = "focus_iana_typed"
SEED = 20260924
START = datetime(2026, 9, 24, 9, 12, 0, tzinfo=timezone.utc)
SESSION_TITLE = "Note the IANA reserved domains"

DISPLAY = {
    "display_id": "1",
    "bounds_global_px": [0, 0, 1512, 982],
    "scale_factor": 2.0,
    "orientation": 0,
}
SAFARI_APP = "pid:611:Safari"
TEXTEDIT_APP = "pid:902:TextEdit"
SAFARI_WINDOW = {"window_id": "3120", "bounds": [0, 25, 1512, 957]}
TEXTEDIT_WINDOW = {"window_id": "3377", "bounds": [300, 120, 800, 600]}
IANA_TITLE = "IANA-managed Reserved Domains"
IANA_URL = "iana.org/domains/reserved"

LINE_1 = "Reserved for documentation: example.com, example.net, example.org"
LINE_2 = "Source: IANA reserved domains page"
SAFARI_PAGE = [
    "IANA-managed Reserved Domains",
    "Certain domains are set aside, and nominally registered to IANA, for specific "
    "policy or technical purposes.",
    "Example domains",
    "As described in RFC 2606 and RFC 6761, a number of domains such as example.com "
    "and example.org are maintained for documentation purposes.",
    "Test IDN top-level domains",
]

# Timeline (seconds from START).
TO_TEXTEDIT = 10.4     # Cmd-Tab to TextEdit
NEW_DOCUMENT = 12.1    # Cmd-N: an empty "Untitled" window
TYPE_1 = (15.0, 35.0)  # typing the first line
TYPE_2 = (37.0, 47.0)  # Enter, then the second line
STOPPED_AT = 60.0
CHARS_PER_SECOND_1 = len(LINE_1) / (TYPE_1[1] - TYPE_1[0])
CHARS_PER_SECOND_2 = len(LINE_2) / (TYPE_2[1] - TYPE_2[0])


class Generator:
    def __init__(self) -> None:
        self.rng = random.Random(SEED)
        self.session_id = str(self._uuid())
        self.rows: list[tuple[float, dict]] = []

    def _uuid(self) -> uuid.UUID:
        return uuid.UUID(int=self.rng.getrandbits(128), version=4)

    def _time(self, offset: float) -> datetime:
        return START + timedelta(seconds=offset, microseconds=self.rng.randrange(1, 999))

    def native(self, offset: float, kind: dict, app_id: str, title: str,
               metadata: dict | None = None) -> None:
        window = SAFARI_WINDOW if app_id == SAFARI_APP else TEXTEDIT_WINDOW
        x, y, w, h = window["bounds"]
        meta = dict(metadata or {})
        meta["focus_session_id"] = self.session_id
        self.rows.append((offset, {
            "id": str(self._uuid()),
            "timestamp": rfc3339(self._time(offset)),
            "kind_json": compact(kind),
            "window_json": compact({
                "window_id": window["window_id"],
                "app_id": app_id,
                "title": title,
                "bounds_global_px": window["bounds"],
                "z_order": 0,
                "is_fullscreen": False,
            }),
            "display_topology_json": compact([DISPLAY]),
            "primary_display_id": DISPLAY["display_id"],
            "cursor_x": x + self.rng.randrange(40, w - 40),
            "cursor_y": y + self.rng.randrange(60, h - 40),
            "ui_scale": DISPLAY["scale_factor"],
            "artifact_ids_json": "[]",
            "metadata_json": metadata_text(meta),
            "display_ids_spanned_json": None,
        }))

    def focus_frame(self, offset: float, app_id: str, title: str,
                    boxes: list[tuple[float, float, float, str]]) -> None:
        """A frame whose OCR lines are *boxes*: ``(x, y, width, text)`` in
        points, each one line (14 points) high."""
        _dx, _dy, dw, dh = DISPLAY["bounds_global_px"]
        elements = [
            {
                "text": text,
                "confidence": 1.0,
                "bbox_normalized": [round(x / dw, 4), round(y / dh, 4),
                                    round(w / dw, 4), round(14 / dh, 4)],
            }
            for x, y, w, text in boxes
        ]
        self.native(offset, {"type": "DwellSnapshot"}, app_id, title, {
            "focus_capture": True,
            "ocr": {
                "elements": elements,
                "full_text": "\n".join(text for *_xyz, text in boxes),
                "processing_time_ms": self.rng.randrange(180, 420),
            },
            "screenshot_path": f"{SCREENSHOTS_TOKEN}/{self._uuid()}.jpg",
        })


def safari_screen() -> list[tuple[float, float, float, str]]:
    return [
        (10, 4, 400, "Safari File Edit View History Bookmarks Window Help"),
        (200, 40, 260, IANA_TITLE),
        (480, 40, 160, "Example Domain"),
        (600, 66, 220, IANA_URL),
        *((120, 160 + 40 * i, 1000, line) for i, line in enumerate(SAFARI_PAGE)),
    ]


def document_lines(t: float) -> list[str]:
    """The TextEdit document's lines at ``t``."""
    if t < TYPE_1[0]:
        return []
    first = LINE_1[:int((min(t, TYPE_1[1]) - TYPE_1[0]) * CHARS_PER_SECOND_1)]
    if t < TYPE_2[0]:
        return [first] if first else []
    second = LINE_2[:int((min(t, TYPE_2[1]) - TYPE_2[0]) * CHARS_PER_SECOND_2)]
    return [LINE_1, second] if second else [LINE_1]


def textedit_screen(t: float) -> list[tuple[float, float, float, str]]:
    return [
        (10, 4, 420, "TextEdit File Edit Format View Window Help"),
        (330, 150, 140, "Helvetica Regular"),
        *((330, 220 + 24 * i, 700, line) for i, line in enumerate(document_lines(t))),
    ]


def build() -> Generator:
    gen = Generator()
    gen.native(TO_TEXTEDIT, {"type": "AppSwitch", "from_app": SAFARI_APP,
                             "to_app": TEXTEDIT_APP}, TEXTEDIT_APP, "")
    gen.native(NEW_DOCUMENT, {"type": "WindowTitleChange"}, TEXTEDIT_APP, "Untitled")
    t = 0.9
    while t < STOPPED_AT:
        if t < TO_TEXTEDIT:
            gen.focus_frame(t, SAFARI_APP, IANA_TITLE, safari_screen())
        elif t >= NEW_DOCUMENT:
            gen.focus_frame(t, TEXTEDIT_APP, "Untitled", textedit_screen(t))
        t += 2.0
    gen.rows.sort(key=lambda item: item[0])
    return gen


def manifest_for(gen: Generator) -> dict:
    rows = [row for _, row in gen.rows]
    return {
        "name": NAME,
        "description": (
            "1-minute typing-only focus recording (the test Mac's session 46b767ef): "
            "Safari on the IANA reserved domains page, then two lines typed into a "
            "new TextEdit document; nothing copied or pasted."
        ),
        "source": {
            "kind": "generated",
            "by": "worker/tests/journeys/record/generate_focus_iana_typed.py",
            "note": "Rebuilt from the live session's actions; its model answers are "
                    "recorded into cassettes/focus_iana_typed.jsonl on the test Mac.",
        },
        "schema_version": schema_version(),
        "recorded_start": rows[0]["timestamp"],
        "recorded_end": rows[-1]["timestamp"],
        "focus_session": {
            "session_id": gen.session_id,
            "title": SESSION_TITLE,
            "started_at": (START - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "stopped_at": (START + timedelta(seconds=STOPPED_AT)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "row_counts": row_counts(rows),
    }


def main() -> None:
    gen = build()
    out = FIXTURES_DIR / NAME
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out, [row for _, row in gen.rows])
    write_manifest(out, manifest_for(gen))
    print(f"wrote {len(gen.rows)} rows to {out}")


if __name__ == "__main__":
    main()
