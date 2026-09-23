"""J-focus clipboard: copy and paste steps only where the clipboard shows them.

The live run of 2026-09-23 (qwen3.5:4b) turned two typed lines in TextEdit
into "Copy the text ... / Paste the copied text" with no clipboard event
recorded.  ``focus_dailynews`` records four copies (``ClipboardChange``):
its Copy and Paste steps stay.  The same recording with its clipboard
events removed is a session in which nothing was copied: no step may say
copy or paste, whatever the model writes.
"""

from __future__ import annotations

import os
import re
import sqlite3

import pytest

from agenthandover_worker.skill_render import render_packet
from journeys.cassette import CASSETTE_DIR, load_entries
from journeys.focus_run import start_focus_run, tick_until_skill
from journeys.packet import parse_packet

_CLAIM = re.compile(r"\b(copy|copies|copied|copying|paste|pastes|pasted|pasting)\b", re.I)

# The requests the no-clipboard run makes that the focus_dailynews cassette
# does not hold: the SOP prompt carries the "no copy, cut or paste was
# recorded" note (WP-Q1); the annotations and diffs replay as they are.
NO_CLIPBOARD_MISSING_KEYS: tuple[str, ...] = (
    "480e9411054222d764723ea0b6e5e7d37a1501c6711d4af5d0e138c99ffc65fc",  # the SOP call
)
_ENTRIES = load_entries(CASSETTE_DIR / "focus_dailynews.jsonl")
needs_no_clipboard_recording = pytest.mark.skipif(
    any(key not in _ENTRIES for key in NO_CLIPBOARD_MISSING_KEYS)
    and os.environ.get("AH_RECORD") != "1",
    reason=(
        "cassettes/focus_dailynews.jsonl lacks the requests of the recording "
        "without its clipboard events (WP-Q1): "
        + ", ".join(k for k in NO_CLIPBOARD_MISSING_KEYS if k not in _ENTRIES)
        + " (and the calls they unblock); re-record with AH_RECORD=1 on the test Mac"
    ),
)


def _claims(text: str) -> list[str]:
    return [line for line in text.splitlines() if _CLAIM.search(line)]


def test_recorded_copies_keep_their_copy_and_paste_steps(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    run = start_focus_run(
        events_db=events_db, worker=worker, clock=clock,
        temp_home=temp_home, cassette=cassette,
    )
    proc = tick_until_skill(run, clock)
    actions = [str(s.get("action") or "") for s in proc["steps"]]
    assert sum(a.lower().startswith("copy") for a in actions) >= 3, actions
    assert sum(a.lower().startswith("paste") for a in actions) >= 3, actions
    # Every copy/paste was supported: none rewritten, none flagged.
    assert not [s for s in proc["steps"] if (s.get("evidence") or {}).get("clipboard")]
    assert "Unconfirmed:" not in render_packet(proc, target="claude-skill")


@needs_no_clipboard_recording
def test_a_session_without_clipboard_events_has_no_copy_or_paste_steps(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    run = start_focus_run(
        events_db=events_db, worker=worker, clock=clock,
        temp_home=temp_home, cassette=cassette, journey="focus_dailynews",
    )
    with sqlite3.connect(run.fixture.db_path) as conn:
        removed = conn.execute(
            "DELETE FROM events WHERE json_extract(kind_json, '$.type') "
            "IN ('ClipboardChange', 'PasteDetected')"
        ).rowcount
    assert removed == 4
    proc = tick_until_skill(run, clock)
    packet = parse_packet(render_packet(proc, target="claude-skill"))
    for step in packet.steps():
        head = step.text.splitlines()[0]
        assert not _CLAIM.search(head), step.text
        assert not _claims(step.fields.get("Verify", "")), step.text
