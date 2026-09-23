"""WP-M2: native-app and terminal evidence from the daemon's rows to a
Skill step.

The rows are written the way the daemon writes them (the daemon's own
migrations, ``focused_element_json`` and ``ShellCommand``/``SystemSleep``
kinds as ``crates/common/src/event.rs`` serialises them), then read through
``WorkerDB``, the focus timeline, the SOP prompt and the step evidence.  The
SOP model call is replaced by a canned answer; everything else is the
product code.  (The J-passive Excel instance and a developer variant of
J-focus extend WP-T4's journeys.)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from agenthandover_worker.db import WorkerDB
from agenthandover_worker.episode_builder import EpisodeBuilder
from agenthandover_worker.event_helpers import describe_focused_field, focused_field
from agenthandover_worker.focus_processor import FocusProcessor
from agenthandover_worker.procedure_schema import _step_evidence
from agenthandover_worker.sop_generator import SOPGenerator, _format_timeline_entry
from agenthandover_worker.task_segmenter import (
    AnnotatedFrame,
    SegmenterConfig,
    TaskSegmenter,
    _split_into_contiguous_segments,
    _sleep_starts,
)

BASE = datetime(2026, 9, 21, 9, 0, 0, tzinfo=timezone.utc)

# What the app's reader returns for an Excel cell (FocusedElementReader).
EXCEL_CELL = {
    "role": "AXCell",
    "value": "1,250.00",
    "column_header": "Amount",
    "document": "file:///Users/me/Q3%20expenses.xlsx",
    "ancestors": [
        {"role": "AXRow", "description": "Row 4", "index": 3},
        {"role": "AXTable", "title": "Expenses", "index": 0},
        {"role": "AXScrollArea"},
    ],
}


def _ts(dt: datetime) -> str:
    """The daemon's timestamp text (chrono ``to_rfc3339``)."""
    return dt.isoformat()


def _insert(
    conn: sqlite3.Connection,
    event_id: str,
    at: datetime,
    kind: dict,
    *,
    window: dict | None = None,
    focused: dict | None = None,
    annotation: dict | None = None,
    metadata: dict | None = None,
) -> None:
    conn.execute(
        "INSERT INTO events (id, timestamp, kind_json, window_json, display_topology_json, "
        "primary_display_id, metadata_json, focused_element_json, "
        "annotation_status, scene_annotation_json) "
        "VALUES (?, ?, ?, ?, '[]', 'main', ?, ?, ?, ?)",
        (
            event_id,
            _ts(at),
            json.dumps(kind),
            json.dumps(window) if window else None,
            json.dumps(metadata or {}),
            json.dumps(focused) if focused else None,
            "completed" if annotation else "pending",
            json.dumps(annotation) if annotation else None,
        ),
    )
    conn.commit()


def _annotation(app: str, what: str, location: str = "") -> dict:
    return {
        "app": app,
        "location": location,
        "task_context": {"what_doing": what, "is_workflow": True},
    }


# ---------------------------------------------------------------------------
# The focused field
# ---------------------------------------------------------------------------


class TestFocusedField:
    def test_cell_takes_its_column_header_value_and_context(self) -> None:
        field = focused_field({"focused_element_json": json.dumps(EXCEL_CELL)})
        assert field == {
            "role": "cell",
            "label": "Amount",
            "value": "1,250.00",
            "context": ["Expenses", "Row 4"],
            "document": "file:///Users/me/Q3%20expenses.xlsx",
        }
        assert describe_focused_field(field) == (
            'cell "Amount" = "1,250.00" (in Expenses > Row 4)'
        )

    def test_unlabelled_field_inherits_the_nearest_labelled_ancestor(self) -> None:
        element = {
            "role": "AXTextField",
            "ancestors": [{"role": "AXGroup"}, {"role": "AXToolbar", "description": "Search"}],
        }
        field = focused_field({"focused_element_json": element})
        assert field == {"role": "text field", "label": "Search"}

    def test_no_element_or_no_role_is_none(self) -> None:
        assert focused_field({}) is None
        assert focused_field({"focused_element_json": None}) is None
        assert focused_field({"focused_element_json": "{}"}) is None
        assert describe_focused_field(None) == ""


# ---------------------------------------------------------------------------
# The prompt and the step evidence
# ---------------------------------------------------------------------------


class TestPromptAndEvidence:
    def test_timeline_entry_quotes_field_and_commands_verbatim(self) -> None:
        text = _format_timeline_entry(
            0, _annotation("Terminal", "Running tests"), None, _ts(BASE),
            focused_field={"role": "cell", "label": "Amount", "value": "1,250.00"},
            shell_commands=[
                {"cwd": "/Users/me/app", "command": 'git commit -m "fix: rounding"'},
                {"cwd": "", "command": "make test\nmake lint"},
            ],
        )
        assert '  Focused field: cell "Amount" = "1,250.00"' in text
        assert '  >>> SHELL COMMAND (in /Users/me/app): git commit -m "fix: rounding"' in text
        assert "  >>> SHELL COMMAND: make test\\nmake lint" in text

    def test_entry_without_them_is_unchanged(self) -> None:
        ann = _annotation("Notes", "Writing")
        assert _format_timeline_entry(0, ann, None, _ts(BASE)) == _format_timeline_entry(
            0, ann, None, _ts(BASE), focused_field=None, shell_commands=[],
        )

    def test_procedure_keeps_the_recorded_evidence(self) -> None:
        evidence = {
            "frames": ["f1"],
            "dom_snapshot_ids": [],
            "focused_fields": [{"role": "cell", "label": "Amount", "event_id": "f1"}],
            "shell_commands": [{"command": "make", "cwd": "/a"}],
        }
        assert _step_evidence(evidence) == evidence
        assert _step_evidence({"frames": ["f1"]}) == {"frames": ["f1"], "dom_snapshot_ids": []}


class TestFocusSessionCarriesNativeEvidence:
    """A focus recording with an Excel frame and terminal commands, read
    from a daemon database, gives steps whose evidence holds the cell's
    label and value and the exact commands."""

    def _session(self, conn: sqlite3.Connection) -> list[str]:
        _insert(
            conn, "f1", BASE, {"type": "DwellSnapshot"},
            window={"app_id": "pid:501:Microsoft Excel", "title": "Q3 expenses.xlsx"},
            focused=EXCEL_CELL,
            annotation=_annotation("Microsoft Excel", "Entering an expense amount", "Q3 expenses.xlsx"),
        )
        # Typed in the terminal before the terminal frame was captured.
        _insert(conn, "c1", BASE + timedelta(seconds=20),
                {"type": "ShellCommand", "cwd": "/Users/me/ledger", "command": "python export.py --month 2026-09"})
        _insert(conn, "c2", BASE + timedelta(seconds=25),
                {"type": "ShellCommand", "cwd": "/Users/me/ledger", "command": 'git commit -am "September export"'})
        _insert(
            conn, "f2", BASE + timedelta(seconds=30), {"type": "DwellSnapshot"},
            window={"app_id": "pid:733:Terminal", "title": "ledger"},
            annotation=_annotation("Terminal", "Exporting and committing the ledger"),
        )
        # After the session's last frame: not part of it.
        _insert(conn, "c3", BASE + timedelta(minutes=5),
                {"type": "ShellCommand", "cwd": "/", "command": "ls"})
        return ["f1", "f2"]

    def test_steps_carry_the_cell_and_the_exact_commands(self, tmp_db_path: Path, write_conn) -> None:
        frame_ids = self._session(write_conn)
        with WorkerDB(tmp_db_path) as db:
            events = [db.get_event_by_id(eid) for eid in frame_ids]
            timeline = FocusProcessor(None, None, SOPGenerator())._collect_timeline(db, events)

        assert timeline[0]["focused_field"]["label"] == "Amount"
        assert timeline[0]["shell_commands"] == []
        assert [c["command"] for c in timeline[1]["shell_commands"]] == [
            "python export.py --month 2026-09",
            'git commit -am "September export"',
        ]

        answer = {
            "title": "Record the monthly expense export",
            "description": "Enter the amount, export and commit.",
            "steps": [
                {"step_number": 1, "action": "Enter the amount in the Amount column",
                 "app": "Microsoft Excel", "location": "Q3 expenses.xlsx",
                 "input": "1,250.00", "verify": "", "frames": [1]},
                {"step_number": 2, "action": "Export and commit the ledger",
                 "app": "Terminal", "location": "/Users/me/ledger",
                 "input": "python export.py --month 2026-09", "verify": "", "frames": [2]},
            ],
            "variables": [],
        }
        prompts: list[str] = []

        def canned(**kwargs):
            prompts.append(kwargs["prompt"])
            return json.dumps(answer), 1.0

        with patch("agenthandover_worker.sop_generator._call_ollama", side_effect=canned):
            result = SOPGenerator().generate_from_focus(timeline, "Monthly export")

        assert result.success, result.error
        assert 'Focused field: cell "Amount" = "1,250.00" (in Expenses > Row 4)' in prompts[0]
        assert (
            '>>> SHELL COMMAND (in /Users/me/ledger): git commit -am "September export"'
            in prompts[0]
        )
        excel, terminal = result.sop["steps"]
        assert excel["evidence"]["focused_fields"] == [{
            "role": "cell", "label": "Amount", "value": "1,250.00",
            "context": ["Expenses", "Row 4"],
            "document": "file:///Users/me/Q3%20expenses.xlsx",
            "event_id": "f1",
        }]
        assert "shell_commands" not in excel["evidence"]
        assert [c["command"] for c in terminal["evidence"]["shell_commands"]] == [
            "python export.py --month 2026-09",
            'git commit -am "September export"',
        ]
        assert [c["event_id"] for c in terminal["evidence"]["shell_commands"]] == ["c1", "c2"]


# ---------------------------------------------------------------------------
# Sleep boundaries
# ---------------------------------------------------------------------------


def _sleep(conn: sqlite3.Connection, event_id: str, slept: datetime, woke: datetime) -> None:
    _insert(conn, event_id, woke, {
        "type": "SystemSleep", "slept_at": _ts(slept), "woke_at": _ts(woke),
    })


class TestSleepBoundaries:
    def test_db_lists_each_sleep_once(self, tmp_db_path: Path, write_conn) -> None:
        slept, woke = BASE + timedelta(minutes=10), BASE + timedelta(minutes=25)
        _sleep(write_conn, "s1", slept, woke)
        # The daemon reports a sleep again after a restart.
        _sleep(write_conn, "s1-again", slept, woke)
        _sleep(write_conn, "s0", BASE - timedelta(days=2), BASE - timedelta(days=2, hours=-8))
        with WorkerDB(tmp_db_path) as db:
            assert db.get_sleep_boundaries(since=_ts(BASE)) == [(_ts(slept), _ts(woke))]
            assert len(db.get_sleep_boundaries()) == 2

    def _frame(self, eid: str, at: datetime) -> AnnotatedFrame:
        return AnnotatedFrame(
            event_id=eid, timestamp=_ts(at), annotation={}, what_doing="Reviewing invoices",
            is_workflow=True, app="Preview",
        )

    def test_a_segment_never_spans_a_sleep(self) -> None:
        frames = [
            self._frame("a", BASE),
            self._frame("b", BASE + timedelta(minutes=2)),
            # Laptop lid closed for 12 minutes: well under the 30-minute gap.
            self._frame("c", BASE + timedelta(minutes=15)),
        ]
        sleep = [(_ts(BASE + timedelta(minutes=2, seconds=30)), _ts(BASE + timedelta(minutes=14)))]

        whole = _split_into_contiguous_segments(frames, 0, 1800)
        assert [[f.event_id for f in s.frames] for s in whole] == [["a", "b", "c"]]

        split = _split_into_contiguous_segments(frames, 0, 1800, _sleep_starts(sleep))
        assert [[f.event_id for f in s.frames] for s in split] == [["a", "b"], ["c"]]

    def test_interruption_merging_stops_at_a_sleep(self) -> None:
        segmenter = TaskSegmenter(SegmenterConfig())
        first = _split_into_contiguous_segments(
            [self._frame("a", BASE), self._frame("b", BASE + timedelta(minutes=1))], 0,
        )[0]
        second = _split_into_contiguous_segments(
            [self._frame("c", BASE + timedelta(minutes=11)),
             self._frame("d", BASE + timedelta(minutes=12))], 0,
        )[0]
        second.segment_id = "second"
        frames = first.frames + second.frames

        merged = segmenter.classify_interruptions([first, second], frames)
        assert len(merged) == 1, "a 10-minute pause merges without a sleep"

        sleep = [(_ts(BASE + timedelta(minutes=2)), _ts(BASE + timedelta(minutes=10)))]
        kept = segmenter.classify_interruptions([first, second], frames, sleep_boundaries=sleep)
        assert len(kept) == 2

    def test_episode_builder_closes_episodes_at_a_sleep(self) -> None:
        def event(eid: str, at: datetime, kind: dict) -> dict:
            return {
                "id": eid, "timestamp": _ts(at), "kind_json": json.dumps(kind),
                "window_json": json.dumps({"app_id": "com.apple.Preview", "title": "invoice.pdf - Preview"}),
                "metadata_json": "{}",
            }

        events = [
            event("a", BASE, {"type": "DwellSnapshot"}),
            event("s", BASE + timedelta(minutes=5), {
                "type": "SystemSleep",
                "slept_at": _ts(BASE + timedelta(minutes=1)),
                "woke_at": _ts(BASE + timedelta(minutes=5)),
            }),
            event("b", BASE + timedelta(minutes=6), {"type": "DwellSnapshot"}),
        ]
        episodes = EpisodeBuilder().process_events(events)
        assert [[e["id"] for e in ep.events] for ep in episodes] == [["a"], ["b"]]
