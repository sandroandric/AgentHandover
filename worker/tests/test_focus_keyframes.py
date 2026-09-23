"""Focus keyframes: which frames of a recording reach the model, and when.

``select_keyframes`` picks at most 60 frames from row signals alone; the
SOP's 20 are annotated before the SOP call, the other keyframes only after
the Skill is written, and the rest are never annotated.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agenthandover_worker.db import WorkerDB
from agenthandover_worker.focus_processor import (
    FOCUS_KEYFRAMES,
    SOP_KEYFRAMES,
    FocusProcessor,
    select_keyframes,
)
from test_focus_processor import (
    _create_test_db,
    _make_mock_annotator,
    _make_mock_differ,
    _make_mock_sop_generator,
)

_START = datetime(2026, 3, 3, 10, 0, tzinfo=timezone.utc)


def _frame(i: int, *, title: str = "Inbox", ocr: str = "", seconds: float | None = None,
           url: str = "") -> dict:
    when = _START + timedelta(seconds=i if seconds is None else seconds)
    metadata = {"focus_session_id": "s1"}
    if ocr:
        metadata["ocr"] = {"full_text": ocr}
    if url:
        metadata["url"] = url
    return {
        "id": f"f{i}",
        "timestamp": when.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "window_json": json.dumps({"app_id": "com.google.Chrome", "title": title}),
        "metadata_json": json.dumps(metadata),
        "annotation_status": "pending",
    }


class TestSelectKeyframes:

    def test_short_sessions_keep_every_frame(self):
        frames = [_frame(i) for i in range(5)]
        assert select_keyframes(frames, 60) == frames

    def test_first_and_last_of_every_dwell_and_the_session(self):
        # Three dwells of 30 frames each: Inbox, Compose, Sent.
        frames = [
            _frame(i, title=("Inbox", "Compose", "Sent")[i // 30]) for i in range(90)
        ]
        picked = {f["id"] for f in select_keyframes(frames, 8)}
        assert {"f0", "f29", "f30", "f59", "f60", "f89"} <= picked
        assert len(picked) == 8

    def test_url_change_starts_a_dwell(self):
        frames = [
            _frame(i, url="https://a.example/1" if i < 40 else "https://a.example/2")
            for i in range(80)
        ]
        picked = {f["id"] for f in select_keyframes(frames, 4)}
        assert picked == {"f0", "f39", "f40", "f79"}

    def test_copy_click_ocr_change_and_pause_outrank_quiet_frames(self):
        # One frame every 10 s on one page; a 2-minute pause before f50.
        frames = [
            _frame(i, ocr="same words on screen", seconds=i * 10 + (120 if i >= 50 else 0))
            for i in range(100)
        ]
        # f20 shows new text (so f20 and f21 differ from their predecessor).
        frames[20]["metadata_json"] = json.dumps({
            "focus_session_id": "s1", "ocr": {"full_text": "a new dialog opened"},
        })
        click = datetime.strptime(frames[70]["timestamp"], "%Y-%m-%dT%H:%M:%S.000Z").replace(
            tzinfo=timezone.utc,
        )
        picked = select_keyframes(
            frames, 7, click_times=[click], clipboard_ids={"f40"},
        )
        ids = [f["id"] for f in picked]
        assert ids == sorted(ids, key=lambda x: int(x[1:]))  # time order
        assert {"f0", "f99", "f40", "f20", "f21", "f50", "f70"} == set(ids)


def _insert(db_path: Path, frames: list[dict], screenshots: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    for f in frames:
        img = screenshots / f"{f['id']}.jpg"
        img.write_bytes(b"\xff\xd8\xff")
        meta = json.loads(f["metadata_json"])
        meta["screenshot_path"] = str(img)
        conn.execute(
            "INSERT INTO events (id, timestamp, kind_json, window_json, metadata_json) "
            "VALUES (?, ?, '{\"type\":\"DwellSnapshot\"}', ?, ?)",
            (f["id"], f["timestamp"], f["window_json"], json.dumps(meta)),
        )
    conn.commit()
    conn.close()


class TestSessionOrder:

    def test_the_sop_waits_for_twenty_annotations_only(self, tmp_path):
        """100 frames: 20 annotated, then the SOP call; 40 more afterwards;
        40 never.  Time-to-Skill is 20 annotations plus one SOP call."""
        db_path = _create_test_db(tmp_path)
        screenshots = tmp_path / "shots"
        screenshots.mkdir()
        frames = [
            _frame(i, title=f"Page {i // 10}", ocr=f"page {i // 10} line {i}")
            for i in range(100)
        ]
        _insert(db_path, frames, screenshots)

        order: list[str] = []
        annotator = _make_mock_annotator()
        annotate = annotator.annotate_event.side_effect
        annotator.annotate_event.side_effect = (
            lambda event, **kw: order.append("annotate") or annotate(event, **kw)
        )
        annotator.discard_screenshot.side_effect = (
            lambda event, _dir: Path(json.loads(event["metadata_json"])["screenshot_path"]).unlink()
        )
        generator = _make_mock_sop_generator()
        generate = generator.generate_from_focus.side_effect
        generator.generate_from_focus.side_effect = (
            lambda timeline, title: order.append("sop") or generate(timeline, title)
        )
        processor = FocusProcessor(annotator, _make_mock_differ(), generator)

        with WorkerDB(db_path) as db:
            result = processor.process_session(
                db, "s1", "Task", db.get_focus_session_events("s1"),
                screenshots_dir=str(screenshots),
            )
            assert result.success
            assert order == ["annotate"] * SOP_KEYFRAMES + ["sop"]
            timeline = generator.generate_from_focus.call_args.args[0]
            assert len(timeline) == SOP_KEYFRAMES

            stats = processor.annotate_remaining(db)
            assert stats["annotated"] == FOCUS_KEYFRAMES - SOP_KEYFRAMES
            assert order.count("annotate") == FOCUS_KEYFRAMES
            assert processor.annotate_remaining(db)["annotated"] == 0

        conn = sqlite3.connect(str(db_path))
        statuses = dict(conn.execute(
            "SELECT annotation_status, COUNT(*) FROM events GROUP BY 1"
        ).fetchall())
        conn.close()
        assert statuses == {"completed": FOCUS_KEYFRAMES, "skipped": 100 - FOCUS_KEYFRAMES}
        # Frames that are never annotated do not keep their JPEG.
        assert len(list(screenshots.iterdir())) == FOCUS_KEYFRAMES

    def test_ollama_down_changes_nothing(self, tmp_path):
        db_path = _create_test_db(tmp_path)
        screenshots = tmp_path / "shots"
        screenshots.mkdir()
        _insert(db_path, [_frame(i, title=f"P{i // 5}") for i in range(90)], screenshots)
        annotator = _make_mock_annotator()
        annotator.ready.return_value = False
        processor = FocusProcessor(annotator, _make_mock_differ(), _make_mock_sop_generator())
        with WorkerDB(db_path) as db:
            result = processor.process_session(
                db, "s1", "Task", db.get_focus_session_events("s1"),
                screenshots_dir=str(screenshots),
            )
            assert result.deferred
            assert processor.annotate_remaining(db)["annotated"] == 0
        annotator.annotate_event.assert_not_called()
        annotator.discard_screenshot.assert_not_called()
        conn = sqlite3.connect(str(db_path))
        assert conn.execute(
            "SELECT COUNT(*) FROM events WHERE annotation_status = 'pending'"
        ).fetchone()[0] == 90
        conn.close()
