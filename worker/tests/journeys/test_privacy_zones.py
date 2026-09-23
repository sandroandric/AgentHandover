"""J-privacy, zones: a password manager's window never survives as a file.

The first line is the daemon: it evaluates ``[privacy.zones]`` for the
focused window before it asks the app for a frame, so a 1Password window
produces no frame row and no JPEG at all
(``crates/daemon/tests/privacy_zones_test.rs``: blocked app, blocked title,
blocked page URL, metadata-only app, and a failed observation tick all
send zero capture requests).

This journey drives the second line, the worker, the way a user meets it:

1. The user blocks ``1password*`` in config.toml (with comments), then runs
   ``agenthandover-setup-vlm --remote`` to switch to a cloud VLM.  The
   zone block and every comment survive the command byte for byte.
2. Frames that reached the database anyway (written by an older daemon or
   before the zone was configured) are settled without a model: the
   1Password frame is marked ``privacy_blocked`` and its JPEG is deleted,
   the metadata-only frame keeps its metadata and loses its JPEG, while
   Ollama is down.
3. At the next worker start, JPEGs no pending event needs are swept.
"""

from __future__ import annotations

import builtins
import json
import os
import sqlite3
import sys
import time
import tomllib
from pathlib import Path

from agenthandover_worker import main as wm
from agenthandover_worker import setup_vlm
from agenthandover_worker.config import load_config

ZONES = """\
# Hand-written by the user: keep the password manager out.
[privacy.zones]
# 1Password, any version
blocked = ["1password*"]
metadata_only = ["Messages"]   # chats: app and time only
"""


def _run_setup_vlm_remote(monkeypatch) -> None:
    """``agenthandover-setup-vlm --remote`` as a user answers it: consent,
    provider 1 (OpenAI), use the key already in the environment."""
    answers = iter(["y", "1", "y"])
    monkeypatch.setenv("OPENAI_API_KEY", "sk-journey-not-a-real-key")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    setup_vlm.main(["--remote"])


def _frames(db_path: Path) -> list[tuple[str, dict]]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT id, metadata_json FROM events "
            "WHERE json_extract(metadata_json, '$.screenshot_path') IS NOT NULL "
            "ORDER BY timestamp"
        ).fetchall()
    finally:
        conn.close()
    return [(row[0], json.loads(row[1])) for row in rows]


def _make_passive_frame(db_path: Path, frame_id: str, app_id: str, title: str) -> Path:
    """Turn a recorded frame into a passive frame of *app_id* (as the daemon
    writes it: ``pid:<pid>:<name>``); returns its JPEG."""
    conn = sqlite3.connect(str(db_path))
    try:
        window, meta = conn.execute(
            "SELECT window_json, metadata_json FROM events WHERE id = ?", (frame_id,)
        ).fetchone()
        window = json.loads(window) if window else {}
        window.update({"app_id": app_id, "title": title})
        meta = json.loads(meta)
        meta.pop("focus_session_id", None)
        meta.pop("focus_capture", None)
        conn.execute(
            "UPDATE events SET window_json = ?, metadata_json = ?, "
            "annotation_status = 'pending' WHERE id = ?",
            (json.dumps(window), json.dumps(meta), frame_id),
        )
        conn.commit()
    finally:
        conn.close()
    return Path(meta["screenshot_path"])


def _status(db_path: Path, frame_id: str) -> tuple[str, str | None]:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT annotation_status, scene_annotation_json FROM events WHERE id = ?",
            (frame_id,),
        ).fetchone()
    finally:
        conn.close()


def test_blocked_frames_leave_no_file_even_after_setup_vlm(
    temp_home, events_db, worker, monkeypatch,
) -> None:
    # 1. Zones configured by hand, then setup-vlm --remote mid-journey.
    config_file = temp_home.config_file
    config_file.write_text(config_file.read_text(encoding="utf-8") + "\n" + ZONES,
                           encoding="utf-8")
    _run_setup_vlm_remote(monkeypatch)

    after = config_file.read_text(encoding="utf-8")
    assert after.endswith(ZONES), "the zone block and its comments survive byte for byte"
    assert tomllib.loads(after)["vlm"]["mode"] == "remote"
    assert load_config().privacy_zones["blocked"] == ["1password*"]

    # 2. Frames that reached the database anyway.
    fixture = events_db("focus_dailynews")
    frames = _frames(fixture.db_path)
    assert len(frames) >= 3
    other_id, vault_id, chat_id = frames[0][0], frames[1][0], frames[2][0]
    conn = sqlite3.connect(str(fixture.db_path))
    conn.execute("UPDATE events SET annotation_status = 'skipped'")
    conn.commit()
    conn.close()
    # An ordinary frame first in the batch: with Ollama down it stays
    # pending, and must not hold up the privacy rows behind it.
    other_jpeg = _make_passive_frame(fixture.db_path, other_id, "pid:77:TextEdit", "notes.txt")
    vault_jpeg = _make_passive_frame(fixture.db_path, vault_id, "pid:4242:1Password", "Vault")
    chat_jpeg = _make_passive_frame(fixture.db_path, chat_id, "pid:88:Messages", "Alice")
    for jpeg in (vault_jpeg, chat_jpeg, other_jpeg):
        assert jpeg.is_file()

    w = worker(fixture)
    assert w.scene_annotator is not None
    stats = wm._process_annotations(
        w.db, w.scene_annotator, fixture.screenshots_dir,
        privacy_checker=w.privacy_checker, activity_classifier=None,
    )
    assert stats["blocked"] == 1 and stats["annotated"] == 0, stats

    assert _status(fixture.db_path, vault_id) == ("privacy_blocked", "")
    assert not vault_jpeg.exists(), "no file for the 1Password frame"
    status, annotation = _status(fixture.db_path, chat_id)
    assert status == "metadata_only"
    assert json.loads(annotation)["privacy_tier"] == "metadata_only"
    assert not chat_jpeg.exists()
    assert _status(fixture.db_path, other_id)[0] == "pending"
    assert other_jpeg.is_file(), "the ordinary frame waits for Ollama with its JPEG"

    # 3. Next worker start (an hour later): the orphan sweep.
    old = time.time() - 3600
    for jpeg in fixture.screenshots_dir.iterdir():
        os.utime(jpeg, (old, old))
    stray = fixture.screenshots_dir / "left-by-an-older-daemon.jpg"
    stray.write_bytes(other_jpeg.read_bytes())
    os.utime(stray, (old, old))
    fresh = fixture.screenshots_dir / "being-written-now.jpg"
    fresh.write_bytes(other_jpeg.read_bytes())
    removed = wm._sweep_orphan_screenshots(w.db, fixture.screenshots_dir)
    assert not stray.exists()
    assert fresh.exists(), "a file younger than the daemon's insert window is kept"
    assert other_jpeg.is_file(), "a pending frame's JPEG is kept"
    remaining = {p.name for p in fixture.screenshots_dir.iterdir()}
    assert remaining == {other_jpeg.name, fresh.name}, remaining
    assert removed >= 1
