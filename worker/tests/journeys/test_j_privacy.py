"""J-privacy (worker side): what passive capture of a password manager, a
password pasted into Gmail and a tokenised reset link leaves behind.

``passive_privacy`` holds the rows the daemon writes as of WP-V1 (no
clipboard preview by default, extension URLs reduced to origin + path).
The journey ticks the worker over them on the recorded cassette and checks
what reached the model and what was stored.

Not here: the daemon-side halves (no row at all for the 1Password frame once
WP-V2 lands in Rust, the storage-writer URL and clipboard gates, pinned by
``crates/daemon`` tests) and ``agenthandover uninstall --purge-data`` in a
temp prefix, which runs on the test Mac.
"""

from __future__ import annotations

import json
import os
import sqlite3

import pytest

from journeys.cassette import CASSETTE_DIR, load_entries

TICKS = 4

# The activity classifier's call on the reset-password frame, as the
# worker makes it since WP-C9 (annotation model, num_ctx 4096,
# num_predict 16) with the prompt WP-F1 redacts (no reset token): neither
# branch recorded this request, and the test Mac's Ollama was down when
# they were merged.  Every test here ticks through it.
UNRECORDED_CLASSIFIER_KEY = "51f238e10b29df3b4d9394b88319d59f60dc36cfae36276ac8d56931c03a626b"
pytestmark = pytest.mark.skipif(
    UNRECORDED_CLASSIFIER_KEY not in load_entries(CASSETTE_DIR / "passive_privacy.jsonl")
    and os.environ.get("AH_RECORD") != "1",
    reason=(
        f"cassettes/passive_privacy.jsonl has no entry for {UNRECORDED_CLASSIFIER_KEY} "
        "(the redacted classifier call at the annotation context): re-record "
        "with AH_RECORD=1 on the test Mac"
    ),
)


def _run(events_db, worker, clock, temp_home, cassette):
    tape = cassette("passive_privacy")
    fixture = events_db("passive_privacy", rebase_to=clock.now())
    journey = worker(fixture)
    for _ in range(TICKS):
        journey.run_tick()
        clock.advance(seconds=30)
    return fixture, journey, tape


def _rows(fixture) -> list[dict]:
    conn = sqlite3.connect(str(fixture.db_path))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM events ORDER BY timestamp")]
    finally:
        conn.close()


def _frames(fixture, app_suffix: str) -> list[dict]:
    return [
        r for r in _rows(fixture)
        if r["window_json"] and json.loads(r["window_json"])["app_id"].endswith(app_suffix)
        and "screenshot_path" in json.loads(r["metadata_json"])
    ]


def _prompts(tape) -> list[str]:
    return [json.dumps(e["request"], ensure_ascii=False) for e in tape.served]


def test_privacy_passive_frames_are_processed(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    """The Gmail and reset-page frames reach annotation (the journey's other
    assertions would hold vacuously if nothing were annotated), and the rows
    as the daemon wrote them carry no clipboard preview and no query string."""
    fixture, _journey, _tape = _run(events_db, worker, clock, temp_home, cassette)
    chrome = _frames(fixture, ":Google Chrome")
    assert [r["annotation_status"] for r in chrome] == ["completed", "completed"]
    for row in _rows(fixture):
        kind = json.loads(row["kind_json"])
        if kind["type"] == "ClipboardChange":
            assert "content_preview" not in kind
        url = json.loads(row["metadata_json"]).get("url")
        if url:
            assert "?" not in url


def test_privacy_password_manager_frame_is_never_annotated_and_its_jpeg_is_deleted(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    fixture, _journey, tape = _run(events_db, worker, clock, temp_home, cassette)
    [frame] = _frames(fixture, ":1Password")
    shot = json.loads(frame["metadata_json"])["screenshot_path"]
    assert frame["annotation_status"] == "privacy_blocked"
    assert not frame["scene_annotation_json"]
    assert not any("1Password" in p for p in _prompts(tape))
    assert not (fixture.screenshots_dir / shot.rsplit("/", 1)[-1]).exists()


def test_privacy_password_on_screen_never_reaches_the_model_or_the_store(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    fixture, journey, tape = _run(events_db, worker, clock, temp_home, cassette)
    password = fixture.manifest["secrets"]["password"]
    gmail_prompts = [p for p in _prompts(tape) if "Gmail" in p]
    assert gmail_prompts
    assert not any(password in p for p in gmail_prompts)
    assert not any(password in (r["scene_annotation_json"] or "") for r in _rows(fixture))
    for root in (journey.knowledge_base.root, journey.state_dir):
        for path in root.rglob("*.json"):
            assert password not in path.read_text(encoding="utf-8", errors="replace"), path


def test_privacy_stored_urls_carry_no_query_string(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    """The reset page's token is on screen (address bar OCR) but neither the
    prompt nor any URL the worker stores keeps it."""
    fixture, journey, tape = _run(events_db, worker, clock, temp_home, cassette)
    token = fixture.manifest["secrets"]["reset_token"]
    reset_prompts = [p for p in _prompts(tape) if "Reset your password" in p]
    assert reset_prompts
    assert not any(token in p for p in _prompts(tape))
    # The address bar line reaches the model as host and path.
    assert all("accounts.example.com/reset\\n" in p for p in reset_prompts)
    annotations = [
        json.loads(r["scene_annotation_json"]) for r in _rows(fixture)
        if r["scene_annotation_json"]
    ]
    assert annotations
    assert not any(token in json.dumps(a) for a in annotations)
    for path in journey.knowledge_base.root.rglob("*.json"):
        assert token not in path.read_text(encoding="utf-8", errors="replace"), path
