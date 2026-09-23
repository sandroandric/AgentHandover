"""J-ollama-down: J-focus while Ollama is down for its first 40 requests,
and J-focus while Ollama is slow to answer the first annotation.

The worker starts against a cassette that answers 503; while it does, the
focus session waits: no frame is annotated, no JPEG deleted, the signal
stays.  The annotator backs off on the journey clock (60 s), so each tick
moves the clock past the backoff and spends one probe on the outage.  Once
the cassette answers again the session completes, and the Skill is the one
J-focus writes (the same golden packet).

The slow case holds the first annotation longer than a warm call takes
but within the annotation timeout (``[vlm] annotation_timeout_seconds``),
as a cold model load on a small Mac does (90 s against the default 180 s).
The journey runs it scaled down, 2 s against a 4 s timeout, so it costs
seconds of wall time; its control holds the call past a 1 s timeout and
sees it deferred and asked again, which shows the configured timeout is
the one the call runs under.
"""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.request

from agenthandover_worker.config import VlmConfig
from agenthandover_worker.model_profiles import get_profile
from agenthandover_worker.scene_annotator import annotation_system_prompt
from agenthandover_worker.skill_render import render_packet
from journeys.focus_run import needs_focus_sop_recording, start_focus_run, tick_until_skill
from journeys.packet import assert_matches_golden

DOWN_REQUESTS = 40
MAX_OUTAGE_TICKS = 3 * DOWN_REQUESTS
# A cold model load measured on a 16 GB Mac, and the scaled journey.
COLD_LOAD_SECONDS = 90
SLOW_SECONDS = 2
SLOW_TIMEOUT_SECONDS = 4
SHORT_TIMEOUT_SECONDS = 1


def _focus_statuses(fixture) -> dict:
    conn = sqlite3.connect(str(fixture.db_path))
    try:
        return dict(conn.execute(
            "SELECT annotation_status, COUNT(*) FROM events "
            "WHERE json_extract(metadata_json, '$.focus_session_id') IS NOT NULL GROUP BY 1"
        ).fetchall())
    finally:
        conn.close()


@needs_focus_sop_recording
def test_ollama_down_defers_the_session_then_writes_the_same_skill(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    run = start_focus_run(
        events_db=events_db, worker=worker, clock=clock,
        temp_home=temp_home, cassette=cassette, ollama_down_for=DOWN_REQUESTS,
    )
    frames = sorted(p.name for p in run.fixture.screenshots_dir.iterdir())
    focus_rows = sum(_focus_statuses(run.fixture).values())
    signal = temp_home.data_dir / "focus-session.json"

    outage_ticks = 0
    while run.tape.fault_state()["down"] > 0:
        assert outage_ticks < MAX_OUTAGE_TICKS, run.tape.fault_state()
        tick = run.worker.run_tick()
        outage_ticks += 1
        assert tick["focus_sops"] == 0
        assert sorted(p.name for p in run.fixture.screenshots_dir.iterdir()) == frames
        assert _focus_statuses(run.fixture) == {"pending": focus_rows}
        assert json.loads(signal.read_text(encoding="utf-8"))["status"] == "stopped"
        assert run.procedures() == []
        clock.advance(seconds=61)
    assert outage_ticks > 1

    proc = tick_until_skill(run, clock)
    assert _focus_statuses(run.fixture).get("pending", 0) == 0
    assert_matches_golden("focus_dailynews.claude-skill.md", render_packet(proc, target="claude-skill"))


def _pending_focus_screenshots(fixture) -> list[str]:
    conn = sqlite3.connect(str(fixture.db_path))
    try:
        return [
            json.loads(meta)["screenshot_path"].rsplit("/", 1)[-1]
            for (meta,) in conn.execute(
                "SELECT metadata_json FROM events WHERE annotation_status = 'pending' "
                "AND json_extract(metadata_json, '$.focus_session_id') IS NOT NULL "
                "AND json_extract(metadata_json, '$.screenshot_path') IS NOT NULL"
            )
        ]
    finally:
        conn.close()


def _hold_first_annotation(run, seconds: float) -> None:
    url = f"{run.tape.host}/__cassette/slow?seconds={seconds}&n=1&path=/api/generate"
    with urllib.request.urlopen(url) as reply:
        assert json.load(reply)["slow_remaining"] == 1


def _served_like(run, request: dict) -> list[dict]:
    return [
        e for e in run.tape.served
        if e["request"].get("prompt") == request.get("prompt")
        and e["request"].get("system") == request.get("system")
    ]


def test_default_annotation_timeout_outlasts_a_cold_model_load() -> None:
    assert VlmConfig().annotation_timeout_seconds >= 2 * COLD_LOAD_SECONDS


@needs_focus_sop_recording
def test_ollama_slow_first_annotation_still_succeeds(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    """The first annotation is answered after 2 s: the annotation timeout
    (4 s here) outlasts it, nothing is deferred, and the Skill is J-focus's."""
    run = start_focus_run(
        events_db=events_db, worker=worker, clock=clock,
        temp_home=temp_home, cassette=cassette,
        vlm={"annotation_timeout_seconds": SLOW_TIMEOUT_SECONDS},
    )
    _hold_first_annotation(run, SLOW_SECONDS)

    proc = tick_until_skill(run, clock)

    [(path, request)] = run.tape.delayed
    assert path == "/api/generate"
    profile = get_profile(request["model"])
    assert request.get("system") == annotation_system_prompt(profile.ann_system)
    # Answered once: the worker waited for it instead of timing out and
    # asking again on a later tick.
    assert len(_served_like(run, request)) == 1
    assert _focus_statuses(run.fixture).get("pending", 0) == 0
    assert_matches_golden("focus_dailynews.claude-skill.md", render_packet(proc, target="claude-skill"))


@needs_focus_sop_recording
def test_ollama_slower_than_the_timeout_is_deferred_then_asked_again(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    """Control: held past a 1 s timeout, the first annotation times out, the
    session waits (nothing deleted), the call is asked again after the
    backoff, and the Skill is still J-focus's."""
    run = start_focus_run(
        events_db=events_db, worker=worker, clock=clock,
        temp_home=temp_home, cassette=cassette,
        vlm={"annotation_timeout_seconds": SHORT_TIMEOUT_SECONDS},
    )
    _hold_first_annotation(run, SLOW_SECONDS)

    first = run.worker.run_tick()
    assert first["focus_sops"] == 0
    # The frames still to annotate keep their JPEGs (the keyframe pass may
    # already have dropped near-duplicates, as it does on every run).
    pending = _pending_focus_screenshots(run.fixture)
    assert pending
    for name in pending:
        assert (run.fixture.screenshots_dir / name).exists(), name
    # The cassette notes the held request once it stops holding it.
    deadline = time.monotonic() + 10 * SLOW_SECONDS
    while not run.tape.delayed and time.monotonic() < deadline:
        time.sleep(0.05)
    [(path, request)] = run.tape.delayed
    assert path == "/api/generate"

    clock.advance(seconds=61)
    proc = tick_until_skill(run, clock)
    assert len(_served_like(run, request)) == 2
    assert _focus_statuses(run.fixture).get("pending", 0) == 0
    assert_matches_golden("focus_dailynews.claude-skill.md", render_packet(proc, target="claude-skill"))
