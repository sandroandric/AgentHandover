"""J-ollama-down: J-focus while Ollama is down for its first 40 requests.

The worker starts against a cassette that answers 503; while it does, the
focus session waits: no frame is annotated, no JPEG deleted, the signal
stays.  The annotator backs off on the journey clock (60 s), so each tick
moves the clock past the backoff and spends one probe on the outage.  Once
the cassette answers again the session completes, and the Skill is the one
J-focus writes (the same golden packet).
"""

from __future__ import annotations

import json
import sqlite3

from agenthandover_worker.skill_render import render_packet
from journeys.focus_run import start_focus_run, tick_until_skill
from journeys.packet import assert_matches_golden

DOWN_REQUESTS = 40
MAX_OUTAGE_TICKS = 3 * DOWN_REQUESTS


def _focus_statuses(fixture) -> dict:
    conn = sqlite3.connect(str(fixture.db_path))
    try:
        return dict(conn.execute(
            "SELECT annotation_status, COUNT(*) FROM events "
            "WHERE json_extract(metadata_json, '$.focus_session_id') IS NOT NULL GROUP BY 1"
        ).fetchall())
    finally:
        conn.close()


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
