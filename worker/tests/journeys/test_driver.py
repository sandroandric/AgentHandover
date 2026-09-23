"""The journey driver, clock and cassette wiring.

The full journeys (J-focus, J-passive, J-ollama-down, ...) belong to WP-T4
and need recorded cassettes; these tests pin the harness they run on.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from agenthandover_worker import bundle_compiler, staleness_detector, trigger_spool
from agenthandover_worker.config import load_config
from journeys.driver import Clock


def _statuses(db_path) -> dict:
    conn = sqlite3.connect(str(db_path))
    try:
        return dict(conn.execute(
            "SELECT annotation_status, COUNT(*) FROM events GROUP BY 1"
        ).fetchall())
    finally:
        conn.close()


def test_ticks_with_ollama_unreachable_keep_the_focus_session(
    events_db, worker, clock, temp_home,
) -> None:
    """Default config points Ollama at a closed port: nothing is annotated,
    no frame is deleted, the stopped session stays queued for later."""
    fixture = events_db("focus_dailynews", rebase_to=clock.now())
    signal = fixture.write_focus_signal(temp_home.data_dir)
    frames = fixture.manifest["row_counts"]["with_screenshot"]

    journey = worker(fixture)
    assert journey.v2_schema_ok
    assert journey.vector_kb is None  # health check could not reach Ollama

    for _ in range(3):
        result = journey.run_tick()
        assert result["focus_sops"] == 0
        assert result["annotations"]["annotated"] == 0
        assert result["passive_sops"] is None  # focus session pending
        clock.advance(seconds=2)

    assert _statuses(fixture.db_path) == {"pending": fixture.manifest["row_counts"]["events"]}
    assert len(list(fixture.screenshots_dir.iterdir())) == frames
    assert json.loads(signal.read_text(encoding="utf-8"))["status"] == "stopped"
    # run_pipeline consumed the first 100 unprocessed rows per tick.
    conn = sqlite3.connect(str(fixture.db_path))
    try:
        processed = conn.execute("SELECT COUNT(*) FROM events WHERE processed = 1").fetchone()[0]
    finally:
        conn.close()
    assert processed == 300
    assert (temp_home.data_dir / "sops-index.json").is_file()


def test_passive_discovery_follows_the_clock(events_db, worker, clock) -> None:
    fixture = events_db("focus_dailynews", rebase_to=clock.now())
    journey = worker(fixture)

    # The timer starts with the worker, as main() seeds it at loop entry.
    assert journey.run_tick()["passive_sops"] is None
    clock.advance(hours=1, minutes=59)
    assert journey.run_tick()["passive_sops"] is None
    clock.advance(minutes=1)
    assert journey.run_tick()["passive_sops"] is not None
    assert journey.run_tick()["passive_sops"] is None


def test_tick_drains_the_trigger_spool_through_mains_handlers(
    events_db, worker, clock, temp_home,
) -> None:
    fixture = events_db("focus_dailynews", rebase_to=clock.now())
    journey = worker(fixture)
    trigger_spool.write_trigger(temp_home.data_dir, "search-query", {"query": ""})

    assert journey.run_tick()["triggers_consumed"] == 1
    assert trigger_spool.pending(temp_home.data_dir) == []
    result = json.loads(
        (temp_home.data_dir / "search-query-result.json").read_text(encoding="utf-8")
    )
    assert result == {"error": "Empty search query", "results": []}
    assert journey.run_tick()["triggers_consumed"] == 0


def test_clock_is_the_default_now_for_freshness(clock: Clock) -> None:
    proc = {"staleness": {"last_observed": clock.now().isoformat()}}
    fresh = staleness_detector.procedure_freshness(proc)
    clock.advance(days=60)
    aged = staleness_detector.procedure_freshness(proc)
    assert aged < fresh
    assert bundle_compiler.procedure_freshness(proc) == aged
    explicit = staleness_detector.procedure_freshness(proc, now=clock.now() - timedelta(days=60))
    assert explicit == fresh


def test_clock_is_aware_and_only_moves_forward() -> None:
    with pytest.raises(ValueError):
        Clock(datetime(2026, 1, 1))
    clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(ValueError):
        clock.advance(seconds=-1)
    assert clock.advance(timedelta(hours=1)) == datetime(2026, 1, 1, 1, tzinfo=timezone.utc)


def test_cassette_fixture_points_the_worker_at_the_cassette(cassette, temp_home) -> None:
    before = load_config().vlm.ollama_host
    tape = cassette("cassette_selftest")
    cfg = load_config()
    assert cfg.path == temp_home.config_file
    assert cfg.vlm.ollama_host == tape.host != before
    assert os.environ["OLLAMA_HOST"] == tape.host
    assert cfg.knowledge.query_api_enabled is False
