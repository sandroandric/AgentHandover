"""J-compute: one volatile hour of passive capture costs a bounded amount
of model time.

``passive_volatile_hour`` is 1,200 frames captured at the daemon's ceiling
while the user switches context every 20-60 s.  The worker ticks over the
hour (15 s of clock per tick) on the recorded cassette.  Once more frames
have been annotated than the bound allows, the run stops: the bound is
already broken, and the cassette holds only the calls up to that point.

The hourly gate (``[vlm] annotation_max_frames_per_hour`` = 60 and
``annotation_max_compute_minutes_per_hour`` = 10, WP-F2) holds the hour to
60 annotated frames and 600 model seconds; the cache key folds the
volatile text ("updated 12s ago", "Ln 312, Col 1", counters) so a screen
seen before is answered without the model.  The daily cap
(``[vlm] annotation_max_compute_minutes_per_day``, WP-C5) is checked with a
1-minute cap and the hourly gate opened wide: passive annotation stops at
it, the rows stay pending with their screenshots.

The model-time bounds and pause reasons are read from what the worker
publishes (WP-C1): ``worker-status.json``'s ``compute.last_hour`` (model
seconds per call type, the per-frame wall-time histogram), its
``annotation_budget`` and ``annotation_paused_reason``, written by the
driver's heartbeat each tick.  That the published seconds are the
cassette's own ``total_duration`` sum is pinned by
``test_compute_status_accounts_for_every_model_call``.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field

import pytest


MAX_ANNOTATED = 60
MAX_MODEL_SECONDS = 600
FTS_REFRESH_BUDGET_S = 0.100
TICK_SECONDS = 15
HOUR_TICKS = 3600 // TICK_SECONDS
CAP_MINUTES = 1
# The hourly gate opened wide, so the daily cap is what stops the run.
NO_HOURLY_GATE = {
    "annotation_max_frames_per_hour": 100_000,
    "annotation_max_compute_minutes_per_hour": 10_000,
}


@dataclass
class ComputeRun:
    fixture: object
    worker: object
    tape: object
    ticks: int = 0
    queue_writes: int = 0
    statuses: dict = field(default_factory=dict)
    results: list = field(default_factory=list)


def _statuses(db_path) -> dict:
    conn = sqlite3.connect(str(db_path))
    try:
        return dict(conn.execute(
            "SELECT annotation_status, COUNT(*) FROM events "
            "WHERE json_extract(metadata_json, '$.screenshot_path') IS NOT NULL GROUP BY 1"
        ).fetchall())
    finally:
        conn.close()


def _run(events_db, worker, clock, temp_home, cassette, *, vlm=None, stop=None) -> ComputeRun:
    """Tick over the hour until nothing is pending and, without ``stop``,
    until the 60-frame bound is broken; with ``stop``, until ``stop(run)``."""
    tape = cassette("passive_volatile_hour", vlm=vlm)
    fixture = events_db("passive_volatile_hour", rebase_to=clock.now())
    run = ComputeRun(fixture=fixture, worker=worker(fixture), tape=tape)
    queue = temp_home.data_dir / "curation-queue.json"
    last_seen = None
    for _ in range(HOUR_TICKS):
        run.results.append(run.worker.run_tick())
        run.ticks += 1
        if queue.exists():
            stat = queue.stat()
            seen = (stat.st_ino, stat.st_mtime_ns)
            run.queue_writes += seen != last_seen
            last_seen = seen
        run.statuses = _statuses(fixture.db_path)
        if not run.statuses.get("pending"):
            break
        if stop is None and run.statuses.get("completed", 0) > MAX_ANNOTATED:
            break
        if stop is not None and stop(run):
            break
        clock.advance(seconds=TICK_SECONDS)
    return run


def test_compute_volatile_hour_annotates_at_most_60_frames(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    """The hourly gate closes at 60 annotated frames: the rest of the hour
    stays pending with its screenshots, and the budget says why."""
    run = _run(events_db, worker, clock, temp_home, cassette)
    assert run.ticks == HOUR_TICKS, (run.ticks, run.statuses)
    assert run.statuses.get("completed", 0) <= MAX_ANNOTATED, (run.ticks, run.statuses)
    budget = run.worker.annotation_budget
    assert budget.pause_reason() == "annotation_hourly_budget_exhausted", budget.status()
    status = run.worker.status()
    assert status["annotation_paused_reason"] == "annotation_hourly_budget_exhausted"
    assert status["annotation_budget"]["hour_frames"] <= MAX_ANNOTATED, status
    assert run.results[-1]["annotations"]["budget_exhausted"]
    assert run.statuses.get("pending", 0) > 0
    for shot in _pending_screenshots(run.fixture.db_path):
        assert (run.fixture.screenshots_dir / shot.rsplit("/", 1)[-1]).exists(), shot
    # Repeat screens are answered by the cache, not the model.
    stats = run.worker.scene_annotator.stats
    assert stats["cache_hits"] > stats["annotated"], stats


def test_compute_volatile_hour_spends_at_most_600_model_seconds(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    run = _run(events_db, worker, clock, temp_home, cassette)
    # The hour's work was done (not cut short at the 60-frame bound).
    assert run.ticks == HOUR_TICKS, (run.ticks, run.statuses)
    last_hour = run.worker.status()["compute"]["last_hour"]
    assert last_hour["model_seconds"] <= MAX_MODEL_SECONDS, last_hour


def test_compute_writes_no_artifact_and_refreshes_fts_quickly(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    run = _run(events_db, worker, clock, temp_home, cassette)
    assert run.statuses.get("completed", 0) > 0
    assert not list(temp_home.data_dir.rglob("*.bin"))

    searcher = run.worker.activity_searcher
    started = time.perf_counter()
    searcher.refresh_index()
    elapsed = time.perf_counter() - started
    assert elapsed < FTS_REFRESH_BUDGET_S, f"events_fts refresh took {elapsed * 1000:.0f} ms"


def test_compute_status_accounts_for_every_model_call(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    """WP-C1: ``worker-status.json`` accounts for every model call the
    cassette answered, per call type, with Ollama's own seconds, and puts
    each frame the model answered in the wall-time histogram once."""
    run = _run(events_db, worker, clock, temp_home, cassette)
    status = run.worker.status()
    compute = status["compute"]
    last_hour = compute["last_hour"]
    # The hour ran on a fake clock in a few real seconds: all of it is
    # within the last hour.
    assert last_hour["model_seconds_by_call_type"] == compute["model_seconds_by_call_type"]

    served = [e for e in run.tape.served if e["path"] in ("/api/generate", "/api/embed")]
    by_type = last_hour["model_calls_by_call_type"]
    assert sum(by_type.values()) == len(served), (by_type, len(served))
    assert by_type.get("embed", 0) == sum(e["path"] == "/api/embed" for e in served)
    assert last_hour["model_seconds"] == pytest.approx(run.tape.model_seconds(), abs=0.01)
    # Passive capture's cost is the per-frame steps: annotation, activity
    # classification (reasoning) and the frame diff.  Under the hourly gate
    # the cache answers most frames, so the diff, not annotation, is the
    # largest.  The SOP miner has not run (its 2-hour timer is not due
    # within the hour): no sop or synthesis seconds.
    seconds = last_hour["model_seconds_by_call_type"]
    assert seconds["annotation"] > 0, seconds
    assert set(seconds) <= {"annotation", "reasoning", "diff", "embed", "preload"}, seconds
    frames = sum(last_hour["annotation_wall_ms_histogram"].values())
    assert frames == by_type["annotation"] - last_hour["annotation_retries"]
    assert status["v2_annotations_today"] == run.statuses.get("completed", 0)


def test_compute_curation_queue_is_not_rewritten_every_tick(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    """With no app request in the hour, nothing changes the curation queue:
    the loop writes it at most once (WP-C5: only when a trigger changed it)."""
    run = _run(events_db, worker, clock, temp_home, cassette)
    assert run.ticks > 1
    assert run.queue_writes <= 1, (run.queue_writes, run.ticks)


def _exhausted(run: ComputeRun) -> bool:
    return bool((run.results[-1].get("annotations") or {}).get("budget_exhausted"))


def test_compute_daily_cap_stops_passive_annotation(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    """At a 1-minute daily compute cap passive annotation stops: later ticks
    annotate nothing, the rest stays pending and keeps its screenshots."""
    run = _run(
        events_db, worker, clock, temp_home, cassette,
        vlm={"annotation_max_compute_minutes_per_day": CAP_MINUTES, **NO_HOURLY_GATE},
        stop=_exhausted,
    )
    budget = run.worker.annotation_budget
    status = run.worker.status()
    assert _exhausted(run), (run.ticks, budget.status(), run.statuses)
    assert budget.pause_reason() == "annotation_budget_exhausted", budget.status()
    assert budget.compute_seconds >= CAP_MINUTES * 60
    assert status["annotation_paused_reason"] == "annotation_budget_exhausted"
    assert status["annotation_budget"]["compute_minutes"] >= CAP_MINUTES
    assert status["annotation_budget"]["max_compute_minutes"] == CAP_MINUTES
    completed = run.statuses.get("completed", 0)

    for _ in range(2):
        clock.advance(seconds=TICK_SECONDS)
        tick = run.worker.run_tick()
        assert tick["annotations"]["annotated"] == 0
        assert tick["annotations"]["budget_exhausted"]
    statuses = _statuses(run.fixture.db_path)
    assert statuses.get("completed", 0) == completed
    assert statuses.get("pending", 0) > 0
    for shot in _pending_screenshots(run.fixture.db_path):
        assert (run.fixture.screenshots_dir / shot.rsplit("/", 1)[-1]).exists(), shot


def _pending_screenshots(db_path) -> list[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return [
            json.loads(meta)["screenshot_path"] for (meta,) in conn.execute(
                "SELECT metadata_json FROM events WHERE annotation_status = 'pending' "
                "AND json_extract(metadata_json, '$.screenshot_path') IS NOT NULL"
            )
        ]
    finally:
        conn.close()
