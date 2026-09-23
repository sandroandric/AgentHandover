"""J-compute: one volatile hour of passive capture costs a bounded amount
of model time.

``passive_volatile_hour`` is 1,200 frames captured at the daemon's ceiling
while the user switches context every 20-60 s.  The worker ticks over the
hour (15 s of clock per tick) on the recorded cassette.  Once more frames
have been annotated than the budget allows, the run stops: the bound is
already broken, and the cassette holds only the calls up to that point.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field

import pytest

from journeys.conftest import JOURNEY_MODELS

MAX_ANNOTATED = 60
MAX_MODEL_SECONDS = 600
FTS_REFRESH_BUDGET_S = 0.100
TICK_SECONDS = 15
HOUR_TICKS = 3600 // TICK_SECONDS


@dataclass
class ComputeRun:
    fixture: object
    worker: object
    tape: object
    ticks: int = 0
    queue_writes: int = 0
    statuses: dict = field(default_factory=dict)


def _statuses(db_path) -> dict:
    conn = sqlite3.connect(str(db_path))
    try:
        return dict(conn.execute(
            "SELECT annotation_status, COUNT(*) FROM events "
            "WHERE json_extract(metadata_json, '$.screenshot_path') IS NOT NULL GROUP BY 1"
        ).fetchall())
    finally:
        conn.close()


def _run(events_db, worker, clock, temp_home, cassette) -> ComputeRun:
    temp_home.configure(vlm=JOURNEY_MODELS)
    tape = cassette("passive_volatile_hour")
    fixture = events_db("passive_volatile_hour", rebase_to=clock.now())
    run = ComputeRun(fixture=fixture, worker=worker(fixture), tape=tape)
    queue = temp_home.data_dir / "curation-queue.json"
    last_seen = None
    for _ in range(HOUR_TICKS):
        run.worker.run_tick()
        run.ticks += 1
        if queue.exists():
            stat = queue.stat()
            seen = (stat.st_ino, stat.st_mtime_ns)
            run.queue_writes += seen != last_seen
            last_seen = seen
        run.statuses = _statuses(fixture.db_path)
        if run.statuses.get("completed", 0) > MAX_ANNOTATED or not run.statuses.get("pending"):
            break
        clock.advance(seconds=TICK_SECONDS)
    return run


@pytest.mark.xfail(strict=True, reason=(
    "WP-C4/WP-C5: nothing gates the model per frame beyond the stale-skip "
    "(_StaleTracker, scene_annotator.py:604-643, which resets on every title "
    "change): _process_annotations (main.py:221) sends every changed frame "
    "to the VLM, with no annotation cache or keyframe gate, so a volatile "
    "hour annotates hundreds of frames"
))
def test_compute_volatile_hour_annotates_at_most_60_frames(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    run = _run(events_db, worker, clock, temp_home, cassette)
    assert run.statuses.get("completed", 0) <= MAX_ANNOTATED, (run.ticks, run.statuses)


@pytest.mark.xfail(strict=True, reason=(
    "WP-C3b/WP-C5: each passive annotation on qwen3.5:4b costs ~16 cassette "
    "seconds (image prompt, ~430 output tokens) plus a frame diff and an "
    "activity-classifier call, so 60 annotated frames alone exceed 600 s"
))
def test_compute_volatile_hour_spends_at_most_600_model_seconds(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    run = _run(events_db, worker, clock, temp_home, cassette)
    assert run.tape.model_seconds() <= MAX_MODEL_SECONDS, run.tape.model_seconds()


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


@pytest.mark.xfail(strict=True, reason=(
    "WP-C5: the loop rewrites curation-queue.json on every tick: "
    "_write_curation_queue(procedure_curator, force=True) at main.py:4808 "
    "runs unconditionally after the trigger drain (the driver mirrors it), "
    "not only when a trigger changed the queue"
))
def test_compute_curation_queue_is_written_once(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    run = _run(events_db, worker, clock, temp_home, cassette)
    assert run.ticks > 1
    assert run.queue_writes == 1, (run.queue_writes, run.ticks)
