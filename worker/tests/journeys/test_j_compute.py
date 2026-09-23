"""J-compute: one volatile hour of passive capture costs a bounded amount
of model time.

``passive_volatile_hour`` is 1,200 frames captured at the daemon's ceiling
while the user switches context every 20-60 s.  The worker ticks over the
hour (15 s of clock per tick) on the recorded cassette.  Once more frames
have been annotated than the bound allows, the run stops: the bound is
already broken, and the cassette holds only the calls up to that point.

The daily cap (``[vlm] annotation_max_compute_minutes_per_day``, WP-C5) is
checked with a 5-minute cap: passive annotation stops at it, the rows stay
pending with their screenshots.  ``worker-status.json`` (the heartbeat that
publishes the pause reason) is not driven before WP-S6; the journey reads
the budget the loop charges instead.
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
CAP_MINUTES = 5


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


_UNGATED = (
    "passive annotation has no hourly gate: _process_annotations takes 5 "
    "pending frames every tick (batch_size=5, main.py:233) and only the daily "
    "AnnotationBudget (1,500 frames / 60 min) bounds it; the annotation cache "
    "hits only frames whose screen text did not change (MAX_HAMMING = 3, "
    "annotation_cache.py:28, while an editor's 'Ln N' or a page's 'updated Ns "
    "ago' moves the 64-bit simhash of a short screen by 5-10 bits), so the "
    "volatile hour passes 60 annotated frames after 13 ticks (38 model calls, "
    "23 cache hits)"
)


@pytest.mark.xfail(strict=True, reason=_UNGATED)
def test_compute_volatile_hour_annotates_at_most_60_frames(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    run = _run(events_db, worker, clock, temp_home, cassette)
    assert run.statuses.get("completed", 0) <= MAX_ANNOTATED, (run.ticks, run.statuses)


@pytest.mark.xfail(strict=True, reason=(
    "cannot hold while the 60-frame bound is broken (the run stops there, "
    "3 minutes into the hour, having spent ~370 cassette seconds): " + _UNGATED
))
def test_compute_volatile_hour_spends_at_most_600_model_seconds(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    run = _run(events_db, worker, clock, temp_home, cassette)
    # The hour's work was done (not cut short at the 60-frame bound).
    assert run.statuses.get("completed", 0) <= MAX_ANNOTATED, (run.ticks, run.statuses)
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
    """At a 5-minute daily compute cap passive annotation stops: later ticks
    annotate nothing, the rest stays pending and keeps its screenshots."""
    run = _run(
        events_db, worker, clock, temp_home, cassette,
        vlm={"annotation_max_compute_minutes_per_day": CAP_MINUTES}, stop=_exhausted,
    )
    budget = run.worker.annotation_budget
    assert _exhausted(run), (run.ticks, budget.status(), run.statuses)
    assert budget.compute_seconds >= CAP_MINUTES * 60
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
