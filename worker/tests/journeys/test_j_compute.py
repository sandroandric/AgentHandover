"""J-compute: one volatile hour of passive capture costs a bounded amount
of model time.

``passive_volatile_hour`` is 1,200 frames captured at the daemon's ceiling
while the user switches context every 20-60 s.  The worker ticks over the
hour (15 s of clock per tick) on the recorded cassette.  Once more frames
have been sent to the model than the bound allows, the run stops: the
bound is already broken, and the cassette holds only the calls up to that
point.

The hourly gate (``[vlm] annotation_max_frames_per_hour`` = 60 and
``annotation_max_compute_minutes_per_hour`` = 10, WP-F2) holds the hour to
60 frames sent to the model and 600 model seconds; the cache key folds the
volatile text ("updated 12s ago", "Ln 312, Col 1", counters) so a screen
seen before is answered without the model.  Since WP-F6 a cache hit is
free under the frame cap (it makes no model call; the diff and
classification calls after it are charged to the compute cap), so the
bound is on model calls, not on completed rows: the hour now annotates
every frame the cache answers.  The daily cap
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

import os

import pytest

from journeys.cassette import CASSETTE_DIR, load_entries


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


def _model_annotated(run: ComputeRun) -> int:
    """Frames the annotator sent to the model (cache hits excluded)."""
    return run.worker.scene_annotator.stats["annotated"]


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
        if stop is None and _model_annotated(run) > MAX_ANNOTATED:
            break
        if stop is not None and stop(run):
            break
        clock.advance(seconds=TICK_SECONDS)
    return run


# Requests the hour makes since WP-F6 (cache hits no longer close the
# hourly frame cap, so the run reaches frames the WP-F2 recording never
# did) that ``passive_volatile_hour.jsonl`` does not hold.  These are the
# misses a replay shows; the first is the annotation of the first frame
# past the old 60-row stop, which fails (HTTP 500, the row deferred)
# every tick after it, so recording it reveals the calls behind it.
# Re-record incrementally with AH_RECORD=1 on the test Mac.
F6_MISSING_KEYS = (
    "44038ed6cbebbdbc5b135e9cd62321be2612cfa570b3e187104468a54823a698",
    "a27b4a24b1328f9518b348ea8b93cd1db0beb9b57199f0c6c549c4c5718e6ac8",
    "bb64064041e26a48809677a1efd130dcc4e922ccfe527cb4efcbb3514e3cd0ca",
    "c8fa94403d4e4143084dfc938ebd1dabb5dfa42e2e6391c52aee23b4a8b5c00c",
    "12d85b60339a2d9b27bdf96d029858d198c380c8291087232515809d9a0dcd74",
    "9b14fb80a511035dd08abbd71ce4c66c422323d37f139070dffe6f6c6d66e8a4",
    "cd6e6e2e588997f9b6d7f3e1fe0fc526865084898a253ec846b3286b9cadaab2",
    "02861cd371489660707fb668a691987705083817cae02c5d2cd43e7491740da0",
)
_HOUR_ENTRIES = load_entries(CASSETTE_DIR / "passive_volatile_hour.jsonl")
needs_f6_recording = pytest.mark.skipif(
    any(key not in _HOUR_ENTRIES for key in F6_MISSING_KEYS)
    and os.environ.get("AH_RECORD") != "1",
    reason=(
        "cassettes/passive_volatile_hour.jsonl lacks the requests the hour makes "
        "once cache hits are free (WP-F6): "
        + ", ".join(k for k in F6_MISSING_KEYS if k not in _HOUR_ENTRIES)
        + " (and the calls they unblock); re-record with AH_RECORD=1 on the test Mac"
    ),
)


@needs_f6_recording
def test_compute_volatile_hour_annotates_at_most_60_frames(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    """The hour sends at most 60 frames to the model; cache hits are free
    and published as hits.  Whatever is still pending keeps its
    screenshot, and when the gate closed, the budget says why."""
    run = _run(events_db, worker, clock, temp_home, cassette)
    assert run.ticks == HOUR_TICKS or not run.statuses.get("pending"), (
        run.ticks, run.statuses,
    )
    stats = run.worker.scene_annotator.stats
    assert stats["annotated"] <= MAX_ANNOTATED, (stats, run.statuses)
    status = run.worker.status()
    published = status["annotation_budget"]
    assert published["hour_frames"] <= MAX_ANNOTATED, status
    # The run is one hour of fake clock: every hit is still in the hour.
    assert published["hour_cache_hits"] == stats["cache_hits"], (published, stats)
    # Repeat screens are answered by the cache, not the model.
    assert stats["cache_hits"] > stats["annotated"], stats
    if run.results[-1]["annotations"]["budget_exhausted"]:
        assert status["annotation_paused_reason"] == "annotation_hourly_budget_exhausted"
    for shot in _pending_screenshots(run.fixture.db_path):
        assert (run.fixture.screenshots_dir / shot.rsplit("/", 1)[-1]).exists(), shot


# The activity classifier's call the volatile hour repeats 28 times, as
# the worker makes it since WP-C9 (annotation model, num_ctx 4096,
# num_predict 16).  Its only recording (WP-C9's) includes a 146 s model
# load, which the replay publishes on every one of the 28 calls; recorded
# with the model resident, it costs under a second.
COLD_CLASSIFIER_KEY = "0901961c0f16ce7a1755d49d8727b724889b6cc98cbc14f31331fd5ef707d98f"
_COLD_LOAD_SECONDS = 10


def _recorded_load_seconds(key: str) -> float:
    entry = _HOUR_ENTRIES.get(key)
    return (entry["response"].get("load_duration") or 0) / 1e9 if entry else 0.0


@pytest.mark.skipif(
    _recorded_load_seconds(COLD_CLASSIFIER_KEY) >= _COLD_LOAD_SECONDS
    and os.environ.get("AH_RECORD") != "1",
    reason=(
        f"cassettes/passive_volatile_hour.jsonl entry {COLD_CLASSIFIER_KEY} was "
        "recorded with a cold model load (146 s load_duration, replayed on each "
        "of its 28 calls): delete that entry and re-record it with AH_RECORD=1 "
        "on the test Mac with qwen3.5:4b resident"
    ),
)
@needs_f6_recording
def test_compute_volatile_hour_spends_at_most_600_model_seconds(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    run = _run(events_db, worker, clock, temp_home, cassette)
    # The hour's work was done (not cut short at the 60-frame bound).
    assert run.ticks == HOUR_TICKS, (run.ticks, run.statuses)
    last_hour = run.worker.status()["compute"]["last_hour"]
    assert last_hour["model_seconds"] <= MAX_MODEL_SECONDS, last_hour


@needs_f6_recording
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


@needs_f6_recording
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


@needs_f6_recording
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
