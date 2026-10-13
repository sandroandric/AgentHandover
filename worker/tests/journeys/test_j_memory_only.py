"""J-memory-only: with ``[skills] enabled = false`` the worker's own tick
fills Memory and calls no generative model.

The J-memory frames as the daemon writes them, none annotated (in Memory
only nothing ever annotates them), on the same cassette.  The worker
ticks over them past the 2-hour passive-discovery interval and across a
day boundary with the once-per-day pass allowed; then:

* the five moments J-memory keeps are stored (the invoice's two frames in
  one row) and the 1Password and Dock frames are not, without any
  annotation: the writer applies the privacy zones and the system-UI
  rule itself;
* every JPEG is gone (embedded, then deleted by the writer);
* every request Ollama received was ``/api/embed``: no screen
  description, no Skill, no summary, no preload;
* the day's upkeep ran (the daily state moved to the new day) without a
  generation plan;
* the worker status says Skills are off;
* search finds each moment.
"""

from __future__ import annotations

import sqlite3
from datetime import timedelta

import pytest

from agenthandover_worker import main as wm
from agenthandover_worker.memory import MemoryIndex, query_embedder
from agenthandover_worker.model_profiles import MEMORY_MODEL
from agenthandover_worker.ollama import switch_off_generation
from journeys import test_j_memory as jm

pytestmark = jm.pytestmark


@pytest.fixture(autouse=True)
def _generation_back_on():
    """The worker switches generation off process-wide; later tests in this
    process build their own worker or call Ollama directly."""
    yield
    switch_off_generation(None)


def _unannotated(monkeypatch) -> None:
    """J-memory's frames with every annotation removed."""
    frames = [
        (fid, seconds, app, title, image, "pending", "")
        for fid, seconds, app, title, image, _status, _doing in jm.FRAMES
    ]
    monkeypatch.setattr(jm, "FRAMES", frames)


def test_j_memory_only(temp_home, cassette, worker, clock, monkeypatch) -> None:
    tape = cassette(jm.CASSETTE)
    temp_home.configure(
        ollama_host=tape.host,
        extra="[memory]\nenabled = true\n\n[skills]\nenabled = false\n",
    )
    _unannotated(monkeypatch)
    start = clock.now() - timedelta(minutes=10)
    db_path = jm._load_frames(temp_home.data_dir, start)
    journey = worker(db_path)
    ctx = journey.ctx
    # No component that calls a generative model was built.
    for name in (
        "llm_reasoner", "behavioral_synthesizer", "activity_classifier", "sop_enhancer",
        "scene_annotator", "frame_differ", "sop_generator", "focus_processor", "task_segmenter",
    ):
        assert getattr(ctx, name) is None, name
    assert not ctx.model_scheduler.split
    ctx.run_daily_jobs = True
    first_day = journey.ctx.local_now().date().isoformat()

    # Past passive discovery's 2-hour interval, then into the next day.
    for _ in range(3):
        journey.run_tick()
        clock.advance(seconds=30)
    clock.advance(hours=3)
    journey.run_tick()
    clock.advance(hours=24)
    journey.run_tick()
    next_day = journey.ctx.local_now().date().isoformat()

    assert {entry["path"] for entry in tape.served} == {"/api/embed"}
    assert not tape.misses

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM memory_frames ORDER BY first_ts")]
        statuses = dict(conn.execute("SELECT id, annotation_status FROM events"))
        daily = conn.execute(
            "SELECT value FROM worker_state WHERE key = ?", (wm.DAILY_JOBS_STATE_KEY,),
        ).fetchone()
    finally:
        conn.close()

    assert [r["id"] for r in rows] == sorted(jm.KEPT)
    assert rows[0]["frames"] == 2
    assert not any(r["app"] in ("1Password", "Dock") for r in rows)
    assert "completed" not in statuses.values(), "nothing was annotated"
    shots = temp_home.data_dir / "screenshots"
    assert sorted(p.name for p in shots.iterdir()) == [], "every JPEG deleted"
    assert first_day != next_day and daily is not None and daily[0] == next_day

    # (missing_models and vlm_available are set by main() at startup, which
    # the driver does not run: test_memory_only covers them.)
    assert journey.status()["skills_enabled"] is False

    index = MemoryIndex(
        db_path, model=MEMORY_MODEL, dim=jm.DIM,
        embed_query=query_embedder(tape.host, MEMORY_MODEL, jm.DIM),
    )
    try:
        for query, app in jm.QUERIES:
            hits = index.search(query, limit=5)
            assert hits and hits[0]["app"] == app, (query, [(h["app"], h["score"]) for h in hits])
    finally:
        index.close()
