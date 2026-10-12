"""J-memory: passive frames go into Memory through the worker's own tick,
and the product's search path finds them.

Eight frames as the daemon writes them (JPEGs in the data directory's
``screenshots/``, rows in ``events.db`` built by the daemon's migrations),
already annotated and diffed as the passive pipeline leaves them, except
a 1Password window still pending (the privacy zone blocks it) and the
Dock (system UI, skipped by the annotator).  Two consecutive frames of
the Acme invoice show the same screen.  With ``[memory] enabled = true``
the worker ticks over them on the recorded cassette (EmbeddingGemma 2 at
512 dimensions); then:

* five moments are stored, the invoice's two frames in one row;
* neither the 1Password nor the Dock frame is in Memory;
* every JPEG is gone (embedded, then deleted; the blocked one by the
  privacy path);
* ``MemoryIndex`` (query prefix, numpy scoring, moments) puts the right
  moment first for each query.

The frames are synthetic (``fixtures/memory_frames``, drawn by a script:
coloured window layouts, no text), so the window titles carry most of the
signal.  The 52-query benchmark over real screenshots (``~/emb-bench``)
runs on the test Mac.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from agenthandover_worker.memory import MemoryIndex, embed_text, query_embedder
from agenthandover_worker.model_profiles import EMBEDDING_PROFILES, MEMORY_MODEL
from daemon_db import create_daemon_db
from journeys.cassette import CASSETTE_DIR, load_entries, request_key

FRAMES_DIR = Path(__file__).parent / "fixtures" / "memory_frames"
CASSETTE = "memory"
DIM = 512

ANNOTATED = "completed"
# (id, seconds after the start, app, window title, image, annotation status,
#  what the user was doing)
FRAMES = [
    ("mem-01", 0, "Google Chrome", "Invoice INV-2041 - Acme Billing", "invoice",
     ANNOTATED, "Reviewing the Acme invoice INV-2041 before paying it"),
    ("mem-02", 20, "Google Chrome", "Invoice INV-2041 - Acme Billing", "invoice",
     ANNOTATED, "Reviewing the Acme invoice INV-2041 before paying it"),
    ("mem-03", 40, "1Password", "Personal vault", "vault", "pending", ""),
    ("mem-04", 60, "Slack", "#design - Acme", "slack",
     ANNOTATED, "Discussing the landing page design in the design channel"),
    ("mem-05", 80, "Xcode", "AgentHandover - Build Succeeded", "xcode",
     ANNOTATED, "Building the AgentHandover app in Xcode"),
    ("mem-06", 100, "Terminal", "zsh - cargo test", "terminal",
     ANNOTATED, "Running the Rust test suite in the terminal"),
    ("mem-07", 120, "Figma", "Landing page - Hero", "figma",
     ANNOTATED, "Editing the hero section of the landing page"),
    ("mem-08", 140, "Dock", "Dock", "dock", "skipped", ""),
]
KEPT = {"mem-01", "mem-04", "mem-05", "mem-06", "mem-07"}

# Query -> the app of the moment that must come first.
QUERIES = [
    ("the Acme invoice I was paying", "Google Chrome"),
    ("design discussion in Slack", "Slack"),
    ("Xcode build of AgentHandover", "Xcode"),
    ("running cargo tests in the terminal", "Terminal"),
    ("editing the landing page hero in Figma", "Figma"),
]


def _frame_request(app: str, title: str, image: str) -> dict:
    jpeg = (FRAMES_DIR / f"{image}.jpg").read_bytes()
    return {
        "model": MEMORY_MODEL,
        "input": [{"text": embed_text(app, title), "image": base64.b64encode(jpeg).decode()}],
        "dimensions": DIM,
    }


def _query_request(query: str) -> dict:
    prefix = EMBEDDING_PROFILES[MEMORY_MODEL].query_prefix
    return {"model": MEMORY_MODEL, "input": [prefix + query], "dimensions": DIM}


def memory_keys() -> list[str]:
    """The /api/embed requests this journey makes with EmbeddingGemma 2."""
    keys = [
        request_key("/api/embed", _frame_request(app, title, image))
        for fid, _s, app, title, image, _st, _d in FRAMES if fid in KEPT
    ]
    keys += [request_key("/api/embed", _query_request(q)) for q, _app in QUERIES]
    return list(dict.fromkeys(keys))


_TAPE = CASSETTE_DIR / f"{CASSETTE}.jsonl"
MISSING = [k for k in memory_keys() if k not in (load_entries(_TAPE) if _TAPE.is_file() else {})]
pytestmark = pytest.mark.skipif(
    bool(MISSING) and os.environ.get("AH_RECORD") != "1",
    reason=(
        f"cassettes/{CASSETTE}.jsonl has no EmbeddingGemma 2 answers for "
        f"{', '.join(MISSING)} (multimodal /api/embed of the J-memory frames and "
        "queries): record with AH_RECORD=1 on the test Mac"
    ),
)


def _load_frames(data_dir: Path, start) -> Path:
    db_path = create_daemon_db(data_dir / "events.db")
    shots = data_dir / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        for fid, seconds, app, title, image, status, doing in FRAMES:
            shot = shots / f"{fid}.jpg"
            shutil.copyfile(FRAMES_DIR / f"{image}.jpg", shot)
            annotation = None
            if status == ANNOTATED:
                annotation = json.dumps({
                    "app": app, "location": title,
                    "task_context": {"what_doing": doing},
                })
            elif status == "skipped":
                annotation = json.dumps({"skip_reason": "system_ui"})
            when = start + timedelta(seconds=seconds)
            conn.execute(
                "INSERT INTO events (id, timestamp, kind_json, window_json, "
                "display_topology_json, primary_display_id, metadata_json, "
                "annotation_status, scene_annotation_json, frame_diff_json) "
                "VALUES (?, ?, '{\"FocusChange\":{}}', ?, '[]', 'main', ?, ?, ?, ?)",
                (
                    fid, when.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                    json.dumps({"app_id": f"pid:{100 + seconds}:{app}", "title": title}),
                    json.dumps({"screenshot_path": str(shot)}),
                    status, annotation,
                    json.dumps({"diff_type": "app_switch"}) if status == ANNOTATED else None,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def test_j_memory(temp_home, cassette, worker, clock) -> None:
    tape = cassette(CASSETTE)
    temp_home.configure(ollama_host=tape.host, extra="[memory]\nenabled = true\n")
    start = clock.now() - timedelta(minutes=10)
    db_path = _load_frames(temp_home.data_dir, start)
    journey = worker(db_path)
    for _ in range(3):
        journey.run_tick()
        clock.advance(seconds=30)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM memory_frames ORDER BY first_ts")]
        statuses = dict(conn.execute("SELECT id, annotation_status FROM events"))
    finally:
        conn.close()

    assert [r["id"] for r in rows] == sorted(KEPT)
    invoice = rows[0]
    assert invoice["frames"] == 2
    assert invoice["first_ts"] < invoice["last_ts"]
    assert invoice["description"].startswith("Reviewing the Acme invoice INV-2041")
    assert {r["dim"] for r in rows} == {DIM} and {r["model"] for r in rows} == {MEMORY_MODEL}
    assert statuses["mem-03"] == "privacy_blocked"
    assert not any(r["app"] in ("1Password", "Dock") for r in rows)
    shots = temp_home.data_dir / "screenshots"
    assert sorted(p.name for p in shots.iterdir()) == [], "every JPEG deleted"

    index = MemoryIndex(
        db_path, model=MEMORY_MODEL, dim=DIM,
        embed_query=query_embedder(tape.host, MEMORY_MODEL, DIM),
    )
    try:
        for query, app in QUERIES:
            hits = index.search(query, limit=5)
            assert hits and hits[0]["app"] == app, (query, [(h["app"], h["score"]) for h in hits])
    finally:
        index.close()
