"""J-forget: what the user forgets is gone from every search and from disk.

The worker as ``main()`` builds it (``[memory] enabled``, agents allowed)
ingests four annotated frames through its own tick: the Acme invoice in
Chrome this morning, and in the last hour the design chat in Slack and a
cargo test run in the Terminal.  Each frame is everywhere AgentHandover
keeps a frame (``memory_world.World``: event, OCR, annotation, activity
index, preserved text and the vectors made from them, thumbnail,
artifact, task segment, workflow observation) and the worker's Memory
pass embeds it into ``memory_frames``.  A fifth frame lands just before
the forget, its screenshot not yet embedded.  Today's and yesterday's
daily summaries exist.

The user forgets the last hour through the query API the app and the CLI
use (``POST /memory/forget``).  Then nothing from that hour is found by
``POST /search`` (the activity index), ``POST /search/semantic`` (the
vector store), ``POST /memory/search`` or MCP ``search_memory``; its
screenshots, thumbnails, artifacts and today's summary are gone; the
morning is still found everywhere.

Ollama is a stand-in answering ``/api/embed`` by keyword (the invoice,
the design chat and the test run each point along their own axis), so
the journey needs no recorded cassette: what it proves is where the data
goes, not how well a model ranks it.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import replace
from datetime import timedelta

from agenthandover_worker import main as worker_main
from agenthandover_worker.mcp_server import build_mcp_server
from agenthandover_worker.ollama import OllamaClient, OllamaUnavailable
from agenthandover_worker.query_api import TOKEN_HEADER
from memory_world import World, axis

KEYWORDS = (("invoice", 1), ("banner", 2), ("design", 2), ("cargo", 3), ("terminal", 3))


def _vector(item) -> list[float]:
    text = (item["text"] if isinstance(item, dict) else item).lower()
    return axis(next((i for word, i in KEYWORDS if word in text), 9))


def fake_ollama(self, method, path, payload, timeout):
    if path != "/api/embed":
        raise OllamaUnavailable(f"{path}: only /api/embed is answered in J-forget")
    return {"embeddings": [_vector(item) for item in payload["input"]]}


def _post(srv, path: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{srv.port}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", TOKEN_HEADER: srv.token}, method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _found(srv, mcp_search) -> dict[str, set[str]]:
    """What each search surface finds for the three subjects."""
    found: dict[str, set[str]] = {"search": set(), "semantic": set(), "memory": set(), "mcp": set()}
    for query in ("acme invoice", "hero banner design", "cargo test terminal"):
        word = query.split()[1]
        status, body = _post(srv, "/search", {"query": word})
        assert status == 200, body
        found["search"] |= {r["event_id"] for r in body["results"]}
        status, body = _post(srv, "/search/semantic", {"query": query})
        assert status == 200, body
        found["semantic"] |= {r["source_id"] for r in body["results"]}
        status, body = _post(srv, "/memory/search", {"query": query, "limit": 100})
        assert status == 200, body
        found["memory"] |= {m["id"] for m in body["moments"] if m["score"] > 0.5}
        result = mcp_search(query)
        assert result["mode"] == "semantic", result
        found["mcp"] |= {m["description"] for m in result["results"] if m["score"] > 0.5}
    return found


def test_j_forget(temp_home, kb_root, worker, clock, monkeypatch) -> None:
    monkeypatch.setattr(OllamaClient, "_request", fake_ollama)
    temp_home.configure(
        ollama_host="http://127.0.0.1:9",
        extra="[memory]\nenabled = true\nagents_access = true\n",
    )
    now = clock.now()
    world = World(temp_home.path, data_dir=temp_home.data_dir, knowledge_dir=kb_root)
    world.frame("ev-am", now - timedelta(hours=3), "Google Chrome", "Invoice INV-2041",
                "Paying the Acme invoice", memory=False, vector=axis(1))
    world.frame("ev-1", now - timedelta(minutes=40), "Slack", "#design",
                "Discussing the hero banner design", memory=False, vector=axis(2))
    world.frame("ev-2", now - timedelta(minutes=20), "Terminal", "zsh - cargo test",
                "Running the cargo test suite in the terminal", memory=False, vector=axis(3))
    world.segment("seg-am", ["ev-am"], now - timedelta(hours=3))
    world.segment("seg-hour", ["ev-1", "ev-2"], now - timedelta(minutes=40))
    world.observation("obs-hour", ["ev-1", "ev-2"], now - timedelta(minutes=40))
    yesterday = (now - timedelta(days=1)).date().isoformat()
    today = now.date().isoformat()
    world.daily(yesterday, "Acme invoices")
    world.daily(today, "the hero banner and cargo tests")

    journey = worker(world.db)
    journey.run_tick()
    assert world.ids("SELECT id FROM memory_frames") == {"ev-am", "ev-1", "ev-2"}
    assert journey.activity_searcher.refresh_index() >= 0
    # Captured just now: its screenshot waits for the next Memory pass.
    world.frame("ev-3", now - timedelta(minutes=2), "Terminal", "zsh - cargo test",
                "Reading the cargo test failures", memory=False, vector=axis(3))
    journey.activity_searcher.refresh_index()

    cfg = journey.ctx.cfg
    journey.ctx.cfg = replace(
        cfg, knowledge=replace(cfg.knowledge, query_api_enabled=True, query_api_port=0),
    )
    srv = worker_main._start_query_api(journey.ctx)
    assert srv is not None
    mcp = build_mcp_server()
    tool = mcp._tool_manager.get_tool("search_memory").fn

    def mcp_search(query: str) -> dict:
        return json.loads(tool(query, limit=10))

    try:
        hour = {"ev-1", "ev-2", "ev-3"}
        before = _found(srv, mcp_search)
        assert hour <= before["search"] and "ev-am" in before["search"]
        assert {"ev-1", "ev-2", "ev-3", "ev-1:ocr", "ev-3:ocr"} <= before["semantic"]
        assert before["memory"] == {"ev-am", "ev-1", "ev-2"}
        assert "Discussing the hero banner design" in "".join(before["mcp"])

        status, counts = _post(srv, "/memory/forget", {
            "since": (now - timedelta(hours=1)).isoformat(), "until": now.isoformat(),
        })
        assert status == 200, counts
        assert counts["events"] == 3 and counts["memory_moments"] == 2
        assert counts["daily_summaries"] == 1

        after = _found(srv, mcp_search)
        assert after["search"] == {"ev-am"}
        forgotten = hour | {f"{e}:ocr" for e in hour} | {today}
        assert before["semantic"] & forgotten and not after["semantic"] & forgotten
        assert {"ev-am", "ev-am:ocr"} <= after["semantic"]
        assert after["memory"] == {"ev-am"}
        assert all("Acme" in d for d in after["mcp"]) and after["mcp"]

        # And on disk.
        assert world.files(world.shots) <= {"ev-am"}
        assert world.files(world.thumbs) == {"ev-am"}
        assert world.files(world.artifacts) == {"ev-am"}
        assert world.daily_days() == {yesterday}
        assert world.vector_sources("daily_summary") == {yesterday}
        assert world.ids("SELECT segment_id FROM task_segments") == {"seg-am"}
        assert world.ids("SELECT obs_id FROM workflow_observations") == set()
        assert world.ids("SELECT event_id FROM preserved_text") == {"ev-am"}

        # The worker goes on: its next Memory pass adds nothing back.
        journey.run_tick()
        assert world.ids("SELECT id FROM memory_frames") == {"ev-am"}
    finally:
        srv.stop()
        world.close()
