"""The ``/memory/*`` query API endpoints (``memory_api.MemoryService``)
and the pause file they write."""

from __future__ import annotations

import json
import os
import stat
import time
import urllib.error
import urllib.request
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from agenthandover_worker.config import Config, MemoryConfig
from agenthandover_worker.memory import paused, pause_file, pause_until, write_pause
from agenthandover_worker.memory_api import MemoryService
from agenthandover_worker.ollama import OllamaUnavailable
from agenthandover_worker.query_api import TOKEN_HEADER, QueryAPIServer
from daemon_db import REPO_ROOT
from memory_world import DIM, NOW, World, axis

PAUSE_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "memory_pause"


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


class Embedder:
    """Queries to vectors by name; ``down`` raises like Ollama when off."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.down = False
        self.queries: list[str] = []

    def __call__(self, query: str) -> list[float]:
        self.queries.append(query)
        if self.down:
            raise OllamaUnavailable("Ollama not reachable (fake)")
        return self.vectors[query]


def write_config(path: Path, *, agents: bool) -> Path:
    path.write_text(f"[memory]\nenabled = true\nagents_access = {str(agents).lower()}\n")
    return path


@pytest.fixture()
def api(tmp_path):
    """A query API whose Memory holds three moments: the Acme invoice in
    Chrome (axis 1), the design chat in Slack (axis 2), yesterday's
    standup in Zoom (axis 3)."""
    world = World(tmp_path)
    world.frame("inv", NOW - 30 * timedelta(minutes=1), "Google Chrome", "Invoice INV-2041",
                "Paying the Acme invoice", memory=False)
    world.moment("inv", NOW - timedelta(minutes=30), NOW - timedelta(minutes=25),
                 "Google Chrome", "Invoice INV-2041", "Paying the Acme invoice",
                 frames=4, thumb_id="inv", vector=axis(1))
    world.moment("chat", NOW - timedelta(minutes=10), NOW - timedelta(minutes=10),
                 "Slack", "#design", "Discussing the hero banner", vector=axis(2))
    world.moment("standup", NOW - timedelta(days=1), NOW - timedelta(days=1),
                 "zoom.us", "Standup", "Daily standup call", vector=axis(3))
    embedder = Embedder({"acme invoice": axis(1), "design chat": axis(2), "standup": axis(3)})
    clock = Clock()
    config_path = write_config(tmp_path / "config.toml", agents=False)
    cfg = replace(Config(), path=config_path,
                  memory=replace(MemoryConfig(), enabled=True))
    writer = SimpleNamespace(waiting="Ollama failed; retrying after a backoff",
                             forgotten=lambda: None)
    service = MemoryService(
        cfg, db_path=world.db, data_dir=world.data_dir, knowledge_dir=world.kb.root,
        writer=writer, vector_kb=world.vkb, embed_query=embedder,
        local_tz=timezone.utc, now=clock,
    )
    srv = QueryAPIServer(world.kb, port=0, memory=service)
    srv.start()
    time.sleep(0.05)
    yield SimpleNamespace(srv=srv, world=world, embedder=embedder, clock=clock,
                          service=service, config_path=config_path, cfg=cfg)
    srv.stop()
    service.close()
    world.close()


def call(srv: QueryAPIServer, method: str, path: str, body=None, *, raw: bytes | None = None,
         token: bool = True) -> tuple[int, dict]:
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    headers = {"Content-Type": "application/json"}
    if token:
        headers[TOKEN_HEADER] = srv.token
    req = urllib.request.Request(
        f"http://127.0.0.1:{srv.port}{path}", data=data, headers=headers, method=method,
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# ---------------------------------------------------------------------------
# POST /memory/search
# ---------------------------------------------------------------------------


def test_search_returns_moments_with_the_contracts_fields(api):
    status, body = call(api.srv, "POST", "/memory/search", {"query": "acme invoice"})
    assert status == 200
    assert body["model"] == "embeddinggemma-2:440m"
    first = body["moments"][0]
    assert set(first) == {
        "id", "first_ts", "last_ts", "frames", "app", "window_title", "url_host",
        "description", "score", "thumb_id",
    }
    assert first["id"] == "inv" and first["thumb_id"] == "inv" and first["frames"] == 4
    assert first["first_ts"] == "2026-10-12T11:30:00.000Z"
    assert first["score"] == pytest.approx(1.05)  # vector 1.0 + the keyword boost
    assert {m["id"] for m in body["moments"][1:]} == {"chat", "standup"}  # score 0, a tie


def test_search_filters_and_limit(api):
    def ids(body):
        status, reply = call(api.srv, "POST", "/memory/search", body)
        assert status == 200, reply
        return [m["id"] for m in reply["moments"]]

    assert ids({"query": "standup", "limit": 1}) == ["standup"]
    assert ids({"query": "standup", "app": "SLACK"}) == ["chat"]
    assert ids({"query": "standup", "since": "2026-10-12T11:00:00Z"}) == ["inv", "chat"]
    assert ids({"query": "standup", "until": "2026-10-12T00:00:00Z"}) == ["standup"]


@pytest.mark.parametrize("body, error", [
    ({}, "'query'"),
    ({"query": "  "}, "'query'"),
    ({"query": 3}, "'query'"),
    ({"query": "x", "limit": 0}, "'limit'"),
    ({"query": "x", "limit": 101}, "'limit'"),
    ({"query": "x", "limit": True}, "'limit'"),
    ({"query": "x", "limit": "5"}, "'limit'"),
    ({"query": "x", "app": ""}, "'app'"),
    ({"query": "x", "since": "last week"}, "'since' must be an ISO-8601"),
    ({"query": "x", "since": "2026-10-12T12:00:00Z", "until": "2026-10-12T11:00:00Z"}, "after"),
    ([1], "JSON object"),
])
def test_search_refuses_bad_requests(api, body, error):
    status, reply = call(api.srv, "POST", "/memory/search", body)
    assert status == 400
    assert error in reply["error"]


def test_search_refuses_invalid_json_and_requests_without_the_token(api):
    status, reply = call(api.srv, "POST", "/memory/search", raw=b"{nope")
    assert status == 400 and "Invalid JSON" in reply["error"]
    status, _ = call(api.srv, "POST", "/memory/search", {"query": "x"}, token=False)
    assert status == 403


def test_search_while_memory_is_off_is_503_memory_off(api):
    api.service.memory = replace(api.service.memory, enabled=False)
    status, reply = call(api.srv, "POST", "/memory/search", {"query": "acme invoice"})
    assert (status, reply) == (503, {"error": "memory_off"})


def test_a_query_that_cannot_be_embedded_is_503_model_unavailable(api):
    api.embedder.down = True
    status, reply = call(api.srv, "POST", "/memory/search", {"query": "acme invoice"})
    assert status == 503
    assert reply["error"] == "model_unavailable"
    assert "Ollama not reachable" in reply["detail"]


def test_without_a_memory_service_the_endpoints_are_501(tmp_path):
    world = World(tmp_path)
    srv = QueryAPIServer(world.kb, port=0)
    srv.start()
    try:
        assert call(srv, "POST", "/memory/search", {"query": "x"})[0] == 501
        assert call(srv, "GET", "/memory/status", token=False)[0] == 501
    finally:
        srv.stop()
        world.close()


# ---------------------------------------------------------------------------
# GET /memory/status, POST /memory/pause, POST /memory/resume
# ---------------------------------------------------------------------------


def test_status_reports_what_memory_holds(api):
    status, body = call(api.srv, "GET", "/memory/status", token=False)
    assert status == 200
    assert body == {
        "enabled": True, "paused": False, "paused_until": None, "agents_access": False,
        "model": "embeddinggemma-2:440m", "moments": 3, "frames": 6,
        "oldest": "2026-10-11T12:00:00Z", "newest": "2026-10-12T11:50:00Z",
        "thumbnails": 1, "disk_bytes": body["disk_bytes"],
        "waiting": "Ollama failed; retrying after a backoff",
    }
    thumb = (api.world.thumbs / "inv.ahthumb").stat().st_size
    assert body["disk_bytes"] > 3 * DIM * 4 + thumb


def test_status_reads_the_agents_switch_from_the_file_each_time(api):
    write_config(api.config_path, agents=True)
    assert call(api.srv, "GET", "/memory/status", token=False)[1]["agents_access"] is True


def test_pause_for_minutes_writes_the_contracts_file_and_status(api):
    status, body = call(api.srv, "POST", "/memory/pause", {"minutes": 90})
    assert status == 200
    assert body["paused"] is True
    assert body["paused_until"] == "2026-10-12T13:30:00Z"
    path = pause_file(api.world.data_dir)
    assert path.read_bytes() == (PAUSE_FIXTURES / "timed.json").read_bytes()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert paused(api.world.data_dir, NOW + timedelta(minutes=89))
    assert not paused(api.world.data_dir, NOW + timedelta(minutes=90))
    # After the time the status says not paused, without anyone resuming.
    api.clock.now = NOW + timedelta(minutes=91)
    assert call(api.srv, "GET", "/memory/status", token=False)[1]["paused"] is False


@pytest.mark.parametrize("body", [None, {}, {"minutes": None}])
def test_pause_without_minutes_lasts_until_resumed(api, body):
    raw = b"" if body is None else None
    status, reply = call(api.srv, "POST", "/memory/pause", body, raw=raw)
    assert status == 200
    assert reply["paused"] is True and reply["paused_until"] is None
    path = pause_file(api.world.data_dir)
    assert path.read_bytes() == (PAUSE_FIXTURES / "until_resumed.json").read_bytes()
    status, reply = call(api.srv, "POST", "/memory/resume", raw=b"")
    assert status == 200 and reply["paused"] is False
    assert not path.exists()


@pytest.mark.parametrize("body", [
    {"minutes": 0}, {"minutes": -5}, {"minutes": "30"}, {"minutes": True},
    {"minutes": 1.5}, {"hours": 1}, ["minutes"],
])
def test_pause_refuses_bad_minutes(api, body):
    status, reply = call(api.srv, "POST", "/memory/pause", body)
    assert status == 400
    assert "minutes" in reply["error"]
    assert not pause_file(api.world.data_dir).exists()


def test_pause_writer_and_reader_agree(tmp_path):
    until = datetime(2026, 10, 12, 15, 30, tzinfo=timezone(timedelta(hours=2)))
    write_pause(tmp_path, until)
    assert json.loads(pause_file(tmp_path).read_text()) == {"paused_until": "2026-10-12T13:30:00Z"}
    assert pause_until(tmp_path, NOW) == (True, datetime(2026, 10, 12, 13, 30, tzinfo=timezone.utc))
    write_pause(tmp_path, None)
    assert pause_until(tmp_path, NOW) == (True, None)
    # A rewrite leaves no temp file behind.
    assert [p.name for p in pause_file(tmp_path).parent.iterdir()] == ["memory-pause.json"]
    pause_file(tmp_path).write_text("garbage")
    assert pause_until(tmp_path, NOW) == (True, None)
    pause_file(tmp_path).unlink()
    assert pause_until(tmp_path, NOW) == (False, None)


# ---------------------------------------------------------------------------
# POST /memory/forget
# ---------------------------------------------------------------------------


def test_forget_returns_the_counts_and_search_no_longer_finds_it(api):
    assert call(api.srv, "POST", "/memory/search", {"query": "acme invoice"})[1]["moments"]
    status, counts = call(api.srv, "POST", "/memory/forget", {"moment_id": "inv"})
    assert status == 200
    assert set(counts) == {
        "events", "memory_moments", "thumbnails", "screenshots", "vectors", "daily_summaries",
    }
    assert counts["memory_moments"] == 1 and counts["events"] == 1 and counts["thumbnails"] == 1
    status, body = call(api.srv, "POST", "/memory/search", {"query": "acme invoice"})
    assert "inv" not in [m["id"] for m in body["moments"]]


@pytest.mark.parametrize("body, status", [
    ({}, 400),
    ({"all": True, "moment_id": "inv"}, 400),
    ({"since": "2026-10-12T11:00:00Z"}, 400),
    ({"moment_id": "no-such-moment"}, 404),
])
def test_forget_refuses_ambiguous_empty_or_unknown_selectors(api, body, status):
    code, reply = call(api.srv, "POST", "/memory/forget", body)
    assert code == status
    assert reply["error"]
    assert api.world.ids("SELECT id FROM memory_frames") == {"inv", "chat", "standup"}


def test_forget_waits_for_the_state_lock_and_gives_up_politely(api, monkeypatch):
    import threading

    from agenthandover_worker import memory_api
    from agenthandover_worker.inference_runner import STATE_LOCK

    monkeypatch.setattr(memory_api, "FORGET_LOCK_TIMEOUT_SECONDS", 0.05)
    held, release = threading.Event(), threading.Event()

    def hold():
        with STATE_LOCK:
            held.set()
            release.wait(5)

    t = threading.Thread(target=hold)
    t.start()
    held.wait(5)
    try:
        status, reply = call(api.srv, "POST", "/memory/forget", {"all": True})
    finally:
        release.set()
        t.join()
    assert status == 503 and reply["error"] == "busy"
    assert api.world.ids("SELECT id FROM memory_frames") == {"inv", "chat", "standup"}
