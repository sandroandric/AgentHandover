"""J-commonplace: an agent connected over MCP learns what Commonplace is and
searches it; the CLI's query API answers the same.

The worker's stores hold an approved Skill (embedded the way the worker
embeds it), today's daily summary (and its vector), a Memory moment and a
profile.  An MCP client starts ``agenthandover-mcp`` over stdio with the
journey's ``HOME``, as Claude Code does, and:

1. reads the server's instructions (they introduce Commonplace) and finds
   ``search_commonplace`` first in the tool list;
2. searches with the agents switch off: the Skill and the day are found
   by meaning, Memory says how the user can allow it;
3. the user turns the switch on (``config.toml``, as
   ``agenthandover memory agents on`` writes it) and, in the same session,
   the next search finds the moment, as text only;
4. asks for the profile by kind.

Then the worker's query API (``POST /commonplace/search``, which
``agenthandover search`` calls) gives the same groups.

Ollama is a stand-in HTTP server answering ``/api/embed`` by keyword (the
domain, the invoice and the banner each point along their own axis), so
the journey needs no recorded cassette: what it proves is the hub's
wiring, from the MCP process and from the worker, not how well a model
ranks.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import sys
import threading
import urllib.request
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from agenthandover_worker import main as worker_main
from agenthandover_worker import paths
from agenthandover_worker.model_profiles import EMBEDDING_MODEL, EMBEDDING_PROFILES, MEMORY_MODEL
from agenthandover_worker.procedure_schema import sop_to_procedure
from agenthandover_worker.query_api import TOKEN_HEADER
from agenthandover_worker.stages.daily import summary_text
from agenthandover_worker.vector_kb import VectorKB, VectorKBConfig
from fixtures.vlm_answers import generated_sop_template
from memory_world import World, axis

mcp = pytest.importorskip("mcp", reason="J-commonplace needs the mcp SDK (the worker's [mcp] extra)")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

TOPICS = (("domain", 1), ("invoice", 2), ("banner", 3))
SLUG = "check-domain-availability"


def _vector(item) -> list[float]:
    text = (item["text"] if isinstance(item, dict) else item).lower()
    return axis(next((i for word, i in TOPICS if word in text), 9))


class FakeOllama(http.server.BaseHTTPRequestHandler):
    """``POST /api/embed`` by keyword; anything else is a 404."""

    requests: list[dict] = []

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if self.path != "/api/embed":
            self.send_error(404)
            return
        FakeOllama.requests.append(body)
        items = body["input"] if isinstance(body["input"], list) else [body["input"]]
        reply = json.dumps({"model": body["model"], "embeddings": [_vector(i) for i in items]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)


@pytest.fixture()
def ollama():
    FakeOllama.requests = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _memory_config(*, agents: bool) -> str:
    return f"[memory]\nenabled = true\nagents_access = {str(agents).lower()}\n"


async def _agent_session(env: dict, switch_on) -> dict:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "agenthandover_worker.mcp_server"], env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()

            async def call(**arguments) -> dict:
                result = await session.call_tool("search_commonplace", arguments)
                assert not result.isError, result
                return json.loads(result.content[0].text)

            tools = [t.name for t in (await session.list_tools()).tools]
            off = await call(query="the acme invoice")
            switch_on()
            on = await call(query="the acme invoice", kinds=["memory", "days"])
            profile = await call(query="how do they work", kinds=["profile"])
            since = await call(
                query="the acme invoice", kinds=["memory"],
                since=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            )
    return {
        "instructions": init.instructions, "tools": tools,
        "off": off, "on": on, "profile": profile, "since": since,
    }


def test_j_commonplace(temp_home, kb_root, worker, ollama) -> None:
    temp_home.configure(ollama_host=ollama, extra=_memory_config(agents=False))
    world = World(temp_home.path, data_dir=temp_home.data_dir, knowledge_dir=kb_root)
    assert world.db == paths.db_path()
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()

    # The approved Skill, embedded the way the worker embeds it.
    proc = sop_to_procedure(generated_sop_template("check_domain_availability"))
    proc.update({
        "id": SLUG, "slug": SLUG, "title": "Check Domain Availability",
        "goal": "Find out whether a domain is free to register and what it costs.",
        "lifecycle_state": "agent_ready",
        "constraints": {"trust_level": "suggest", "guardrails": []},
        "staleness": {"last_observed": now.isoformat()},
    })
    world.kb.save_procedure(proc)
    intents = ["Paid the Acme invoice in the billing portal", "Reviewed the hero banner"]
    world.kb.save_daily_summary(today, {
        "active_hours": 3.0, "task_count": 2,
        "tasks": [{"intent": i, "event_ids": ["ev-1"], "urls": ["https://billing.acme.com/inv"]}
                  for i in intents],
        "top_apps": [{"app": "Google Chrome", "minutes": 120}],
    })
    with VectorKB(world.db, VectorKBConfig(ollama_host=ollama)) as vkb:
        worker_main._embed_procedure(vkb, world.kb.get_procedure(SLUG))
        worker_main._embed_in_vector_kb(vkb, "daily_summary", today, summary_text(today, 2, 3.0, intents))
        assert vkb.count("procedure") == 1 and vkb.count("daily_summary") == 1
    world.moment("inv", now - timedelta(minutes=30), now - timedelta(minutes=25),
                 "Google Chrome", "Invoice INV-2041", "Paying the Acme invoice",
                 thumb_id="inv", vector=axis(2))
    world.kb.update_profile({"tools": {"browser": "Google Chrome"}})

    env = {**os.environ, "HOME": str(temp_home.path)}
    answer = asyncio.run(_agent_session(
        env, lambda: temp_home.configure(ollama_host=ollama, extra=_memory_config(agents=True)),
    ))

    # 1. The agent learns what Commonplace is, and the tool comes first.
    assert "Commonplace" in answer["instructions"]
    assert "search_commonplace" in answer["instructions"]
    assert answer["tools"][0] == "search_commonplace"
    assert {"search_skills", "search_memory", "get_skill"} <= set(answer["tools"])

    # 2. Switch off: Skills and days by meaning, Memory explains itself.
    off = answer["off"]
    assert set(off) == {"query", "skills", "memory", "days"}
    assert off["memory"]["mode"] == "not_allowed"
    assert "Memory Settings" in off["memory"]["reason"]
    # Agents are never handed the command that would turn it on.
    assert "agenthandover memory agents on" not in off["memory"]["reason"]
    assert off["days"]["mode"] == "semantic"
    assert [d["date"] for d in off["days"]["results"]] == [today]
    assert off["days"]["results"][0]["excerpt"].startswith("Paid the Acme invoice")
    assert off["skills"]["mode"] == "semantic"
    # "the acme invoice" points away from the domain Skill: nothing over the floor.
    assert off["skills"]["results"] == []

    # 3. Switch on, same session: the moment, as text.
    memory = answer["on"]["memory"]
    assert memory["mode"] == "semantic"
    assert [m["description"] for m in memory["results"]] == ["Paying the Acme invoice"]
    assert set(memory["results"][0]) == {
        "app", "window_title", "url_host", "description", "first_ts", "last_ts", "score",
    }
    assert answer["since"]["memory"]["results"] == []  # the range reaches the search

    # 4. The profile, by kind.
    assert answer["profile"]["profile"]["results"][0]["tools"] == {"browser": "Google Chrome"}

    text = json.dumps(answer)
    for leaked in ("thumb", '"inv"', "ev-1", "billing.acme.com/inv", str(temp_home.data_dir)):
        assert leaked not in text, leaked
    # The MCP server embedded the query with each model's query prefix: the
    # text model's for Skills and days (once for both), Memory's for Memory.
    queries = [
        (r["model"], r["input"][0]) for r in FakeOllama.requests
        if "acme invoice" in json.dumps(r["input"])
    ]
    text_query = (EMBEDDING_MODEL, f"{EMBEDDING_PROFILES[EMBEDDING_MODEL].query_prefix}the acme invoice")
    memory_query = (MEMORY_MODEL, f"{EMBEDDING_PROFILES[MEMORY_MODEL].query_prefix}the acme invoice")
    assert set(queries) == {text_query, memory_query}
    assert queries.count(text_query) == 2  # one per search that asked for skills or days

    # The worker's query API, which `agenthandover search` asks, gives the same groups.
    journey = worker(world.db)
    cfg = journey.ctx.cfg
    journey.ctx.cfg = replace(
        cfg, knowledge=replace(cfg.knowledge, query_api_enabled=True, query_api_port=0),
    )
    srv = worker_main._start_query_api(journey.ctx)
    assert srv is not None
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{srv.port}/commonplace/search",
            data=json.dumps({"query": "the acme invoice", "kinds": ["memory", "days"]}).encode(),
            headers={"Content-Type": "application/json", TOKEN_HEADER: srv.token},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            api = json.loads(resp.read())
        assert api == answer["on"]
        # Without a request for a domain Skill there is none; with one, the Skill.
        req = urllib.request.Request(
            f"http://127.0.0.1:{srv.port}/commonplace/search",
            data=json.dumps({"query": "is this domain free", "kinds": ["skills"]}).encode(),
            headers={"Content-Type": "application/json", TOKEN_HEADER: srv.token},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            skills = json.loads(resp.read())["skills"]
        assert skills["mode"] == "semantic"
        assert [s["slug"] for s in skills["results"]] == [SLUG]
    finally:
        srv.stop()
        world.close()
