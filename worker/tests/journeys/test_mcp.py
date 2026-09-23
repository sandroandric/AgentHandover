"""J-mcp: an agent pairs with AgentHandover and asks it for a Skill.

The worker has approved a Skill (knowledge base) and embedded it (vector
store in ``events.db``).  The journey then does what a user and their
agent do:

1. ``agenthandover connect claude-code`` installs the Skill where Claude
   Code reads it and registers the MCP server through the Claude Code
   CLI (a stub here that records its argv; the real CLI would change this
   machine's Claude Code configuration).
2. ``agenthandover connect mcp`` prints the server's tools.
3. An MCP client starts ``agenthandover-mcp`` over stdio, as Claude Code
   does, lists the tools and calls ``search_skills`` and ``get_skill``.

Embeddings come from the ``mcp`` cassette (nomic-embed-text, recorded
with ``AH_RECORD=1``).
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agenthandover_worker import main as worker_main
from agenthandover_worker import paths
from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.procedure_schema import sop_to_procedure
from agenthandover_worker.vector_kb import VectorKB, VectorKBConfig
from fixtures.vlm_answers import generated_sop_template

SLUG = "check-domain-availability"
QUERY = "find out whether a web domain name can still be registered"


def _approved_skill(kb_root: Path) -> dict:
    proc = sop_to_procedure(generated_sop_template("check_domain_availability"))
    # Approved for agents: what the approval leaves (trust "suggest") and a
    # goal, so the hand-off gate (ledger.handoff_gate) lets it through.
    proc.update({
        "id": SLUG, "slug": SLUG, "title": "Check Domain Availability",
        "goal": "Find out whether a domain is free to register and what it costs.",
        "lifecycle_state": "agent_ready",
        "constraints": {"trust_level": "suggest", "guardrails": []},
    })
    kb = KnowledgeBase(root=kb_root)
    kb.ensure_structure()
    kb.save_procedure(proc)
    return kb.get_procedure(SLUG)


def _stub_claude(tmp_path: Path) -> tuple[Path, Path]:
    """A ``claude`` that records its argv and reports our server connected
    once it has been added."""
    log = tmp_path / "claude-argv.jsonl"
    added = tmp_path / "claude-added"
    script = tmp_path / "claude"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        f"log, added = pathlib.Path({str(log)!r}), pathlib.Path({str(added)!r})\n"
        "with log.open('a') as fh:\n"
        "    fh.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1:3] == ['mcp', 'add']:\n"
        "    added.write_text(sys.argv[-1])\n"
        "elif sys.argv[1:3] == ['mcp', 'list'] and added.exists():\n"
        "    print('agenthandover: ' + added.read_text() + ' - \\u2713 Connected')\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script, log


def _connect(agent: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "agenthandover_worker.agent_connect", agent],
        env=env, capture_output=True, text=True, timeout=120,
    )


async def _ask_the_server(env: dict) -> dict:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agenthandover_worker.mcp_server"],
        env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            search = await session.call_tool("search_skills", {"query": QUERY})
            skill = await session.call_tool("get_skill", {"slug": SLUG})
            ready = await session.call_tool("list_ready_skills", {})
    return {
        "tools": [t.name for t in tools.tools],
        "search": json.loads(search.content[0].text),
        "skill": json.loads(skill.content[0].text),
        "ready": json.loads(ready.content[0].text),
    }


def test_an_agent_pairs_and_finds_the_skill_by_meaning(
    temp_home, kb_root, events_db, cassette, tmp_path: Path,
) -> None:
    tape = cassette("mcp")
    proc = _approved_skill(kb_root)

    # The worker's database, with the Skill embedded the way the worker does it.
    loaded = events_db("focus_dailynews")
    assert loaded.db_path == paths.db_path()
    with VectorKB(loaded.db_path, VectorKBConfig(ollama_host=tape.host)) as vkb:
        worker_main._embed_procedure(vkb, proc)
        assert vkb.count("procedure") == 1

    claude, argv_log = _stub_claude(tmp_path)
    env = {**os.environ, "HOME": str(temp_home.path), "AH_CLAUDE_BIN": str(claude)}

    # 1. Pair Claude Code.
    paired = _connect("claude-code", env)
    assert paired.returncode == 0, paired.stdout + paired.stderr
    skill_file = temp_home.path / ".claude" / "skills" / SLUG / "SKILL.md"
    assert skill_file.is_file()
    calls = [json.loads(line) for line in argv_log.read_text().splitlines()]
    add = next(c for c in calls if c[:2] == ["mcp", "add"])
    assert add[:8] == ["mcp", "add", "--scope", "user", "--transport", "stdio", "agenthandover", "--"]
    assert Path(add[8]).is_absolute() and add[8].endswith("agenthandover-mcp")
    assert "connected" in paired.stdout

    # 2. The tool list `connect mcp` prints ...
    shown = _connect("mcp", env)
    assert shown.returncode == 0, shown.stderr

    # 3. ... is the list the running server gives an agent.
    answer = asyncio.run(_ask_the_server(env))
    for name in answer["tools"]:
        assert f"  - {name}(" in shown.stdout
    assert {"search_skills", "get_skill", "report_execution_start"} <= set(answer["tools"])

    search = answer["search"]
    assert search["mode"] == "semantic", search
    assert search["results"][0]["slug"] == SLUG
    assert search["results"][0]["title"] == "Check Domain Availability"

    assert [s["slug"] for s in answer["ready"]["ready"]] == [SLUG]
    assert answer["ready"]["held_back"] == []
    # get_skill returns the same packet body Claude Code reads from disk.
    packet_body = answer["skill"]["packet"].split("\n---\n", 1)[1]
    disk_body = skill_file.read_text(encoding="utf-8").split("\n---\n", 1)[1]
    assert packet_body == disk_body
    assert "report_execution_start" in packet_body


def test_without_the_worker_database_search_says_it_fell_back(
    temp_home, kb_root, tmp_path: Path,
) -> None:
    _approved_skill(kb_root)
    env = {**os.environ, "HOME": str(temp_home.path)}
    params_env = env

    async def ask() -> dict:
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "agenthandover_worker.mcp_server"],
            env=params_env,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("search_skills", {"query": "domain"})
        return json.loads(result.content[0].text)

    search = asyncio.run(ask())
    assert search["mode"] == "keyword_fallback"
    assert "no worker database" in search["reason"]
    assert [r["slug"] for r in search["results"]] == [SLUG]
