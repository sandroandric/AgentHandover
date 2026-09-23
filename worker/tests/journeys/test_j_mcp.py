"""J-mcp: an agent connected over MCP finds the approved Skill, runs it and
reports back.

After J-approve's promotion, ``agenthandover-mcp`` (the console script the
package installs) is spawned over stdio with the MCP SDK's client and the
journey's ``HOME``, exactly as Claude Code spawns it.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from agenthandover_worker.execution_monitor import ExecutionMonitor, ExecutionStatus
from agenthandover_worker.skill_render import render_packet
from journeys.focus_run import approve_for_agents, start_focus_run, tick_until_skill
from journeys.packet import parse_packet

mcp = pytest.importorskip("mcp", reason="J-mcp needs the mcp SDK (the worker's [mcp] extra)")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

TOOLS = {
    "list_ready_skills", "list_all_skills", "get_skill", "search_skills",
    "get_user_profile", "report_execution_start", "report_step_result",
    "report_execution_complete",
}


def _server_command() -> str:
    script = Path(sys.executable).parent / "agenthandover-mcp"
    found = str(script) if script.is_file() else shutil.which("agenthandover-mcp")
    assert found, "agenthandover-mcp is not installed in the test environment"
    return found


@dataclass
class AgentSession:
    tools: set[str]
    ready: list[dict]
    skill: dict
    started: dict
    step_result: dict
    completed: dict


async def _agent_runs_skill(home: Path, slug: str, *, deviate: str) -> AgentSession:
    params = StdioServerParameters(
        command=_server_command(), args=[], env={**os.environ, "HOME": str(home)},
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            async def call(tool: str, **arguments) -> dict | list:
                result = await session.call_tool(tool, arguments)
                assert not result.isError, result
                return json.loads(result.content[0].text)

            tools = {t.name for t in (await session.list_tools()).tools}
            ready = await call("list_ready_skills")
            skill = await call("get_skill", slug=slug)
            started = await call("report_execution_start", slug=slug, agent_id="journey")
            step_id = parse_packet(skill["packet"]).steps()[0].step_id
            step_result = await call(
                "report_step_result",
                execution_id=started["execution_id"], step_id=step_id, status="deviated",
                actual_action=deviate, notes=f"agent did: {deviate}",
            )
            completed = await call(
                "report_execution_complete", execution_id=started["execution_id"],
            )
    return AgentSession(tools, ready, skill, started, step_result, completed)


def _journey(events_db, worker, clock, temp_home, cassette):
    run = start_focus_run(
        events_db=events_db, worker=worker, clock=clock,
        temp_home=temp_home, cassette=cassette,
    )
    slug = tick_until_skill(run, clock)["id"]
    approve_for_agents(run, clock, slug)
    kb = run.worker.knowledge_base
    before = kb.get_procedure(slug)
    session = asyncio.run(
        _agent_runs_skill(temp_home.path, slug, deviate="Opened the digest from Drafts instead"),
    )
    return kb, slug, before, kb.get_procedure(slug), session


def test_mcp_agent_gets_the_packet_and_reports_one_run(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    kb, slug, before, after, session = _journey(events_db, worker, clock, temp_home, cassette)

    assert session.tools == TOOLS
    assert [s["slug"] for s in session.ready] == [slug]
    # get_skill hands the agent the same packet every target renders.
    assert session.skill["packet"] == render_packet(before, target="mcp")
    packet_ids = [s.step_id for s in parse_packet(session.skill["packet"]).steps()]
    assert session.step_result == {"recorded": True, "step_id": packet_ids[0]}
    assert session.completed["final_status"] == "deviated"

    # The run log (the runs table the worker and the query API read) holds
    # exactly this run; the procedure JSON the app reads counts it.
    runs = ExecutionMonitor(kb).get_history(slug)
    assert [(r.execution_id, r.procedure_slug) for r in runs] == [
        (session.started["execution_id"], slug),
    ]
    assert after != before
    assert after["execution_stats"]["total"] == 1
    assert after["execution_stats"]["deviated"] == 1
    assert [a["step_id"] for a in after["observed_alternatives"]] == [packet_ids[0]]


def test_mcp_deviation_lands_on_the_reported_step(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    """The deviation is recorded against the packet's first step, and the
    run lists every step of the procedure (no extra '(dynamic)' step)."""
    kb, slug, before, _after, session = _journey(events_db, worker, clock, temp_home, cassette)
    [run] = ExecutionMonitor(kb).get_history(slug)
    packet_ids = [s.step_id for s in parse_packet(session.skill["packet"]).steps()]
    assert [s.step_id for s in run.steps] == packet_ids
    assert len(run.steps) == len(before["steps"])
    deviated = [s for s in run.steps if s.status is ExecutionStatus.DEVIATED]
    assert [s.step_id for s in deviated] == [packet_ids[0]]
    assert [d["step_id"] for d in run.deviations] == [packet_ids[0]]


def test_mcp_finished_run_updates_last_confirmed(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    """A run that reached the end (here with a deviation) confirms the
    Skill is still current."""
    _kb, _slug, before, after, _session = _journey(events_db, worker, clock, temp_home, cassette)
    confirmed = (after.get("staleness") or {}).get("last_confirmed")
    assert confirmed and confirmed != (before.get("staleness") or {}).get("last_confirmed")
