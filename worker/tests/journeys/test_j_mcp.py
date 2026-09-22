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

    # The run log holds exactly this run, and the procedure JSON changed.
    runs = json.loads((kb.root / "observations" / "executions.json").read_text())["records"]
    assert [(r["execution_id"], r["procedure_slug"]) for r in runs] == [
        (session.started["execution_id"], slug),
    ]
    assert after != before
    assert [a["step_id"] for a in after["observed_alternatives"]] == [packet_ids[0]]


@pytest.mark.xfail(strict=True, reason=(
    "WP-I3 (#28): ExecutionMonitor.start_execution numbers expected steps "
    "str(i) (execution_monitor.py:131) while the packet and report_step_result "
    "use the procedure's step_id (step_1, ...), so _find_step "
    "(execution_monitor.py:369) misses and the deviation lands on a new "
    "'(dynamic)' step instead of the packet's first step"
))
def test_mcp_deviation_lands_on_the_reported_step(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    kb, slug, before, _after, session = _journey(events_db, worker, clock, temp_home, cassette)
    runs = json.loads((kb.root / "observations" / "executions.json").read_text())["records"]
    steps = runs[0]["steps"]
    first_action = before["steps"][0].get("action")
    deviated = [s for s in steps if s["status"] == "deviated"]
    assert len(steps) == len(before["steps"])
    assert [s["expected_action"] for s in deviated] == [first_action]


@pytest.mark.xfail(strict=True, reason=(
    "WP-I3: a finished run with a deviation does not confirm the Skill's "
    "freshness: SkillImprover._handle_deviation (skill_improver.py:96) never "
    "writes staleness.last_confirmed, only _handle_success (skill_improver.py:63) does"
))
def test_mcp_finished_run_updates_last_confirmed(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    _kb, _slug, before, after, _session = _journey(events_db, worker, clock, temp_home, cassette)
    confirmed = (after.get("staleness") or {}).get("last_confirmed")
    assert confirmed and confirmed != (before.get("staleness") or {}).get("last_confirmed")
