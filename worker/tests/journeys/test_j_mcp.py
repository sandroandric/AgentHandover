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
from datetime import timedelta
from pathlib import Path

import pytest

from agenthandover_worker.execution_monitor import ExecutionMonitor, ExecutionStatus
from agenthandover_worker.skill_render import render_packet
from journeys.focus_run import (
    approve_for_agents,
    needs_focus_sop_recording,
    start_focus_run,
    tick_until_skill,
)
from journeys.packet import parse_packet

# Every test here starts from the focus_dailynews Skill, so it skips
# together with that cassette's SOP entry.
pytestmark = needs_focus_sop_recording

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


# Approved long ago and never given trust: the hand-off gate holds it back.
STALE_SLUG = "weekly-expense-report"


def _stale_observe_procedure(now) -> dict:
    return {
        "schema_version": "3.2.0", "id": STALE_SLUG, "slug": STALE_SLUG,
        "title": "Weekly expense report",
        "goal": "File the week's receipts as one expense report.",
        "outcome": "The report is submitted.",
        "steps": [{"step_id": "step_1", "action": "Open the expenses page"}],
        "lifecycle_state": "agent_ready",
        "constraints": {"trust_level": "observe", "guardrails": []},
        "staleness": {"last_observed": (now - timedelta(days=150)).isoformat()},
    }


def _journey(events_db, worker, clock, temp_home, cassette):
    run = start_focus_run(
        events_db=events_db, worker=worker, clock=clock,
        temp_home=temp_home, cassette=cassette,
    )
    slug = tick_until_skill(run, clock)["id"]
    kb = run.worker.knowledge_base
    kb.save_procedure(_stale_observe_procedure(clock.now()))
    approve_for_agents(run, clock, slug)
    before = kb.get_procedure(slug)
    session = asyncio.run(
        _agent_runs_skill(temp_home.path, slug, deviate="Opened the digest from Drafts instead"),
    )
    return kb, slug, before, kb.get_procedure(slug), session, run


def test_mcp_agent_gets_the_packet_and_reports_one_run(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    kb, slug, before, after, session, _run = _journey(events_db, worker, clock, temp_home, cassette)

    assert session.tools == TOOLS
    # Only the approved focus Skill is ready; the stale, observe-trust one
    # is neither listed nor installed for Claude Code.
    assert [s["slug"] for s in session.ready["ready"]] == [slug]
    assert not (temp_home.path / ".claude" / "skills" / STALE_SLUG).exists()
    [held] = session.ready["held_back"]
    assert held["slug"] == STALE_SLUG, held
    assert any("trust level is 'observe'" in r for r in held["reasons"]), held
    assert any(r.startswith("freshness") for r in held["reasons"]), held
    assert (temp_home.path / ".claude" / "skills" / slug / "SKILL.md").is_file()
    # get_skill hands the agent the same packet every target renders.
    assert session.skill["packet"] == render_packet(before, target="mcp")
    packet_ids = [s.step_id for s in parse_packet(session.skill["packet"]).steps()]
    assert session.step_result == {"recorded": True, "step_id": packet_ids[0], "deviated": True}
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
    kb, slug, before, _after, session, _run = _journey(events_db, worker, clock, temp_home, cassette)
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
    _kb, _slug, before, after, _session, _run = _journey(events_db, worker, clock, temp_home, cassette)
    confirmed = (after.get("staleness") or {}).get("last_confirmed")
    assert confirmed and confirmed != (before.get("staleness") or {}).get("last_confirmed")


def test_mcp_deviation_asks_the_user_in_the_next_digest(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    """WP-L5: the agent deviated at a step the Skill's decision (a
    hypothesis, from one recording) applies at.  The decision is no longer
    trusted: the ledger holds it unknown on the run's evidence, the next
    digest asks the user how they decide there, and the next agent is told
    to ask instead of assuming."""
    kb, slug, before, after, session, run = _journey(
        events_db, worker, clock, temp_home, cassette,
    )
    assert before["knowledge"]["decisions"]["status"] == "hypothesis"
    execution_id = session.started["execution_id"]
    decisions = after["knowledge"]["decisions"]
    assert decisions["status"] == "unknown" and decisions["source"] == "agent"
    assert decisions["evidence"] == [{"kind": "run", "id": execution_id}]

    # The daily batch asks what the run raised, as a review card within the
    # weekly budget the focus recording's questions share, then writes the
    # digest, as the worker writes it and the app reads it.
    from agenthandover_worker import questions
    from agenthandover_worker.daily_digest import DigestGenerator

    questions.refresh(
        run.worker.db, kb, temp_home.data_dir, now=clock.now(),
        weekly_budget=run.worker.cfg.knowledge.weekly_question_budget,
    )
    step_id = parse_packet(session.skill["packet"]).steps()[0].step_id
    run_question = f"run:{slug}:{execution_id}:{step_id}"
    cards = json.loads((temp_home.data_dir / questions.MICRO_QUESTIONS_FILE).read_text())
    assert run_question in [card["id"] for card in cards["questions"]]
    day = clock.now().strftime("%Y-%m-%d")
    generator = DigestGenerator(kb, db=run.worker.db)
    generator.save_digest(generator.generate(day))
    saved = json.loads((kb.root / "observations" / "digests" / f"{day}.json").read_text())
    [section] = [s for s in saved["sections"] if s["title"] == "Questions for you"]
    [question] = [item for item in section["items"] if item["question_id"] == run_question]
    assert "Opened the digest from Drafts instead" in question["title"]
    # It is the Skill's decisions question: the general one is not asked.
    assert f"slot:{slug}:decisions" not in [
        row["question_id"] for row in run.worker.db.get_micro_questions()
    ]

    # The next agent gets the question in its packet, and the decision as
    # not yet known.
    packet = render_packet(after, target="mcp")
    assert question["title"] in packet
    assert (
        "- Decisions (how the user chooses between options): an agent's run did "
        "not go as the recordings suggested. Ask the user; do not assume."
    ) in packet
