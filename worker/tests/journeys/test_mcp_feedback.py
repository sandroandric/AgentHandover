"""J-mcp: an agent runs a Skill through the MCP server and reports back.

The agent talks to the real FastMCP server over an in-memory MCP session
(the protocol a coding agent speaks), in a temp ``HOME``.  The Skill is
written, and later regenerated, by the worker's own ``ProcedureWriter``.

Asserted: the packet shows the step ids; a step reported by its packet id
and a step reported by its number land on those steps; exactly one
deviation is recorded, on the deviated step; the procedure carries the
run (``execution_stats``, ``last_confirmed``, one observed alternative);
the worker's trust advisor reads the same run; and a regeneration keeps
all of it.

Not covered here: a regeneration through the focus-recording pipeline
(``_process_focus_sessions_v2``), which needs a recorded model cassette
(WP-T4); its merge is pinned in ``test_procedure_writer.py``.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import anyio
import pytest

pytest.importorskip("mcp", reason="the MCP server needs the 'mcp' extra")

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from agenthandover_worker.mcp_server import build_mcp_server  # noqa: E402

SLUG = "export-weekly-report"


def _template(description: str) -> dict:
    """An SOP template as the generator hands it to ``write_procedure``."""
    return {
        "slug": SLUG,
        "title": "Export the weekly report",
        "description": description,
        "confidence_avg": 0.78,
        "episode_count": 1,
        "apps_involved": ["Google Chrome"],
        "steps": [
            {"step": "Open the analytics dashboard", "confidence": 0.9,
             "parameters": {"app": "Google Chrome", "location": "https://analytics.example.com",
                            "verify": "The dashboard shows this week"}},
            {"step": "Export the weekly table as CSV", "confidence": 0.9,
             "parameters": {"app": "Google Chrome", "verify": "A CSV file is downloaded"}},
            {"step": "Upload the CSV to the team drive", "confidence": 0.9,
             "parameters": {"app": "Google Chrome", "verify": "The file is listed in the drive"}},
        ],
    }


async def _agent_run() -> dict:
    """What a coding agent does with the Skill; returns the tool replies."""
    server = build_mcp_server()
    replies: dict = {}

    async def call(name: str, **args) -> dict:
        result = await session.call_tool(name, args)
        assert not result.isError, result.content
        return json.loads(result.content[0].text)

    async with create_connected_server_and_client_session(server) as session:
        replies["skill"] = await call("get_skill", slug=SLUG)
        start = await call("report_execution_start", slug=SLUG, agent_id="journey-agent")
        replies["start"] = start
        eid = start["execution_id"]
        replies["steps"] = [
            # By the packet id, no claim about the action.
            await call("report_step_result", execution_id=eid, step_id="step_1"),
            # By its number, deviated.
            await call(
                "report_step_result", execution_id=eid, step_id="2", status="deviated",
                actual_action="Copied the weekly table into a spreadsheet by hand",
                notes="The CSV export button was missing",
            ),
            # By the packet id, a paraphrase of the expected action.
            await call(
                "report_step_result", execution_id=eid, step_id="step_3",
                actual_action="Uploaded the CSV to the team drive",
            ),
        ]
        replies["complete"] = await call("report_execution_complete", execution_id=eid)
    return replies


def test_deviation_lands_on_its_step_and_survives_regeneration(
    events_db, worker, kb_root,
) -> None:
    journey = worker(events_db("focus_dailynews"))
    journey.procedure_writer.write_procedure(
        _template("Weekly analytics export"), source="focus", source_id="session-1",
    )

    replies = anyio.run(_agent_run)

    # The packet prints the ids the agent reports.
    packet = replies["skill"]["packet"]
    assert "1. [step_1] Open the analytics dashboard" in packet
    assert "2. [step_2] Export the weekly table as CSV" in packet
    assert replies["start"]["expected_steps"] == ["step_1", "step_2", "step_3"]

    # Reported by id and by number: each lands on its step.
    assert [(r["step_id"], r["deviated"]) for r in replies["steps"]] == [
        ("step_1", False), ("step_2", True), ("step_3", False),
    ]
    assert replies["complete"]["final_status"] == "deviated"
    assert replies["complete"]["deviations"] == 1

    proc = journey.knowledge_base.get_procedure(SLUG)
    alternatives = proc["observed_alternatives"]
    assert [(a["step_id"], a["observed_action"]) for a in alternatives] == [
        ("step_2", "Copied the weekly table into a spreadsheet by hand"),
    ]
    assert alternatives[0]["detail"] == "The CSV export button was missing"
    stats = proc["execution_stats"]
    assert (stats["total"], stats["deviated"], stats["deviations"]) == (1, 1, 1)
    assert proc["staleness"]["last_confirmed"] == stats["last_run"]
    assert proc["execution_confidence"] == 0.5  # one deviated run counts half
    assert proc["confidence_avg"] == 0.78  # synthesis grounding is untouched

    # The worker (a different process in production) reads the same run.
    assert journey.trust_advisor._get_execution_stats(SLUG)["total"] == 1
    with closing(sqlite3.connect(journey.db_path)) as conn:
        assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
    assert not (kb_root / "observations" / "runs.db").exists()

    # Regeneration: the worker writes the Skill again from a new template.
    journey.procedure_writer.write_procedure(
        _template("Weekly analytics export, regenerated"),
        source="focus", source_id="session-2",
    )
    regenerated = journey.knowledge_base.get_procedure(SLUG)
    assert regenerated["description"] == "Weekly analytics export, regenerated"
    assert regenerated["observed_alternatives"] == alternatives
    assert regenerated["execution_stats"] == stats
    assert regenerated["execution_confidence"] == 0.5
    assert regenerated["staleness"]["last_confirmed"] == stats["last_run"]
    assert regenerated["lifecycle_state"] == proc["lifecycle_state"]
    # The alternative still points at the step it was observed on.
    step_2 = next(s for s in regenerated["steps"] if s["step_id"] == "step_2")
    assert step_2["action"] == "Export the weekly table as CSV"
