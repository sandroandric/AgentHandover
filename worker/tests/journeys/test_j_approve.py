"""J-approve: the app's "Approve for agents" turns the focus draft into a
Skill Claude Code can run.

After J-focus's run leaves the draft, the journey writes the one trigger
the app writes (``lifecycle-promote`` to ``agent_ready``) and runs one
tick.  The main loop's idle poll backoff is not driven here (the driver
ticks on demand); its ceiling is pinned in ``test_trigger_spool.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agenthandover_worker import trigger_spool
from agenthandover_worker.skill_render import description, render_packet
from journeys.focus_run import approve_for_agents, start_focus_run, tick_until_skill


@pytest.mark.parametrize("adapter", ["openclaw", "claude-skill"])
def test_approve_for_agents_walks_to_agent_ready_and_writes_the_command(
    adapter, events_db, worker, clock, temp_home, cassette,
) -> None:
    run = start_focus_run(
        events_db=events_db, worker=worker, clock=clock,
        temp_home=temp_home, cassette=cassette, adapter=adapter,
    )
    draft = tick_until_skill(run, clock)
    slug = draft["id"]
    commands = temp_home.path / ".claude" / "commands"
    command = commands / f"ah-{slug}.md"

    assert draft["lifecycle_state"] == "draft"
    assert not command.exists()

    tick = approve_for_agents(run, clock, slug)

    assert tick["triggers_consumed"] == 1
    assert trigger_spool.pending(run.worker.state_dir) == []
    proc = run.worker.knowledge_base.get_procedure(slug)
    assert proc["lifecycle_state"] == "agent_ready"
    walked = [(h["from_state"], h["to_state"]) for h in proc["lifecycle_history"]]
    assert walked[-3:] == [
        ("draft", "reviewed"), ("reviewed", "verified"), ("verified", "agent_ready"),
    ]
    # The approval did not fork the procedure (no ``unknown.json`` beside it).
    procedure_files = sorted(
        p.name for p in (run.worker.knowledge_base.root / "procedures").glob("*.json")
    )
    assert procedure_files == [f"{slug}.json"]

    # Claude Code's /ah-<slug> is the packet; the index lists it with its description.
    assert command.read_text(encoding="utf-8") == render_packet(proc, target="claude-command")
    index = (commands / "agenthandover-list.md").read_text(encoding="utf-8")
    assert f"- **{proc['title']}** (`/ah-{slug}`): {description(proc)}" in index

    if adapter == "claude-skill":
        skill = temp_home.path / ".claude" / "skills" / slug / "SKILL.md"
        assert skill.read_text(encoding="utf-8") == render_packet(proc, target="claude-skill")
    assert _files_under(temp_home.path / ".claude") >= {f"commands/ah-{slug}.md"}


def _files_under(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
