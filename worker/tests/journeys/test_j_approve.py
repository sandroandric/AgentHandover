"""J-approve: the app's "Approve for agents" turns the focus draft into a
Skill Claude Code can run.

After J-focus's run leaves the draft, the journey writes the one trigger
the app writes (``lifecycle-promote`` to ``agent_ready``) and runs one
tick.  The main loop's idle poll backoff is not driven here (the driver
ticks on demand); its ceiling is pinned in ``test_trigger_spool.py``.

Claude Code gets the Skill as ``~/.claude/skills/<slug>/SKILL.md`` (WP-P2)
and the ``/agenthandover-list`` command points at it; no ``ah-<slug>.md``
command is written any more.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agenthandover_worker import trigger_spool
from agenthandover_worker.skill_render import description, render_packet
from journeys.focus_run import approve_for_agents, start_focus_run, tick_until_skill


@pytest.mark.parametrize("adapter", ["openclaw", "claude-skill"])
def test_approve_for_agents_walks_to_agent_ready_and_writes_the_skill(
    adapter, events_db, worker, clock, temp_home, cassette,
) -> None:
    run = start_focus_run(
        events_db=events_db, worker=worker, clock=clock,
        temp_home=temp_home, cassette=cassette, adapter=adapter,
    )
    draft = tick_until_skill(run, clock)
    slug = draft["id"]
    commands = temp_home.path / ".claude" / "commands"
    skill = temp_home.path / ".claude" / "skills" / slug / "SKILL.md"

    assert draft["lifecycle_state"] == "draft"
    assert not skill.exists()

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

    # Claude Code's /<slug> is the packet; the index lists it with its description.
    assert skill.read_text(encoding="utf-8") == render_packet(proc, target="claude-skill")
    index = (commands / "agenthandover-list.md").read_text(encoding="utf-8")
    assert f"- **{proc['title']}** (`/{slug}`): {description(proc)}" in index
    assert _files_under(temp_home.path / ".claude") >= {
        f"skills/{slug}/SKILL.md", "commands/agenthandover-list.md",
    }
    if adapter == "openclaw":
        assert _files_under(temp_home.path / ".claude") == {
            f"skills/{slug}/SKILL.md", "commands/agenthandover-list.md",
        }


def test_auto_approved_draft_reaches_no_agent_until_approved_for_agents(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    """The export ``[sop] auto_approve`` runs right after a focus recording
    (``_focus_v2_export``, every adapter the worker has) writes the draft
    to no agent directory: every export goes through the hand-off gate.
    The same slug appears after promotion."""
    from agenthandover_worker import main as wm

    run = start_focus_run(
        events_db=events_db, worker=worker, clock=clock,
        temp_home=temp_home, cassette=cassette, adapter="claude-skill",
    )
    draft = tick_until_skill(run, clock)
    slug = draft["id"]
    skills = temp_home.path / ".claude" / "skills"
    openclaw = run.worker.sops_dir
    exported = wm._focus_v2_export(
        sop_templates=[draft], title=draft["title"], inference_time=0.0,
        **run.worker._export_kwargs(),
    )

    assert exported == 0
    assert run.worker.knowledge_base.get_procedure(slug)["lifecycle_state"] == "draft"
    assert not (skills / slug).exists()
    assert not [p for p in openclaw.rglob("*") if slug in p.name]

    approve_for_agents(run, clock, slug)

    assert (skills / slug / "SKILL.md").is_file()


def test_claude_skill_adapter_writes_only_the_skill_and_the_index_command(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    run = start_focus_run(
        events_db=events_db, worker=worker, clock=clock,
        temp_home=temp_home, cassette=cassette, adapter="claude-skill",
    )
    slug = tick_until_skill(run, clock)["id"]
    approve_for_agents(run, clock, slug)
    assert _files_under(temp_home.path / ".claude") == {
        f"skills/{slug}/SKILL.md", "commands/agenthandover-list.md",
    }


def _files_under(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
