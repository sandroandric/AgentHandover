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

import sqlite3
import time
from contextlib import closing
from pathlib import Path

import pytest

from agenthandover_worker import trigger_spool
from agenthandover_worker.skill_render import description, render_packet
from journeys.focus_run import (
    approve_for_agents,
    needs_focus_sop_recording,
    start_focus_run,
    tick_until_skill,
)


@needs_focus_sop_recording
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
    # The approval read and wrote the procedures table (WP-S2); the file
    # is the row's export, byte for byte.
    with closing(sqlite3.connect(run.worker.db_path)) as conn:
        rows = conn.execute(
            "SELECT slug, lifecycle_state, procedure_json FROM procedures"
        ).fetchall()
    assert [row[:2] for row in rows] == [(slug, "agent_ready")]
    export = run.worker.knowledge_base.root / "procedures" / f"{slug}.json"
    assert export.read_text(encoding="utf-8") == rows[0][2]

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


@needs_focus_sop_recording
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
        **run.worker.export_kwargs(),
    )

    assert exported == 0
    assert run.worker.knowledge_base.get_procedure(slug)["lifecycle_state"] == "draft"
    assert not (skills / slug).exists()
    assert not [p for p in openclaw.rglob("*") if slug in p.name]

    approve_for_agents(run, clock, slug)

    assert (skills / slug / "SKILL.md").is_file()


@needs_focus_sop_recording
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


# --- WP-C7: approving while the inference thread runs a long model call ---

# The model call the approval overlaps is held this long by the cassette
# (``/__cassette/slow``, first /api/generate request only).
SLOW_CALL_SECONDS = 6.0
# The app's approval must be done this soon after its trigger.
APPROVE_WITHIN_SECONDS = 2.0
SEEDED_SLUG = "check-domain-availability"


def _seed_draft(worker) -> str:
    """A draft Skill waiting for approval (the review record and the
    procedure a finished recording leaves), from a recorded SOP answer."""
    from agenthandover_worker.procedure_schema import sop_to_procedure
    from fixtures.vlm_answers import generated_sop_template

    proc = sop_to_procedure(generated_sop_template("check_domain_availability"))
    proc.update({
        "id": SEEDED_SLUG, "slug": SEEDED_SLUG, "title": "Check Domain Availability",
        "source": "focus", "lifecycle_state": "draft",
    })
    worker.knowledge_base.save_procedure(proc)
    worker.db.save_generated_sop(
        slug=SEEDED_SLUG, title=proc["title"], source="focus", sop_template=proc,
        confidence=proc["confidence_avg"], source_id="focus-earlier", auto_approve=False,
    )
    return SEEDED_SLUG


def _wait_for(condition, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.02)


def _approve_during_the_slow_call(worker, tape, long_job: str) -> None:
    """Start the inference thread, let *long_job* reach its slow model call,
    approve the seeded draft for agents, and check the approval finished
    within :data:`APPROVE_WITHIN_SECONDS` while the call still runs."""
    slug = _seed_draft(worker)
    tape.set_slow(SLOW_CALL_SECONDS, n=1, path="/api/generate")
    runner = worker.start_inference()
    worker.run_tick()
    # The slow request has reached the cassette (its fault budget is spent).
    _wait_for(lambda: tape.fault_state()["slow_seconds"] == 0.0)
    assert runner.status()["running"] == long_job

    trigger_spool.write_trigger(
        worker.state_dir, "lifecycle-promote",
        {"procedure_slug": slug, "to_state": "agent_ready"},
    )
    started = time.monotonic()
    tick = worker.run_tick()
    elapsed = time.monotonic() - started

    assert tick["triggers_consumed"] == 1
    assert elapsed < APPROVE_WITHIN_SECONDS, f"approval took {elapsed:.2f} s"
    proc = worker.knowledge_base.get_procedure(slug)
    assert proc["lifecycle_state"] == "agent_ready"
    skill = worker.cfg.export.skills_dir / slug / "SKILL.md"
    assert skill.read_text(encoding="utf-8") == render_packet(proc, target="claude-skill")
    # The heartbeat of that tick: the long call still runs, and the
    # approval's model work (embedding, style) waits behind it at P0.
    inference = worker.status()["inference"]
    assert inference["running"] == long_job
    assert inference["annotation_backlog"] >= 0
    assert inference["lower_capture_cadence"] is worker.annotation_budget.asks_lower_capture(
        inference["annotation_backlog"],
    )
    assert f"P0 enrich-procedure:{slug}" in inference["queued"]
    assert inference["queued"][0].startswith("P0 ")

    # Stop before the queued enrichment runs (the cassette holds no answer
    # for it); the long job finishes its remaining calls at full speed.
    tape.reset_faults()
    assert runner.stop(timeout=60)


def test_approval_completes_while_passive_annotation_waits_for_the_model(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    tape = cassette("passive_volatile_hour")
    fixture = events_db("passive_volatile_hour", rebase_to=clock.now())
    _approve_during_the_slow_call(
        worker(fixture, adapter="claude-skill"), tape, "P2 passive-annotation",
    )


@needs_focus_sop_recording
def test_approval_completes_while_a_focus_recording_is_processed(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    run = start_focus_run(
        events_db=events_db, worker=worker, clock=clock,
        temp_home=temp_home, cassette=cassette, adapter="claude-skill",
    )
    _approve_during_the_slow_call(run.worker, run.tape, "P0 focus")
