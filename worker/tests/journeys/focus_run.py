"""The focus-recording run shared by J-focus, J-approve, J-mcp and
J-ollama-down: load ``focus_dailynews``, write the app's "stopped" signal,
tick the worker on the ``focus_dailynews`` cassette until the Skill (the v3
procedure in the knowledge base) exists.

When the worker asks its post-recording questions (``focus-questions.json``
with status ``pending``), the run answers them the way the app's "Skip"
button does: it rewrites the file with status ``skipped``, so the worker
merges each question's default.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from agenthandover_worker import trigger_spool
from journeys.cassette import Cassette
from journeys.conftest import TempHome
from journeys.driver import Clock, JourneyWorker
from journeys.fixture_loader import LoadedFixture

FOCUS_CASSETTE = "focus_dailynews"
TICK_SECONDS = 30


@dataclass
class FocusRun:
    fixture: LoadedFixture
    worker: JourneyWorker
    tape: Cassette
    ticks: list[dict] = field(default_factory=list)
    questions_skipped: int = 0
    # How many model calls the cassette had served when the Skill was
    # generated (SOP, synthesis, questions): the focus processor annotates
    # the keyframes the SOP did not need (``annotate_remaining``) only in
    # the next tick's annotation phase.
    served_at_skill: int | None = None

    def procedures(self) -> list[dict]:
        return self.worker.knowledge_base.list_procedures()


def skip_pending_questions(state_dir: Path) -> int:
    """Answer a pending ``focus-questions.json`` with the app's Skip; returns
    how many questions it held (0 when there was nothing pending)."""
    path = state_dir / "focus-questions.json"
    if not path.is_file():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "pending":
        return 0
    data["status"] = "skipped"
    path.write_text(json.dumps(data), encoding="utf-8")
    return len(data.get("questions") or [])


def start_focus_run(
    *,
    events_db: Callable[..., LoadedFixture],
    worker: Callable[..., JourneyWorker],
    clock: Clock,
    temp_home: TempHome,
    cassette: Callable[..., Cassette],
    adapter: str = "openclaw",
    ollama_down_for: int = 0,
    journey: str = FOCUS_CASSETTE,
    tier: str = "standard",
) -> FocusRun:
    """Models, cassette, fixture and signal in place; no tick run yet.

    ``ollama_down_for`` asks the cassette (``/__cassette/down``) to answer
    that many API requests with HTTP 503 before the worker starts.
    ``journey`` names the cassette and ``tier`` the RAM tier whose models
    the worker runs."""
    tape = cassette(journey, tier=tier)
    if ollama_down_for:
        with urllib.request.urlopen(f"{tape.host}/__cassette/down?n={ollama_down_for}") as reply:
            assert json.load(reply)["down"] == ollama_down_for
    fixture = events_db("focus_dailynews", rebase_to=clock.now())
    fixture.write_focus_signal(temp_home.data_dir)
    run = FocusRun(fixture=fixture, worker=worker(fixture, adapter=adapter), tape=tape)
    processor = run.worker.focus_processor
    annotate_remaining = processor.annotate_remaining

    def remaining_after_the_skill(db):
        # Every tick's annotation phase calls it; the first call with
        # keyframes left over is the tick after the SOP was generated.
        if run.served_at_skill is None and processor._remaining:
            run.served_at_skill = len(tape.served)
        return annotate_remaining(db)

    processor.annotate_remaining = remaining_after_the_skill
    return run


def tick_until_skill(run: FocusRun, clock: Clock, *, max_ticks: int = 6) -> dict:
    """Tick until the knowledge base holds the procedure; returns it."""
    state_dir = run.worker.state_dir
    for _ in range(max_ticks):
        run.ticks.append(run.worker.run_tick())
        run.questions_skipped += skip_pending_questions(state_dir)
        procedures = run.procedures()
        if procedures:
            assert len(procedures) == 1, [p.get("id") for p in procedures]
            return procedures[0]
        clock.advance(seconds=TICK_SECONDS)
    raise AssertionError(
        f"no procedure after {max_ticks} ticks: {run.ticks!r}; "
        f"state dir holds {sorted(p.name for p in state_dir.iterdir())}"
    )


def approve_for_agents(run: FocusRun, clock: Clock, slug: str) -> dict:
    """What the app's "Approve for agents" does: one ``lifecycle-promote``
    trigger in the spool, then the worker's next tick.  Returns that tick."""
    trigger_spool.write_trigger(
        run.worker.state_dir, "lifecycle-promote",
        {"procedure_slug": slug, "to_state": "agent_ready"},
    )
    clock.advance(seconds=TICK_SECONDS)
    tick = run.worker.run_tick()
    run.ticks.append(tick)
    return tick
