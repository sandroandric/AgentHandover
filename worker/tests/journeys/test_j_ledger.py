"""J-ledger: what three runs of one routine teach, slot by slot.

Skeleton: J-focus's recording plus two more runs of the same routine (one
skipping an item, one diverging) and a micro-question skipped through the
app's answer file.  The ledger's schema (``proc["knowledge"]``, see
``agenthandover_worker/ledger.py``) landed with WP-L1; the learners that
fill it (WP-L2..L4) and the two extra run fixtures have not, so the test
fails at the missing fixture until then.
"""

from __future__ import annotations

import pytest

from agenthandover_worker.ledger import readiness
from journeys.focus_run import needs_focus_sop_recording, start_focus_run, tick_until_skill

RUNS = ("focus_dailynews", "focus_dailynews_skip", "focus_dailynews_diverge")


@needs_focus_sop_recording
@pytest.mark.xfail(strict=True, reason=(
    "waits on WP-L2..WP-L4 (contrast learning, triggers and divergence, "
    "micro-questions) and the focus_dailynews_skip / focus_dailynews_diverge fixtures"
))
def test_three_runs_fill_the_ledger_with_evidence(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    run = start_focus_run(
        events_db=events_db, worker=worker, clock=clock,
        temp_home=temp_home, cassette=cassette,
    )
    slug = tick_until_skill(run, clock)["id"]
    for name in RUNS[1:]:
        extra = events_db(name, rebase_to=clock.now())
        extra.write_focus_signal(temp_home.data_dir)
        run.worker.run_tick()

    proc = run.worker.knowledge_base.get_procedure(slug)
    knowledge = proc["knowledge"]
    # The selection criteria come from contrasting chosen with skipped
    # items across the three runs (WP-L2), each run cited.
    decisions = knowledge["decisions"]
    assert decisions["status"] == "hypothesis"
    assert decisions["source"] == "contrast"
    assert len({e["id"] for e in decisions["evidence"] if e["kind"] == "instance"}) == 3
    # The diverging run adds a branch (WP-L3).
    [branch] = proc["branches"]
    assert branch["condition"]
    # What starts the routine and how the user checks it recur across the
    # three runs (WP-L3; unit pins in test_run_signals.py).
    for slot in ("trigger", "verification"):
        assert knowledge[slot]["status"] == "hypothesis"
        assert len({e["id"] for e in knowledge[slot]["evidence"] if e["kind"] == "instance"}) == 3
    # The micro-question was skipped (WP-L4): no slot claims a user answer.
    assert not any(
        e["kind"] == "answer" for entry in knowledge.values() for e in entry["evidence"]
    )
    assert readiness(proc).ready
