"""J-ledger: what three runs of one routine teach, slot by slot.

Skeleton: J-focus's recording plus two more runs of the same routine (one
skipping an item, one diverging) and a micro-question answered through the
app's answer file.  It waits on the learning ledger (WP-L1..L4) and on the
two extra run fixtures; the test fails at the missing fixture until then.
"""

from __future__ import annotations

import pytest

from journeys.focus_run import start_focus_run, tick_until_skill

RUNS = ("focus_dailynews", "focus_dailynews_skip", "focus_dailynews_diverge")


@pytest.mark.xfail(strict=True, reason=(
    "waits on WP-L1..WP-L4 (the ledger, micro-questions and slot readiness) "
    "and the focus_dailynews_skip / focus_dailynews_diverge fixtures"
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

    ledger = run.worker.knowledge_base.get_procedure(slug)["ledger"]
    criteria = ledger["slots"]["selection_criteria"]
    assert criteria["status"] == "hypothesis"
    assert len({e["instance_id"] for e in criteria["evidence"]}) == 3
    [branch] = ledger["slots"]["branches"]["value"]
    assert branch["condition"]
    assert ledger["slots"]["skipped_item"]["status"] == "unknown"
    assert ledger["readiness"]["computed_from"] == "slots"
