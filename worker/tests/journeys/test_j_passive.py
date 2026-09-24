"""J-passive: three weeks of passive capture with two planted routines.

Skeleton: the ``passive_3weeks`` fixture and its replay scorer landed with
WP-M4 (``test_passive_replay.py``); WP-M3's workflow-key mining (the
observation log, local-time recurrence, value and the review-queue cap) is
proved model-free on the same fixture in ``test_workflow_observations.py``.
The journey still waits on the ``passive_3weeks`` cassette (SOP generation
through real models) and on the daily batch, which the driver cannot run
until WP-S6 splits ``main()`` (see ``driver.py``).  The assertions are the
journey's contract; the test fails at the missing cassette until then.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest

from journeys.cassette import CASSETTE_DIR, load_entries
from journeys.driver import PASSIVE_DISCOVERY_INTERVAL

DAYS = 21

# The task_segmenter.label_cluster calls this run makes since the reasoner
# has its own daily budget (the VLM fallback queue it used to share filled
# up and they were skipped); new to it besides the calls the xfail waits on.
REASONER_BUDGET_MISSING_KEYS: tuple[str, ...] = (
    "0002042db6069791754e4ad62dcf8e812579f2e30832735252238447e34ab334",
    "131e1a805ed0bb011ffe7700abd25fabd50f210243df582d617b9c6f90ae23ff",
    "406ec3685f01dc7b8576c210840cf09bf1764d08228ba4e4ff1a17789ba96421",
    "7fb3f1d49dbd46aa24c4796fde8c41930bb44a1990b3b20f3ec7cb094ad052e4",
    "988564cf2b7f24343d9da09cda5524dce1d32ea2eb8c8dce70cbe8b79c21438b",
)
_PATH = CASSETTE_DIR / "passive_3weeks.jsonl"
_ENTRIES = load_entries(_PATH) if _PATH.exists() else {}
needs_label_recording = pytest.mark.skipif(
    any(key not in _ENTRIES for key in REASONER_BUDGET_MISSING_KEYS)
    and os.environ.get("AH_RECORD") != "1",
    reason=(
        "cassettes/passive_3weeks.jsonl lacks the cluster-label calls the "
        "reasoner's own budget no longer skips: "
        + ", ".join(k for k in REASONER_BUDGET_MISSING_KEYS if k not in _ENTRIES)
        + " (and the calls they unblock); record with AH_RECORD=1 on the test Mac"
    ),
)


@needs_label_recording
@pytest.mark.xfail(strict=True, reason=(
    "waits on the passive_3weeks cassette and WP-S6 (daily batch driven by "
    "run_tick); the mining itself is pinned in test_workflow_observations.py"
))
def test_three_weeks_discover_exactly_the_two_planted_routines(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    cassette("passive_3weeks")
    fixture = events_db("passive_3weeks", rebase_to=clock.now() + timedelta(days=DAYS))
    journey = worker(fixture)
    end = clock.now() + timedelta(days=DAYS)
    while clock.now() < end:
        journey.run_tick()
        clock.advance(PASSIVE_DISCOVERY_INTERVAL)

    planted = fixture.manifest["planted_routines"]
    candidates = journey.knowledge_base.list_procedures()
    assert sorted(p["id"] for p in candidates) == sorted(r["slug"] for r in planted)
    weekly = next(r for r in planted if r["cadence"] == "weekly")
    proc = journey.knowledge_base.get_procedure(weekly["slug"])
    assert proc["recurrence"]["pattern"] == "weekly"
    assert proc["recurrence"]["weekday"] == weekly["local_weekday"]
    # WP-L3: its lead-in varies, so its trigger is the local clock the runs
    # share (model-free pin: test_workflow_observations.py).
    assert proc["knowledge"]["trigger"]["status"] == "hypothesis"
    assert proc["learned"]["trigger"]["weekday"] == weekly["local_weekday"]
    for p in candidates:
        assert p["evidence"]["total_observations"] >= 3
        assert any(s.get("location", "").startswith("http") for s in p["steps"])
        assert any(isinstance(s.get("anchor"), dict) for s in p["steps"])
    assert len(journey.db.list_generated_sops(status="draft")) <= 3
