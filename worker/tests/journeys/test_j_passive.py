"""J-passive: three weeks of passive capture with two planted routines.

Skeleton: it waits on the ``passive_3weeks`` fixture and the workflow-key
mining of WP-M3/WP-M4, and on the daily batch, which the driver cannot run
until WP-S6 splits ``main()`` (see ``driver.py``).  The assertions are the
journey's contract; the test fails at the missing fixture until then.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from journeys.conftest import JOURNEY_MODELS
from journeys.driver import PASSIVE_DISCOVERY_INTERVAL

DAYS = 21


@pytest.mark.xfail(strict=True, reason=(
    "waits on WP-M3/WP-M4 (workflow-key mining, the passive_3weeks fixture "
    "with two planted routines) and WP-S6 (daily batch driven by run_tick)"
))
def test_three_weeks_discover_exactly_the_two_planted_routines(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    temp_home.configure(vlm=JOURNEY_MODELS)
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
    for p in candidates:
        assert p["evidence"]["total_observations"] >= 3
        assert any(s.get("location", "").startswith("http") for s in p["steps"])
        assert any(isinstance(s.get("anchor"), dict) for s in p["steps"])
    assert len(journey.db.list_generated_sops(status="draft")) <= 3
