"""J-passive: three weeks of passive capture with two planted routines.

The journey runs on the replay's path (:func:`journeys.replay.replaying`):
the fixture's rows reach ``events.db`` as the virtual clock passes their
recorded timestamps, the worker's own tick runs every 15 minutes (passive
discovery on its 2-hour cadence), and the models answer from the
``passive_3weeks`` cassette, recorded on that path.  The process's local
time zone is the fixture's (UTC): workflow mining reads the local weekday
and minute from it, as it reads the user's on a Mac.

The full journey (:func:`test_three_weeks_discover_exactly_the_two_planted_routines`)
also runs the worker's once-per-day pass through the tick, as ``main()``
does, and the user approves the drafts from the review queue at the end of
the three weeks, as a real user who looks at the queue now and then:
``[sop] auto_approve`` is off by default, so a passive Skill reaches the
knowledge base only by approval, and a draft learns from its later runs
meanwhile as evidence only (WP-F12, ``workflow_mining.learn_on_draft``).
The cassette lacks the model calls of the daily pass and the approvals, so
it waits on a recording.  What the drafts show before either is pinned on
the same replay without the daily pass (:func:`passive_drafts`); the
learning itself is pinned model-free in
``test_workflow_observations.test_drafts_learn_from_later_runs_until_approved_at_the_end``.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Iterator

import pytest

from agenthandover_worker import trigger_spool
from journeys.cassette import CASSETTE_DIR
from journeys.driver import JourneyWorker
from journeys.fixture_loader import read_labels, read_manifest
from journeys.replay import Replay, replaying

NAME = "passive_3weeks"
DAYS = 21
CASSETTE = CASSETTE_DIR / f"{NAME}.jsonl"
MANIFEST = read_manifest(NAME)
PLANTED = {r["slug"]: r for r in MANIFEST["planted_routines"]}
WEEKLY = next(r for r in MANIFEST["planted_routines"] if r["cadence"] == "weekly")
# The review queue a three-week run may leave the user (PLAN.md, J-passive).
MAX_REVIEW_QUEUE = 3



def _replay(*, daily_jobs: bool, on_tick=None):
    start = datetime.fromisoformat(MANIFEST["recorded_start"]).replace(
        hour=0, minute=0, second=0, microsecond=0)
    return replaying(
        fixture=NAME, labels=read_labels(NAME),
        start=start, end=start + timedelta(days=DAYS),
        cassette=CASSETTE,
        # The tier the cassette is recorded on, not the RAM tier of the
        # machine running the test (conftest's rule).
        vlm={"tier": "recommended"},
        daily_jobs=daily_jobs, on_tick=on_tick,
    )


@pytest.fixture(scope="module")
def utc_local_time() -> Iterator[None]:
    """The process's local time zone is the fixture's for the module."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TZ", MANIFEST["timezone"])
        time.tzset()
        yield
    time.tzset()


@pytest.fixture(scope="module")
def passive_drafts(utc_local_time) -> Iterator[Replay]:
    """Three weeks of passive discovery alone (no daily pass, no
    approval): every model call is in the cassette."""
    with _replay(daily_jobs=False) as run:
        yield run


def _drafts(worker: JourneyWorker) -> dict[str, dict]:
    """The review queue's drafts by slug, each with its template."""
    return {
        d["slug"]: worker.db.get_generated_sop(d["sop_id"])
        for d in worker.db.get_generated_sops(status="draft")
    }


def _routine_of(run: Replay) -> dict[str, str]:
    """Skill slug -> the planted routine its observations are (the replay's
    scoring: a true positive spans exactly one routine)."""
    return {c.slug: c.routine for c in run.report.candidates if c.verdict == "true_positive"}


def test_discovery_drafts_exactly_the_two_planted_routines(passive_drafts: Replay) -> None:
    """No chimera and no one-off: one draft per planted routine, with the
    weekly one's weekday, its runs as observations and URLs for steps."""
    run = passive_drafts
    assert run.report.failures(min_recall=1.0, min_precision=1.0, max_chimeras=0) == []
    drafts = _drafts(run.worker)
    routine_of = _routine_of(run)
    assert sorted(routine_of) == sorted(drafts)
    assert sorted(routine_of.values()) == sorted(PLANTED)
    assert len(drafts) <= MAX_REVIEW_QUEUE

    by_routine = {routine_of[slug]: d["sop_json"] for slug, d in drafts.items()}
    weekly = by_routine[WEEKLY["slug"]]
    assert weekly["recurrence"]["pattern"] == "weekly"
    assert weekly["recurrence"]["weekday"] == WEEKLY["local_weekday"]
    for template in by_routine.values():
        assert len(template["observations"]) >= 3
        assert any(
            str(s["parameters"].get("location", "")).startswith("http")
            for s in template["steps"]
        )


def test_the_drafts_carry_dom_anchors(passive_drafts: Replay) -> None:
    """WP-F12: each draft's steps are anchored, model-free, to the elements
    its runs clicked (``workflow_mining.anchor_draft_steps``), each anchor
    resting on the recorded clicks."""
    for draft in _drafts(passive_drafts.worker).values():
        anchored = [s for s in draft["sop_json"]["steps"] if isinstance(s.get("anchor"), dict)]
        assert anchored
        for step in anchored:
            assert step["anchor"]["selector"] and step["evidence"]["click_ids"]


def test_three_weeks_discover_exactly_the_two_planted_routines(local_tz) -> None:
    """The whole journey: three weeks through the worker's tick with its
    daily pass; the drafts wait in the review queue, learning from their
    later runs, until the user approves them at the end."""
    local_tz(MANIFEST["timezone"])
    queue_sizes: list[int] = []

    def watch_queue(worker: JourneyWorker) -> None:
        queue_sizes.append(len(worker.db.get_generated_sops(status="draft")))

    with _replay(daily_jobs=True, on_tick=watch_queue) as run:
        # Discovery, and a digest with content for every day with work.
        assert run.report.failures(min_recall=1.0, min_precision=1.0, max_chimeras=0) == []
        assert run.report.days
        assert max(queue_sizes) <= MAX_REVIEW_QUEUE
        assert run.worker.knowledge_base.list_procedures() == []  # nothing approved yet

        # The user approves what the review queue shows; the next tick
        # consumes the triggers.
        for draft in run.worker.db.get_generated_sops(status="draft"):
            trigger_spool.write_trigger(
                run.worker.state_dir, "approve", {"sop_id": draft["sop_id"], "action": "approve"},
            )
        run.worker.run_tick()

        kb = run.worker.knowledge_base
        routine_of = _routine_of(run)
        assert sorted(p["id"] for p in kb.list_procedures()) == sorted(routine_of)
        assert sorted(routine_of.values()) == sorted(PLANTED)
        procs = {routine_of[slug]: kb.get_procedure(slug) for slug in routine_of}

        weekly = procs[WEEKLY["slug"]]
        assert weekly["recurrence"]["pattern"] == "weekly"
        assert weekly["recurrence"]["weekday"] == WEEKLY["local_weekday"]
        # WP-L3: its lead-in varies, so its trigger is the local clock the
        # runs share (model-free pin: test_workflow_observations.py).
        assert weekly["knowledge"]["trigger"]["status"] == "hypothesis"
        assert weekly["learned"]["trigger"]["weekday"] == WEEKLY["local_weekday"]
        assert weekly["learned"]["trigger"]["description"] == "Most Mondays around 10:30"
        # The daily routine's draft learned from every later run (WP-F12).
        daily = next(p for slug, p in procs.items() if slug != WEEKLY["slug"])
        assert len(daily["evidence"]["observations"]) == 15
        assert daily["learned"]["done"]["support"] == "15 of 15 runs"
        for proc in procs.values():
            assert proc["evidence"]["total_observations"] >= 3
            assert any(s.get("location", "").startswith("http") for s in proc["steps"])
