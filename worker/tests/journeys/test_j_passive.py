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
does, and the user approves each draft from the review queue on the tick it
appears: ``[sop] auto_approve`` is off by default, so a passive Skill
reaches the knowledge base only by approval, and live recognition records
the later runs of an approved Skill only.  The cassette lacks the model
calls of both (the daily pass's session links and embeddings, the
approval's embeddings and style analysis), so it waits on a recording.
What the drafts show before either is pinned now, on the same replay
without the daily pass (:func:`passive_drafts`).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Iterator

import pytest

from agenthandover_worker import trigger_spool
from journeys.cassette import CASSETTE_DIR, load_entries
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

# The calls the full journey makes that the cassette lacks, found by
# running it with AH_RECORD=1 AH_RECORD_UPSTREAM at a closed port (5,889
# distinct requests): the ones whose request depends on no other missing
# answer.  The rest follow from these answers and cannot be keyed before
# they are recorded: the session linker's other embeddings (it embeds each
# intent once; at a closed port it retried every pair, 5,810 keys) and its
# YES/NO checks (they depend on the vectors), the profile's embeddings and
# the behavioural synthesis (both carry the style analysis's voice
# profile).  Record with AH_RECORD=1 on the test Mac until the journey
# misses nothing.
DAILY_AND_APPROVAL_MISSING_KEYS: tuple[str, ...] = (
    # The daily summary's embedding, one per day with work (stages.daily).
    "c8fc20b77844f04cd4f85e7b8bf38307e9b696045b329fd10515608ac16a0da3",  # 2026-08-31
    "1ee5d88c81a32ba356951a14a890bf5387ff5b7bae5ba035420a9a33fd5c5da3",  # 2026-09-01
    "ba64b47a097a9f3adcb5c95fdcbd8d07194d58a3bb928a25395b388ae6746313",  # 2026-09-02
    "d9600a55ffffdecd5e17019a858be4e753499dec6aac68fee40dcd5c88218a99",  # 2026-09-03
    "034105419c8d12db13b40e4ee1628f2aaf02faf623dbb5e5fa88f74f97b5eb8b",  # 2026-09-04
    "74410f4a07c147c131349fbd5b5e366edb054d37bde1cac04c62f62e906a2690",  # 2026-09-07
    "5c3440624e5bb8874cc7f8da96c91b3e4e8c900df9fb1d413a795290084c0c68",  # 2026-09-08
    "f720cd90f0e4ec7a829cb5cf5ab437c6f178c01a9aa403143406dbd37a41496c",  # 2026-09-09
    "32b2ce96027da9a8f2f28c304d8fb648b2efa83787f1ddb8cf82f5460b1bc660",  # 2026-09-10
    "e7d7b2292eb2fae335d16c7e569069aebfd2d193e8eeb8869f0052fc6d38f86b",  # 2026-09-11
    "6d8626e0f1609575c06f2ec727a405654b1b619ef0f498e1224536ed08caf1f8",  # 2026-09-14
    "29e607e87c3e73567ec314bfbbfc935ba9af1d6d7621538e80fba92a9daee0fa",  # 2026-09-15
    "5caf1fb4dc0a3834e8f7b8067ede5477d8fef09b03096abc4afb209424a8b769",  # 2026-09-16
    "ec62ca70fe0ab296d8549e7eb0f59e1e4c7954f9050e48d477b2e2732dfb09aa",  # 2026-09-17
    "76bfe871abb86b57e4b25636494aa518917fa44276e3f9ef7263e67cea553c87",  # 2026-09-18
    # The session linker's first intent embedding (stages.daily, day 3).
    "783426732123516105e760924f9131a055e04ba39190a7b17a0805d68a9a8a85",
    # Approving the Stripe payout draft: the procedure's two embeddings
    # and the style analysis.
    "8424c83987c5f7c8a0599005b5d73271d075e4d6871ac82d25772ecb680a35be",
    "bf4397fa3455f661c7c73595e745ff08f70488bf7717337363330eeaf522fc3f",
    "4c60822764a168f68a1ae59be1b108e1b51645baee8a271a86fa872abdb8404d",
    # Approving the expense report draft: the same three.
    "0b11da012e18597113955b693298cbbfc7593939fe3d5f75f4572c42e2055c39",
    "f6e7cb68c776642f4a09e71a5989a6c44b00aeb1b42a3ee5e8faa6ebefacdfd5",
    "fb186f03f006a5b0fa0a64d0abb937bea2fc37e42c173efd52451ee45dff92ab",
)
_ENTRIES = load_entries(CASSETTE) if CASSETTE.exists() else {}
needs_daily_recording = pytest.mark.skipif(
    any(key not in _ENTRIES for key in DAILY_AND_APPROVAL_MISSING_KEYS)
    and os.environ.get("AH_RECORD") != "1",
    reason=(
        f"cassettes/{NAME}.jsonl lacks the daily pass's and the approvals' "
        "model calls: "
        + ", ".join(k for k in DAILY_AND_APPROVAL_MISSING_KEYS if k not in _ENTRIES)
        + " (and the calls they unblock); record with AH_RECORD=1 on the test Mac"
    ),
)


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


@pytest.mark.xfail(strict=True, reason=(
    "passive steps carry no DOM anchor: sop_generator.generate_from_passive "
    "builds the template without a timeline (_vlm_sop_to_template(mode="
    "'passive')), the passive prompt numbers frames per demonstration, and "
    "no passive frame carries click_targets (only focus_processor joins "
    "ClickIntents to frames), so _anchor_for_step has neither cited frames "
    "nor clicks; how passive steps are grounded is the lead's decision"
))
def test_the_drafts_carry_dom_anchors(passive_drafts: Replay) -> None:
    for draft in _drafts(passive_drafts.worker).values():
        assert any(isinstance(s.get("anchor"), dict) for s in draft["sop_json"]["steps"])


@needs_daily_recording
def test_three_weeks_discover_exactly_the_two_planted_routines(local_tz) -> None:
    """The whole journey: three weeks through the worker's tick with its
    daily pass, the user approving each draft as it appears."""
    local_tz(MANIFEST["timezone"])
    queue_sizes: list[int] = []
    approved: set[str] = set()

    def approve_new_drafts(worker: JourneyWorker) -> None:
        # The app's review queue: approve what it shows (the next tick
        # consumes the trigger).
        drafts = worker.db.get_generated_sops(status="draft")
        queue_sizes.append(len(drafts))
        for draft in drafts:
            if draft["sop_id"] not in approved:
                approved.add(draft["sop_id"])
                trigger_spool.write_trigger(
                    worker.state_dir, "approve", {"sop_id": draft["sop_id"], "action": "approve"},
                )

    with _replay(daily_jobs=True, on_tick=approve_new_drafts) as run:
        # Discovery, and a digest with content for every day with work.
        assert run.report.failures(min_recall=1.0, min_precision=1.0, max_chimeras=0) == []
        assert run.report.days
        assert max(queue_sizes) <= MAX_REVIEW_QUEUE

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
        for proc in procs.values():
            assert proc["evidence"]["total_observations"] >= 3
            assert any(s.get("location", "").startswith("http") for s in proc["steps"])
