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

# WP-F12: the interrupted Monday is one observation now (the 40-minute
# call no longer splits it), so the expense report's passive SOP request
# changed: discovery waits on its recording.
DISCOVERY_MISSING_KEYS: tuple[str, ...] = (
    "69d22760a76f69b91aedb45db80086af6ec93f97ea3c39b0787395a2ff58125c",
)

# The calls the full journey makes that the cassette lacks, found by
# running it with AH_RECORD=1 AH_RECORD_UPSTREAM at a closed port: the ones
# whose request depends on no other missing answer.  The rest follow from
# these answers and cannot be keyed before they are recorded: the session
# linker's later batches and its YES/NO checks (they depend on the
# vectors), and after the approval the behavioural synthesis (it carries
# the style analysis's voice profile).  Before WP-F12 the closed-port run
# listed 6,153 distinct requests (5,811 of them the linker's per-pair
# embeddings); after it, 321 (the linker embeds each day's new intents in
# one batch and asks at most 10 new YES/NO pairs a day).
# Record with AH_RECORD=1 on the test Mac until the journey misses nothing.
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
    # The profile's embedding, from the third summary on.
    "b3d99754909c847b4c5ed40c36cdb5a2dcfae8e9334cadd55cea61134ed86d1e",  # 2026-09-02
    "3c6d501fd93162728acfb723d18cdc59f476039e26319382accecac7e7f81995",  # 2026-09-03
    "791b20e2e8f00120b60e97b43650190bf5181e060393df94e13b714b991a6cac",  # 2026-09-04
    "1daf3a0681117c619361711da31af8d0a9db097916d64fa1e1fd1204c31deba7",  # 2026-09-07
    "31d1d551e73bb5c0809a3a82eac2478d679e9b7a7bcaab2f8b8b6de83d9a52f6",  # 2026-09-08
    "95c8476235dcb8ecc61a15680905a1e1061b731202f60b5c57494df5283af51d",  # 2026-09-09
    "26ac105e0a62cc5d3f57b2bb83e006f579c518f7219a7470ea973012e593bd97",  # 2026-09-10
    "1714af954f356a4c4cf5180bda4d824dcdee7ec795f9bfa82dcb3121d947c153",  # 2026-09-11
    "8b6518a3da480abe5d7804a55aeb31e19e83e4dd6a1b88385d8e12b386403291",  # 2026-09-14
    "ac2afb49de4766ed27ba01e8fc06890b836c80aa33482336ed1ec0b5d3897840",  # 2026-09-15
    "86511cbc66dd6600c4d08a66bc2c2f0bfa4b2079098660693d04f0218e7c5d83",  # 2026-09-16
    "5fe84b91b7dd60ff7104659f5675eb77ea676ab38601d15be26e5ae13ec364e9",  # 2026-09-17
    "192aaf0f39f1a1240a4259b91dc4431e311ead21e6accfdb8e78e66d2be7a5fb",  # 2026-09-18
    # The session linker's first batch of intent embeddings (day 3).
    "c4ac6160207dd102c5ceb5338b3b934f7d612768ba1acfb4e692f476683338dc",
    # The approvals' calls (the procedures' embeddings, the style analysis)
    # follow the expense report's discovery answer
    # (DISCOVERY_MISSING_KEYS) and the approval at the end: they are keyed
    # once that is recorded.
)
_ENTRIES = load_entries(CASSETTE) if CASSETTE.exists() else {}
needs_discovery_recording = pytest.mark.skipif(
    any(key not in _ENTRIES for key in DISCOVERY_MISSING_KEYS)
    and os.environ.get("AH_RECORD") != "1",
    reason=(
        f"cassettes/{NAME}.jsonl lacks the expense report's passive SOP request "
        "now that the interrupted Monday is one observation: "
        + ", ".join(k for k in DISCOVERY_MISSING_KEYS if k not in _ENTRIES)
        + "; record with AH_RECORD=1 on the test Mac"
    ),
)
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


@needs_discovery_recording
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


@needs_discovery_recording
def test_the_drafts_carry_dom_anchors(passive_drafts: Replay) -> None:
    """WP-F12: each draft's steps are anchored, model-free, to the elements
    its runs clicked (``workflow_mining.anchor_draft_steps``), each anchor
    resting on the recorded clicks."""
    for draft in _drafts(passive_drafts.worker).values():
        anchored = [s for s in draft["sop_json"]["steps"] if isinstance(s.get("anchor"), dict)]
        assert anchored
        for step in anchored:
            assert step["anchor"]["selector"] and step["evidence"]["click_ids"]


@needs_discovery_recording
@needs_daily_recording
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
