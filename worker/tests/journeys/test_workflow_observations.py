"""Workflow observations over three weeks of passive capture (WP-M3).

The ``passive_3weeks`` fixture goes through the observation log the way
passive discovery drives it (a 24-hour window every two hours on the
fixture's own clock, each finished work session recorded once), with no
model in the loop: sessions, workflow keys, grouping, recurrence, value and
the review queue are all model-free.  The labels say what must come out:
exactly the two planted routines, each with every instance and nothing
from another task, the Monday routine weekly on Monday, the payout routine
daily.

What these tests do not cover is the rest of J-passive (SOP generation from
the promoted workflows through the cassette and the daily batch driven by
``run_tick``); see ``test_j_passive.py``.

WP-M5 adds live recognition on the same fixture: a Skill's session is
tagged while it runs and recorded as the Skill's observation at its end.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from agenthandover_worker import workflow_mining as wm
from agenthandover_worker.daily_digest import DigestGenerator
from agenthandover_worker.db import WorkerDB
from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.procedure_schema import sop_to_procedure
from journeys.fixture_loader import load_fixture, parse_timestamp, read_labels, read_manifest

NAME = "passive_3weeks"
TICK = timedelta(hours=2)
STRIPE = "stripe-payout-reconciliation"
EXPENSES = "weekly-expense-report"


def _unit_of(labels: dict) -> dict[str, str]:
    unit = {}
    for routine in labels["routines"]:
        for inst in routine["instances"]:
            for eid in inst["event_ids"]:
                unit[eid] = routine["slug"]
    for n, task in enumerate(labels["one_off_tasks"]):
        for eid in task["event_ids"]:
            unit[eid] = f"task:{n}"
    return unit


def _instance_of(labels: dict) -> dict[str, str]:
    return {
        eid: inst["instance_id"]
        for routine in labels["routines"] for inst in routine["instances"]
        for eid in inst["event_ids"]
    }


def _walk(db: WorkerDB, start: datetime, end: datetime) -> None:
    """Passive discovery's observation pass every two hours."""
    now = start
    while now <= end:
        events = db.get_annotated_events_in_window(hours=24, now=now)
        if events:
            wm.record_observations(db, events, now=now, sleep_boundaries=db.get_sleep_boundaries())
        now += TICK


def _load(tmp_path: Path, shift_instance: tuple[str, timedelta] | None = None):
    fixture = load_fixture(NAME, tmp_path / "data")
    labels = read_labels(NAME)
    if shift_instance is not None:
        instance_id, delta = shift_instance
        ids = next(
            inst["event_ids"] for r in labels["routines"] for inst in r["instances"]
            if inst["instance_id"] == instance_id
        )
        conn = sqlite3.connect(str(fixture.db_path))
        for eid in ids:
            (ts,) = conn.execute("SELECT timestamp FROM events WHERE id = ?", (eid,)).fetchone()
            moved = parse_timestamp(ts) + delta
            conn.execute("UPDATE events SET timestamp = ? WHERE id = ?", (moved.isoformat(), eid))
        conn.commit()
        conn.close()
    manifest = read_manifest(NAME)
    start = parse_timestamp(manifest["recorded_start"]).replace(hour=0, minute=0, second=0, microsecond=0)
    end = parse_timestamp(manifest["recorded_end"]) + timedelta(days=1, hours=2)
    return fixture, labels, start, end


@pytest.fixture()
def mined(tmp_path: Path):
    fixture, labels, start, end = _load(tmp_path)
    with WorkerDB(fixture.db_path) as db:
        _walk(db, start, end)
        yield db, labels, end


def _routine_of(candidate: wm.Candidate, unit_of: dict[str, str]) -> Counter:
    return Counter(
        unit_of.get(eid, "noise")
        for o in candidate.observations for eid in json.loads(o["event_ids_json"])
    )


def test_three_weeks_yield_exactly_the_two_planted_workflows(mined) -> None:
    db, labels, end = mined
    unit_of, instance_of = _unit_of(labels), _instance_of(labels)
    # Every finished work session is one observation.
    observations = db.get_workflow_observations()
    assert len(observations) == 15 + 4 + len(labels["one_off_tasks"])

    candidates = wm.review_candidates(db, now=end)
    by_routine = {}
    for cand in candidates:
        units = _routine_of(cand, unit_of)
        assert len(units) == 1, f"chimera: {units}"  # nothing from another task
        by_routine[next(iter(units))] = cand
    assert set(by_routine) == {STRIPE, EXPENSES}

    for slug, cand in by_routine.items():
        planted = next(r for r in labels["routines"] if r["slug"] == slug)
        seen = {instance_of[eid] for o in cand.observations for eid in json.loads(o["event_ids_json"])}
        assert seen == {i["instance_id"] for i in planted["instances"]}  # every instance
    # The interrupted Monday is two sessions (a 40-minute call between).
    assert len(by_routine[STRIPE].observations) == 15
    assert len(by_routine[EXPENSES].observations) == 4

    assert by_routine[STRIPE].recurrence["pattern"] == "daily"
    weekly = by_routine[EXPENSES].recurrence
    assert (weekly["pattern"], weekly["weekday"], weekly["day"]) == ("weekly", 0, "monday")
    # Highest value first; every step of both routines is in the browser.
    assert candidates[0] is by_routine[STRIPE]
    for cand in candidates:
        assert cand.value["automatable_fraction"] == 1.0
        assert cand.value["hours_per_month"] > 0


def test_observations_outlive_raw_retention(mined) -> None:
    """Twenty-one days on, the daemon has purged every event and the
    segment TTL every segment; the observation log still carries both
    workflows, their recurrence and their extracted evidence."""
    db, labels, end = mined
    later = end + timedelta(days=21)
    for n in range(3):
        db.save_task_segment(
            segment_id=f"seg-{n}", cluster_id=n, task_label="", event_ids=[],
            apps=[], start_time=end.isoformat(), end_time=end.isoformat(),
        )
    conn = sqlite3.connect(db._get_writable_path())
    conn.execute("DELETE FROM events")  # the daemon's raw retention
    conn.commit()
    conn.close()
    assert db.purge_task_segments(wm.SEGMENT_TTL_DAYS, now=later) == 3
    assert db.get_task_segments() == []

    candidates = wm.review_candidates(db, now=end)
    assert len(candidates) == 2
    for cand in candidates:
        assert cand.recurrence is not None
        for obs in cand.observations:
            evidence = json.loads(obs["evidence_json"])
            assert evidence["event_count"] >= 2
            assert evidence["timing_patterns"]["total_duration_seconds"] > 0
    # No per-frame vectors anywhere in the worker's rows: an observation is
    # a few kilobytes at most.
    sizes = [len(json.dumps(o)) for o in db.get_workflow_observations()]
    assert max(sizes) < 8_000


def test_local_time_los_angeles(tmp_path: Path, local_tz) -> None:
    """In Los Angeles, a payout instance moved to 21:05 local (04:05 UTC the
    next day) belongs to its local day: the day's events, its observation
    and the digest's recurring workflows; the Monday routine (03:30 local)
    is still weekly on Monday."""
    local_tz("America/Los_Angeles")
    labels = read_labels(NAME)
    stripe = next(r for r in labels["routines"] if r["slug"] == STRIPE)
    # A morning instance on a Wednesday.
    inst = next(
        i for i in stripe["instances"]
        if datetime.fromisoformat(i["start"]).weekday() == 2 and not i["variant"]
    )
    started = datetime.fromisoformat(inst["start"])
    local_day = started.date().isoformat()
    evening = started.replace(hour=21, minute=5) + timedelta(hours=7)  # 21:05 PDT
    fixture, labels, start, end = _load(tmp_path, (inst["instance_id"], evening - started))

    with WorkerDB(fixture.db_path) as db:
        day_events = {e["id"] for e in db.get_annotated_events_for_date(local_day)}
        assert set(inst["event_ids"]) <= day_events

        _walk(db, start, end)
        obs = next(
            o for o in db.get_workflow_observations()
            if set(json.loads(o["event_ids_json"])) == set(inst["event_ids"])
        )
        assert (obs["started_local_date"], obs["started_local_minute"] // 60) == (local_day, 21)

        kb = KnowledgeBase(root=tmp_path / "knowledge")
        kb.ensure_structure()
        highlights = DigestGenerator(kb, db=db)._workflow_highlights(local_day)
        assert any("a month" in h.detail for h in highlights)

        candidates = wm.review_candidates(db, now=end)
        unit_of = _unit_of(labels)
        by_routine = {next(iter(_routine_of(c, unit_of))): c for c in candidates}
        assert by_routine[STRIPE].recurrence["pattern"] == "daily"
        weekly = by_routine[EXPENSES].recurrence
        assert (weekly["pattern"], weekly["weekday"]) == ("weekly", 0)
        assert by_routine[STRIPE].obs_ids.count(obs["obs_id"]) == 1


def test_the_digest_says_how_long_a_workflow_takes(mined, tmp_path: Path) -> None:
    db, labels, end = mined
    stripe_day = datetime.fromisoformat(labels["routines"][0]["instances"][-1]["start"]).date().isoformat()
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    highlights = DigestGenerator(kb, db=db)._workflow_highlights(stripe_day)
    (payout,) = [h for h in highlights if "(daily)" in h.detail]
    assert payout.title.startswith("Recurring workflow:")
    assert payout.detail.endswith("a month.") and "about " in payout.detail


def _insert_frame(db_path: Path, event_id: str, when: datetime, app: str, title: str) -> None:
    """A captured frame the annotator has not labelled (it joins no work
    session): what was on screen before a run."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO events (id, timestamp, kind_json, window_json, display_topology_json, "
        "primary_display_id, metadata_json) VALUES (?, ?, ?, ?, '[]', 'main', '{}')",
        (event_id, when.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z", '{"type":"DwellSnapshot"}',
         json.dumps({"window_id": "1", "app_id": f"pid:1:{app}", "title": title,
                     "bounds_global_px": [0, 0, 100, 100], "z_order": 0, "is_fullscreen": False})),
    )
    conn.commit()
    conn.close()


def test_the_weekly_routine_is_triggered_by_its_monday_clock(tmp_path: Path, local_tz) -> None:
    """WP-L3: the Monday expense report starts after a different app each
    week (Slack, Mail, a calendar reminder), so no lead-in recurs; its runs
    share the local weekday and time, so the ledger's ``trigger`` slot is a
    time-based hypothesis resting on those runs, and the calendar event
    seen before one of them is the trigger's value.  The Stripe payout is
    done when "Send now" is clicked in Slack, in every run."""
    local_tz("UTC")
    fixture, labels, start, end = _load(tmp_path)
    expenses = next(r for r in labels["routines"] if r["slug"] == EXPENSES)
    lead_ins = [
        ("Slack", "finance (Channel) - Acme - Slack"),
        ("Mail", "Receipts for last week - Inbox"),
        ("Calendar", "Submit expenses - Calendar"),
    ]
    assert len(expenses["instances"]) == len(lead_ins)
    for n, (inst, (app, title)) in enumerate(zip(expenses["instances"], lead_ins)):
        began = parse_timestamp(inst["start"])
        _insert_frame(fixture.db_path, f"lead-{n}", began - timedelta(minutes=2), app, title)

    with WorkerDB(fixture.db_path) as db:
        _walk(db, start, end)
        candidates = wm.review_candidates(db, now=end)
    unit_of = _unit_of(labels)
    by_routine = {next(iter(_routine_of(c, unit_of))): c for c in candidates}
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    procs = {}
    for slug, cand in by_routine.items():
        # The draft's template carries its observations (as passive
        # discovery's does); saving it computes the ledger.
        template = {"slug": slug, "title": cand.task_label,
                    "steps": [{"step": "step_1", "action": cand.task_label}]}
        template.update(cand.template_evidence())
        kb.save_procedure(sop_to_procedure(template))
        procs[slug] = kb.get_procedure(slug)

    weekly = procs[EXPENSES]
    trigger = weekly["knowledge"]["trigger"]
    assert (trigger["status"], trigger["source"]) == ("hypothesis", "observation")
    learned = weekly["learned"]["trigger"]
    assert (learned["kind"], learned["weekday"], learned["time"]) == ("time", 0, "10:30")
    assert learned["weekday"] == expenses["local_weekday"]
    assert learned["description"] == "Most Mondays around 10:30"
    # Each Monday it rests on is one of the routine's observations.
    assert {e["kind"] for e in trigger["evidence"]} == {"instance"}
    assert len(trigger["evidence"]) == 3
    assert {e["id"] for e in trigger["evidence"]} <= set(by_routine[EXPENSES].obs_ids)
    assert learned["calendar_event"] == "Submit expenses"
    assert trigger["value_ref"] == "learned.trigger.calendar_event"
    assert weekly["learned"]["done"]["label"] == "Submit"

    daily = procs[STRIPE]
    assert daily["learned"]["trigger"]["description"] == "Most weekdays around 09:15"
    done = daily["knowledge"]["done_criteria"]
    assert (done["status"], done["value_ref"]) == ("hypothesis", "learned.done")
    assert daily["learned"]["done"]["label"] == "Send now"
    assert daily["learned"]["done"]["support"] == "15 of 15 runs"


def test_a_skill_is_recognised_while_it_runs(tmp_path: Path) -> None:
    """WP-M5 recognise-and-attach.  The payout routine became a Skill in
    its first week; in the second week, live recognition (every two
    minutes, no model) tags the Tuesday session with the Skill before it
    ends and records it as the Skill's observation once the user has been
    idle for ten minutes, before any passive discovery pass sees it.  The
    later passes keep that one observation."""
    fixture, labels, start, end = _load(tmp_path)
    routine = next(r for r in labels["routines"] if r["slug"] == STRIPE)
    tuesday = next(i for i in routine["instances"] if i["instance_id"].endswith("20260908"))
    first_week = datetime.fromisoformat("2026-09-05T00:00:00+00:00")
    with WorkerDB(fixture.db_path) as db:
        _walk(db, start, first_week)
        unit_of = _unit_of(labels)
        [cand] = [c for c in wm.review_candidates(db, now=first_week)
                  if set(_routine_of(c, unit_of)) == {STRIPE}]
        db.set_observations_promoted(cand.obs_ids, STRIPE, promoted_at=first_week)

        stamps = sorted(
            parse_timestamp(r["timestamp"]) for r in db.get_events_by_ids(tuesday["event_ids"])
        )
        now, tagged_at = stamps[0], None
        while now <= stamps[-1] + timedelta(minutes=12):
            if wm.recognize(db, None, now=now) == STRIPE and tagged_at is None:
                tagged_at = now
            now += timedelta(seconds=wm.RECOGNIZE_INTERVAL_SECONDS)
        assert tagged_at is not None and tagged_at < stamps[-1]  # while in progress

        [obs] = [o for o in db.get_workflow_observations(since=stamps[0] - timedelta(minutes=1))
                 if o["started_utc"] <= stamps[-1].strftime("%Y-%m-%dT%H:%M:%S.999Z")]
        assert obs["promoted_slug"] == STRIPE and obs["promoted_at"] is None
        assert set(json.loads(obs["event_ids_json"])) <= set(tuesday["event_ids"])
        # It carries its run signals (WP-L3) like any recorded session.
        assert json.loads(obs["evidence_json"])["signals"]["done"]["label"] == "Send now"

        # Passive discovery's later passes update that row, never add one.
        _walk(db, first_week, end)
        holding = [o for o in db.get_workflow_observations()
                   if set(json.loads(o["event_ids_json"])) & set(tuesday["event_ids"])]
        assert [o["obs_id"] for o in holding] == [obs["obs_id"]]
        assert holding[0]["promoted_slug"] == STRIPE
