"""Regression pins for workflow mining (WP-M3): step signatures and keys,
sessions, the observation log, grouping, recurrence, value and the review
queue.  The journey-level proof over the three-week fixture is
``journeys/test_workflow_observations.py``."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from agenthandover_worker import workflow_mining as wm
from agenthandover_worker.db import WorkerDB, _local_day_bounds
from agenthandover_worker.pattern_detector import PatternDetector
from agenthandover_worker.task_segmenter import AnnotatedFrame
from daemon_db import kb_db


def _ann(location: str, *, app: str = "Google Chrome", typed: str = "", chosen: str = "",
         what: str = "working", workflow: bool = True) -> dict:
    return {
        "app": app,
        "location": location,
        "task_context": {"what_doing": what, "is_workflow": workflow},
        "key_text": {"typed_text": typed},
        "visible_items": [{"label": chosen, "chosen": True}] if chosen else [],
    }


def _event(eid: str, ts: datetime, ann: dict) -> dict:
    return {
        "id": eid,
        "timestamp": ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "annotation_status": "completed",
        "scene_annotation_json": json.dumps(ann),
    }


def _frame(location: str, **kw) -> AnnotatedFrame:
    return AnnotatedFrame.from_event(_event("x", datetime(2026, 9, 1, tzinfo=timezone.utc), _ann(location, **kw)))


# The same payout workflow in two orders.
PAYOUT = [
    ("https://dashboard.stripe.com/payouts", {"chosen": "Payouts"}),
    ("https://dashboard.stripe.com/payouts/po_1QxAbCdEfGh23Jk", {"chosen": "$1,204.00"}),
    ("https://docs.google.com/spreadsheets/d/1pQx7RkTz0Lm4aVn8sWc2YbE5dJh3uFg/edit", {"typed": "$1,204.00"}),
    ("https://docs.google.com/spreadsheets/d/1pQx7RkTz0Lm4aVn8sWc2YbE5dJh3uFg/edit", {"chosen": "D240"}),
    ("https://app.slack.com/client/T04ACME9/C05FINANCE", {"typed": "reconciled"}),
]


def _sigs(steps) -> list[tuple[str, str, str]]:
    return wm.collapse([wm.step_signature(_frame(loc, **kw)) for loc, kw in steps])


# ---------------------------------------------------------------------------
# Signatures and keys
# ---------------------------------------------------------------------------


class TestSignatures:
    @pytest.mark.parametrize(("location", "expected"), [
        ("https://acme.io/reports/123", "acme.io/reports/{id}"),
        ("https://www.acme.io/reports/123/?tab=2#top", "acme.io/reports/{id}"),
        ("https://dashboard.stripe.com/payouts/po_1QxAbCdEfGh23Jk", "dashboard.stripe.com/payouts/{id}"),
        ("https://x.io/runs/3f2b9c1e-8a4d-4c2b-9e1f-0a1b2c3d4e5f/logs", "x.io/runs/{id}/logs"),
        ("https://github.com/org/repo/commit/9fceb02d0ae598e95dc970b74767f19372d61af8", "github.com/org/repo/commit/{id}"),
        ("https://acme.io/invoices/INV-2044", "acme.io/invoices/{id}"),
        # Issue keys: each morning's tickets differ, the workflow does not.
        ("https://jira.atlassian.net/browse/PROJ-101", "jira.atlassian.net/browse/{id}"),
        ("https://linear.app/acme/issue/ENG-42/fix-login", "linear.app/acme/issue/{id}/fix-login"),
        ("https://acme.io/blog/release-2024", "acme.io/blog/release-2024"),
        ("https://mail.google.com/mail/u/0/#inbox/FMfcgz", "mail.google.com/mail/u/{id}"),
        # A document's own key names the document: two documents stay two.
        ("https://docs.google.com/document/d/1hQ4hiringPlan/edit", "docs.google.com/document/d/1hQ4hiringPlan/edit"),
        ("https://app.slack.com/client/T04ACME9/C05FINANCE", "app.slack.com/client/T04ACME9/C05FINANCE"),
        ("https://acme.io/settings/my_documents", "acme.io/settings/my_documents"),
        ("~/code/acme-api/auth_service.py", "~/code/acme-api/auth_service.py"),
        ("", ""),
    ])
    def test_host_or_doc(self, location: str, expected: str) -> None:
        assert wm.host_or_doc(location) == expected

    def test_action_verb_comes_from_the_annotation_structure(self) -> None:
        assert wm.action_verb(_ann("u", typed="42")) == "enter"
        assert wm.action_verb(_ann("u", chosen="Submit")) == "click"
        assert wm.action_verb(_ann("u")) == "view"
        # A v0.4 annotation (legacy keys) is read through as_compact.
        legacy = {"app": "Chrome", "task_context": {"what_doing": "x", "likely_next": "y"},
                  "key_text": {"typed_text": "hello", "email_addresses": []}}
        assert wm.action_verb(legacy) == "enter"

    def test_key_is_run_length_collapsed_and_order_sensitive(self) -> None:
        steps = _sigs(PAYOUT)
        doubled = _sigs([PAYOUT[0], PAYOUT[0], *PAYOUT[1:]])
        assert doubled == steps
        assert wm.workflow_key(doubled) == wm.workflow_key(steps)
        swapped = _sigs([PAYOUT[2], *PAYOUT[:2], *PAYOUT[3:]])
        assert wm.workflow_key(swapped) != wm.workflow_key(steps)

    def test_record_ids_do_not_change_the_key(self) -> None:
        other = [(loc.replace("po_1QxAbCdEfGh23Jk", "po_1ZzYyXxWwVv98Tt"), kw) for loc, kw in PAYOUT]
        assert wm.workflow_key(_sigs(other)) == wm.workflow_key(_sigs(PAYOUT))


class TestSameWorkflow:
    def test_reordered_variant_matches_through_the_aligner(self) -> None:
        normal = _sigs(PAYOUT)
        sheet_first = _sigs([PAYOUT[3], *PAYOUT])
        assert wm.workflow_key(normal) != wm.workflow_key(sheet_first)
        assert wm.align_score(normal, sheet_first) >= wm.ALIGN_THRESHOLD
        assert wm.same_workflow(normal, sheet_first)

    def test_bag_jaccard_matches_a_longer_repeat(self) -> None:
        short = _sigs(PAYOUT)
        longer = _sigs(PAYOUT + PAYOUT[2:])  # a second sheet/slack round
        assert wm.bag_jaccard(set(short), set(longer)) >= wm.BAG_THRESHOLD
        assert wm.same_workflow(short, longer)

    def test_two_documents_on_one_host_are_different_workflows(self) -> None:
        a = _sigs([("https://docs.google.com/document/d/1hQ4hiringPlan/edit", {"typed": "x"})])
        b = _sigs([("https://docs.google.com/document/d/1postmortemSep2/edit", {"typed": "y"})])
        assert not wm.same_workflow(a, b)

    def test_unrelated_workflows_do_not_match(self) -> None:
        expense = _sigs([
            ("https://mail.google.com/mail/u/0/#search/receipt", {"chosen": "Search"}),
            ("https://www.expensify.com/reports", {"chosen": "New report"}),
            ("https://www.expensify.com/report?reportID=0012345", {"typed": "$12"}),
        ])
        assert not wm.same_workflow(_sigs(PAYOUT), expense)


# ---------------------------------------------------------------------------
# Sessions and the observation log
# ---------------------------------------------------------------------------


def _session_events(prefix: str, start: datetime, steps=PAYOUT, gap_s: int = 90) -> list[dict]:
    return [
        _event(f"{prefix}-{i}", start + timedelta(seconds=gap_s * i), _ann(loc, **kw))
        for i, (loc, kw) in enumerate(steps)
    ]


class TestSessions:
    def test_idle_gap_of_ten_minutes_splits(self) -> None:
        t0 = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)
        events = [
            _event("a", t0, _ann("https://a.io/x")),
            _event("b", t0 + timedelta(minutes=9, seconds=59), _ann("https://a.io/y")),
            _event("c", t0 + timedelta(minutes=19, seconds=59), _ann("https://a.io/z")),
        ]
        assert [[f.event_id for f in s] for s in wm.split_sessions(events)] == [["a", "b"], ["c"]]

    def test_noise_frames_are_dropped_and_sleep_splits(self) -> None:
        t0 = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)
        events = [
            _event("a", t0, _ann("https://a.io/x")),
            _event("yt", t0 + timedelta(minutes=1), _ann("https://youtube.com/watch", workflow=False)),
            _event("b", t0 + timedelta(minutes=2), _ann("https://a.io/y")),
            _event("c", t0 + timedelta(minutes=4), _ann("https://a.io/z")),
        ]
        slept = [((t0 + timedelta(minutes=3)).isoformat(), (t0 + timedelta(minutes=3, seconds=30)).isoformat())]
        assert [[f.event_id for f in s] for s in wm.split_sessions(events, slept)] == [["a", "b"], ["c"]]


@pytest.fixture()
def db(tmp_db_path: Path):
    with WorkerDB(tmp_db_path) as worker_db:
        yield worker_db


def _insert(write_conn, events: list[dict]) -> None:
    for ev in events:
        write_conn.execute(
            "INSERT INTO events (id, timestamp, kind_json, window_json, display_topology_json, "
            "primary_display_id, artifact_ids_json, metadata_json, annotation_status, "
            "scene_annotation_json, processed) VALUES (?, ?, '{}', '{}', '[]', '', '[]', '{}', ?, ?, 1)",
            (ev["id"], ev["timestamp"], ev["annotation_status"], ev["scene_annotation_json"]),
        )
    write_conn.commit()


class TestObservationLog:
    def test_finished_sessions_are_recorded_once_with_local_fields(
        self, db: WorkerDB, write_conn, local_tz,
    ) -> None:
        local_tz("America/Los_Angeles")
        # 21:00 local on Friday 2026-09-04 is 04:00 UTC on Saturday.
        start = datetime(2026, 9, 5, 4, 0, tzinfo=timezone.utc)
        events = _session_events("s", start)
        in_progress = _session_events("p", start + timedelta(hours=2))
        _insert(write_conn, events + in_progress)
        now = start + timedelta(hours=2, minutes=8)  # the second session is live

        window = db.get_annotated_events_in_window(hours=24, now=now)
        assert wm.record_observations(db, window, now=now) == 1
        assert wm.record_observations(db, window, now=now) == 1  # an upsert, not a copy
        (obs,) = db.get_workflow_observations()
        assert obs["started_local_date"] == "2026-09-04"
        assert obs["started_local_weekday"] == 4  # Friday
        assert obs["started_local_minute"] == 21 * 60
        assert obs["duration_s"] == 360.0
        assert json.loads(obs["event_ids_json"]) == [e["id"] for e in events]
        assert json.loads(obs["hosts_json"]) == ["dashboard.stripe.com", "docs.google.com", "app.slack.com"]
        evidence = json.loads(obs["evidence_json"])
        assert evidence["event_count"] == len(events)
        assert evidence["timing_patterns"]["total_duration_seconds"] == 360.0
        # Distilled up to the idle cutoff (the live session starts later).
        assert db.get_state(wm.DISTILLED_STATE_KEY) == "2026-09-05T05:58:00.000Z"

    def test_a_session_cut_by_the_window_start_is_not_recorded_again(
        self, db: WorkerDB, write_conn,
    ) -> None:
        """Passive discovery runs every few minutes over a sliding 24-hour
        window; when the window's start falls inside an already recorded
        session, its tail is not a second observation."""
        start = datetime(2026, 9, 1, 9, tzinfo=timezone.utc)
        events = _session_events("s", start)
        _insert(write_conn, events)
        now = start + timedelta(hours=1)
        assert wm.record_observations(db, db.get_annotated_events_in_window(hours=24, now=now), now=now) == 1
        # A day later the window starts between the session's first frames.
        later = start + timedelta(hours=24, minutes=2)
        tail = db.get_annotated_events_in_window(hours=24, now=later)
        assert 2 <= len(tail) < len(events)
        assert wm.record_observations(db, tail, now=later) == 0
        (obs,) = db.get_workflow_observations()
        assert json.loads(obs["event_ids_json"]) == [e["id"] for e in events]

    def test_a_forty_minute_instance_interrupted_by_a_call_is_one_observation(
        self, db: WorkerDB, write_conn,
    ) -> None:
        """The interruption story is the session split alone (the segmenter's
        interruption merging is gone): a payout reconciliation that takes 40
        minutes because a Zoom call lands in its middle is one observation,
        since no gap between frames reaches the idle gap."""
        start = datetime(2026, 9, 1, 9, tzinfo=timezone.utc)
        call = {"app": "zoom.us", "what": "on a call with the finance team"}
        timeline = [
            (0, PAYOUT[0]), (2, PAYOUT[1]),
            *((4 + 7 * k, ("", call)) for k in range(4)),  # minutes 4..25
            (33, PAYOUT[2]), (36, PAYOUT[3]), (40, PAYOUT[4]),
        ]
        events = []
        for i, (minute, (loc, kw)) in enumerate(timeline):
            ann = _ann(loc, **kw)
            if loc == "":
                ann["task_context"]["activity_type"] = "communication"
            events.append(_event(f"w-{i}", start + timedelta(minutes=minute), ann))
        _insert(write_conn, events)

        now = start + timedelta(hours=1)
        window = db.get_annotated_events_in_window(hours=24, now=now)
        assert wm.record_observations(db, window, now=now) == 1
        (obs,) = db.get_workflow_observations()
        assert obs["duration_s"] == 40 * 60
        assert json.loads(obs["event_ids_json"]) == [e["id"] for e in events]

    def test_a_single_frame_session_is_not_an_observation(self, db: WorkerDB, write_conn) -> None:
        now = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
        _insert(write_conn, [_event("one", now - timedelta(hours=1), _ann("https://a.io/x"))])
        assert wm.record_observations(db, db.get_annotated_events_in_window(hours=24, now=now), now=now) == 0

    def test_promotion_survives_a_re_record(self, db: WorkerDB, write_conn) -> None:
        start = datetime(2026, 9, 1, 9, tzinfo=timezone.utc)
        _insert(write_conn, _session_events("s", start))
        now = start + timedelta(hours=1)
        window = db.get_annotated_events_in_window(hours=24, now=now)
        wm.record_observations(db, window, now=now)
        (obs,) = db.get_workflow_observations()
        db.set_observations_promoted([obs["obs_id"]], "payout", promoted_at=now)
        wm.record_observations(db, window, now=now)
        (again,) = db.get_workflow_observations()
        assert again["promoted_slug"] == "payout" and again["promoted_at"]

    def test_unpromoted_observations_expire_promoted_ones_stay(self, db: WorkerDB, write_conn) -> None:
        now = datetime(2026, 12, 1, tzinfo=timezone.utc)
        for name, days in (("old", 100), ("old-promoted", 99), ("recent", 10)):
            start = now - timedelta(days=days)
            _insert(write_conn, _session_events(name, start))
            wm.record_observations(
                db, db.get_annotated_events_in_window(hours=24, now=start + timedelta(hours=1)),
                now=start + timedelta(hours=1),
            )
        promoted = next(o for o in db.get_workflow_observations()
                        if json.loads(o["event_ids_json"])[0].startswith("old-promoted"))
        db.set_observations_promoted([promoted["obs_id"]], "payout")
        assert db.purge_unpromoted_observations(wm.UNPROMOTED_TTL_DAYS, now=now) == 1
        assert len(db.get_workflow_observations()) == 2


# ---------------------------------------------------------------------------
# Recurrence (local time), value, the review queue
# ---------------------------------------------------------------------------


def _obs(day: str, minute: int = 9 * 60, duration: float = 400.0, key: str = "k", steps=None,
         hosts=("dashboard.stripe.com", "docs.google.com", "app.slack.com"), obs_id: str = "") -> dict:
    d = date.fromisoformat(day)
    return {
        "obs_id": obs_id or f"obs-{key}-{day}-{minute}",
        "workflow_key": key,
        "started_utc": f"{day}T{minute // 60:02d}:{minute % 60:02d}:00.000Z",
        "started_local_date": day,
        "started_local_weekday": d.weekday(),
        "started_local_minute": minute,
        "duration_s": duration,
        "apps_json": "[]",
        "hosts_json": json.dumps(list(hosts)),
        "step_sig_json": json.dumps([list(s) for s in (steps or _sigs(PAYOUT))]),
        "evidence_json": "{}",
        "source": "passive",
        "task_label": "Reconcile payout",
        "event_ids_json": "[]",
        "promoted_slug": None,
        "promoted_at": None,
    }


def _workdays(first: date, n: int) -> list[str]:
    out, d = [], first
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


class TestRecurrence:
    detector = PatternDetector(None, min_observations=3)

    def test_weekly_on_the_local_weekday(self) -> None:
        mondays = ["2026-08-31", "2026-09-07", "2026-09-14"]
        active = set(_workdays(date(2026, 8, 31), 15))
        pattern = self.detector.detect_from_observations(
            [_obs(d, minute=10 * 60 + 30) for d in mondays],
            today=date(2026, 9, 19), active_dates=active,
        )
        assert pattern is not None
        assert (pattern.pattern, pattern.weekday, pattern.day, pattern.time) == ("weekly", 0, "monday", "10:30")

    def test_daily_on_workdays(self) -> None:
        days = _workdays(date(2026, 8, 31), 15)
        pattern = self.detector.detect_from_observations(
            [_obs(d) for d in days], today=date(2026, 9, 19), active_dates=set(days),
        )
        assert pattern is not None and pattern.pattern == "daily"
        assert pattern.observations == 15

    def test_daily_needs_min_observations(self) -> None:
        days = _workdays(date(2026, 9, 14), 2)
        assert self.detector.detect_from_observations(
            [_obs(d) for d in days], today=date(2026, 9, 15), active_dates=set(days),
        ) is None

    def test_same_weekday_without_a_weekly_gap_is_not_weekly(self) -> None:
        # Three Mondays, but four weeks apart: the median gap is 28 days.
        mondays = ["2026-07-27", "2026-08-24", "2026-09-21"]
        assert self.detector.detect_from_observations(
            [_obs(d) for d in mondays], today=date(2026, 9, 22),
            active_dates=set(_workdays(date(2026, 7, 27), 40)),
        ) is None

    def test_window_is_42_local_days(self) -> None:
        old = ["2026-06-01", "2026-06-08", "2026-06-15"]
        assert self.detector.detect_from_observations(
            [_obs(d) for d in old], today=date(2026, 9, 1),
        ) is None


class TestValueAndQueue:
    def test_value_is_median_duration_times_monthly_occurrences(self) -> None:
        obs = [_obs(d, duration=dur) for d, dur in
               zip(_workdays(date(2026, 9, 1), 5), (300, 360, 420, 3000, 400))]
        value = wm.workflow_value(obs, history_days=30)
        assert value["median_duration_s"] == 400.0
        assert value["occurrences_per_month"] == 5.0
        assert value["value_score"] == 2000.0
        assert value["automatable_fraction"] == 1.0
        assert wm.about_hours({"hours_per_month": 3.4}) == "about 3 hours a month"
        assert wm.about_hours({"hours_per_month": 0.25}) == "about 15 minutes a month"

    def _save(self, db: WorkerDB, rows: list[dict]) -> None:
        for row in rows:
            assert db.save_workflow_observation(row)

    def test_three_observations_make_a_candidate_two_do_not(self, db: WorkerDB) -> None:
        now = datetime(2026, 9, 20, tzinfo=timezone.utc)
        self._save(db, [_obs(d, key="a") for d in _workdays(date(2026, 9, 1), 2)])
        assert wm.review_candidates(db, now=now) == []
        self._save(db, [_obs("2026-09-10", key="a")])
        (cand,) = wm.review_candidates(db, now=now)
        assert len(cand.observations) == 3
        assert cand.template_evidence()["episode_count"] == 3

    def test_single_step_workflows_are_never_candidates(self, db: WorkerDB) -> None:
        inbox = _sigs([("https://mail.google.com/mail/u/0/#inbox", {})])
        self._save(db, [_obs(d, key="inbox", steps=inbox) for d in _workdays(date(2026, 9, 1), 10)])
        assert wm.review_candidates(db, now=datetime(2026, 9, 20, tzinfo=timezone.utc)) == []

    def test_queue_admits_three_per_week_highest_value_first(self, db: WorkerDB) -> None:
        now = datetime(2026, 9, 20, tzinfo=timezone.utc)
        for n in range(5):
            steps = _sigs([(f"https://tool{n}.io/a", {"chosen": "Go"}), (f"https://tool{n}.io/b", {"typed": "x"})])
            self._save(db, [
                _obs(d, key=f"w{n}", steps=steps, duration=100.0 * (n + 1), hosts=[f"tool{n}.io"])
                for d in _workdays(date(2026, 9, 1), 3)
            ])
        cands = wm.review_candidates(db, now=now)
        assert [c.source_id for c in cands] == ["w4", "w3", "w2"]
        # Two promoted this week: one slot left, for the most valuable of the rest.
        for c, slug in zip(cands[:2], ("four", "three")):
            db.set_observations_promoted(c.obs_ids, slug, promoted_at=now)
        assert [c.source_id for c in wm.review_candidates(db, now=now)] == ["w2"]
        # A week later the cap is free again.
        later = now + timedelta(days=8)
        assert [c.source_id for c in wm.review_candidates(db, now=later)] == ["w2", "w1", "w0"]

    def test_a_promoted_workflow_absorbs_new_observations(self, db: WorkerDB) -> None:
        now = datetime(2026, 11, 30, tzinfo=timezone.utc)
        first = [_obs(d, key="a") for d in _workdays(date(2026, 9, 1), 3)]  # > 42 days ago
        self._save(db, first)
        db.set_observations_promoted([o["obs_id"] for o in first], "payout", promoted_at=now - timedelta(days=80))
        self._save(db, [_obs(d, key="a2", steps=_sigs([PAYOUT[3], *PAYOUT]))
                        for d in _workdays(date(2026, 11, 20), 4)])
        assert wm.review_candidates(db, now=now) == []
        assert {o["promoted_slug"] for o in db.get_workflow_observations()} == {"payout"}

    def test_a_failed_generation_waits_for_new_evidence(self, db: WorkerDB) -> None:
        now = datetime(2026, 9, 20, tzinfo=timezone.utc)
        self._save(db, [_obs(d, key="a") for d in _workdays(date(2026, 9, 1), 3)])
        (cand,) = wm.review_candidates(db, now=now)
        db.record_failed_generation("passive", cand.source_id, "boom", context={"obs_ids": cand.obs_ids})
        assert wm.review_candidates(db, now=now) == []
        self._save(db, [_obs("2026-09-15", key="a")])
        (again,) = wm.review_candidates(db, now=now)
        assert len(again.observations) == 4


    def test_a_retry_regenerates_from_the_failed_observations(
        self, db: WorkerDB, write_conn, tmp_path: Path,
    ) -> None:
        """The retry trigger finds the workflow's observations through the
        failure's ``obs_ids``, feeds their frames to the generator, points
        the draft at them and promotes them."""
        from unittest.mock import MagicMock, patch

        from agenthandover_worker.main import _process_retry_triggers
        from agenthandover_worker.sop_generator import GeneratedSOP
        from fixtures.vlm_answers import generated_sop_template

        start = datetime(2026, 9, 1, 9, tzinfo=timezone.utc)
        sessions = [_session_events(f"d{n}", start + timedelta(days=n)) for n in range(3)]
        _insert(write_conn, [e for s in sessions for e in s])
        now = start + timedelta(days=3)
        wm.record_observations(db, db.get_annotated_events_in_window(hours=96, now=now), now=now)
        obs_ids = [o["obs_id"] for o in db.get_workflow_observations()]
        assert len(obs_ids) == 3
        failure_id = db.record_failed_generation(
            "passive", "key", "boom", title="Reconcile payout", context={"obs_ids": obs_ids},
        )
        generator = MagicMock()
        generator.generate_from_passive.return_value = GeneratedSOP(
            sop=generated_sop_template("check_domain_availability"), success=True,
        )
        writer = MagicMock()
        writer.get_sops_dir.return_value = tmp_path / "sops"
        with patch("agenthandover_worker.main._status_dir", return_value=tmp_path):
            _process_retry_triggers(
                {"failure_id": failure_id}, db, sop_generator=generator,
                openclaw_writer=writer, index_generator=MagicMock(), sop_auto_approve=False,
            )
        (demos,), _ = generator.generate_from_passive.call_args
        assert [len(d) for d in demos] == [len(s) for s in sessions]
        (sop,) = db.get_generated_sops()
        template = db.get_generated_sop(sop["sop_id"])["sop_json"]
        assert [o["obs_id"] for o in template["observations"]] == obs_ids
        assert template["episode_count"] == 3
        promoted = {o["promoted_slug"] for o in db.get_workflow_observations()}
        assert promoted == {sop["slug"]}
        assert db.get_failed_generations() == []


# ---------------------------------------------------------------------------
# Local days (#165)
# ---------------------------------------------------------------------------


class TestLocalDay:
    def test_bounds_follow_the_local_zone_and_dst(self, local_tz) -> None:
        local_tz("America/Los_Angeles")
        assert _local_day_bounds("2026-09-04") == ("2026-09-04T07:00:00", "2026-09-05T07:00:00")
        # 2026-11-01: PDT -> PST, a 25-hour day.
        assert _local_day_bounds("2026-11-01") == ("2026-11-01T07:00:00", "2026-11-02T08:00:00")

    def test_a_21h_local_frame_belongs_to_its_local_day(self, db: WorkerDB, write_conn, local_tz) -> None:
        local_tz("America/Los_Angeles")
        evening = datetime(2026, 9, 5, 4, 0, tzinfo=timezone.utc)  # Fri 21:00 PDT
        midnight = datetime(2026, 9, 5, 7, 0, tzinfo=timezone.utc)  # Sat 00:00 PDT exactly
        _insert(write_conn, [
            _event("evening", evening, _ann("https://a.io/x")),
            {**_event("midnight", midnight, _ann("https://a.io/y")), "timestamp": "2026-09-05T07:00:00+00:00"},
        ])
        assert [e["id"] for e in db.get_annotated_events_for_date("2026-09-04")] == ["evening"]
        assert [e["id"] for e in db.get_annotated_events_for_date("2026-09-05")] == ["midnight"]


# ---------------------------------------------------------------------------
# Skills and their observations: recognise, attach, link, diverge (WP-M5)
# ---------------------------------------------------------------------------

PAYOUT_SKILL = {
    "id": "payout",
    "title": "Reconcile payout",
    "steps": [
        {"action": "Open the payouts list", "app": "Google Chrome", "location": PAYOUT[0][0]},
        {"action": "Open today's payout", "app": "Google Chrome", "location": PAYOUT[1][0]},
        {"action": "Enter the amount in the sheet", "app": "Google Chrome", "location": PAYOUT[2][0]},
        {"action": "Post in #finance", "app": "Google Chrome", "location": PAYOUT[4][0]},
    ],
}
# The routine, but on a failed payout the user emails the bank first.
DETOUR = [
    PAYOUT[0],
    (PAYOUT[1][0], {"chosen": "Payout failed"}),
    ("https://mail.google.com/mail/u/0/#inbox", {"typed": "Hi, the payout failed"}),
    *PAYOUT[2:],
]


def _kb(tmp_path: Path, *procs: dict):
    from agenthandover_worker.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(root=tmp_path / "kb", db_path=kb_db(tmp_path / "kb"))
    kb.ensure_structure()
    for proc in procs:
        kb.save_procedure(json.loads(json.dumps(proc)))
    return kb


class TestSkillsAndObservations:
    def _promote(self, db: WorkerDB, first: date = date(2026, 9, 1)) -> list[dict]:
        rows = [_obs(d, key="a") for d in _workdays(first, 3)]
        for row in rows:
            assert db.save_workflow_observation(row)
        db.set_observations_promoted(
            [r["obs_id"] for r in rows], "payout", promoted_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
        )
        return rows

    def test_a_session_is_recognised_in_progress_and_attached_at_its_end(
        self, db: WorkerDB, write_conn,
    ) -> None:
        self._promote(db)
        start = datetime(2026, 9, 8, 9, tzinfo=timezone.utc)
        events = _session_events("live", start)  # five frames, 90 s apart
        _insert(write_conn, events)
        # Two frames, two steps: too few to tell.
        assert wm.recognize(db, None, now=start + timedelta(minutes=2)) is None
        assert db.get_state(wm.LIVE_SESSION_STATE_KEY) is None
        # Three steps (payouts, a payout, the sheet): the Skill's opening.
        assert wm.recognize(db, None, now=start + timedelta(minutes=4)) == "payout"
        assert json.loads(db.get_state(wm.LIVE_SESSION_STATE_KEY))["first_event_id"] == "live-0"
        assert wm.recognize(db, None, now=start + timedelta(minutes=8)) == "payout"
        assert db.get_workflow_observations(since=start) == []  # still in progress
        # Idle for ten minutes: the session ended; it is recorded, attached.
        last = start + timedelta(seconds=90 * (len(events) - 1))
        assert wm.recognize(db, None, now=last + timedelta(minutes=10, seconds=30)) is None
        (obs,) = db.get_workflow_observations(since=start)
        assert json.loads(obs["event_ids_json"]) == [e["id"] for e in events]
        assert obs["promoted_slug"] == "payout"
        assert obs["promoted_at"] is None  # an instance, not a promotion
        assert db.get_state(wm.LIVE_SESSION_STATE_KEY) == ""

    def test_a_new_session_ends_the_tagged_one(self, db: WorkerDB, write_conn) -> None:
        self._promote(db)
        start = datetime(2026, 9, 8, 9, tzinfo=timezone.utc)
        _insert(write_conn, _session_events("live", start))
        assert wm.recognize(db, None, now=start + timedelta(minutes=4)) == "payout"
        # The laptop slept; the next frames are a new session.
        slept, woke = start + timedelta(minutes=6, seconds=30), start + timedelta(minutes=7, seconds=30)
        write_conn.execute(
            "INSERT INTO events (id, timestamp, kind_json, window_json, display_topology_json, "
            "primary_display_id, artifact_ids_json, metadata_json, processed) "
            "VALUES ('sleep', ?, ?, '{}', '[]', '', '[]', '{}', 1)",
            (woke.isoformat(), json.dumps(
                {"type": "SystemSleep", "slept_at": slept.isoformat(), "woke_at": woke.isoformat()},
            )),
        )
        write_conn.commit()
        later = start + timedelta(minutes=8)
        _insert(write_conn, [_event("other", later, _ann("https://a.io/x"))])
        wm.recognize(db, None, now=later + timedelta(seconds=30))
        assert [o["promoted_slug"] for o in db.get_workflow_observations(since=start)] == ["payout"]

    def test_an_unrelated_session_is_not_tagged(self, db: WorkerDB, write_conn) -> None:
        self._promote(db)
        start = datetime(2026, 9, 8, 9, tzinfo=timezone.utc)
        other = [(f"https://{h}.io/page", {}) for h in ("crm", "billing", "wiki", "calendar")]
        _insert(write_conn, _session_events("o", start, steps=other))
        assert wm.recognize(db, None, now=start + timedelta(minutes=6)) is None
        assert db.get_state(wm.LIVE_SESSION_STATE_KEY) is None

    def test_a_recorded_skill_holds_its_workflow_group(self, db: WorkerDB, tmp_path: Path) -> None:
        """A recorded Skill has no observation of its own: its workflow's
        group is linked to it by its steps' places, so it is never
        proposed again as a new draft and its recurrence is written."""
        for day in _workdays(date(2026, 9, 1), 3):
            assert db.save_workflow_observation(_obs(day, key="a"))
        now = datetime(2026, 9, 20, tzinfo=timezone.utc)
        assert [c.source_id for c in wm.review_candidates(db, now=now)] == ["a"]  # no kb: unlinked
        kb = _kb(tmp_path, PAYOUT_SKILL)
        assert wm.review_candidates(db, now=now, kb=kb) == []
        rows = db.get_workflow_observations()
        assert {o["promoted_slug"] for o in rows} == {"payout"}
        assert all(o["promoted_at"] is None for o in rows)  # the weekly cap is untouched
        assert wm.refresh_skills(db, kb, now=now) == 1
        proc = kb.get_procedure("payout")
        assert proc["recurrence"]["pattern"] == "daily"
        assert proc["evidence"]["total_observations"] == 3

    def test_attached_observations_teach_the_skill_its_trigger_and_done(
        self, db: WorkerDB, tmp_path: Path,
    ) -> None:
        """Instances attached after the draft (WP-M5) are added to the
        procedure's evidence with their run signals, so what they show
        (WP-L3: a recurring lead-in, the commit that ends them) becomes the
        Skill's learned trigger and done, citing them."""
        kb = _kb(tmp_path, PAYOUT_SKILL)
        self._promote(db)
        later = []
        for n, day in enumerate(_workdays(date(2026, 9, 8), 3)):
            obs = _obs(day, key="a")
            obs["evidence_json"] = json.dumps({"signals": {
                "lead_in": [{"app": "Mail", "place": "", "title": f"Payout failed #{n} - Inbox",
                             "subject": f"Payout failed #{n}", "event_id": f"lead-{n}"}],
                "local_weekday": date.fromisoformat(day).weekday(), "local_minute": 9 * 60,
                "done": {"kind": "post", "label": "Post", "app": "Google Chrome",
                         "place": "app.slack.com", "event_id": f"done-{n}"},
                "verification": [], "corrections": [],
            }})
            assert db.save_workflow_observation(obs)
            later.append(obs["obs_id"])
        now = datetime(2026, 9, 20, tzinfo=timezone.utc)
        assert wm.review_candidates(db, now=now, kb=kb) == []

        proc = kb.get_procedure("payout")
        cited = [o["obs_id"] for o in proc["evidence"]["observations"]]
        # The three instances attached now (the promoted ones were the
        # draft's, set by the promotion, not attached).
        assert cited == later
        assert proc["evidence"]["total_observations"] == 3
        trigger = proc["learned"]["trigger"]
        # Recurring by the subject's shape; described by the latest one.
        assert trigger["description"] == 'Usually starts after the message "Payout failed #2" in Mail'
        assert trigger["instances"] == later
        assert proc["learned"]["done"]["event_ids"] == ["done-0", "done-1", "done-2"]
        ledger = proc["knowledge"]
        assert ledger["trigger"]["status"] == "hypothesis"
        assert {e["id"] for e in ledger["trigger"]["evidence"]} == set(later)
        assert ledger["done_criteria"]["value_ref"] == "learned.done"
        # Attached once: a later pass cites nothing twice.
        wm.review_candidates(db, now=now, kb=kb)
        assert len(kb.get_procedure("payout")["evidence"]["observations"]) == len(cited)

    def test_a_skill_without_places_is_not_linked(self, db: WorkerDB, tmp_path: Path) -> None:
        for day in _workdays(date(2026, 9, 1), 3):
            db.save_workflow_observation(_obs(day, key="a"))
        vague = {"id": "vague", "title": "Vague", "steps": [{"action": "Do the payout"}, {"action": "Tell finance"}]}
        now = datetime(2026, 9, 20, tzinfo=timezone.utc)
        assert [c.source_id for c in wm.review_candidates(db, now=now, kb=_kb(tmp_path, vague))] == ["a"]

    def test_a_skill_is_linked_to_its_own_workflow_only(self, db: WorkerDB, tmp_path: Path) -> None:
        other = [("https://crm.io/leads", {}), ("https://crm.io/leads/42", {"chosen": "Acme"}),
                 ("https://mail.google.com/mail/u/0/", {"typed": "Hello"})]
        for day in _workdays(date(2026, 9, 1), 3):
            db.save_workflow_observation(_obs(day, key="crm", steps=_sigs(other)))
        now = datetime(2026, 9, 20, tzinfo=timezone.utc)
        assert [c.source_id for c in wm.review_candidates(db, now=now, kb=_kb(tmp_path, PAYOUT_SKILL))] == ["crm"]

    def test_a_focus_recording_is_an_observation_of_its_skill(self, db: WorkerDB) -> None:
        events = _session_events("focus", datetime(2026, 9, 1, 9, tzinfo=timezone.utc))
        obs_id = wm.record_focus_observation(db, events, "payout", "Reconcile payout")
        (obs,) = db.get_workflow_observations()
        assert obs["obs_id"] == obs_id
        assert obs["source"] == "focus" and obs["promoted_slug"] == "payout"
        assert obs["task_label"] == "Reconcile payout"
        # The run's signals (WP-L3) are stored as on a passive observation.
        assert set(json.loads(obs["evidence_json"])["signals"]) >= {"lead_in", "done"}
        # Later passive sessions of the workflow belong to the Skill.
        for day in _workdays(date(2026, 9, 8), 3):
            db.save_workflow_observation(_obs(day, key="a"))
        assert wm.review_candidates(db, now=datetime(2026, 9, 20, tzinfo=timezone.utc)) == []
        assert {o["promoted_slug"] for o in db.get_workflow_observations()} == {"payout"}

    def test_a_focus_recording_needs_two_annotated_frames(self, db: WorkerDB) -> None:
        events = _session_events("focus", datetime(2026, 9, 1, 9, tzinfo=timezone.utc))[:1]
        assert wm.record_focus_observation(db, events, "payout") is None
        assert db.get_workflow_observations() == []

    def test_two_alike_detours_become_a_branch_of_the_skill(
        self, db: WorkerDB, write_conn, tmp_path: Path,
    ) -> None:
        kb = _kb(tmp_path, PAYOUT_SKILL)
        self._promote(db)
        for n, day in enumerate((8, 9)):
            start = datetime(2026, 9, day, 9, tzinfo=timezone.utc)
            _insert(write_conn, _session_events(f"detour{n}", start, steps=DETOUR))
        now = datetime(2026, 9, 10, tzinfo=timezone.utc)
        assert wm.record_observations(db, db.get_annotated_events_in_window(hours=48, now=now), now=now) == 2
        assert wm.review_candidates(db, now=now, kb=kb) == []

        proc = kb.get_procedure("payout")
        detours = [o["obs_id"] for o in db.get_workflow_observations(since=datetime(2026, 9, 8, tzinfo=timezone.utc))]
        assert [h["step_index"] for h in proc["branch_hypotheses"]] == [2, 2]
        [branch] = proc["branches"]
        assert branch["condition"] == (
            'the screen shows "Payout failed" (https://dashboard.stripe.com/payouts/po_1QxAbCdEfGh23Jk)'
        )
        assert branch["action"] == "before step 3: working"
        assert branch["alternative_steps"][0]["location"] == "https://mail.google.com/mail/u/0/#inbox"
        assert {e["id"] for e in branch["evidence"] if e["kind"] == "instance"} == set(detours)
        # Mined once: a later pass adds nothing.
        wm.review_candidates(db, now=now, kb=kb)
        assert len(kb.get_procedure("payout")["branch_hypotheses"]) == 2

    def test_an_instance_that_follows_the_steps_adds_no_hypothesis(
        self, db: WorkerDB, write_conn, tmp_path: Path,
    ) -> None:
        kb = _kb(tmp_path, PAYOUT_SKILL)
        self._promote(db)
        start = datetime(2026, 9, 8, 9, tzinfo=timezone.utc)
        _insert(write_conn, _session_events("same", start))
        now = start + timedelta(hours=1)
        wm.record_observations(db, db.get_annotated_events_in_window(hours=4, now=now), now=now)
        wm.review_candidates(db, now=now, kb=kb)
        assert kb.get_procedure("payout").get("branch_hypotheses", []) == []
