"""Regression pins for WP-L3: what the events around each observed run
say about its trigger, its done point, how the user checks the result and
what they corrected (``run_signals``, ``correction_detector.field_corrections``),
and how three runs fill the ledger's ``trigger``, ``done_criteria`` and
``verification`` slots.  The journey-level proofs are J-ledger and J-passive
(``journeys/test_j_ledger.py``, ``journeys/test_j_passive.py``) and the
model-free pass over the three-week fixture in
``journeys/test_workflow_observations.py``."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agenthandover_worker import run_signals as rs
from agenthandover_worker import workflow_mining as wm
from agenthandover_worker.correction_detector import field_corrections
from agenthandover_worker.db import WorkerDB
from agenthandover_worker.ledger import derive_knowledge, refresh_knowledge
from agenthandover_worker.skill_render import render_body
from daemon_db import create_daemon_db

# A Monday.
DAY = datetime(2026, 9, 7, 9, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _utc(local_tz) -> None:
    local_tz("UTC")


def _ts(when: datetime) -> str:
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _window(app: str, title: str) -> str:
    app_id = app if "." in app and " " not in app else f"pid:1:{app}"
    return json.dumps({"window_id": "1", "app_id": app_id, "title": title,
                       "bounds_global_px": [0, 0, 100, 100], "z_order": 0, "is_fullscreen": False})


class Rows:
    """Event rows as the daemon writes them, built in time order."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, when: datetime, kind: dict, *, app: str = "", title: str = "",
            metadata: dict | None = None, element: dict | None = None,
            annotation: dict | None = None) -> str:
        eid = f"ev-{len(self.rows):03d}"
        self.rows.append({
            "id": eid,
            "timestamp": _ts(when),
            "kind_json": json.dumps(kind),
            "window_json": _window(app, title) if app else None,
            "metadata_json": json.dumps(metadata or {}),
            "focused_element_json": json.dumps(element) if element else None,
            "scene_annotation_json": json.dumps(annotation) if annotation else None,
            "annotation_status": "completed" if annotation else "pending",
        })
        return eid

    def frame(self, when: datetime, app: str, title: str, url: str = "", *,
              work: bool = True, **kw) -> str:
        return self.add(when, {"type": "DwellSnapshot"}, app=app, title=title,
                        annotation={"app": app, "location": url,
                                    "task_context": {"what_doing": title, "is_workflow": work}},
                        **kw)

    def click(self, when: datetime, label: str, url: str, tag: str = "button") -> str:
        return self.add(
            when, {"type": "ClickIntent", "target_description": f"{label}<{tag}>"},
            app="com.google.Chrome", title=url,
            metadata={"url": url, "target": {"tagName": tag, "innerText": label}},
        )

    def submit(self, when: datetime, label: str, url: str) -> str:
        return self.add(
            when,
            {"type": "FormSubmit", "url": url,
             "form_anchor": {"tag": "form", "method": "post",
                             "submitter": {"tagName": "button", "text": label}}},
            app="com.google.Chrome", title=url, metadata={"url": url},
        )

    def write(self, db_path: Path) -> Path:
        create_daemon_db(db_path)
        conn = sqlite3.connect(str(db_path))
        for row in self.rows:
            conn.execute(
                "INSERT INTO events (id, timestamp, kind_json, window_json, "
                "display_topology_json, primary_display_id, metadata_json, "
                "focused_element_json, scene_annotation_json, annotation_status) "
                "VALUES (?, ?, ?, ?, '[]', 'main', ?, ?, ?, ?)",
                (row["id"], row["timestamp"], row["kind_json"], row["window_json"],
                 row["metadata_json"], row["focused_element_json"],
                 row["scene_annotation_json"], row["annotation_status"]),
            )
        conn.commit()
        conn.close()
        return db_path


def _field(value: str, label: str = "Company", role: str = "AXTextField") -> dict:
    return {"role": role, "title": label, "value": value}


NEWS = "https://news.example.com/"
SHEET = "https://docs.example.com/sheet/daily"
SENT = "https://mail.example.com/sent"


def _run(rows: Rows, start: datetime, *, lead_app: str = "Mail",
         lead_title: str = "Daily digest #41 - Inbox", corrected: bool = False) -> tuple[datetime, datetime]:
    """One run of the daily routine: a message arrives, the user reads the
    news, fills the sheet, submits it and checks the sent view."""
    if lead_app:
        # Reading the message is not work on the task (a passive session
        # would otherwise start with it).
        rows.frame(start - timedelta(minutes=3), lead_app, lead_title, work=False)
    # AgentHandover's own window between the message and the run.
    rows.frame(start - timedelta(minutes=1), "AgentHandover", "AgentHandover", work=False)
    rows.click(start - timedelta(seconds=4), "Top stories", NEWS, tag="a")
    rows.frame(start, "Google Chrome", "News", NEWS)
    rows.frame(start + timedelta(minutes=1), "Google Chrome", "Sheet", SHEET,
               element=_field("Acme Inc"))
    if corrected:
        rows.frame(start + timedelta(minutes=1, seconds=40), "Google Chrome", "Sheet", SHEET,
                   element=_field("Acme, Inc."))
    rows.submit(start + timedelta(minutes=2), "Save row", SHEET)
    end = start + timedelta(minutes=2, seconds=5)
    rows.frame(end, "Google Chrome", "Sheet", SHEET)
    # The check, within a minute of the commit, and a later page that is not.
    rows.frame(start + timedelta(minutes=2, seconds=40), "Google Chrome", "Sent", SENT)
    rows.frame(start + timedelta(minutes=4), "Google Chrome", "Other", "https://other.example.com/")
    return start, end


# ---------------------------------------------------------------------------
# field_corrections
# ---------------------------------------------------------------------------


def _ax(when: datetime, value: str, **kw) -> dict:
    return {"id": f"ax-{value}-{when.second}", "timestamp": _ts(when),
            "window_json": _window("Numbers", "Invoices"),
            "focused_element_json": json.dumps(_field(value, **kw))}


def test_typing_continued_is_not_a_correction() -> None:
    events = [_ax(DAY, "Ac"), _ax(DAY + timedelta(seconds=5), "Acme"),
              _ax(DAY + timedelta(seconds=9), "Acme Inc")]
    assert field_corrections(events) == []


def test_a_changed_value_is_a_correction_from_before_to_after() -> None:
    events = [_ax(DAY, "Acme Inc"), _ax(DAY + timedelta(seconds=20), "Acme"),
              _ax(DAY + timedelta(seconds=30), "Acme, Inc.")]
    [correction] = field_corrections(events)
    assert (correction["field"], correction["before"], correction["after"]) == (
        "Company", "Acme Inc", "Acme, Inc.",
    )
    assert correction["event_ids"] == [events[0]["id"], events[2]["id"]]


def test_a_later_new_input_or_a_revert_or_a_search_is_not_a_correction() -> None:
    later = [_ax(DAY, "Acme Inc"), _ax(DAY + timedelta(minutes=10), "Globex")]
    reverted = [_ax(DAY, "Acme"), _ax(DAY + timedelta(seconds=5), "Acne"),
                _ax(DAY + timedelta(seconds=9), "Acme")]
    search = [_ax(DAY, "invoice 1042", role="AXSearchField"),
              _ax(DAY + timedelta(seconds=5), "invoice 1043", role="AXSearchField")]
    assert field_corrections(later) == []
    assert field_corrections(reverted) == []
    assert field_corrections(search) == []


def test_a_dom_field_correction() -> None:
    def snap(n: int, value: str) -> dict:
        return {"id": f"dom-{n}", "timestamp": _ts(DAY + timedelta(seconds=10 * n)),
                "metadata_json": json.dumps({"url": SHEET, "anchors": [
                    {"tag": "input", "selector": "#amount", "id": "amount", "name": "Amount",
                     "interacted": True, "value": value},
                    {"tag": "form", "selector": "form"},
                ]})}
    [correction] = field_corrections([snap(0, "1,250.00"), snap(1, "1,205.00")])
    assert (correction["field"], correction["before"], correction["after"]) == (
        "Amount", "1,250.00", "1,205.00",
    )


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def test_instance_signals_read_the_lead_in_the_commit_and_the_check(tmp_path: Path) -> None:
    rows = Rows()
    start, end = _run(rows, DAY, corrected=True)
    with WorkerDB(rows.write(tmp_path / "events.db")) as db:
        signals = rs.signals_from_db(db, start, end)

    # The message before the run, with its subject; not AgentHandover's
    # own window, not the click that opened the run's first page.
    [lead] = signals["lead_in"]
    assert (lead["app"], lead["subject"]) == ("Mail", "Daily digest #41")
    assert (signals["local_weekday"], signals["local_minute"]) == (0, 9 * 60)
    done = signals["done"]
    assert (done["kind"], done["label"], done["place"]) == ("form", "Save row", "docs.example.com/sheet/daily")
    # The sent view within a minute of the commit; the sheet itself (the
    # commit's own page) and the page four minutes on are not checks.
    assert [v["place"] for v in signals["verification"]] == ["mail.example.com/sent"]
    [correction] = signals["corrections"]
    assert (correction["before"], correction["after"]) == ("Acme Inc", "Acme, Inc.")


def test_a_click_labelled_send_is_a_commit() -> None:
    rows = Rows()
    rows.frame(DAY, "Google Chrome", "Channel", "https://chat.example.com/c/1")
    rows.click(DAY + timedelta(seconds=30), "Send now", "https://chat.example.com/c/1")
    rows.click(DAY + timedelta(seconds=40), "Sending options and settings for later", "https://chat.example.com/c/1")
    signals = rs.instance_signals(rows.rows, start=DAY, end=DAY + timedelta(minutes=1))
    assert (signals["done"]["kind"], signals["done"]["label"]) == ("send", "Send now")
    assert rs._commit(rows.rows[2]) is None  # a long label is not a control's name


def test_privacy_zones_hold_for_the_lead_in(tmp_path: Path) -> None:
    """A blocked window (a default one, 1Password; one the worker already
    marked blocked) never reaches the signals; a metadata-only window
    gives its app and nothing it showed."""
    rows = Rows()
    rows.frame(DAY - timedelta(minutes=4), "1Password", "Bank login - 1Password", work=False)
    blocked = rows.frame(DAY - timedelta(minutes=3), "Mail", "Diagnosis results - Inbox", work=False)
    private = rows.frame(DAY - timedelta(minutes=2), "Messages", "Dana: the password is x", work=False)
    rows.frame(DAY, "Google Chrome", "News", NEWS)
    rows.frame(DAY + timedelta(minutes=1), "Google Chrome", "News", NEWS)
    for row in rows.rows:
        if row["id"] == blocked:
            row["annotation_status"] = "privacy_blocked"
        if row["id"] == private:
            row["annotation_status"] = "metadata_only"
    with WorkerDB(rows.write(tmp_path / "events.db")) as db:
        signals = rs.signals_from_db(db, DAY, DAY + timedelta(minutes=1))
    assert signals["lead_in"] == [
        {"app": "Messages", "place": "", "title": "", "event_id": private},
    ]


# ---------------------------------------------------------------------------
# Three runs fill the ledger
# ---------------------------------------------------------------------------


def _procedure(tmp_path: Path, runs: int, **run_kw) -> dict:
    rows = Rows()
    spans = [_run(rows, DAY + timedelta(days=n), **run_kw) for n in range(runs)]
    with WorkerDB(rows.write(tmp_path / "events.db")) as db:
        observations = [
            {"date": s.date().isoformat(), "type": "focus", "session_id": f"session-{n}",
             "signals": rs.signals_from_db(db, s, e)}
            for n, (s, e) in enumerate(spans)
        ]
    return {
        "id": "daily-sheet", "title": "Daily sheet",
        "steps": [{"step_id": "step_1", "action": "Fill the sheet",
                   "evidence": {"frames": ["ev-003"], "dom_snapshot_ids": []}}],
        "evidence": {"observations": observations, "total_observations": runs},
    }


def test_three_runs_make_trigger_done_and_verification_hypotheses(tmp_path: Path) -> None:
    proc = _procedure(tmp_path, 3, corrected=True)
    knowledge = refresh_knowledge(proc)
    learned = proc["learned"]
    runs = [f"session-{n}" for n in range(3)]

    trigger = knowledge["trigger"]
    assert (trigger["status"], trigger["source"], trigger["value_ref"]) == (
        "hypothesis", "observation", "learned.trigger",
    )
    assert [e["id"] for e in trigger["evidence"]] == runs
    # The digest's number changes from day to day; the subject recurs.
    assert learned["trigger"]["kind"] == "event"
    assert learned["trigger"]["subject"] == "Daily digest #41"
    assert learned["trigger"]["description"] == 'Usually starts after the message "Daily digest #41" in Mail'

    verification = knowledge["verification"]
    assert verification["status"] == "hypothesis"
    assert {e["id"] for e in verification["evidence"] if e["kind"] == "instance"} == set(runs)
    [check] = learned["verification"]["checks"]
    assert (check["place"], check["support"]) == ("mail.example.com/sent", "3 of 3 runs")
    assert len(learned["verification"]["corrections"]) == 3
    assert any(e["kind"] == "frame" for e in verification["evidence"])

    done = knowledge["done_criteria"]
    assert (done["status"], done["value_ref"]) == ("hypothesis", "learned.done")
    assert len([e for e in done["evidence"] if e["kind"] == "frame"]) == 3

    body = render_body(proc)
    assert 'Observed start: Usually starts after the message "Daily digest #41" in Mail (3 of 3 runs' in body
    assert "Done when the form is submitted (\"Save row\") on docs.example.com/sheet/daily" in body
    assert "## Verification" in body
    assert 'The user corrected "Company" from "Acme Inc" to "Acme, Inc."' in body


def test_two_runs_leave_the_trigger_unknown(tmp_path: Path) -> None:
    proc = _procedure(tmp_path, 2)
    knowledge = refresh_knowledge(proc)
    assert knowledge["trigger"]["status"] == "unknown"
    assert knowledge["verification"]["status"] == "unknown"
    # One commit ends every run: the done point needs no three runs.
    assert knowledge["done_criteria"]["value_ref"] == "learned.done"
    assert "trigger" not in proc["learned"]


def test_learned_items_are_dropped_when_the_runs_no_longer_show_them(tmp_path: Path) -> None:
    proc = _procedure(tmp_path, 3)
    refresh_knowledge(proc)
    assert "trigger" in proc["learned"]
    for obs in proc["evidence"]["observations"]:
        obs["signals"] = {"lead_in": [], "done": None, "verification": [], "corrections": []}
    knowledge = refresh_knowledge(proc)
    assert "learned" not in proc
    assert knowledge["trigger"]["status"] == "unknown"


def _clock_instances(starts: list[datetime], lead_apps: list[str]) -> dict:
    return {
        "evidence": {"observations": [
            {"obs_id": f"obs-{n}", "signals": {
                "lead_in": [{"app": app, "place": "", "title": f"{app} window {n}"}] if app else [],
                "local_weekday": s.weekday(), "local_minute": s.hour * 60 + s.minute,
            }}
            for n, (s, app) in enumerate(zip(starts, lead_apps))
        ]},
    }


def test_disagreeing_lead_ins_with_a_shared_clock_make_a_time_trigger() -> None:
    mondays = [DAY + timedelta(weeks=n, minutes=m) for n, m in enumerate((88, 95, 110))]
    tuesday = DAY + timedelta(days=15, hours=5)
    proc = _clock_instances(mondays + [tuesday], ["Slack", "Mail", "Calendar", "Notes"])
    trigger = rs.learn(proc)["trigger"]
    # 10:28, 10:35 and 10:50: the median, to the quarter hour.
    assert (trigger["kind"], trigger["weekday"], trigger["time"]) == ("time", 0, "10:30")
    assert trigger["description"] == "Most Mondays around 10:30"
    assert trigger["instances"] == ["obs-0", "obs-1", "obs-2"]
    knowledge = derive_knowledge(proc)
    assert knowledge["trigger"]["status"] == "hypothesis"
    assert [e["id"] for e in knowledge["trigger"]["evidence"]] == ["obs-0", "obs-1", "obs-2"]


def test_a_daily_clock_is_most_weekdays_not_one_weekday() -> None:
    starts = [DAY + timedelta(days=d, minutes=(d * 3) % 10) for d in range(12) if d % 7 < 5]
    proc = _clock_instances(starts, [""] * len(starts))
    trigger = rs.learn(proc)["trigger"]
    assert (trigger["weekday"], trigger["description"]) == (None, "Most weekdays around 09:00")


def test_runs_at_scattered_times_teach_no_trigger() -> None:
    starts = [DAY + timedelta(days=d, hours=h) for d, h in ((0, 0), (1, 3), (2, 6), (3, 8))]
    assert "trigger" not in rs.learn(_clock_instances(starts, [""] * 4))


def test_a_calendar_event_in_the_lead_in_is_the_triggers_value(tmp_path: Path) -> None:
    rows = Rows()
    rows.frame(DAY - timedelta(minutes=2), "Calendar", "Expense review - Calendar")
    rows.frame(DAY, "Google Chrome", "Reports", "https://expenses.example.com/reports")
    signals = rs.instance_signals(rows.rows, start=DAY, end=DAY + timedelta(minutes=5))
    assert signals["lead_in"][0]["calendar_event"] == "Expense review"

    starts = [DAY + timedelta(weeks=n) for n in range(3)]
    proc = _clock_instances(starts, ["Slack", "Mail", ""])
    proc["evidence"]["observations"][2]["signals"]["lead_in"] = signals["lead_in"]
    learned = rs.learn(proc)
    assert learned["trigger"]["kind"] == "time"
    assert learned["trigger"]["calendar_event"] == "Expense review"
    assert derive_knowledge(proc)["trigger"]["value_ref"] == "learned.trigger.calendar_event"


# ---------------------------------------------------------------------------
# Passive observations carry the signals into the draft
# ---------------------------------------------------------------------------


def test_recorded_observations_carry_their_signals_into_the_pointers(tmp_path: Path) -> None:
    rows = Rows()
    _run(rows, DAY)
    with WorkerDB(rows.write(tmp_path / "events.db")) as db:
        events = db.get_annotated_events_in_window(hours=24, now=DAY + timedelta(hours=2))
        assert wm.record_observations(db, events, now=DAY + timedelta(hours=2)) == 1
        [obs] = db.get_workflow_observations()
    [pointer] = wm.observation_pointers([obs])
    assert pointer["signals"]["done"]["label"] == "Save row"
    assert pointer["signals"]["lead_in"][0]["subject"] == "Daily digest #41"
