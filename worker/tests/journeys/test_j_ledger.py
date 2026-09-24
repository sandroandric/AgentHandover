"""J-ledger: what four runs of one routine teach, slot by slot.

The runs are the daily-news fixtures (``record/generate_focus_dailynews.py``):
J-focus's recording, a morning that leaves a story out (a video with no
transcript), and two mornings that take the same detour (a paywalled story
read through an archive).  WP-L4's micro-questions are skipped through
the app's answer files: each morning's in the Q&A window
(``focus-questions.json``), any review card left open with the card's
Skip (a blank ``micro-answer``); unit pin
``test_questions.py::test_a_skipped_question_leaves_its_slot_as_it_was``.
The ledger's schema (``proc["knowledge"]``, see
``agenthandover_worker/ledger.py``) landed with WP-L1, the learners with
WP-L2 (contrast), WP-L3 (run signals) and WP-M5 (divergence branches).

The whole journey needs the model's answers for the three extra runs
(annotation, SOP, synthesis), which are not recorded yet (the questions
call no model since WP-L4).  The parts that call no model run now over the
same fixtures, loaded into one events database the way the daemon keeps
every morning: what the runs' events teach about the trigger, the done
criterion and the check (``run_signals``), the selection criterion the
chosen headlines carry (``contrast``), and the branch the two detours
make (``branch_mining``).  For the last two, the frames are described the
way a faithful annotator would (app from the window, location from the
page the browser extension reported under that window title, screen text
from the OCR, list entries from the front page's OCR): the model is what
the recordings will supply.
"""

from __future__ import annotations

import json
import os
import re

import pytest

from agenthandover_worker import branch_mining, run_signals, trigger_spool
from agenthandover_worker import questions as qs
from agenthandover_worker.contrast import criteria_hypotheses, ground_visible_items
from agenthandover_worker.db import WorkerDB
from agenthandover_worker.event_helpers import display_name, parse_timestamp
from agenthandover_worker.evidence_tracker import ObservationEvidence, append_observation
from agenthandover_worker.ledger import readiness, refresh_knowledge
from agenthandover_worker.task_segmenter import AnnotatedFrame
from journeys.cassette import CASSETTE_DIR, load_entries
from journeys.fixture_loader import LoadedFixture
from journeys.focus_run import (
    FOCUS_CASSETTE,
    TICK_SECONDS,
    needs_focus_sop_recording,
    skip_pending_questions,
    start_focus_run,
    tick_until_skill,
)

RUNS = (
    "focus_dailynews", "focus_dailynews_skip",
    "focus_dailynews_diverge", "focus_dailynews_diverge2",
)
SKIP_RUN, DIVERGE_RUNS = RUNS[1], RUNS[2:]
FRONT_PAGE = "https://news.ycombinator.com/"
PAYWALL_LINE = "Subscribe to continue reading"
ARCHIVE_HOST = "archive.example.net"

# The first model request of each extra run (the annotation of its first
# keyframe; a standalone run of the fixture on the focus_dailynews
# cassette misses it).  The SOP and synthesis calls that follow depend on
# these answers, so they cannot be named before the recording.
EXTRA_RUN_MISSING_KEYS: dict[str, str] = {
    "focus_dailynews_skip": "93414f3ff1c29740ef33e93342b84ab116e26245258d0a156141b0d57e72ca15",
    "focus_dailynews_diverge": "3f20f2202b8b75ccc6c40b2170af5a05737a407a66918dd0780c0b499a5791b0",
    "focus_dailynews_diverge2": "28d4fe0afe5a63a264b5b2efab28a3dcc96f61f38b689f00c4bf6c0b3eb75941",
}
_ENTRIES = load_entries(CASSETTE_DIR / f"{FOCUS_CASSETTE}.jsonl")
_MISSING = [f"{name} {key}" for name, key in EXTRA_RUN_MISSING_KEYS.items() if key not in _ENTRIES]
needs_extra_run_recordings = pytest.mark.skipif(
    bool(_MISSING) and os.environ.get("AH_RECORD") != "1",
    reason=(
        f"cassettes/{FOCUS_CASSETTE}.jsonl lacks the model calls of the extra "
        "daily-news runs: " + ", ".join(_MISSING)
        + " (and the SOP and synthesis calls each unblocks); "
        "re-record J-ledger with AH_RECORD=1 on the test Mac"
    ),
)


@needs_focus_sop_recording
@needs_extra_run_recordings
def test_runs_fill_the_ledger_with_evidence(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    run = start_focus_run(
        events_db=events_db, worker=worker, clock=clock,
        temp_home=temp_home, cassette=cassette,
    )
    tick_until_skill(run, clock)
    sessions = {_session(run.fixture)}
    for name in RUNS[1:]:
        # The next morning: the run lands in the same events database.
        clock.advance(days=1)
        extra = events_db(name, rebase_to=clock.now())
        extra.write_focus_signal(temp_home.data_dir)
        sessions.add(_session(extra))
        _tick_until_observed(run, clock, _session(extra))
    _skip_micro_questions(run, clock, temp_home.data_dir)

    # Every run is an instance of the one Skill.
    [proc] = run.procedures()
    knowledge = proc["knowledge"]
    # The selection criteria come from contrasting chosen with skipped
    # items across the runs (WP-L2), each run cited.
    decisions = knowledge["decisions"]
    assert decisions["status"] == "hypothesis"
    assert decisions["source"] == "contrast"
    assert {e["id"] for e in decisions["evidence"] if e["kind"] == "instance"} == sessions
    # The two diverging runs add a branch (WP-M5).
    [branch] = proc["branches"]
    assert PAYWALL_LINE in branch["condition"]
    # What starts the routine and how the user checks it recur across the
    # runs (WP-L3; unit pins in test_run_signals.py).
    for slot in ("trigger", "verification"):
        assert knowledge[slot]["status"] == "hypothesis"
        assert {e["id"] for e in knowledge[slot]["evidence"] if e["kind"] == "instance"} == sessions
    # The micro-questions were skipped (WP-L4): no slot claims a user answer.
    assert not any(
        e["kind"] == "answer" for entry in knowledge.values() for e in entry["evidence"]
    )
    assert readiness(proc).ready


def _skip_micro_questions(run, clock, state_dir) -> None:
    """Skip every micro-question the runs raised, as the app does: the Q&A
    window's were skipped in ``focus-questions.json`` while ticking; each
    review card still open in ``micro-questions.json`` gets the card's
    Skip, a ``micro-answer`` with a blank answer.  One more tick reads
    both, and every question asked is closed as skipped."""
    path = state_dir / qs.MICRO_QUESTIONS_FILE
    cards = json.loads(path.read_text(encoding="utf-8"))["questions"] if path.is_file() else []
    for card in cards:
        trigger_spool.write_trigger(
            state_dir, "micro-answer", {"question_id": card["id"], "answer": ""},
        )
    clock.advance(seconds=TICK_SECONDS)
    run.ticks.append(run.worker.run_tick())
    assert run.ticks[-1]["triggers_consumed"] == len(cards)
    asked = run.worker.db.get_micro_questions()
    assert run.questions_skipped and asked
    assert {row["status"] for row in asked} == {"skipped"}, asked


def _tick_until_observed(run, clock, session_id: str, *, max_ticks: int = 6) -> None:
    """Tick (skipping the post-recording questions, as the app's Skip
    does) until a procedure cites the run *session_id* as an instance."""
    for _ in range(max_ticks):
        clock.advance(seconds=TICK_SECONDS)
        run.ticks.append(run.worker.run_tick())
        run.questions_skipped += skip_pending_questions(run.worker.state_dir)
        if any(
            isinstance(obs, dict) and obs.get("session_id") == session_id
            for proc in run.procedures()
            for obs in (proc.get("evidence") or {}).get("observations") or []
        ):
            return
    raise AssertionError(f"run {session_id} is no procedure's instance: {run.ticks[-1]!r}")


# ---------------------------------------------------------------------------
# The parts that call no model
# ---------------------------------------------------------------------------


def _load_runs(events_db) -> list[LoadedFixture]:
    """Every run in one events database, on the mornings they were recorded."""
    return [events_db(name) for name in RUNS]


def _session(fixture: LoadedFixture) -> str:
    return fixture.manifest["focus_session"]["session_id"]


def _run_events(db: WorkerDB, fixture: LoadedFixture) -> tuple[list[dict], list[dict]]:
    """The run's focus events, and every event from its first to its last."""
    focus = db.get_focus_session_events(_session(fixture))
    times = [parse_timestamp(e["timestamp"]) for e in focus]
    return focus, db.get_events_between(min(times), max(times))


def _annotated_frames(db: WorkerDB, fixture: LoadedFixture) -> list[AnnotatedFrame]:
    """The run's focus frames as a faithful annotator describes them."""
    focus, around = _run_events(db, fixture)
    page_urls: dict[str, str] = {}
    for event in around:
        metadata = json.loads(event["metadata_json"])
        if metadata.get("title") and metadata.get("url"):
            page_urls[metadata["title"]] = metadata["url"]
    frames = []
    for event in focus:
        metadata = json.loads(event["metadata_json"])
        if not metadata.get("focus_capture"):
            continue
        window = json.loads(event["window_json"])
        app = display_name(window["app_id"])
        location = page_urls.get(window["title"], window["title"]) if app == "Google Chrome" else ""
        frames.append(AnnotatedFrame(
            event_id=event["id"], timestamp=event["timestamp"],
            annotation={"app": app, "location": location,
                        "_screen_text": metadata["ocr"]["full_text"]},
            app=app, location=location,
        ))
    return frames


def test_the_runs_teach_trigger_done_and_check(events_db) -> None:
    """Each run's events, read as the focus pipeline reads them
    (``run_signals_from_db``), make the trigger (the same time every
    morning), the done criterion (Send in the channel) and the check (the
    channel after posting) hypotheses that cite every run."""
    fixtures = _load_runs(events_db)
    proc: dict = {"id": "daily-news-digest", "steps": [], "goal": "Post the daily news digest"}
    with WorkerDB(fixtures[0].db_path) as db:
        for fixture in fixtures:
            focus, _ = _run_events(db, fixture)
            append_observation(proc, ObservationEvidence(
                date=focus[0]["timestamp"][:10], type="focus", duration_minutes=0,
                session_id=_session(fixture), event_count=len(focus),
                signals=run_signals.run_signals_from_db(db, focus),
            ), once_per_session=True)
    knowledge = refresh_knowledge(proc)
    sessions = {_session(f) for f in fixtures}
    learned = proc["learned"]

    assert learned["trigger"]["kind"] == "time"
    assert learned["trigger"]["description"].startswith("Most ")
    assert learned["done"]["label"] == "Send now"
    assert learned["done"]["place"].startswith("app.slack.com/")
    [check] = learned["verification"]["checks"]
    assert check["title"] == "daily-news (Channel) - Example Co - Slack"
    for slot in ("trigger", "done_criteria", "verification"):
        assert knowledge[slot]["status"] == "hypothesis"
        assert knowledge[slot]["source"] == "observation"
        assert {e["id"] for e in knowledge[slot]["evidence"] if e["kind"] == "instance"} == sessions


def test_the_chosen_headlines_carry_a_criterion(events_db) -> None:
    """Contrasting the headline clicked with the ones left on the front
    page, run by run, gives the routine's criterion: the "Show HN" post,
    picked every morning and never skipped."""
    fixtures = _load_runs(events_db)
    instances = []
    with WorkerDB(fixtures[0].db_path) as db:
        for fixture in fixtures:
            _, around = _run_events(db, fixture)
            front_ocr, lists = "", []
            for event in around:
                kind = json.loads(event["kind_json"])["type"]
                metadata = json.loads(event["metadata_json"])
                window = json.loads(event["window_json"] or "{}")
                if metadata.get("focus_capture") and window.get("title") == "Hacker News":
                    front_ocr = metadata["ocr"]["full_text"]
                elif kind == "ClickIntent" and metadata.get("url") == FRONT_PAGE:
                    clicked = metadata["target"]["innerText"]
                    headlines = [
                        m.group(1) for line in front_ocr.splitlines()
                        if (m := re.match(r"^\d+\. (.+)$", line))
                    ]
                    assert clicked in headlines
                    lists.append(ground_visible_items(
                        [{"label": h, "chosen": h == clicked} for h in headlines],
                        None, front_ocr,
                    ))
            assert len(lists) == 3
            instances.append({"id": _session(fixture), "lists": lists})

    best = criteria_hypotheses(instances)[0]
    assert (best["attribute"], best["value"]) == ("keyword", "show")
    assert best["counter"] == 0
    assert {e["id"] for e in best["evidence"]} == {_session(f) for f in fixtures}


def _mine_divergences(events_db) -> tuple[dict, dict[str, str]]:
    """The first run as the Skill (one step per place it visited), the
    other runs' divergences recorded on it."""
    fixtures = _load_runs(events_db)
    with WorkerDB(fixtures[0].db_path) as db:
        frames = {f.name: _annotated_frames(db, f) for f in fixtures}
    steps: list[dict] = []
    for frame in frames[RUNS[0]]:
        if not steps or (steps[-1]["app"], steps[-1]["location"]) != (frame.app, frame.location):
            steps.append({"action": f"Open {frame.location or frame.app}",
                          "app": frame.app, "location": frame.location})
    proc: dict = {"id": "daily-news-digest", "steps": steps}
    for fixture in fixtures[1:]:
        assert branch_mining.record_divergences(proc, _session(fixture), frames[fixture.name])
    return proc, {f.name: _session(f) for f in fixtures}


def _instances(entry: dict) -> set[str]:
    return {e["id"] for e in entry["evidence"] if e["kind"] == "instance"}


def test_the_paywall_detour_makes_a_branch(events_db) -> None:
    """The two mornings that read a paywalled story through the archive
    agree (the same steps replaced by the same places, a screen that
    shares the paywall's lines): the branch names the paywall and cites
    both, and nothing the skipping morning did becomes a branch."""
    proc, sessions = _mine_divergences(events_db)
    diverging = {sessions[name] for name in DIVERGE_RUNS}

    paywall = [b for b in proc["branches"] if PAYWALL_LINE in b["condition"]]
    assert len(paywall) == 1
    [branch] = paywall
    assert branch["source"] == "divergence"
    assert _instances(branch) == diverging
    assert "research.example.com" in branch["condition_state"]["ocr_snippet"]
    # The skip left its own hypothesis (the video glanced at instead of
    # the second story's steps) that no other run agrees with.
    [skipped] = [h for h in proc["branch_hypotheses"]
                 if branch_mining._instance(h) == sessions[SKIP_RUN]]
    assert [s["location"].split("/")[2] for s in skipped["alternative_steps"]] == [
        "video.example.com"]
    assert all(_instances(b) == diverging for b in proc["branches"])


def test_one_detour_makes_one_branch(events_db) -> None:
    """The paywalled story's detour (the archived copy, the note, back to
    the archived copy) is one branch: it replaces the Skill's "note the
    story, back to the story" steps with the detour's own steps, the
    story opened on the paywalled site being the Skill's "open the story"
    step."""
    proc, sessions = _mine_divergences(events_db)
    [branch] = proc["branches"]
    assert PAYWALL_LINE in branch["condition"]
    steps = proc["steps"]
    assert [steps[i]["app"] for i in range(branch["step_index"], branch["rejoin_index"])] == [
        "Notes", "Google Chrome"]
    assert [s["location"].split("/")[2] if s["location"] else s["app"]
            for s in branch["alternative_steps"]] == [ARCHIVE_HOST, "Notes", ARCHIVE_HOST]
    # Each diverging run left exactly one hypothesis.
    for name in DIVERGE_RUNS:
        assert len([h for h in proc["branch_hypotheses"]
                    if branch_mining._instance(h) == sessions[name]]) == 1
