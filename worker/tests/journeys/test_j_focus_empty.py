"""J-focus, empty leg: a focus session recorded while observation was off.

Live on the test Mac (2026-09-24) a session started while the daemon was
down had no events; after the stop the worker logged "has no events yet,
will retry" every tick and the app showed "Analyzing ..." for good.  The
worker now waits a grace after the stop, then closes the session as
nothing recorded (``focus-result.json``) and removes the signal; a session
still recording is left alone, and the wait is logged once.
"""

from __future__ import annotations

import json
import logging
import os
import time

from agenthandover_worker import main as wm
from journeys.focus_run import needs_focus_sop_recording, start_focus_run, tick_until_skill

SESSION = "0e7f0c55-daemon-was-down"


def _signal(state_dir, status: str, *, stopped_seconds_ago: float) -> None:
    path = state_dir / "focus-session.json"
    path.write_text(json.dumps({
        "session_id": SESSION, "title": "Note the IANA reserved domains",
        "started_at": "2026-09-24T09:10:00Z", "status": status,
    }), encoding="utf-8")
    written = time.time() - stopped_seconds_ago
    os.utime(path, (written, written))


def test_a_session_with_no_events_is_closed_as_nothing_recorded(
    events_db, worker, clock, temp_home, caplog,
) -> None:
    journey = worker(events_db("focus_dailynews", rebase_to=clock.now()))
    state_dir = temp_home.data_dir
    signal_path = state_dir / "focus-session.json"
    result_path = state_dir / wm.FOCUS_RESULT_FILE

    # Still recording, however long: left alone.
    _signal(state_dir, "recording", stopped_seconds_ago=3600)
    journey.run_tick()
    assert signal_path.exists() and not result_path.exists()

    # Stopped a moment ago: the daemon's last frames may still come.
    _signal(state_dir, "stopped", stopped_seconds_ago=5)
    with caplog.at_level(logging.INFO, logger="agenthandover_worker"):
        for _ in range(3):
            journey.run_tick()
    assert signal_path.exists() and not result_path.exists()
    waits = [r for r in caplog.records if SESSION in r.getMessage()]
    assert len(waits) == 1, [r.getMessage() for r in waits]

    # The grace after the stop is over: closed, and the app is told why.
    os.utime(signal_path, (time.time() - wm.EMPTY_SESSION_GRACE_SECONDS - 1,) * 2)
    journey.run_tick()
    assert not signal_path.exists()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["session_id"] == SESSION
    assert result["status"] == "nothing_recorded"
    assert result["message"] == "Nothing was recorded: observation was off during this session"
    assert result["stopped_at"] and result["closed_at"]
    assert journey.knowledge_base.list_procedures() == []


@needs_focus_sop_recording
def test_the_next_recorded_session_clears_the_old_outcome(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    result_path = temp_home.data_dir / wm.FOCUS_RESULT_FILE
    temp_home.data_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"session_id": SESSION, "status": "nothing_recorded"}))
    run = start_focus_run(
        events_db=events_db, worker=worker, clock=clock,
        temp_home=temp_home, cassette=cassette,
    )
    tick_until_skill(run, clock)
    assert not result_path.exists()


def test_the_grace_counts_from_the_signals_stop_time(tmp_path) -> None:
    """A signal that says when it was stopped (``stopped_at``) is timed
    from that, not from when the file was last written."""
    signal = {"session_id": "s-stopped-at", "title": "T", "status": "stopped",
              "stopped_at": "2026-09-24T09:00:00Z"}
    path = tmp_path / "focus-session.json"
    path.write_text(json.dumps(signal))
    stop = 1_790_240_400.0  # 2026-09-24T09:00:00Z
    assert not wm._close_empty_focus_session(tmp_path, path, signal, now=stop + 60)
    assert path.exists()
    assert wm._close_empty_focus_session(tmp_path, path, signal, now=stop + 121)
    result = json.loads((tmp_path / wm.FOCUS_RESULT_FILE).read_text())
    assert result["stopped_at"] == "2026-09-24T09:00:00+00:00"
    assert not path.exists()
