"""The reasoner's daily budget, on the worker as ``main()`` builds it.

The LLM reasoner and the behavioral synthesizer used to share their daily
budget with the VLM fallback queue: every translation the event pipeline
rejected queued a fallback job, and once 50 were queued in a day every later
reasoning call (``task_segmenter.label_cluster``, synthesis) was skipped,
logged only at DEBUG.  The queue is gone; the reasoner and the synthesizer
have their own budget (``scheduler.ReasoningBudget``), and when it is spent
``worker-status.json`` says so and the worker logs it at INFO once a day.

No model answers here: the model boundaries are patched, and Ollama is a
closed port.
"""

from __future__ import annotations

import json
import logging
import socket
import sqlite3
import uuid
from datetime import timedelta
from pathlib import Path

import pytest

from agenthandover_worker import behavioral_synthesizer
from agenthandover_worker.llm_reasoning import LLMReasoner
from agenthandover_worker.scheduler import REASONING_BUDGET_REASON
from agenthandover_worker.task_segmenter import AnnotatedFrame
from daemon_db import create_daemon_db

LABEL = "Reconcile the Stripe payout"
SYNTHESIS = json.dumps({"goal": "Reconciles the day's Stripe payout", "confidence": 0.8})


def _closed_port_host() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


def _rejected_events(db_path: Path, n: int, at) -> None:
    """*n* events of an unknown kind with no window or metadata: the
    pipeline translates each one, and the scorer rejects every translation
    (unknown intent, confidence 0.0)."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executemany(
            "INSERT INTO events (id, timestamp, kind_json, window_json, metadata_json, "
            "display_topology_json, primary_display_id, processed) "
            "VALUES (?, ?, ?, '{}', '{}', '[]', 'main', 0)",
            [
                (
                    str(uuid.uuid4()),
                    (at + timedelta(seconds=i)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    json.dumps({"UnknownEvent": {}}),
                )
                for i in range(n)
            ],
        )
        conn.commit()
    finally:
        conn.close()


def _count(db_path: Path, sql: str) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(sql).fetchone()[0]
    finally:
        conn.close()


@pytest.fixture()
def models(monkeypatch):
    """The reasoner's and the synthesizer's model calls, answered here:
    ``models["reasoning"]`` and ``models["synthesis"]`` count them."""
    calls = {"reasoning": 0, "synthesis": 0}

    def reason(self, prompt, system, num_predict=None, think=None):
        calls["reasoning"] += 1
        return LABEL, 2.0

    def synthesize(**kwargs):
        calls["synthesis"] += 1
        return SYNTHESIS, 3.0

    monkeypatch.setattr(LLMReasoner, "_call_ollama", reason)
    monkeypatch.setattr(behavioral_synthesizer, "_call_ollama_text", synthesize)
    return calls


def _frames() -> list[AnnotatedFrame]:
    return [AnnotatedFrame(
        event_id="e1", timestamp="2026-09-15T09:00:00.000Z", annotation={},
        what_doing="Matching payout po_1 against the bank sheet",
        app="Google Chrome", location="https://dashboard.stripe.com/payouts/po_1",
    )]


def _procedure() -> dict:
    return {"id": "stripe-payout", "title": "Stripe payout", "steps": [
        {"step": "Open the payout", "app": "Google Chrome"},
    ]}


def _observations() -> list[list[dict]]:
    return [[{"what_doing": "Open the payout", "app": "Google Chrome"}]]


def _ready(journey) -> None:
    """No model scheduling: Ollama is a closed port, and the calls are
    answered by :func:`models`."""
    journey.llm_reasoner.before_call = None
    journey.behavioral_synthesizer.before_call = None


def test_sixty_rejected_translations_stop_neither_labelling_nor_synthesis(
    temp_home, worker, clock, models,
) -> None:
    db_path = create_daemon_db(temp_home.data_dir / "events.db")
    _rejected_events(db_path, 60, clock.now() - timedelta(minutes=5))
    journey = worker(db_path)
    _ready(journey)

    result = journey.run_tick()
    assert result["pipeline"]["translations"] == 60
    # The translations are only counted: no fallback job, in memory or in
    # the daemon's vlm_queue table.
    assert "vlm_enqueued" not in result["pipeline"]
    assert _count(db_path, "SELECT COUNT(*) FROM vlm_queue") == 0
    assert _count(db_path, "SELECT COUNT(*) FROM events WHERE processed = 1") == 60

    # The same day, cluster labelling and synthesis still reach the model.
    assert journey.task_segmenter._label_cluster_with_llm(_frames()) == LABEL
    insights = journey.behavioral_synthesizer.synthesize(
        "stripe-payout", _procedure(), _observations(), force=True,
    )
    assert insights.goal == "Reconciles the day's Stripe payout"
    assert models == {"reasoning": 1, "synthesis": 1}

    journey.run_tick()
    status = journey.status()
    assert status["reasoning_paused_reason"] is None
    assert status["reasoning_budget"]["calls"] == 2
    assert status["reasoning_budget"]["skipped"] == 0
    for key in ("vlm_queue_pending", "vlm_jobs_today", "vlm_dropped_today"):
        assert key not in status


def test_a_spent_reasoning_budget_is_published_and_logged_once_a_day(
    temp_home, worker, clock, models, caplog,
) -> None:
    temp_home.configure(
        ollama_host=_closed_port_host(), vlm={"reasoning_max_calls_per_day": 2},
    )
    db_path = create_daemon_db(temp_home.data_dir / "events.db")
    journey = worker(db_path)
    _ready(journey)
    segmenter = journey.task_segmenter

    with caplog.at_level(logging.INFO, logger="agenthandover_worker.scheduler"):
        assert segmenter._label_cluster_with_llm(_frames()) == LABEL
        assert journey.behavioral_synthesizer.synthesize(
            "stripe-payout", _procedure(), _observations(), force=True,
        ).goal
        # The day's two calls are spent: both callers are refused.
        assert segmenter._label_cluster_with_llm(_frames()) is None
        assert journey.behavioral_synthesizer.synthesize(
            "stripe-payout", _procedure(), _observations(), force=True,
        ).goal == ""
        assert segmenter._label_cluster_with_llm(_frames()) is None
    assert models == {"reasoning": 1, "synthesis": 1}
    reached = [r for r in caplog.records if "Reasoning budget reached" in r.getMessage()]
    assert [r.levelno for r in reached] == [logging.INFO]
    assert "task_segmenter.label_cluster" in reached[0].getMessage()

    journey.run_tick()
    status = journey.status()
    assert status["reasoning_paused_reason"] == REASONING_BUDGET_REASON
    assert status["reasoning_budget"] | {"date": ""} == {
        "date": "", "calls": 2, "max_calls": 2, "compute_minutes": 0.08,
        "max_compute_minutes": 90.0, "skipped": 3,
    }
    # The counters are persisted, so a restart does not reopen the day.
    saved = json.loads(journey.db.get_state("reasoning_budget"))
    assert (saved["calls"], saved["skipped"]) == (2, 3)

    # The next local day starts over; the reset reports the day's skips.
    caplog.clear()
    clock.advance(days=1)
    with caplog.at_level(logging.INFO, logger="agenthandover_worker.scheduler"):
        journey.run_tick()
        assert segmenter._label_cluster_with_llm(_frames()) == LABEL
    assert journey.status()["reasoning_paused_reason"] is None
    assert any(
        "3 model call(s) skipped over the daily limit" in r.getMessage()
        for r in caplog.records
    )
