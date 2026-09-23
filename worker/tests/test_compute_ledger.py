"""Regression pins for WP-C1: the worker's compute ledger (model seconds
per call type, annotation retries and deferrals, the per-frame wall-time
histogram, and the last-hour roll-up published in ``worker-status.json``).
The journey that reads it is ``journeys/test_j_compute.py``."""

from __future__ import annotations

from agenthandover_worker.ollama import Response
from agenthandover_worker.ops_telemetry import ComputeLedger


class _Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


def _reply(seconds: float) -> Response:
    return Response(text="", total_duration=int(seconds * 1e9), elapsed=0.01)


def test_seconds_and_calls_per_call_type():
    ledger = ComputeLedger(now=_Clock())
    ledger.record_call("annotation", _reply(2.0))
    ledger.record_call("annotation", _reply(1.5))
    ledger.record_call("sop", _reply(40.0))
    snap = ledger.snapshot()
    assert snap["model_seconds_by_call_type"] == {"annotation": 3.5, "sop": 40.0}
    assert snap["model_calls_by_call_type"] == {"annotation": 2, "sop": 1}
    assert snap["model_seconds"] == 43.5


def test_a_reply_without_ollama_durations_counts_its_wall_time():
    ledger = ComputeLedger(now=_Clock())
    ledger.record_call("other", Response(text="", elapsed=0.75))
    assert ledger.snapshot()["model_seconds_by_call_type"] == {"other": 0.75}


def test_retries_deferrals_and_histogram():
    ledger = ComputeLedger(now=_Clock())
    ledger.record_annotation(0.4, 1)      # le_500
    ledger.record_annotation(1.8, 2)      # le_2000, one retry (vision after text)
    ledger.record_annotation(75.0, 1)     # slower than the last edge
    ledger.record_annotation(0.0, 0)      # a cache hit: no model call, not counted
    ledger.record_deferred()
    snap = ledger.snapshot()
    assert snap["annotation_retries"] == 1
    assert snap["annotation_deferred"] == 1
    assert snap["annotation_wall_ms_histogram"] == {
        "le_500": 1, "le_2000": 1, "gt_60000": 1,
    }
    # Buckets are published fastest first.
    assert list(snap["annotation_wall_ms_histogram"]) == ["le_500", "le_2000", "gt_60000"]


def test_last_hour_drops_what_is_older_than_an_hour():
    clock = _Clock()
    ledger = ComputeLedger(now=clock)
    ledger.record_call("annotation", _reply(5.0))
    ledger.record_annotation(5.0, 2)
    clock.t += 1800
    ledger.record_call("sop", _reply(30.0))
    snap = ledger.snapshot()
    assert snap["last_hour"]["model_seconds_by_call_type"] == {"annotation": 5.0, "sop": 30.0}

    clock.t += 1801  # the annotation is now 3601 s old
    snap = ledger.snapshot()
    last_hour = snap["last_hour"]
    assert last_hour["model_seconds_by_call_type"] == {"sop": 30.0}
    assert last_hour["model_seconds"] == 30.0
    assert last_hour["annotation_retries"] == 0
    assert last_hour["annotation_wall_ms_histogram"] == {}
    # Since start-up keeps everything.
    assert snap["model_seconds_by_call_type"] == {"annotation": 5.0, "sop": 30.0}
    assert snap["annotation_retries"] == 1

    clock.t += 3600
    assert ledger.snapshot()["last_hour"]["model_calls_by_call_type"] == {}


def test_reset_clears_both_views():
    ledger = ComputeLedger(now=_Clock())
    ledger.record_call("embed", _reply(0.2))
    ledger.record_deferred()
    ledger.reset()
    snap = ledger.snapshot()
    assert snap["model_seconds_by_call_type"] == {}
    assert snap["annotation_deferred"] == 0
    assert snap["last_hour"]["model_seconds"] == 0.0
