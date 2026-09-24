"""WP-C7: model work runs on one inference thread; the control thread
keeps its heartbeat and triggers going while a model call runs."""

from __future__ import annotations

import threading
import time

import pytest

from agenthandover_worker import inference_runner as ir
from agenthandover_worker.inference_runner import (
    P0, P1, P2, STATE_LOCK, InferenceRunner, StateLock, model_wait, model_work,
)


@pytest.fixture()
def runner():
    runners: list[InferenceRunner] = []

    def make(**kwargs) -> InferenceRunner:
        r = InferenceRunner(**kwargs)
        runners.append(r)
        return r

    yield make
    for r in runners:
        r.stop(timeout=5)
    assert ir._ACTIVE is None


class Gate:
    """A job that blocks until released, so tests can queue behind it."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self) -> None:
        self.started.set()
        assert self.release.wait(5), "gate never released"


def test_jobs_run_on_one_thread_highest_priority_first(runner) -> None:
    r = runner().start()
    order: list[str] = []
    threads: set[int] = set()
    gate = Gate()

    def job(name: str):
        def run() -> None:
            order.append(name)
            threads.add(threading.get_ident())
        return run

    assert r.submit("gate", P2, gate)
    assert gate.started.wait(5)
    assert r.submit("passive", P2, job("passive"))
    assert r.submit("tail", P1, job("tail"))
    assert r.submit("focus", P0, job("focus"))
    assert r.submit("passive-2", P2, job("passive-2"))
    gate.release.set()
    assert r.wait_idle(5)

    assert order == ["focus", "tail", "passive", "passive-2"]
    assert len(threads) == 1 and threading.get_ident() not in threads


def test_a_queued_or_running_key_is_not_queued_again(runner) -> None:
    r = runner().start()
    gate = Gate()
    runs: list[str] = []
    r.submit("passive", P2, gate)
    assert gate.started.wait(5)
    # Running: a resubmission is a no-op that reports the job as there.
    assert r.submit("passive", P2, lambda: runs.append("second"))
    assert r.submit("focus", P0, lambda: runs.append("focus"))
    assert r.submit("focus", P0, lambda: runs.append("focus again"))
    assert r.status()["queued"] == ["P0 focus"]
    gate.release.set()
    assert r.wait_idle(5)
    assert runs == ["focus"]


def test_the_queue_is_bounded(runner) -> None:
    r = runner(max_queued=2).start()
    gate = Gate()
    r.submit("gate", P2, gate)
    assert gate.started.wait(5)
    assert r.submit("a", P0, lambda: None)
    assert r.submit("b", P0, lambda: None)
    assert not r.submit("c", P0, lambda: None)
    assert not r.busy("c")
    gate.release.set()
    assert r.wait_idle(5)


def test_a_failing_job_is_logged_counted_and_backs_off(runner, caplog) -> None:
    now = [100.0]
    r = runner(clock=lambda: now[0]).start()

    def boom() -> None:
        raise RuntimeError("model exploded")

    assert r.submit("passive", P2, boom)
    assert r.wait_idle(5)
    assert "Inference job passive failed" in caplog.text
    assert r.consecutive_failures() == 1
    status = r.status()
    assert status["failed"] == 1 and status["retry_after_seconds"] == {"passive": 5.0}
    # Backing off: not taken again until the backoff has passed.
    assert not r.submit("passive", P2, boom)
    now[0] += 5.0
    assert r.submit("passive", P2, lambda: None)
    assert r.wait_idle(5)
    assert r.consecutive_failures() == 0
    assert r.status()["completed"] == 1


def test_the_idle_hook_runs_when_the_queue_runs_empty(runner) -> None:
    events: list[str] = []
    r = runner(on_idle=lambda: events.append("idle")).start()
    gate = Gate()
    r.submit("gate", P2, gate)
    assert gate.started.wait(5)
    r.submit("a", P0, lambda: events.append("a"))
    r.submit("b", P2, lambda: events.append("b"))
    gate.release.set()
    assert r.wait_idle(5)
    # Once, after the last job: never between two jobs.
    assert events == ["a", "b", "idle"]


def test_status_names_the_running_job_and_how_long_it_has_run(runner) -> None:
    now = [0.0]
    r = runner(clock=lambda: now[0]).start()
    gate = Gate()
    r.submit("focus", P0, gate)
    assert gate.started.wait(5)
    r.submit("passive-annotation", P2, lambda: None)
    now[0] = 42.0
    status = r.status()
    assert status["running"] == "P0 focus"
    assert status["running_seconds"] == 42.0
    assert status["queued"] == ["P2 passive-annotation"]
    assert status["depth"] == 2
    gate.release.set()
    assert r.wait_idle(5)
    assert r.status()["running"] is None and r.status()["depth"] == 0


def test_the_control_thread_runs_while_the_inference_thread_waits_for_a_model(runner) -> None:
    """The state lock is free during a model wait, and only then."""
    r = runner().start()
    in_python = threading.Event()
    in_model = threading.Event()
    leave_python = threading.Event()
    leave_model = threading.Event()

    def job() -> None:
        in_python.set()
        leave_python.wait(5)
        with model_wait("test"):
            in_model.set()
            leave_model.wait(5)

    r.submit("focus", P0, job)
    assert in_python.wait(5)
    # The job runs Python on shared state: the control thread waits.
    got = threading.Event()

    def control_tick() -> None:
        with STATE_LOCK:
            got.set()

    t = threading.Thread(target=control_tick)
    t.start()
    assert not got.wait(0.3)
    leave_python.set()
    assert in_model.wait(5)
    # The job waits for the model: the control thread gets in.
    assert got.wait(5)
    t.join(5)
    leave_model.set()
    assert r.wait_idle(5)


def test_model_work_runs_inline_without_a_runner_and_is_queued_with_one(runner) -> None:
    ran: list[str] = []
    assert model_work("embed", lambda: ran.append(threading.current_thread().name))
    assert ran == [threading.current_thread().name]

    r = runner().start()
    gate = Gate()
    r.submit("focus", P0, gate)
    assert gate.started.wait(5)
    # From the control thread: queued, not run now.
    assert model_work("embed", lambda: ran.append(threading.current_thread().name))
    assert len(ran) == 1 and r.status()["queued"] == ["P0 embed"]
    gate.release.set()
    assert r.wait_idle(5)
    assert ran[-1] == "inference"

    # From the inference thread itself: run in place, inside the job.
    nested: list[str] = []

    def job() -> None:
        model_work("inner", lambda: nested.append("inner"))
        nested.append("after")

    r.submit("outer", P0, job)
    assert r.wait_idle(5)
    assert nested == ["inner", "after"]


def test_a_model_wait_on_the_control_thread_is_logged(runner, caplog) -> None:
    runner().start()
    with STATE_LOCK:
        with model_wait("POST /api/generate qwen"):
            pass
    assert "Model call on the control thread (POST /api/generate qwen)" in caplog.text


def test_stop_drops_waiting_jobs_and_waits_for_the_running_one(runner) -> None:
    r = runner().start()
    gate = Gate()
    ran: list[str] = []
    r.submit("focus", P0, gate)
    assert gate.started.wait(5)
    r.submit("passive", P2, lambda: ran.append("passive"))
    # A model call cannot be interrupted: stop gives up after its timeout.
    assert not r.stop(timeout=0.2)
    assert ir._ACTIVE is None
    gate.release.set()
    r._thread.join(5)
    assert not r._thread.is_alive()
    assert ran == []


def test_the_state_lock_is_reentrant_and_released_at_any_depth() -> None:
    lock = StateLock()
    with lock:
        with lock:
            assert lock.held()
            with lock.released():
                assert not lock.held()
                # Another thread can take it now.
                other = threading.Thread(target=lambda: (lock.acquire(), lock.release()))
                other.start()
                other.join(5)
                assert not other.is_alive()
            assert lock.held()
        assert lock.held()
    assert not lock.held()
    with pytest.raises(RuntimeError):
        lock.release()


def test_an_ollama_request_is_a_model_wait(monkeypatch) -> None:
    """Every Ollama request lets go of the state lock while it waits."""
    from agenthandover_worker import ollama

    seen: list[bool] = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self) -> bytes:
            seen.append(STATE_LOCK.held())
            return b'{"version": "0.34.2"}'

    monkeypatch.setattr(ollama.urllib.request, "urlopen", lambda req, timeout=None: Reply())
    with STATE_LOCK:
        assert ollama.OllamaClient("http://127.0.0.1:9").version() == "0.34.2"
        assert STATE_LOCK.held()
    assert seen == [False]


def test_wait_idle_times_out_while_a_job_runs(runner) -> None:
    r = runner().start()
    gate = Gate()
    r.submit("focus", P0, gate)
    assert gate.started.wait(5)
    started = time.monotonic()
    assert not r.wait_idle(0.2)
    assert time.monotonic() - started < 2
    gate.release.set()
    assert r.wait_idle(5)
