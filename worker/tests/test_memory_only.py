"""Memory only (``[skills] enabled = false``): each guard on its own.

The journey (``journeys/test_j_memory_only.py``) runs the worker's tick end
to end on a cassette; these pin each layer separately, since any one of
them would keep a generative model from running while the others hold:
the config key, the Ollama client's refusal, the jobs the control thread
submits, the retry request, the daily upkeep, the worker status.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agenthandover_worker import main as wm
from agenthandover_worker.config import load_config
from agenthandover_worker.ollama import (
    GenerationOff,
    OllamaClient,
    OllamaUnavailable,
    switch_off_generation,
)
from agenthandover_worker.stages import Tick, control, daily


@pytest.fixture(autouse=True)
def _generation_back_on():
    yield
    switch_off_generation(None)


def _cfg(skills: bool, memory: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        skills=SimpleNamespace(enabled=skills),
        memory=SimpleNamespace(enabled=memory),
        sop=SimpleNamespace(auto_approve=False),
    )


# -- config ----------------------------------------------------------------


def test_skills_are_on_unless_the_file_turns_them_off(tmp_path) -> None:
    assert load_config(tmp_path / "missing.toml").skills.enabled is True
    file = tmp_path / "config.toml"
    file.write_text("[skills]\nenabled = false\n", encoding="utf-8")
    assert load_config(file).skills.enabled is False
    file.write_text("[memory]\nenabled = true\n", encoding="utf-8")
    assert load_config(file).skills.enabled is True


# -- the Ollama client -----------------------------------------------------


def test_switched_off_the_client_refuses_generate_and_preload_without_a_request(monkeypatch) -> None:
    client = OllamaClient("http://127.0.0.1:9")
    sent: list[str] = []
    monkeypatch.setattr(client, "_request", lambda method, path, *a, **k: sent.append(path) or {"embeddings": [[1.0]]})
    switch_off_generation("Skills are off (Memory only)")

    with pytest.raises(GenerationOff, match="Memory only"):
        client.generate("qwen3.5:4b", "describe this screen")
    with pytest.raises(GenerationOff):
        client.preload("qwen3.5:4b", "10m")
    assert sent == []
    # A caller that slipped past the switch defers, as for an outage.
    assert issubclass(GenerationOff, OllamaUnavailable)

    # Memory's embedding still reaches Ollama.
    client.embed("embeddinggemma-2:440m", ["a moment"])
    assert sent == ["/api/embed"]


def test_switched_back_on_the_client_generates_again(monkeypatch) -> None:
    client = OllamaClient("http://127.0.0.1:9")
    monkeypatch.setattr(client, "_request", lambda *a, **k: {"response": "ok", "done_reason": "stop"})
    switch_off_generation("off")
    switch_off_generation(None)
    assert client.generate("qwen3.5:4b", "hello").text == "ok"


# -- the control thread's jobs -------------------------------------------


def _ctx(skills: bool) -> SimpleNamespace:
    return SimpleNamespace(
        cfg=_cfg(skills),
        memory_writer=object(), focus_processor=None, v2_schema_ok=True,
        runner=None, embedding_upgrade=None,
    )


def _job_keys(ctx) -> list[str]:
    return [key for key, _priority, _job in control.submit_model_jobs(ctx, Tick())]


def test_memory_only_submits_memory_and_the_upkeep_and_no_model_job() -> None:
    assert _job_keys(_ctx(skills=False)) == ["memory", "maintenance"]


def test_with_skills_the_generation_jobs_are_submitted_as_before() -> None:
    keys = _job_keys(_ctx(skills=True))
    assert "passive-annotation" in keys and "passive-generation" in keys
    assert "maintenance" not in keys


def test_memory_only_ignores_a_retry_and_queues_no_model_work(monkeypatch) -> None:
    queued: list[str] = []
    monkeypatch.setattr(control, "model_work", lambda key, job: queued.append(key))
    ctx = SimpleNamespace(
        cfg=_cfg(skills=False), db=None, focus_processor=None, sop_generator=None,
        screenshots_dir=None, export_kwargs=lambda: {},
    )
    control.trigger_handlers(ctx)["retry"]({"failure_id": "f-1"})
    assert queued == []

    ctx.cfg = _cfg(skills=True)
    control.trigger_handlers(ctx)["retry"]({"failure_id": "f-1"})
    assert len(queued) == 1 and queued[0].startswith("retry:")


# -- the daily upkeep ------------------------------------------------------


def test_the_upkeep_runs_once_a_day_without_a_generation_plan(monkeypatch) -> None:
    from datetime import datetime, timezone

    days: list[str] = []

    def upkeep(ctx, today) -> None:
        days.append(today)
        ctx.daily_jobs_date = today

    monkeypatch.setattr(daily, "_upkeep", upkeep)
    batch: list[str] = []
    monkeypatch.setattr(daily, "_daily_batch", lambda ctx, day: batch.append(day))
    now = datetime(2026, 10, 14, 9, 0, tzinfo=timezone.utc)
    ctx = SimpleNamespace(run_daily_jobs=True, daily_jobs_date="2026-10-13", local_now=lambda: now)
    tick = Tick()
    assert tick.generation_plan is None

    assert daily.run_maintenance(ctx, tick) is True
    assert daily.run_maintenance(ctx, tick) is False, "once per day"
    assert days == ["2026-10-14"]
    assert batch == [], "no summaries: they need the model"


# -- the worker status -----------------------------------------------------


def test_memory_only_reports_no_model_missing_without_asking_ollama(monkeypatch) -> None:
    def probe(_vlm):
        raise AssertionError("Memory only asks Ollama about no generative model")

    monkeypatch.setattr(wm, "_vlm_backend_available", probe)
    monkeypatch.setattr(wm, "_missing_vlm_models", probe)
    assert wm._generative_models(SimpleNamespace(skills=SimpleNamespace(enabled=False))) == (False, [])


def test_the_status_says_whether_skills_are_on() -> None:
    def written(**extra) -> dict:
        wm._write_worker_status(
            started_at="2026-10-14T00:00:00Z", events_processed_today=0, sops_generated=0,
            last_pipeline_duration_ms=None, consecutive_errors=0, vlm_available=False,
            **extra,
        )
        return json.loads((wm._status_dir() / "worker-status.json").read_text())

    assert written()["skills_enabled"] is True
    assert written(skills_enabled=False, missing_models=[])["skills_enabled"] is False
