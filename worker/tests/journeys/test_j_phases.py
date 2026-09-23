"""J-phases (WP-C9): on a tier with two models, one model is loaded at a time.

The ``focus_dailynews`` recording runs on the Performance tier (qwen3.5:4b
annotates, gemma4:12b-it-qat writes the Skill; since WP-F6 the Recommended
tier runs qwen3.5:4b for both, so it has no model switch to prove) against
its recorded cassette.  The request sequence the cassette served must be phase-ordered:
the annotation model's calls, then the annotation model unloaded and the
SOP model loaded once with the context its calls use, the generation calls
(SOP, synthesis, questions) with no annotation-model call between two of
them, then the SOP model unloaded before the annotation model is used again.
"""

from __future__ import annotations

import os

import pytest

from agenthandover_worker.model_profiles import get_tier, get_profile
from journeys.cassette import CASSETTE_DIR
from journeys.focus_run import start_focus_run, tick_until_skill

JOURNEY = "focus_dailynews_performance"
TIER = get_tier("performance")
QWEN, GEMMA = TIER.annotation_model, TIER.sop_model


def _event(entry: dict) -> tuple[str, str] | None:
    """``(kind, model)`` of one served request: ``call``, ``load`` or
    ``unload``; None for embeddings and the other endpoints."""
    if entry["path"] != "/api/generate":
        return None
    request = entry["request"]
    model = request["model"]
    if request.get("prompt"):
        return ("call", model)
    if entry["response"].get("done_reason") == "unload":
        return ("unload", model)
    return ("load", model)


@pytest.mark.skipif(
    not (CASSETTE_DIR / f"{JOURNEY}.jsonl").exists() and os.environ.get("AH_RECORD") != "1",
    reason=(
        f"cassettes/{JOURNEY}.jsonl is not recorded yet: record it with "
        "AH_RECORD=1 against the test Mac's Ollama 0.34.2 (gemma4:12b-it-qat)"
    ),
)
def test_two_model_focus_run_uses_one_model_at_a_time(
    events_db, worker, clock, temp_home, cassette,
) -> None:
    run = start_focus_run(
        events_db=events_db, worker=worker, clock=clock,
        temp_home=temp_home, cassette=cassette, journey=JOURNEY, tier=TIER.id,
    )
    proc = tick_until_skill(run, clock)
    assert proc["title"] == run.fixture.manifest["focus_session"]["title"]
    # The tick after the Skill annotates the keyframes the SOP did not need.
    run.ticks.append(run.worker.run_tick())

    events = [e for e in map(_event, run.tape.served) if e is not None]
    calls = [model for kind, model in events if kind == "call"]
    assert QWEN in calls and GEMMA in calls

    # Split the sequence into phases at each SOP-model load.
    phases: list[list[tuple[str, str]]] = []
    resident: set[str] = set()
    for kind, model in events:
        if kind == "unload":
            resident.discard(model)
        elif kind == "load":
            if model == GEMMA:
                assert QWEN not in resident, "gemma loaded while qwen was resident"
                phases.append([])
            resident.add(model)
        else:
            assert model in resident or model == QWEN, (
                f"{model} called without its phase's load: {events}"
            )
            resident.add(model)
            if model == GEMMA:
                assert QWEN not in resident, f"qwen resident during generation: {events}"
                phases[-1].append((kind, model))
            else:
                assert GEMMA not in resident, (
                    f"a qwen call inside a generation phase: {events}"
                )

    # Generation phases: the SOP phase, and the export after the questions
    # were answered (skipped) on the next tick.  Every one of them loads
    # gemma once and ends with gemma unloaded.
    assert 1 <= len(phases) <= 2, events
    assert all(phase for phase in phases)
    assert GEMMA not in resident

    # Each load carries the context of the calls it serves, so none of them
    # reloads the model.
    gemma_ctx = get_profile(GEMMA).sop_num_ctx
    for entry in run.tape.served:
        request = entry["request"]
        if entry["path"] == "/api/generate" and request.get("model") == GEMMA:
            if entry["response"].get("done_reason") != "unload":
                assert request["options"]["num_ctx"] == gemma_ctx, request.get("prompt", "")[:80]
