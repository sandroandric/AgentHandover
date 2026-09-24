"""WP-C9: the annotation phase, then the generation phase, one model loaded
at a time; the memory check; load failures as deferrals; per-phase num_ctx.

A fake Ollama records every request in order.  The journey on a recorded
two-model cassette is ``journeys/test_j_phases.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from agenthandover_worker import behavioral_synthesizer, sop_generator
from agenthandover_worker.behavioral_synthesizer import BehavioralSynthesizer, SynthesizerConfig
from agenthandover_worker.llm_reasoning import LLMReasoner, ReasoningConfig, annotation_reasoner
from agenthandover_worker.model_profiles import get_profile
from agenthandover_worker.model_scheduler import (
    FALLBACK_AFTER_SECONDS,
    GB,
    GENERATION_BATCH_WAIT_SECONDS,
    LOAD_BACKOFF_SECONDS,
    LOAD_START_SECONDS,
    LOAD_TIMEOUT_SECONDS,
    DISK_FLOOR_BYTES,
    SWAP_LIMIT_BYTES,
    ModelScheduler,
    before_generation_call,
    parse_swapusage,
    parse_vm_stat,
)
from agenthandover_worker.ollama import OllamaClient, OllamaTimeout, OllamaUnavailable, Response
from agenthandover_worker.sop_generator import SOPGenerator, SOPGeneratorConfig

QWEN = "qwen3.5:4b"
GEMMA = "gemma4:12b-it-qat"
QWEN_SIZE = int(3.4 * GB)
GEMMA_SIZE = int(7.2 * GB)  # the tier table's size, until /api/ps reports one


class FakeOllama(OllamaClient):
    """Loaded models and a log of every request: ``("ps",)``,
    ``("unload", model)``, ``("load", model, num_ctx)``,
    ``("generate", model, num_ctx)``."""

    def __init__(self, loaded: dict[str, int] | None = None) -> None:
        super().__init__("http://fake")
        self.loaded: dict[str, int] = dict(loaded or {})
        self.log: list[tuple] = []
        self.fail_loads = 0
        # Loads that do not answer within their timeout: "queued" (Ollama
        # has not started them: absent from /api/ps) or "loading" (listed
        # in /api/ps, answered by the next request).
        self.stalled_loads: list[str] = []
        self.load_timeouts: list[float | None] = []

    def ps(self, timeout=5.0):
        self.log.append(("ps",))
        return [{"name": m, "size": s} for m, s in self.loaded.items()]

    def unload(self, model, timeout=30.0):
        self.log.append(("unload", model))
        self.loaded.pop(model, None)

    def generate(self, model, prompt, *, options=None, keep_alive=None, **kwargs):
        num_ctx = (options or {}).get("num_ctx")
        if prompt == "" and keep_alive is not None:
            self.log.append(("load", model, num_ctx))
            self.load_timeouts.append(kwargs.get("timeout"))
            if self.stalled_loads:
                stall = self.stalled_loads.pop(0)
                if stall == "loading":
                    self.loaded[model] = GEMMA_SIZE
                raise OllamaTimeout(f"Ollama timed out after {kwargs.get('timeout'):.0f}s")
            if self.fail_loads:
                self.fail_loads -= 1
                raise OllamaUnavailable(
                    "Ollama HTTP 500 at http://fake/api/generate: "
                    "timed out waiting for llama-server to start"
                )
        else:
            self.log.append(("generate", model, num_ctx))
        self.loaded[model] = {QWEN: QWEN_SIZE, GEMMA: GEMMA_SIZE}.get(model, GB)
        return Response(text='{"title": "t", "steps": []}')


@dataclass
class _Config:
    model: str


class FakeGenerator:
    """A generation component: asks the scheduler first, then calls its
    model with the profile's generation context."""

    def __init__(self, fake: FakeOllama) -> None:
        self.config = _Config(GEMMA)
        self.fake = fake

    def call(self) -> None:
        before_generation_call(self)
        self.fake.generate(
            self.config.model, "p",
            options={"num_ctx": get_profile(self.config.model).sop_num_ctx},
        )


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def _scheduler(fake, *, free=16 * GB, clock=None, annotation=QWEN, sop=GEMMA):
    """*memory* holds ``free``, ``swap`` (bytes in use) and ``disk`` (bytes
    free); the tests change them between ticks."""
    memory = {"free": free, "swap": 0, "disk": 100 * GB}
    scheduler = ModelScheduler(
        fake, annotation_model=annotation, sop_model=sop,
        clock=clock or Clock(), free_memory=lambda: memory["free"],
        swap_used=lambda: memory["swap"], free_disk=lambda: memory["disk"],
    )
    return scheduler, memory


def _annotate(fake: FakeOllama, n: int = 2) -> None:
    for _ in range(n):
        fake.generate(QWEN, "frame", options={"num_ctx": get_profile(QWEN).ann_num_ctx})


def _generations(log: list[tuple]) -> list[str]:
    return [entry[1] for entry in log if entry[0] in ("generate", "load")]


GEMMA_CTX = get_profile(GEMMA).sop_num_ctx
QWEN_ANN_CTX = get_profile(QWEN).ann_num_ctx


def test_a_tick_annotates_then_generates_and_unloads_between_the_phases():
    """Pending frames and pending generation: every annotation first, then
    the annotation model is unloaded, the SOP model loaded once with the
    calls' context, used, and unloaded; the embedding model is left alone."""
    fake = FakeOllama({QWEN: QWEN_SIZE, "nomic-embed-text:latest": int(0.3 * GB)})
    scheduler, _ = _scheduler(fake)
    component = FakeGenerator(fake)
    scheduler.register(component)

    scheduler.begin_annotation()
    _annotate(fake)
    plan = scheduler.plan_generation(
        urgent=False, due={"passive_discovery": True, "daily": True},
    )
    assert plan.run and plan.model == GEMMA
    assert plan.allows("passive_discovery") and plan.allows("daily")
    component.call()
    component.call()
    scheduler.end_generation()

    assert fake.log == [
        ("ps",),                              # startup: no SOP model left over
        ("generate", QWEN, QWEN_ANN_CTX),
        ("generate", QWEN, QWEN_ANN_CTX),
        ("ps",),                              # the memory check
        ("unload", QWEN),
        ("load", GEMMA, GEMMA_CTX),
        ("generate", GEMMA, GEMMA_CTX),
        ("generate", GEMMA, GEMMA_CTX),
        ("ps",),                              # learn the loaded size
        ("unload", GEMMA),
    ]
    assert "nomic-embed-text:latest" in fake.loaded

    # The next tick starts with the annotation model alone.
    scheduler.begin_annotation()
    _annotate(fake, 1)
    assert fake.log[-1] == ("generate", QWEN, QWEN_ANN_CTX)
    assert GEMMA not in fake.loaded


def test_no_generation_work_loads_nothing():
    fake = FakeOllama({QWEN: QWEN_SIZE})
    scheduler, _ = _scheduler(fake)
    scheduler.begin_annotation()
    _annotate(fake)
    assert scheduler.plan_generation(urgent=False, due={"passive_discovery": False}).run is False
    scheduler.end_generation()
    assert _generations(fake.log) == [QWEN, QWEN]
    assert ("unload", QWEN) not in fake.log


def test_memory_short_defers_with_the_status_reason_then_runs_when_memory_frees():
    fake = FakeOllama({QWEN: QWEN_SIZE})
    scheduler, memory = _scheduler(fake, free=int(3.1 * GB))
    component = FakeGenerator(fake)
    scheduler.register(component)

    plan = scheduler.plan_generation(urgent=False, due={"daily": True})
    assert plan.run is False
    # 7.2 GB + 1 GB margin; 3.1 GB free + 3.4 GB the annotation model frees.
    assert scheduler.reason == "waiting for memory: need 8.2 GB, 6.5 GB free"
    assert scheduler.status()["generation_waiting"] == scheduler.reason
    assert ("unload", QWEN) not in fake.log
    assert GEMMA not in _generations(fake.log)

    memory["free"] = 6 * GB
    plan = scheduler.plan_generation(urgent=False, due={"daily": True})
    assert plan.allows("daily")
    assert scheduler.reason is None
    component.call()
    assert _generations(fake.log) == [GEMMA, GEMMA]
    assert fake.log.index(("unload", QWEN)) < fake.log.index(("load", GEMMA, GEMMA_CTX))


def test_a_stopped_focus_session_runs_with_memory_short_after_unloading_the_annotation_model():
    fake = FakeOllama({QWEN: QWEN_SIZE})
    scheduler, _ = _scheduler(fake, free=1 * GB)
    component = FakeGenerator(fake)
    scheduler.register(component)

    _annotate(fake)  # the session's keyframes
    plan = scheduler.plan_generation(urgent=True, due={"passive_discovery": False})
    assert plan.run and plan.model == GEMMA and not plan.fallback
    component.call()
    assert fake.log[-3:] == [
        ("unload", QWEN), ("load", GEMMA, GEMMA_CTX), ("generate", GEMMA, GEMMA_CTX),
    ]


def test_the_memory_check_uses_the_size_ollama_reported():
    """After one generation phase the SOP model's size comes from /api/ps,
    not the tier table."""
    fake = FakeOllama({QWEN: QWEN_SIZE})
    scheduler, memory = _scheduler(fake, free=16 * GB)
    component = FakeGenerator(fake)
    scheduler.register(component)
    scheduler.plan_generation(urgent=True, due={})
    component.call()
    fake.loaded[GEMMA] = int(9.5 * GB)  # what /api/ps says once loaded
    scheduler.end_generation()

    memory["free"] = 9 * GB  # room for the table's 7.2 GB, not for 9.5 GB
    assert scheduler.plan_generation(urgent=False, due={"daily": True}).run is False
    assert scheduler.reason == "waiting for memory: need 10.5 GB, 9.0 GB free"


def test_an_already_loaded_sop_model_needs_no_memory_check():
    fake = FakeOllama({GEMMA: GEMMA_SIZE})
    scheduler, _ = _scheduler(fake, free=0)
    assert scheduler.plan_generation(urgent=False, due={"daily": True}).run is True


def test_passive_discovery_is_batched_until_other_work_or_the_wait():
    clock = Clock()
    fake = FakeOllama({QWEN: QWEN_SIZE})
    scheduler, _ = _scheduler(fake, clock=clock)

    assert scheduler.plan_generation(urgent=False, due={"passive_discovery": True}).run is False
    clock.now += GENERATION_BATCH_WAIT_SECONDS - 1
    assert scheduler.plan_generation(urgent=False, due={"passive_discovery": True}).run is False
    # Other generation work runs anyway: the pass joins it.
    plan = scheduler.plan_generation(
        urgent=False, due={"passive_discovery": True, "daily": True},
    )
    assert plan.allows("passive_discovery") and plan.allows("daily")

    # Due again: waits, then runs alone once it has waited long enough.
    assert scheduler.plan_generation(urgent=False, due={"passive_discovery": True}).run is False
    clock.now += GENERATION_BATCH_WAIT_SECONDS
    assert scheduler.plan_generation(
        urgent=False, due={"passive_discovery": True},
    ).allows("passive_discovery")
    # A focus session never waits, and takes the due pass along.
    assert scheduler.plan_generation(urgent=False, due={"passive_discovery": True}).run is False
    assert scheduler.plan_generation(
        urgent=True, due={"passive_discovery": True},
    ).allows("passive_discovery")


def test_a_focus_job_planning_for_itself_leaves_the_batched_wait_alone():
    """WP-C7: the focus job plans without taking the batched jobs, so a
    passive pass that has waited keeps its place in the batch window."""
    clock = Clock()
    fake = FakeOllama({QWEN: QWEN_SIZE})
    scheduler, _ = _scheduler(fake, clock=clock)

    assert scheduler.plan_generation(urgent=False, due={"passive_discovery": True}).run is False
    clock.now += GENERATION_BATCH_WAIT_SECONDS - 10
    focus = scheduler.plan_generation(urgent=True, due={}, take_batched=False)
    assert focus.run and not focus.allows("passive_discovery")
    scheduler.end_generation()
    clock.now += 10
    # Due since before the focus job: its wait is over now.
    assert scheduler.plan_generation(
        urgent=False, due={"passive_discovery": True},
    ).allows("passive_discovery")


def test_after_hours_without_memory_the_jobs_run_on_the_annotation_model():
    clock = Clock()
    fake = FakeOllama({QWEN: QWEN_SIZE})
    scheduler, _ = _scheduler(fake, free=1 * GB, clock=clock)
    component = FakeGenerator(fake)
    scheduler.register(component)

    assert scheduler.plan_generation(urgent=False, due={"daily": True}).run is False
    clock.now += FALLBACK_AFTER_SECONDS
    plan = scheduler.plan_generation(urgent=False, due={"daily": True})
    assert plan.allows("daily") and plan.fallback and plan.model == QWEN
    assert scheduler.status()["fallback"] is True
    assert scheduler.status()["generation_waiting"] == (
        f"generating on {QWEN}: {GEMMA} needs 8.2 GB, 4.4 GB free for 6 h"
    )
    assert scheduler.generation_model == QWEN

    component.call()
    # The resident annotation model writes, with its own generation context;
    # it is never unloaded, and the SOP model is never loaded.
    assert component.config.model == QWEN
    assert ("unload", QWEN) not in fake.log
    assert GEMMA not in _generations(fake.log)
    assert fake.log[-1] == ("generate", QWEN, get_profile(QWEN).sop_num_ctx)
    scheduler.end_generation()
    assert component.config.model == GEMMA
    assert ("unload", QWEN) not in fake.log
    assert scheduler.generation_model == GEMMA


def test_a_failed_load_defers_with_a_backoff_and_is_retried():
    clock = Clock()
    fake = FakeOllama({QWEN: QWEN_SIZE})
    fake.fail_loads = 1
    scheduler, _ = _scheduler(fake, clock=clock)
    component = FakeGenerator(fake)
    scheduler.register(component)

    scheduler.plan_generation(urgent=True, due={})
    with pytest.raises(OllamaUnavailable):
        component.call()
    assert "timed out waiting for llama-server to start" in scheduler.reason
    assert "retrying in 60 s" in scheduler.reason
    assert GEMMA not in [e[1] for e in fake.log if e[0] == "generate"]

    # Within the backoff: no plan, no load attempt.
    loads = sum(1 for e in fake.log if e[0] == "load")
    clock.now += LOAD_BACKOFF_SECONDS - 1
    assert scheduler.plan_generation(urgent=True, due={}).run is False
    with pytest.raises(OllamaUnavailable):
        component.call()
    assert sum(1 for e in fake.log if e[0] == "load") == loads

    clock.now += 2
    assert scheduler.plan_generation(urgent=True, due={}).run is True
    component.call()
    assert fake.log[-1] == ("generate", GEMMA, GEMMA_CTX)


def test_one_model_tiers_are_left_as_they_were():
    fake = FakeOllama({QWEN: QWEN_SIZE})
    scheduler, _ = _scheduler(fake, free=0, sop=QWEN)
    component = FakeGenerator(fake)
    component.config.model = QWEN
    scheduler.register(component)
    scheduler.begin_annotation()
    plan = scheduler.plan_generation(urgent=False, due={"passive_discovery": True})
    assert plan.allows("passive_discovery")
    component.call()
    scheduler.end_generation()
    assert fake.log == [("generate", QWEN, get_profile(QWEN).sop_num_ctx)]
    assert scheduler.status() is None


def test_a_sop_model_left_loaded_by_an_earlier_run_is_unloaded_at_startup():
    fake = FakeOllama({GEMMA: GEMMA_SIZE, QWEN: QWEN_SIZE})
    scheduler, _ = _scheduler(fake)
    scheduler.begin_annotation()
    scheduler.begin_annotation()
    assert fake.log == [("ps",), ("unload", GEMMA)]


def test_a_tick_that_raised_mid_phase_still_unloads_the_sop_model():
    fake = FakeOllama({QWEN: QWEN_SIZE})
    scheduler, _ = _scheduler(fake)
    component = FakeGenerator(fake)
    scheduler.register(component)
    scheduler.begin_annotation()
    scheduler.plan_generation(urgent=True, due={})
    component.call()  # ...and the tick raised before end_generation
    scheduler.begin_annotation()
    assert fake.log[-1] == ("unload", GEMMA)


# ---------------------------------------------------------------------------
# Free memory
# ---------------------------------------------------------------------------


VM_STAT = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    89886.
Pages active:                                 237024.
Pages inactive:                               244299.
Pages speculative:                              9644.
Pages throttled:                                   0.
Pages wired down:                             282775.
"""


def test_free_memory_counts_free_inactive_and_speculative_pages():
    assert parse_vm_stat(VM_STAT) == 16384 * (89886 + 244299 + 9644)
    assert parse_vm_stat("Pages free: 1.\n") is None
    assert parse_vm_stat("") is None


def test_swap_usage_is_read_from_sysctl():
    # `sysctl -n vm.swapusage` on a Mac; units are binary.
    assert parse_swapusage(
        "total = 2048.00M  used = 971.81M  free = 1076.19M  (encrypted)\n"
    ) == int(971.81 * 1024 ** 2)
    assert parse_swapusage(
        "vm.swapusage: total = 45056.00M  used = 44.10G  free = 1.00G  (encrypted)"
    ) == int(44.10 * 1024 ** 3)
    assert parse_swapusage("total = 0.00M  used = 0.00M  free = 0.00M") == 0
    assert parse_swapusage("") is None
    assert parse_swapusage("vm.swapusage: unknown oid") is None


def test_heavy_swap_refuses_the_load_then_it_runs_when_swap_drops():
    """The test Mac panicked loading a 7 GB model with ~44 GB swapped out:
    enough free pages is not enough when the Mac is already swapping."""
    fake = FakeOllama({QWEN: QWEN_SIZE})
    scheduler, memory = _scheduler(fake, free=16 * GB)
    component = FakeGenerator(fake)
    scheduler.register(component)
    memory["swap"] = 44 * GB

    assert scheduler.plan_generation(urgent=False, due={"daily": True}).run is False
    assert scheduler.reason == (
        "waiting: 44.0 GB of swap in use (limit 8.0 GB): the Mac is short of memory"
    )
    assert scheduler.status()["generation_waiting"] == scheduler.reason
    assert ("unload", QWEN) not in fake.log
    assert GEMMA not in _generations(fake.log)

    memory["swap"] = SWAP_LIMIT_BYTES  # at the limit is not over it
    assert scheduler.plan_generation(urgent=False, due={"daily": True}).allows("daily")
    assert scheduler.reason is None
    component.call()
    assert ("load", GEMMA, GEMMA_CTX) in fake.log


def test_a_nearly_full_disk_refuses_the_load():
    fake = FakeOllama({QWEN: QWEN_SIZE})
    scheduler, memory = _scheduler(fake)
    memory["disk"] = DISK_FLOOR_BYTES - 1
    assert scheduler.plan_generation(urgent=False, due={"daily": True}).run is False
    assert scheduler.reason == (
        "waiting: 10.0 GB of disk free (floor 10.0 GB): swap cannot grow"
    )


def test_a_focus_recording_under_swap_pressure_runs_on_the_resident_model():
    """A stopped focus recording is processed, but loads nothing: the
    resident annotation model writes it, and the status says why."""
    fake = FakeOllama({QWEN: QWEN_SIZE})
    scheduler, memory = _scheduler(fake)
    component = FakeGenerator(fake)
    scheduler.register(component)
    memory["swap"] = 20 * GB

    plan = scheduler.plan_generation(urgent=True, due={})
    assert plan.run and plan.fallback and plan.model == QWEN
    assert scheduler.status()["generation_waiting"] == (
        f"generating on {QWEN}: not loading {GEMMA}, 20.0 GB of swap in use "
        "(limit 8.0 GB): the Mac is short of memory"
    )
    component.call()
    assert ("unload", QWEN) not in fake.log
    assert GEMMA not in _generations(fake.log)
    assert fake.log[-1] == ("generate", QWEN, get_profile(QWEN).sop_num_ctx)


def test_batched_work_under_swap_pressure_falls_back_after_the_wait():
    clock = Clock()
    fake = FakeOllama({QWEN: QWEN_SIZE})
    scheduler, memory = _scheduler(fake, clock=clock)
    memory["swap"] = 20 * GB
    assert scheduler.plan_generation(urgent=False, due={"daily": True}).run is False
    clock.now += FALLBACK_AFTER_SECONDS
    plan = scheduler.plan_generation(urgent=False, due={"daily": True})
    assert plan.allows("daily") and plan.fallback and plan.model == QWEN


def test_pressure_does_not_matter_once_the_sop_model_is_loaded_or_when_unknown():
    fake = FakeOllama({GEMMA: GEMMA_SIZE})
    scheduler, memory = _scheduler(fake)
    memory["swap"] = 40 * GB
    memory["disk"] = 0
    plan = scheduler.plan_generation(urgent=False, due={"daily": True})
    assert plan.run and not plan.fallback

    fake = FakeOllama({QWEN: QWEN_SIZE})
    scheduler, memory = _scheduler(fake)
    memory["swap"] = None
    memory["disk"] = None
    assert scheduler.plan_generation(urgent=False, due={"daily": True}).run is True


# ---------------------------------------------------------------------------
# num_ctx per call type comes from the profile
# ---------------------------------------------------------------------------


def _options_sent(call) -> dict:
    sent = []

    def fake_generate(self, model, prompt, **kwargs):
        sent.append(kwargs.get("options") or {})
        return Response(text='{"title": "t", "steps": [{"action": "a"}]}')

    with patch.object(OllamaClient, "generate", fake_generate):
        call()
    assert sent
    return sent[0]


@pytest.mark.parametrize("model", [QWEN, GEMMA])
def test_every_generation_call_uses_the_profiles_generation_context(model):
    """SOP generation, synthesis and the reasoner (Q&A, labels, curation)
    share the profile's ``sop_num_ctx``: a different context would make
    Ollama reload the model in the middle of the phase."""
    ctx = get_profile(model).sop_num_ctx
    generator = SOPGenerator(SOPGeneratorConfig(model=model))
    assert _options_sent(lambda: generator._call("p"))["num_ctx"] == ctx
    reasoner = LLMReasoner(ReasoningConfig(model=model))
    assert _options_sent(lambda: reasoner.reason_text("p"))["num_ctx"] == ctx
    synthesizer = BehavioralSynthesizer(SynthesizerConfig(model=model))
    options = _options_sent(lambda: synthesizer.synthesize(
        "slug", {"title": "t", "steps": [{"action": "a"}]}, [[{"app": "x"}]], force=True,
    ))
    assert options["num_ctx"] == ctx


def test_the_classifier_reasoner_uses_the_annotation_context():
    reasoner = annotation_reasoner(QWEN, "http://fake")
    options = _options_sent(lambda: reasoner.reason_text("p", think=False))
    assert options["num_ctx"] == get_profile(QWEN).ann_num_ctx
    assert options["num_predict"] == 16
    assert reasoner.config.model == QWEN


# ---------------------------------------------------------------------------
# A model that fails to load defers the Skill; it never fails it
# ---------------------------------------------------------------------------


def _unavailable(*args, **kwargs):
    raise OllamaUnavailable(
        "Ollama HTTP 500 at http://fake/api/generate: "
        "timed out waiting for llama-server to start"
    )


def test_a_failed_load_defers_focus_and_passive_generation():
    generator = SOPGenerator(SOPGeneratorConfig(model=GEMMA))
    timeline = [{"annotation": {"app": "Chrome", "task_context": {"what_doing": "x"}},
                 "timestamp": "2026-09-15T09:00:00"}]
    with patch.object(sop_generator, "_call_ollama", _unavailable):
        focus = generator.generate_from_focus(timeline, "Task")
        passive = generator.generate_from_passive([timeline, timeline], task_label="Task")
        enriched = generator.generate_from_passive_enriched(
            [timeline, timeline], task_label="Task",
        )
    for result in (focus, passive, enriched):
        assert result.deferred and not result.success
        assert "llama-server" in result.error


def test_a_failed_load_on_a_retry_attempt_defers_instead_of_failing():
    """The first answer is unparseable; the retry meets a model that fails
    to load: the session is deferred, not recorded as a failure."""
    generator = SOPGenerator(SOPGeneratorConfig(model=GEMMA))
    timeline = [{"annotation": {"app": "Chrome", "task_context": {"what_doing": "x"}},
                 "timestamp": "2026-09-15T09:00:00"}]
    answers = iter([("not json", 1.0)])

    def first_garbage_then_unavailable(**kwargs):
        try:
            return next(answers)
        except StopIteration:
            _unavailable()

    with patch.object(sop_generator, "_call_ollama", first_garbage_then_unavailable):
        result = generator.generate_from_focus(timeline, "Task")
    assert result.deferred and not result.success


def test_a_generation_model_that_cannot_load_defers_the_registered_components():
    """The scheduler's refusal reaches every generation component as the
    deferral each one already handles."""
    fake = FakeOllama({QWEN: QWEN_SIZE})
    fake.fail_loads = 99
    scheduler, _ = _scheduler(fake)
    generator = SOPGenerator(SOPGeneratorConfig(model=GEMMA))
    reasoner = LLMReasoner(ReasoningConfig(model=GEMMA))
    synthesizer = BehavioralSynthesizer(SynthesizerConfig(model=GEMMA))
    scheduler.register(generator, reasoner, synthesizer)
    scheduler.plan_generation(urgent=True, due={})
    timeline = [{"annotation": {"app": "Chrome", "task_context": {"what_doing": "x"}},
                 "timestamp": "2026-09-15T09:00:00"}]

    with patch.object(behavioral_synthesizer, "_call_ollama_text", _unavailable):
        assert generator.generate_from_focus(timeline, "Task").deferred
        result = reasoner.reason_text("p")
        assert not result.success and "could not be loaded" in result.error
        insights = synthesizer.synthesize("s", {"title": "t"}, [[{"app": "x"}]], force=True)
    assert not insights.goal and not insights.strategy
    assert not [e for e in fake.log if e[0] == "generate"]


def test_a_load_ollama_never_starts_is_deferred_after_the_start_wait():
    """Ollama queues the load (no memory) and never lists the model in
    /api/ps: the phase is deferred after LOAD_START_SECONDS, not after the
    full load timeout, and nothing is generated."""
    fake = FakeOllama({QWEN: QWEN_SIZE})
    fake.stalled_loads = ["queued"]
    scheduler, _ = _scheduler(fake)
    component = FakeGenerator(fake)
    scheduler.register(component)

    scheduler.plan_generation(urgent=True, due={})
    with pytest.raises(OllamaUnavailable):
        component.call()
    assert fake.load_timeouts == [LOAD_START_SECONDS]
    assert "did not start loading" in scheduler.reason
    assert "retrying in 60 s" in scheduler.reason
    assert GEMMA not in fake.loaded
    assert not [e for e in fake.log if e[0] == "generate"]
    assert fake.log[-1] == ("ps",)


def test_a_slow_load_that_ollama_started_is_waited_for():
    """The model is in /api/ps after the start wait: the load is under way,
    the request is repeated for the rest of the load timeout and the phase
    runs."""
    fake = FakeOllama({QWEN: QWEN_SIZE})
    fake.stalled_loads = ["loading"]
    scheduler, _ = _scheduler(fake)
    component = FakeGenerator(fake)
    scheduler.register(component)

    scheduler.plan_generation(urgent=True, due={})
    component.call()
    assert fake.load_timeouts == [
        LOAD_START_SECONDS, LOAD_TIMEOUT_SECONDS - LOAD_START_SECONDS,
    ]
    assert scheduler.reason is None
    assert fake.log[-1] == ("generate", GEMMA, GEMMA_CTX)
