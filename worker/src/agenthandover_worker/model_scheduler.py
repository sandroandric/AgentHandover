"""Phase-ordered model use: annotate first, then write Skills, one model at a time.

On a tier whose annotation model differs from its SOP model
(``qwen3.5:2b-q4_K_M`` and ``qwen3.5:4b`` on 8 GB Macs, ``qwen3.5:4b`` and
``gemma4:12b-it-qat`` on 24 GB, ``qwen3.5:4b`` and ``gemma4:31b-it-qat`` on
48 GB), the two may not fit in memory together once ordinary apps are open: Ollama
evicts one to load the other, and under pressure a load can fail ("timed
out waiting for llama-server to start").  The worker therefore runs each tick in two phases:

1. the **annotation phase**: every frame it annotates this tick (focus
   keyframes, passive frames, frame diffs, the activity classifier), on the
   annotation model;
2. the **generation phase**, only when generation work is pending: SOP
   generation, behavioural synthesis, Q&A and the daily model jobs, on the
   SOP model.

The phases run as jobs on the inference thread (:mod:`inference_runner`):
each job that annotates starts with :meth:`~ModelScheduler.begin_annotation`,
each job that generates plans with :meth:`~ModelScheduler.plan_generation`,
and the runner calls :meth:`~ModelScheduler.end_generation` when its queue
runs empty.

:class:`ModelScheduler` owns the switch between them:

* the first generation call of a phase (every component that calls the
  SOP model is :meth:`~ModelScheduler.register`-ed and asks first) unloads
  the annotation model and loads the SOP model with the context size the
  calls use, so they do not reload it; the end of the phase unloads the
  SOP model.  A phase with no generation call loads nothing.  The embedding
  model is never unloaded.  What is loaded is read from Ollama's
  ``/api/ps``.
* :meth:`~ModelScheduler.plan_generation` decides which jobs the phase
  runs.  Work the user waits for (a stopped focus recording, answered
  questions) always runs.  A passive-discovery pass is batched: it waits
  until other generation work runs anyway or it has been due for
  :data:`GENERATION_BATCH_WAIT_SECONDS`.
* before planning to load the SOP model it checks free memory
  (``vm_stat``: free + inactive + speculative pages, plus what unloading the
  annotation model gives back) against the SOP model's size and
  :data:`MEMORY_MARGIN_BYTES`.  Without room, the jobs wait ("waiting for
  memory: need 8.2 GB, 3.1 GB free") and are retried on the next tick;
  after :data:`FALLBACK_AFTER_SECONDS` they run on the annotation model
  instead, which is already resident, and the worker status says so.  A
  focus recording is always processed, on the SOP model, unless the Mac
  is already swapping hard or its disk is nearly full (below).
* before loading the SOP model it also checks the swap in use
  (``sysctl vm.swapusage``) against :data:`SWAP_LIMIT_BYTES` and the free
  disk against :data:`DISK_FLOOR_BYTES` (swap files live on the startup
  disk, so a full disk means swap cannot grow).  Either one refuses the
  load: batched jobs wait with the reason in the status, as for memory,
  and a focus recording is processed on the resident annotation model
  instead, which loads nothing.  A Mac that loaded a 7 GB model with tens
  of GB already swapped out has run out of memory altogether (seen on the
  16 GB test Mac: a kernel panic, the compressor full, ~44 GB of swap).
* a load that fails (Ollama answers 5xx or times out) defers the phase with
  an exponential backoff: the calls raise ``OllamaUnavailable``, which the
  generators treat as "try again later", never as a failed Skill.  A load
  Ollama has not even started after :data:`LOAD_START_SECONDS` (the model
  is absent from ``/api/ps``: Ollama is still waiting for memory) is
  deferred then, not after the full load timeout; dropping the request
  withdraws it from Ollama's queue, so it does not hold up the annotation
  model's calls.

On a tier with one model for both (Recommended, since WP-F6) none of
this applies: the generation phase runs whenever work is due, as
before.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Callable

from agenthandover_worker.model_profiles import MODEL_SIZES_GB, get_profile
from agenthandover_worker.ollama import (
    OllamaClient, OllamaError, OllamaTimeout, OllamaUnavailable,
)

logger = logging.getLogger(__name__)

GB = 1_000_000_000  # decimal, as ``ollama list`` and the tier table count

# A passive-discovery pass that nobody waits for is due every 2 hours, or
# after 5 idle minutes; it joins the next generation phase that runs
# anyway, or waits this long for one.  (The daily jobs are not batched:
# they come once a day.)
BATCHED_JOBS = frozenset({"passive_discovery"})
GENERATION_BATCH_WAIT_SECONDS = 30 * 60

# Headroom over the SOP model's size (its /api/ps size includes the context
# at the calls' num_ctx; the tier table's download size does not, and the
# margin covers that context, about 1 GB for gemma4:12b at 16384).
MEMORY_MARGIN_BYTES = 1 * GB
# Batched work that has waited this long for memory runs on the annotation
# model instead.
FALLBACK_AFTER_SECONDS = 6 * 3600

# The SOP model is not loaded while more swap than this is in use or less
# disk than this is free (swap cannot grow).  ``agenthandover doctor`` warns
# at the same values (crates/cli/src/machine.rs pins them to these).
SWAP_LIMIT_BYTES = 8 * GB
DISK_FLOOR_BYTES = 10 * GB

# Memory's embedding model is small (~0.7 GB), and swap in use stays high
# long after the pressure that caused it has passed (many 16 GB Macs sit
# above SWAP_LIMIT_BYTES all day).  So background embedding waits on what
# says the Mac is short of memory now: little free memory, or swap that
# is growing; the disk floor applies as above.
EMBED_FREE_FLOOR_BYTES = 2 * GB
SWAP_GROWTH_WINDOW_SECONDS = 300.0
SWAP_GROWTH_LIMIT_BYTES = 256_000_000
# Embedding starts only once swap has been watched this long.
SWAP_WATCH_MIN_SECONDS = 60.0

# A failed SOP-model load is retried after this, doubling up to the cap.
LOAD_BACKOFF_SECONDS = 60.0
LOAD_BACKOFF_MAX_SECONDS = 30 * 60.0
# Ollama's own load timeout is 5 minutes; wait a little longer for its answer.
LOAD_TIMEOUT_SECONDS = 360.0
# A model that is not in /api/ps this long after the load request is not
# loading (Ollama lists a runner from the moment it starts it): defer.
LOAD_START_SECONDS = 60.0
GENERATION_KEEP_ALIVE = "5m"


# ---------------------------------------------------------------------------
# Free memory
# ---------------------------------------------------------------------------

_PAGE_SIZE_RE = re.compile(r"page size of (\d+) bytes")
_PAGES_RE = re.compile(r"^Pages (free|inactive|speculative):\s+(\d+)\.?\s*$", re.M)


def parse_vm_stat(text: str) -> int | None:
    """Bytes in free, inactive and speculative pages of ``vm_stat`` output,
    or None when the output does not have them."""
    size = _PAGE_SIZE_RE.search(text)
    pages = {name: int(count) for name, count in _PAGES_RE.findall(text)}
    if size is None or set(pages) != {"free", "inactive", "speculative"}:
        return None
    return int(size.group(1)) * sum(pages.values())


def free_memory_bytes() -> int | None:
    """Memory the system can hand to a model load now, or None when it
    cannot be read (not macOS)."""
    try:
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_vm_stat(out)


_SWAP_USED_RE = re.compile(r"used\s*=\s*([\d.]+)([KMGT])")
_UNITS = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}


def parse_swapusage(text: str) -> int | None:
    """Bytes of swap in use from ``sysctl vm.swapusage`` output
    (``total = 2048.00M  used = 971.81M  free = 1076.19M  (encrypted)``),
    or None when the output does not have them."""
    match = _SWAP_USED_RE.search(text)
    if match is None:
        return None
    return int(float(match.group(1)) * _UNITS[match.group(2)])


def swap_used_bytes() -> int | None:
    """Swap in use now, or None when it cannot be read (not macOS)."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_swapusage(out)


def free_disk_bytes() -> int | None:
    """Free space on the startup disk, where macOS keeps its swap files."""
    try:
        return shutil.disk_usage("/").free
    except OSError:
        return None


def before_generation_call(component: object) -> None:
    """Run *component*'s ``before_call`` (set by :meth:`ModelScheduler.register`)
    before a generation-model call; raise :class:`OllamaUnavailable` when the
    generation model could not be loaded, which every caller treats as
    "defer, try again later"."""
    hook = getattr(component, "before_call", None)
    if hook is not None and not hook():
        raise OllamaUnavailable("the generation model could not be loaded; deferred")


def _gb(n: int) -> str:
    return f"{n / GB:.1f} GB"


def _same_model(a: str, b: str) -> bool:
    """Ollama names: a bare name means its ``:latest`` tag."""
    def norm(name: str) -> str:
        return name if ":" in name else f"{name}:latest"
    return norm(a) == norm(b)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationPlan:
    """What this tick's generation phase may run."""

    run: bool
    model: str = ""
    # The batched jobs due now; a plan that runs lets all of them run.
    jobs: frozenset[str] = frozenset()
    fallback: bool = False

    def allows(self, job: str) -> bool:
        return self.run and job in self.jobs


class ModelScheduler:
    """The switch between the annotation and the generation phase.

    *clock* returns seconds (wall clock); *free_memory*, *swap_used* and
    *free_disk* return bytes or None (unknown).  Components registered with :meth:`register` (anything with a
    ``config.model``, such as the SOP generator, the synthesizer and the
    generation reasoner) are pointed at the annotation model while a
    fallback phase runs, and back afterwards.
    """

    def __init__(
        self,
        client: OllamaClient,
        *,
        annotation_model: str,
        sop_model: str,
        clock: Callable[[], float] = time.time,
        free_memory: Callable[[], int | None] = free_memory_bytes,
        swap_used: Callable[[], int | None] = swap_used_bytes,
        free_disk: Callable[[], int | None] = free_disk_bytes,
    ) -> None:
        self.client = client
        self.annotation_model = annotation_model
        self.sop_model = sop_model
        self._clock = clock
        self._free_memory = free_memory
        self._swap_used = swap_used
        self._free_disk = free_disk
        self._components: list[object] = []
        self._due_since: dict[str, float] = {}
        self._plan_model = sop_model
        self._generating = False
        self._startup_checked = False
        self._backoff_until = 0.0
        self._load_failures = 0
        # The loaded size (bytes) Ollama reported for a model, from /api/ps.
        self._sizes: dict[str, int] = {}
        # ``(time, swap in use)`` samples over the last
        # SWAP_GROWTH_WINDOW_SECONDS, for :meth:`embedding_blocker`.
        self._swap_samples: list[tuple[float, int]] = []
        # Why generation is not running (or runs on the fallback), for the
        # worker status; None while it runs normally.
        self.reason: str | None = None

    @property
    def split(self) -> bool:
        """True when annotation and Skills use different models."""
        return not _same_model(self.annotation_model, self.sop_model)

    @property
    def generation_model(self) -> str:
        """The model this tick's generation phase writes with."""
        return self._plan_model

    def register(self, *components: object) -> None:
        """Components that call the generation model: each switches Ollama
        to it before a call (``before_generation_call``)."""
        for component in components:
            if component is not None:
                component.before_call = self.enter_generation  # type: ignore[attr-defined]
                self._components.append(component)

    def status(self) -> dict | None:
        """The ``model_schedule`` block of ``worker-status.json`` (None on a
        one-model tier)."""
        if not self.split:
            return None
        return {
            "annotation_model": self.annotation_model,
            "sop_model": self.sop_model,
            "generation_model": self._plan_model,
            "generation_waiting": self.reason,
            "fallback": not _same_model(self._plan_model, self.sop_model),
        }

    # -- phases ---------------------------------------------------------

    def begin_annotation(self) -> None:
        """Start the annotation phase: no SOP model left loaded.

        Normally :meth:`end_generation` unloaded it; this also covers a tick
        that raised mid-phase and, once, a SOP model left loaded by an
        earlier worker run (read from ``/api/ps``)."""
        if not self.split:
            return
        if self._generating:
            self.end_generation()
        if self._startup_checked:
            return
        try:
            loaded = self.client.ps()
        except OllamaError as exc:
            logger.debug("Ollama /api/ps unavailable: %s", exc)
            return
        self._startup_checked = True
        self._learn_sizes(loaded)
        if any(_same_model(m.get("name", ""), self.sop_model) for m in loaded):
            self._unload(self.sop_model)

    def plan_generation(
        self, *, urgent: bool, due: dict[str, bool], take_batched: bool = True,
    ) -> GenerationPlan:
        """Decide whether this tick has a generation phase.

        *urgent* is work the user is waiting for (a stopped focus recording,
        answered questions); *due* names the batched jobs and whether each
        is due now.  With *take_batched* False the plan lets no batched job
        run and they keep waiting as they were: the focus job
        (inference_runner) plans for itself, and the passive jobs plan for
        themselves when they run.
        """
        now = self._clock()
        if not self._generating:
            self._plan_model = self.sop_model
        for job, is_due in due.items():
            if is_due:
                self._due_since.setdefault(job, now)
            else:
                self._due_since.pop(job, None)
        jobs = frozenset(self._due_since) if take_batched else frozenset()
        if not self.split:
            return self._run(self.sop_model, jobs)
        if not urgent and not jobs:
            return GenerationPlan(run=False)
        if now < self._backoff_until:
            return GenerationPlan(run=False)
        oldest = min(self._due_since.values(), default=now)
        if (
            not urgent
            and jobs <= BATCHED_JOBS
            and now - oldest < GENERATION_BATCH_WAIT_SECONDS
        ):
            return GenerationPlan(run=False)

        try:
            loaded: list[dict] | None = self.client.ps()
        except OllamaError as exc:
            logger.debug("Ollama /api/ps unavailable: %s", exc)
            loaded = None
        if loaded is not None:
            self._learn_sizes(loaded)
            if any(_same_model(m.get("name", ""), self.sop_model) for m in loaded):
                # Already loaded: nothing to load, nothing to check.
                self.reason = None
                return self._run(self.sop_model, jobs)

        pressure = self._pressure()
        if pressure is not None:
            if urgent or now - oldest >= FALLBACK_AFTER_SECONDS:
                self.reason = (
                    f"generating on {self.annotation_model}: not loading "
                    f"{self.sop_model}, {pressure}"
                )
                logger.warning("%s", self.reason)
                return self._run(self.annotation_model, jobs, fallback=True)
            self.reason = f"waiting: {pressure}"
            logger.info("Generation deferred, %s", self.reason)
            return GenerationPlan(run=False)

        short = self._memory_short(loaded) if loaded is not None else None
        if short is None:
            self.reason = None
            return self._run(self.sop_model, jobs)
        need, free = short
        if urgent:
            logger.warning(
                "Memory is short for %s (need %s, %s free); processing the "
                "recorded session anyway after unloading %s",
                self.sop_model, _gb(need), _gb(free), self.annotation_model,
            )
            self.reason = None
            return self._run(self.sop_model, jobs)
        if now - oldest >= FALLBACK_AFTER_SECONDS:
            self.reason = (
                f"generating on {self.annotation_model}: {self.sop_model} "
                f"needs {_gb(need)}, {_gb(free)} free for "
                f"{(now - oldest) / 3600:.0f} h"
            )
            logger.warning("%s", self.reason)
            return self._run(self.annotation_model, jobs, fallback=True)
        self.reason = f"waiting for memory: need {_gb(need)}, {_gb(free)} free"
        logger.info("Generation deferred, %s", self.reason)
        return GenerationPlan(run=False)

    def enter_generation(self) -> bool:
        """Switch to the generation model before the first generation call
        of a phase (later calls in the phase return at once).  Registered
        components call it through ``before_generation_call``.

        Unloads the annotation model and loads the generation model with the
        context size its calls use.  False when the load failed: the phase
        is deferred with a backoff and nothing is generated.
        """
        if not self.split or self._generating:
            return True
        now = self._clock()
        if now < self._backoff_until:
            return False
        model = self._plan_model
        if not _same_model(model, self.annotation_model):
            self._unload(self.annotation_model)
        try:
            self._load(model)
        except OllamaError as exc:
            self._load_failures += 1
            delay = min(
                LOAD_BACKOFF_SECONDS * 2 ** (self._load_failures - 1),
                LOAD_BACKOFF_MAX_SECONDS,
            )
            self._backoff_until = now + delay
            self.reason = f"Ollama could not load {model} ({exc}); retrying in {delay:.0f} s"
            logger.warning("Generation deferred: %s", self.reason)
            self._plan_model = self.sop_model
            return False
        self._load_failures = 0
        self._generating = True
        for component in self._components:
            component.config.model = model  # type: ignore[attr-defined]
        return True

    def end_generation(self) -> None:
        """End the generation phase: unload its model (unless it is the
        annotation model, which the next phase needs) and point the
        components back at the SOP model."""
        if not self._generating:
            return
        model = self._plan_model
        self._generating = False
        self._plan_model = self.sop_model
        for component in self._components:
            component.config.model = self.sop_model  # type: ignore[attr-defined]
        try:
            self._learn_sizes(self.client.ps())
        except OllamaError as exc:
            logger.debug("Ollama /api/ps unavailable: %s", exc)
        if not _same_model(model, self.annotation_model):
            self._unload(model)

    def embedding_blocker(self) -> str | None:
        """Why background embedding (Memory, and the "Upgrade everything"
        re-embed) must wait now, or None: the Mac is short of memory now
        (:meth:`_embedding_pressure`), or a generation phase has the SOP
        model loaded on a two-model tier, where another model beside it is
        what the memory guard exists to prevent."""
        pressure = self._embedding_pressure()
        if pressure is not None:
            return pressure
        if self.split and self._generating:
            return f"{self._plan_model} is writing Skills"
        return None

    def _embedding_pressure(self) -> str | None:
        """For the small embedding model: free memory under
        :data:`EMBED_FREE_FLOOR_BYTES`, swap grown by more than
        :data:`SWAP_GROWTH_LIMIT_BYTES` within
        :data:`SWAP_GROWTH_WINDOW_SECONDS` (or not watched for
        :data:`SWAP_WATCH_MIN_SECONDS` yet), or free disk under
        :data:`DISK_FLOOR_BYTES`.  Swap merely in use is no reason: unlike
        the SOP model's :meth:`_pressure`.  An unknown value is no reason."""
        disk = self._free_disk()
        if disk is not None and disk < DISK_FLOOR_BYTES:
            return (
                f"{_gb(disk)} of disk free (floor {_gb(DISK_FLOOR_BYTES)}): "
                "swap cannot grow"
            )
        free = self._free_memory()
        if free is not None and free < EMBED_FREE_FLOOR_BYTES:
            return (
                f"{_gb(free)} of memory free (needs {_gb(EMBED_FREE_FLOOR_BYTES)}): "
                "the Mac is short of memory"
            )
        swap = self._swap_used()
        if swap is None:
            return None
        now = self._clock()
        self._swap_samples = [
            (t, b) for t, b in self._swap_samples if now - t <= SWAP_GROWTH_WINDOW_SECONDS
        ]
        self._swap_samples.append((now, swap))
        first = self._swap_samples[0][0]
        growth = swap - min(b for _, b in self._swap_samples)
        if growth > SWAP_GROWTH_LIMIT_BYTES:
            return (
                f"swap grew by {growth / 1_000_000:.0f} MB in the last "
                f"{SWAP_GROWTH_WINDOW_SECONDS / 60:.0f} minutes: the Mac is short of memory"
            )
        if now - first < SWAP_WATCH_MIN_SECONDS:
            return "checking that the Mac is not short of memory"
        return None

    # -- internals ------------------------------------------------------

    def _run(self, model: str, jobs: frozenset[str], *, fallback: bool = False) -> GenerationPlan:
        for job in jobs:
            self._due_since.pop(job, None)
        self._plan_model = model
        return GenerationPlan(run=True, model=model, jobs=jobs, fallback=fallback)

    def _load(self, model: str) -> None:
        """Load *model* with the context its calls use; raise
        :class:`OllamaError` when it does not load.

        The first wait is :data:`LOAD_START_SECONDS`.  If Ollama has not
        answered by then and the model is not in ``/api/ps``, the load has
        not started and :class:`OllamaUnavailable` defers it; if it is
        there, the load is under way and the request is repeated (Ollama
        joins it to the running load) for the rest of
        :data:`LOAD_TIMEOUT_SECONDS`.
        """
        options = {"num_predict": 1, "num_ctx": get_profile(model).sop_num_ctx}
        try:
            self.client.preload(
                model, GENERATION_KEEP_ALIVE, options=options,
                timeout=LOAD_START_SECONDS,
            )
            return
        except OllamaTimeout:
            loaded = self.client.ps()
        self._learn_sizes(loaded)
        if not any(_same_model(m.get("name", ""), model) for m in loaded):
            raise OllamaUnavailable(
                f"Ollama did not start loading {model} within "
                f"{LOAD_START_SECONDS:.0f} s (not in /api/ps)"
            )
        self.client.preload(
            model, GENERATION_KEEP_ALIVE, options=options,
            timeout=LOAD_TIMEOUT_SECONDS - LOAD_START_SECONDS,
        )

    def _unload(self, model: str) -> None:
        try:
            self.client.unload(model)
        except OllamaError as exc:
            logger.debug("Unloading %s failed: %s", model, exc)

    def _learn_sizes(self, loaded: list[dict]) -> None:
        for entry in loaded:
            name, size = entry.get("name", ""), entry.get("size")
            if name and isinstance(size, int) and size > 0:
                self._sizes[name] = size

    def _size_of(self, model: str) -> int | None:
        for name, size in self._sizes.items():
            if _same_model(name, model):
                return size
        gb = MODEL_SIZES_GB.get(model)
        return int(gb * GB) if gb else None

    def _pressure(self) -> str | None:
        """Why no model may be loaded now (swap in use over
        :data:`SWAP_LIMIT_BYTES`, free disk under :data:`DISK_FLOOR_BYTES`),
        or None.  An unknown value is no reason to refuse."""
        swap = self._swap_used()
        if swap is not None and swap > SWAP_LIMIT_BYTES:
            return (
                f"{_gb(swap)} of swap in use (limit {_gb(SWAP_LIMIT_BYTES)}): "
                "the Mac is short of memory"
            )
        disk = self._free_disk()
        if disk is not None and disk < DISK_FLOOR_BYTES:
            return (
                f"{_gb(disk)} of disk free (floor {_gb(DISK_FLOOR_BYTES)}): "
                "swap cannot grow"
            )
        return None

    def _memory_short(self, loaded: list[dict]) -> tuple[int, int] | None:
        """``(need, free)`` bytes when the SOP model (not in *loaded*, the
        ``/api/ps`` list) does not fit, else None; *free* counts the
        annotation model's memory, which the phase frees.

        Fits when free memory plus what unloading the annotation model
        returns covers its size and the margin.  An unknown size or free
        memory is no reason to wait: the load itself then decides.  (Nor is
        an unknown loaded set: the caller does not ask then.)
        """
        size = self._size_of(self.sop_model)
        free = self._free_memory()
        if size is None or free is None:
            return None
        reclaim = sum(
            m.get("size") or 0 for m in loaded
            if _same_model(m.get("name", ""), self.annotation_model)
        )
        need = size + MEMORY_MARGIN_BYTES
        available = free + reclaim
        return None if available >= need else (need, available)
