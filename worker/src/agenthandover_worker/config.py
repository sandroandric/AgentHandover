"""The worker's one config loader.

``load_config()`` reads ``config.toml`` once (one ``tomllib`` read) and
returns a frozen :class:`Config`.  Every default the worker applies to a
missing or invalid setting lives here, next to the field it belongs to.

The sections and defaults mirror ``crates/common/src/config.rs`` (the
daemon's reader of the same file) for the fields the worker uses; the
worker-only settings (``[vlm] ollama_host`` and
``annotation_timeout_seconds``, ``[focus]``, ``[features]``,
``[embedding]``, ``[memory]``, ``[export] claude_skills_dir``) are documented in
``config.example.toml``.

A missing file gives the defaults.  A file that is not valid TOML gives
the defaults and a warning.  A value of the wrong type gives that field's
default and a warning; it never stops the worker.
"""

from __future__ import annotations

import logging
import platform
import tomllib
from dataclasses import dataclass, field, replace
from datetime import time as dt_time
from pathlib import Path
from typing import Any

from agenthandover_worker.model_profiles import MEMORY_MODEL

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# ``[export] adapter`` values (the worker's ``--adapter`` choices).
EXPORT_ADAPTERS = ("openclaw", "generic", "skill-md", "claude-skill", "all")


def config_path() -> Path:
    """Where ``config.toml`` lives on this platform (it may not exist)."""
    if platform.system() == "Darwin":
        return (
            Path.home() / "Library" / "Application Support"
            / "agenthandover" / "config.toml"
        )
    return Path.home() / ".config" / "agenthandover" / "config.toml"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VlmConfig:
    """``[vlm]``: the local models (through Ollama) and their budgets.

    ``mode = "local"`` in the file is what the tier writers put there;
    the loader does not read it (cloud mode was removed in 0.5.0 and
    ``model_profiles.migrate_vlm_block`` turns an old remote block local).
    """

    annotation_enabled: bool = True
    # The RAM tier onboarding or ``agenthandover setup`` chose; a model key
    # the file leaves out comes from this tier, else from the tier this
    # Mac's RAM recommends (``model_profiles.TIERS``, the one tier table).
    tier: str = ""
    annotation_model: str = ""
    sop_model: str = ""
    # The LLM reasoner's model; empty means ``sop_model``.
    reasoning_model: str = ""
    # A/B flag: use the tier's ``ab_sop_model`` (48 GB: the 26B MoE) as
    # the SOP model.
    sop_ab: bool = False
    sliding_window_max_age_sec: int = 600
    ollama_host: str = DEFAULT_OLLAMA_HOST
    # Per annotation call; also bounds the model preload at worker start.
    annotation_timeout_seconds: float = 180.0
    # Daily budget for passive annotation (focus recordings are exempt):
    # frames sent to the model, and Ollama's own prompt + generation time.
    # When either is spent, passive rows wait for the next local day.
    annotation_max_frames_per_day: int = 1500
    annotation_max_compute_minutes_per_day: float = 60.0
    # The same per rolling hour, so one busy hour cannot spend the day:
    # frames sent to the model (cache hits free) and Ollama compute of every
    # passive model call (annotation, frame diffs, passive discovery).
    annotation_max_frames_per_hour: int = 60
    annotation_max_compute_minutes_per_hour: float = 10.0
    # Daily budget of the LLM reasoner (cluster labels, session links,
    # corrections, curation) and the behavioral synthesizer: model calls,
    # and their wall-clock minutes.  When either is spent, both wait for
    # the next local day (``reasoning_paused_reason`` in worker-status.json).
    reasoning_max_calls_per_day: int = 300
    reasoning_max_compute_minutes_per_day: float = 90.0

    @property
    def reasoner_model(self) -> str:
        """The model the LLM reasoner runs: the SOP model unless overridden."""
        return self.reasoning_model or self.sop_model


@dataclass(frozen=True)
class LlmConfig:
    """``[llm]``: the optional SOP text enhancer."""

    enhance_sops: bool = True
    max_enhancements_per_day: int = 20
    model: str = ""
    timeout_seconds: int = 60
    temperature: float = 0.3
    max_tokens: int = 800


@dataclass(frozen=True)
class SopConfig:
    """``[sop]``."""

    auto_approve: bool = False


@dataclass(frozen=True)
class IdleJobsConfig:
    """``[idle_jobs]``: when heavy background work may run."""

    require_ac_power: bool = True
    min_battery_percent: int = 50
    max_cpu_percent: int = 30
    max_temp_c: int = 80
    run_window_start: dt_time = dt_time(1, 0)
    run_window_end: dt_time = dt_time(5, 0)

    def scheduler_kwargs(self) -> dict:
        """Keyword arguments for ``scheduler.SchedulerConfig``."""
        return {
            "require_ac_power": self.require_ac_power,
            "min_battery_percent": self.min_battery_percent,
            "max_cpu_percent": self.max_cpu_percent,
            "max_temp_c": self.max_temp_c,
            "run_window_start": self.run_window_start,
            "run_window_end": self.run_window_end,
        }


@dataclass(frozen=True)
class FeaturesConfig:
    """``[features]``: pipeline stages that can be switched off."""

    activity_classification: bool = True
    continuity_tracking: bool = True
    lifecycle_management: bool = True
    curation: bool = True
    runtime_validation: bool = True


@dataclass(frozen=True)
class ExportConfig:
    """``[export]``."""

    # Which writers the worker exports through (``--adapter`` overrides).
    # A pkg install writes ``adapter = "all"``; a missing key keeps the
    # daemon's default, ``openclaw``.
    adapter: str = "openclaw"
    # Root for ClaudeSkillWriter; ``~`` is expanded.
    claude_skills_dir: Path | None = None  # None: ~/.claude/skills

    @property
    def skills_dir(self) -> Path:
        return self.claude_skills_dir or Path.home() / ".claude" / "skills"


@dataclass(frozen=True)
class KnowledgeConfig:
    """``[knowledge]``: the local query API, and how many micro-questions
    (``questions``) may be asked in any seven days (0: none)."""

    query_api_enabled: bool = True
    query_api_port: int = 9477
    weekly_question_budget: int = 5


@dataclass(frozen=True)
class EmbeddingConfig:
    """``[embedding]``: one model for every text embedding (the vector KB
    and the task segmenter), always used with nomic's task prefixes."""

    model: str = "nomic-embed-text"
    # Vectors not updated for this many days are purged once a day.
    retention_days: int = 90
    # Hard cap on stored vectors; the least recently updated go first.
    max_rows: int = 50_000


@dataclass(frozen=True)
class MemoryConfig:
    """``[memory]``: the searchable record of what was on screen (``memory``).

    Off unless the file turns it on (setup writes ``enabled = true`` for new
    installs).  ``crates/common/src/config.rs`` reads the same keys."""

    enabled: bool = False
    model: str = MEMORY_MODEL
    dimensions: int = 256
    # Memory rows older than this are purged once a day.
    retention_days: int = 365
    # Encrypted thumbnails older than this are deleted once a day.
    thumbnail_days: int = 7
    # MCP ``search_memory`` answers only when this is on.
    agents_access: bool = False
    # Frames embedded per rolling hour; past it, frames wait.
    max_frames_per_hour: int = 600


@dataclass(frozen=True)
class FocusConfig:
    """``[focus]``."""

    # A focus recording older than this is closed as stale.
    max_session_hours: float = 4.0


@dataclass(frozen=True)
class OpenClawConfig:
    """``[openclaw]``."""

    workspace_path: str = ""


@dataclass(frozen=True)
class Config:
    """The whole of ``config.toml`` as the worker reads it."""

    path: Path | None = None  # the file read, None when it did not exist
    vlm: VlmConfig = field(default_factory=lambda: _vlm({}))
    llm: LlmConfig = field(default_factory=LlmConfig)
    sop: SopConfig = field(default_factory=SopConfig)
    idle_jobs: IdleJobsConfig = field(default_factory=IdleJobsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    focus: FocusConfig = field(default_factory=FocusConfig)
    openclaw: OpenClawConfig = field(default_factory=OpenClawConfig)
    # ``[privacy.zones]`` as written; ``privacy_zones.PrivacyZoneConfig``
    # owns its defaults and validation.
    privacy_zones: dict = field(default_factory=dict)
    # The parsed file, for ``ConfigValidator``.
    raw: dict = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> Config:
    """Read ``config.toml`` (default: :func:`config_path`) into a Config."""
    file = Path(path) if path is not None else config_path()
    if not file.is_file():
        return Config()
    try:
        with open(file, "rb") as fh:
            raw = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("Could not read %s (%s); using defaults", file, exc)
        return Config()

    def section(name: str) -> dict:
        value = raw.get(name, {})
        if isinstance(value, dict):
            return value
        logger.warning("[%s] in %s is not a table; using defaults", name, file)
        return {}

    privacy = section("privacy")
    zones = privacy.get("zones", {})
    return Config(
        path=file,
        vlm=_vlm(section("vlm")),
        llm=_build(LlmConfig, "llm", section("llm")),
        sop=_build(SopConfig, "sop", section("sop")),
        idle_jobs=_idle_jobs(section("idle_jobs")),
        features=_build(FeaturesConfig, "features", section("features")),
        export=_export(section("export")),
        knowledge=_build(KnowledgeConfig, "knowledge", section("knowledge")),
        embedding=_build(EmbeddingConfig, "embedding", section("embedding")),
        memory=_build(MemoryConfig, "memory", section("memory")),
        focus=_build(FocusConfig, "focus", section("focus")),
        openclaw=_build(OpenClawConfig, "openclaw", section("openclaw")),
        privacy_zones=zones if isinstance(zones, dict) else {},
        raw=raw,
    )


# Fields whose value must be a positive, finite number.
_POSITIVE = {
    ("vlm", "annotation_timeout_seconds"),
    ("vlm", "annotation_max_frames_per_day"),
    ("vlm", "annotation_max_compute_minutes_per_day"),
    ("vlm", "annotation_max_frames_per_hour"),
    ("vlm", "annotation_max_compute_minutes_per_hour"),
    ("vlm", "reasoning_max_calls_per_day"),
    ("vlm", "reasoning_max_compute_minutes_per_day"),
    ("focus", "max_session_hours"),
    ("embedding", "retention_days"),
    ("embedding", "max_rows"),
    ("memory", "dimensions"),
    ("memory", "retention_days"),
    ("memory", "thumbnail_days"),
    ("memory", "max_frames_per_hour"),
}


def _build(cls: type, name: str, values: dict) -> Any:
    """``cls`` with every field present in *values* coerced to the type of
    its default; a value that does not fit keeps the default."""
    defaults = cls()
    kwargs: dict[str, Any] = {}
    for key in cls.__dataclass_fields__:
        if key not in values:
            continue
        default = getattr(defaults, key)
        value = _coerce(values[key], default)
        if value is not None and (name, key) in _POSITIVE and not (
            0 < value < float("inf")
        ):
            value = None
        if value is None:
            logger.warning(
                "[%s] %s = %r is invalid; using %r", name, key, values[key], default,
            )
            continue
        kwargs[key] = value
    return cls(**kwargs)


def _coerce(value: Any, default: Any) -> Any:
    """*value* as the type of *default*, or None when it cannot be."""
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value != 0
        if isinstance(value, str) and value.strip().lower() in (
            "true", "1", "yes", "false", "0", "no",
        ):
            return value.strip().lower() in ("true", "1", "yes")
        return None
    if isinstance(default, int):
        if isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if isinstance(default, float):
        if isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if isinstance(default, str):
        return value if isinstance(value, str) else None
    return None


def _vlm(values: dict) -> VlmConfig:
    """``[vlm]`` with the model keys it leaves out filled from the tier table."""
    from agenthandover_worker.model_profiles import get_tier, recommend_tier

    base = _build(VlmConfig, "vlm", values)
    tier = get_tier(base.tier) if base.tier else None
    if base.tier and tier is None:
        logger.warning("[vlm] tier = %r is not a known tier; using RAM's", base.tier)
    tier = tier or recommend_tier()
    sop_model = base.sop_model or tier.sop_model
    if base.sop_ab and tier.ab_sop_model:
        sop_model = tier.ab_sop_model
    return replace(
        base,
        tier=tier.id,
        annotation_model=base.annotation_model or tier.annotation_model,
        sop_model=sop_model,
    )


def _idle_jobs(values: dict) -> IdleJobsConfig:
    base = _build(
        IdleJobsConfig, "idle_jobs",
        {k: v for k, v in values.items() if k != "run_window_local_time"},
    )
    window = values.get("run_window_local_time")
    if window is None:
        return base
    try:
        start, end = str(window).split("-")
        start_h, start_m = (int(p) for p in start.strip().split(":"))
        end_h, end_m = (int(p) for p in end.strip().split(":"))
        return IdleJobsConfig(
            **{**base.scheduler_kwargs(),
               "run_window_start": dt_time(start_h, start_m),
               "run_window_end": dt_time(end_h, end_m)},
        )
    except ValueError:
        logger.warning(
            "[idle_jobs] run_window_local_time = %r is not HH:MM-HH:MM; using "
            "01:00-05:00", window,
        )
        return base


def _export(values: dict) -> ExportConfig:
    adapter = values.get("adapter", ExportConfig.adapter)
    if adapter not in EXPORT_ADAPTERS:
        logger.warning(
            "[export] adapter = %r is not one of %s; using %r",
            adapter, ", ".join(EXPORT_ADAPTERS), ExportConfig.adapter,
        )
        adapter = ExportConfig.adapter
    configured = values.get("claude_skills_dir")
    if not configured:
        return ExportConfig(adapter=adapter)
    if not isinstance(configured, str):
        logger.warning(
            "[export] claude_skills_dir = %r is not a path; using the default",
            configured,
        )
        return ExportConfig(adapter=adapter)
    return ExportConfig(
        adapter=adapter, claude_skills_dir=Path(configured).expanduser(),
    )
