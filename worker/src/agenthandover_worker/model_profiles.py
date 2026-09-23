"""Model profiles — per-model optimal inference settings.

Each model family has different optimal parameters for annotation and SOP
generation. This module provides a single source of truth so the pipeline
automatically uses the right settings regardless of which model the user picks.

It also holds the one RAM tier table (:data:`TIERS`).  The installer runs
``python -m agenthandover_worker.model_profiles write-tiers`` to write it as
``tiers.json``; the app's onboarding, the CLI (``agenthandover setup``,
``doctor``) and the daemon's config defaults all read that JSON, so the
tiers cannot diverge.  ``resources/tiers.json`` is the committed copy: the
Rust crates compile it in and the Swift tests decode it; a worker test pins
it to :func:`tiers_json`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profile dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelProfile:
    """Optimal inference settings for a specific model."""

    # Identity
    model: str
    family: str  # "qwen3.5", "gemma4"

    # Annotation settings (text-first JSON extraction, vision when needed).
    # The output is constrained by the annotation JSON schema, so
    # temperature 0 keeps it deterministic.  ``ann_num_predict`` bounds the
    # compact 12-leaf schema with its capped lists (typical output ~120
    # tokens); ``ann_num_ctx`` holds the <=1,700-token prompt, an image on
    # a vision call, and the output with headroom.
    ann_system: str = ""
    ann_temperature: float = 0.0
    ann_num_predict: int = 300
    ann_num_ctx: int = 4096
    ann_top_k: int = 40
    ann_top_p: float = 0.95
    ann_presence_penalty: float = 0.0
    ann_think: bool | str = False  # False, True, "low", "medium", "high"

    # SOP generation settings (reasoning, structured output).
    # ``sop_num_ctx`` is the context of every generation-phase call on the
    # model (SOP, synthesis, the reasoner's Q&A, labels and curation), not
    # of the SOP call alone: Ollama reloads a model whose num_ctx changes,
    # so per-call-type contexts would reload it inside one phase (WP-C9).
    # It is the largest call measured plus a ~30% margin.  Measured on the
    # focus_dailynews cassette (prompt_eval_count + num_predict): SOP 6,619
    # prompt tokens at the 20-keyframe cap + num_predict; synthesis 4,225 +
    # 3,000; reasoner calls <= 1,002 + 4,000.  Gemma (num_predict 6,000):
    # 12,619 -> 16,384.  Qwen (num_predict 12,000): 18,619 -> 24,576.
    # Qwen 3.5 4B is the SOP model on the Standard and Recommended (16 GB)
    # tiers, so those tiers' generation phase runs at 24,576; Gemma 4 12B
    # QAT (Performance) at 16,384 and 31B QAT (Max Quality) at 32,768.
    sop_system: str = ""
    sop_temperature: float = 0.3
    sop_num_predict: int = 6000
    sop_num_ctx: int = 16384
    sop_top_k: int = 40
    sop_top_p: float = 0.95
    sop_presence_penalty: float = 0.0
    sop_think: bool | str = False

    # Behavioral synthesis system prompt override (empty = the
    # synthesizer's own BEHAVIORAL_SYSTEM_PROMPT).  Never the SOP prompt:
    # synthesis asks for a different JSON object than SOP generation.
    synth_system: str = ""

    # Diff settings (text-only comparison)
    diff_system: str = ""
    diff_temperature: float = 0.3
    diff_num_predict: int = 500
    diff_top_k: int = 40
    diff_top_p: float = 0.95
    diff_presence_penalty: float = 0.0
    diff_think: bool | str = False

    def ann_options(self) -> dict:
        """Return Ollama options dict for annotation calls."""
        return {
            "num_predict": self.ann_num_predict,
            "num_ctx": self.ann_num_ctx,
            "temperature": self.ann_temperature,
            "top_k": self.ann_top_k,
            "top_p": self.ann_top_p,
            "presence_penalty": self.ann_presence_penalty,
        }

    def sop_options(self) -> dict:
        """Return Ollama options dict for SOP generation calls."""
        return {
            "num_predict": self.sop_num_predict,
            "num_ctx": self.sop_num_ctx,
            "temperature": self.sop_temperature,
            "top_k": self.sop_top_k,
            "top_p": self.sop_top_p,
            "presence_penalty": self.sop_presence_penalty,
        }

    def diff_options(self) -> dict:
        """Return Ollama options dict for frame diff calls."""
        return {
            "num_predict": self.diff_num_predict,
            "temperature": self.diff_temperature,
            "top_k": self.diff_top_k,
            "top_p": self.diff_top_p,
            "presence_penalty": self.diff_presence_penalty,
        }


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_GEMMA_ANN_SYSTEM = (
    "You are a screen analysis expert. Always respond with valid JSON only. "
    "No markdown, no explanation, no code fences. Just the JSON object."
)

_GEMMA_SOP_SYSTEM = (
    "You are a workflow analysis expert. Generate structured Skills from "
    "observed user workflows. Respond with valid JSON only."
)

_GEMMA_DIFF_SYSTEM = (
    "You are a screen change detector. Compare two consecutive screen "
    "annotations and describe the user's action. Respond with valid JSON only."
)


# ---------------------------------------------------------------------------
# Known profiles
# ---------------------------------------------------------------------------

_PROFILES: dict[str, ModelProfile] = {
    # --- Qwen 3.5 family (default for 8GB machines) ---
    "qwen3.5:2b": ModelProfile(
        model="qwen3.5:2b",
        family="qwen3.5",
        # Qwen has baked-in presence_penalty=1.5, top_k=20
        # Don't override — let its defaults work
        ann_think=False,
        sop_system="",
        sop_num_predict=12000,
        sop_num_ctx=24576,
        sop_temperature=0.3,
        sop_think=False,  # Qwen thinking eats all tokens
        diff_num_predict=500,
        diff_think=False,
    ),
    "qwen3.5:4b": ModelProfile(
        model="qwen3.5:4b",
        family="qwen3.5",
        ann_think=False,
        sop_num_predict=12000,
        sop_num_ctx=24576,
        sop_temperature=0.3,
        sop_think=False,
        diff_num_predict=500,
        diff_think=False,
    ),

    # --- Gemma 4 family ---
    # Best practices from Google: temp=1.0, top_p=0.95, top_k=64
    # Our testing: temp=0.3 for JSON, presence_penalty=1.0 for extraction
    # Image before text for optimal multimodal performance

    "gemma4": ModelProfile(
        model="gemma4",
        family="gemma4",
        ann_system=_GEMMA_ANN_SYSTEM,
        ann_top_k=64,
        ann_top_p=0.95,
        ann_presence_penalty=1.0,
        ann_think=False,
        sop_system=_GEMMA_SOP_SYSTEM,
        sop_temperature=0.3,
        sop_num_predict=6000,
        sop_num_ctx=16384,
        sop_top_k=64,
        sop_top_p=0.95,
        sop_presence_penalty=1.0,
        sop_think=True,
        diff_system=_GEMMA_DIFF_SYSTEM,
        diff_temperature=0.3,
        diff_num_predict=500,
        diff_top_k=64,
        diff_top_p=0.95,
        diff_presence_penalty=1.0,
        diff_think=False,
    ),
    # QAT 12B — the SOP model of the 24GB tier (v0.4.0 also used it on 16GB;
    # WP-F6 moved 16GB to qwen3.5:4b).  Settings
    # mirror the base ``gemma4`` profile, which is exactly what the
    # dailynews A/B validation ran on (the 12B QAT inherited the gemma4
    # family profile via fallback and produced the winning output).
    "gemma4:12b-it-qat": ModelProfile(
        model="gemma4:12b-it-qat",
        family="gemma4",
        ann_system=_GEMMA_ANN_SYSTEM,
        ann_top_k=64,
        ann_top_p=0.95,
        ann_presence_penalty=1.0,
        ann_think=False,
        sop_system=_GEMMA_SOP_SYSTEM,
        sop_temperature=0.3,
        sop_num_predict=6000,
        sop_num_ctx=16384,
        sop_top_k=64,
        sop_top_p=0.95,
        sop_presence_penalty=1.0,
        sop_think=True,
        diff_system=_GEMMA_DIFF_SYSTEM,
        diff_temperature=0.3,
        diff_num_predict=500,
        diff_top_k=64,
        diff_top_p=0.95,
        diff_presence_penalty=1.0,
        diff_think=False,
    ),
    "gemma4:e4b": ModelProfile(
        model="gemma4:e4b",
        family="gemma4",
        ann_system=_GEMMA_ANN_SYSTEM,
        ann_top_k=64,
        ann_top_p=0.95,
        ann_presence_penalty=1.0,
        ann_think=False,
        sop_system=_GEMMA_SOP_SYSTEM,
        sop_temperature=0.3,
        sop_num_predict=6000,
        sop_num_ctx=16384,
        sop_top_k=64,
        sop_top_p=0.95,
        sop_presence_penalty=1.0,
        sop_think=True,
        diff_system=_GEMMA_DIFF_SYSTEM,
        diff_temperature=0.3,
        diff_num_predict=500,
        diff_top_k=64,
        diff_top_p=0.95,
        diff_presence_penalty=1.0,
        diff_think=False,
    ),
    "gemma4:e4b-it-q8_0": ModelProfile(
        model="gemma4:e4b-it-q8_0",
        family="gemma4",
        ann_system=_GEMMA_ANN_SYSTEM,
        ann_top_k=64,
        ann_top_p=0.95,
        ann_presence_penalty=1.0,
        ann_think=False,
        sop_system=_GEMMA_SOP_SYSTEM,
        sop_temperature=0.3,
        sop_num_predict=6000,
        sop_num_ctx=16384,
        sop_top_k=64,
        sop_top_p=0.95,
        sop_presence_penalty=1.0,
        sop_think=True,
        diff_system=_GEMMA_DIFF_SYSTEM,
        diff_temperature=0.3,
        diff_num_predict=500,
        diff_top_k=64,
        diff_top_p=0.95,
        diff_presence_penalty=1.0,
        diff_think=False,
    ),
    "gemma4:e2b": ModelProfile(
        model="gemma4:e2b",
        family="gemma4",
        ann_system=_GEMMA_ANN_SYSTEM,
        ann_top_k=64,
        ann_top_p=0.95,
        ann_presence_penalty=0.0,  # E2B overflows with penalty
        ann_think=False,
        sop_system=_GEMMA_SOP_SYSTEM,
        sop_temperature=0.3,
        sop_num_predict=6000,
        sop_num_ctx=16384,
        sop_top_k=64,
        sop_top_p=0.95,
        sop_presence_penalty=0.0,
        sop_think=True,
        diff_system=_GEMMA_DIFF_SYSTEM,
        diff_temperature=0.3,
        diff_num_predict=500,
        diff_top_k=64,
        diff_top_p=0.95,
        diff_think=False,
    ),
    "gemma4:26b": ModelProfile(
        model="gemma4:26b",
        family="gemma4",
        ann_system=_GEMMA_ANN_SYSTEM,
        ann_top_k=64,
        ann_top_p=0.95,
        ann_presence_penalty=1.0,
        ann_think=False,
        sop_system=_GEMMA_SOP_SYSTEM,
        sop_temperature=0.3,
        sop_num_predict=6000,
        sop_num_ctx=32768,  # 256K capable, give more room
        sop_top_k=64,
        sop_top_p=0.95,
        sop_presence_penalty=1.0,
        sop_think=True,
        diff_system=_GEMMA_DIFF_SYSTEM,
        diff_temperature=0.3,
        diff_num_predict=500,
        diff_top_k=64,
        diff_top_p=0.95,
        diff_presence_penalty=1.0,
        diff_think=False,
    ),
    # MoE 26B (4B active) QAT: registered for the 48 GB tier's A/B arm
    # (``[vlm] sop_ab = true``); same settings as ``gemma4:26b``.
    "gemma4:26b-a4b-it-qat": ModelProfile(
        model="gemma4:26b-a4b-it-qat",
        family="gemma4",
        ann_system=_GEMMA_ANN_SYSTEM,
        ann_top_k=64,
        ann_top_p=0.95,
        ann_presence_penalty=1.0,
        ann_think=False,
        sop_system=_GEMMA_SOP_SYSTEM,
        sop_temperature=0.3,
        sop_num_predict=6000,
        sop_num_ctx=32768,
        sop_top_k=64,
        sop_top_p=0.95,
        sop_presence_penalty=1.0,
        sop_think=True,
        diff_system=_GEMMA_DIFF_SYSTEM,
        diff_temperature=0.3,
        diff_num_predict=500,
        diff_top_k=64,
        diff_top_p=0.95,
        diff_presence_penalty=1.0,
        diff_think=False,
    ),
    "gemma4:31b": ModelProfile(
        model="gemma4:31b",
        family="gemma4",
        ann_system=_GEMMA_ANN_SYSTEM,
        ann_top_k=64,
        ann_top_p=0.95,
        ann_presence_penalty=1.0,
        ann_think=False,
        sop_system=_GEMMA_SOP_SYSTEM,
        sop_temperature=0.3,
        sop_num_predict=6000,
        sop_num_ctx=32768,
        sop_top_k=64,
        sop_top_p=0.95,
        sop_presence_penalty=1.0,
        sop_think=True,
        diff_system=_GEMMA_DIFF_SYSTEM,
        diff_temperature=0.3,
        diff_num_predict=500,
        diff_top_k=64,
        diff_top_p=0.95,
        diff_presence_penalty=1.0,
        diff_think=False,
    ),
    # QAT 31B — the v0.4.0 default for the 48GB max-quality tier.  Same
    # settings as ``gemma4:31b`` (32K SOP context) but the QAT quant runs
    # in ~18GB instead of 20GB while keeping near-original quality.
    "gemma4:31b-it-qat": ModelProfile(
        model="gemma4:31b-it-qat",
        family="gemma4",
        ann_system=_GEMMA_ANN_SYSTEM,
        ann_top_k=64,
        ann_top_p=0.95,
        ann_presence_penalty=1.0,
        ann_think=False,
        sop_system=_GEMMA_SOP_SYSTEM,
        sop_temperature=0.3,
        sop_num_predict=6000,
        sop_num_ctx=32768,
        sop_top_k=64,
        sop_top_p=0.95,
        sop_presence_penalty=1.0,
        sop_think=True,
        diff_system=_GEMMA_DIFF_SYSTEM,
        diff_temperature=0.3,
        diff_num_predict=500,
        diff_top_k=64,
        diff_top_p=0.95,
        diff_presence_penalty=1.0,
        diff_think=False,
    ),
    "gemma4:31b-it-q8_0": ModelProfile(
        model="gemma4:31b-it-q8_0",
        family="gemma4",
        ann_system=_GEMMA_ANN_SYSTEM,
        ann_top_k=64,
        ann_top_p=0.95,
        ann_presence_penalty=1.0,
        ann_think=False,
        sop_system=_GEMMA_SOP_SYSTEM,
        sop_temperature=0.3,
        sop_num_predict=6000,
        sop_num_ctx=32768,
        sop_top_k=64,
        sop_top_p=0.95,
        sop_presence_penalty=1.0,
        sop_think=True,
        diff_system=_GEMMA_DIFF_SYSTEM,
        diff_temperature=0.3,
        diff_num_predict=500,
        diff_top_k=64,
        diff_top_p=0.95,
        diff_presence_penalty=1.0,
        diff_think=False,
    ),
}


def get_profile(model: str) -> ModelProfile:
    """Get the optimal profile for a model.

    Falls back to a generic profile based on family detection if the exact
    model tag isn't registered.
    """
    if model in _PROFILES:
        return _PROFILES[model]

    # Try family match (e.g. "gemma4:something-custom" → gemma4 profile)
    for prefix in ("gemma4:", "qwen3.5:"):
        if model.startswith(prefix) and prefix.rstrip(":") in _PROFILES:
            base = _PROFILES.get(f"{prefix}latest") or _PROFILES.get(prefix.rstrip(":"))
            if base:
                logger.info("Using %s profile for unknown model %s", prefix, model)
                return ModelProfile(**{**base.__dict__, "model": model})

    # Gemma family detection
    if "gemma4" in model.lower() or "gemma-4" in model.lower():
        base = _PROFILES["gemma4"]
        logger.info("Using gemma4 profile for %s", model)
        return ModelProfile(**{**base.__dict__, "model": model})

    # Qwen family detection
    if "qwen" in model.lower():
        base = _PROFILES["qwen3.5:2b"]
        logger.info("Using qwen profile for %s", model)
        return ModelProfile(**{**base.__dict__, "model": model})

    # Unknown model — return safe defaults
    logger.warning("No profile for model '%s', using generic defaults", model)
    return ModelProfile(model=model, family="unknown")


# ---------------------------------------------------------------------------
# Ollama version check
# ---------------------------------------------------------------------------

# The QAT Gemma 4 tags need a recent Ollama (older ones answer a pull with
# HTTP 412, "requires a newer version of Ollama"); 0.34.0 is the version the
# tier table's models were measured on (test Mac, Ollama 0.34.2).
MINIMUM_OLLAMA_VERSION_FOR_GEMMA4 = "0.34.0"


def version_at_least(version: str, minimum: str) -> bool:
    """True when *version* (e.g. ``"0.34.2-rc1"``) is >= *minimum*."""
    try:
        have = [int(p) for p in version.split("-")[0].split(".")]
        need = [int(p) for p in minimum.split(".")]
    except ValueError:
        return False
    return have >= need


def check_ollama_version(ollama_host: str = "http://localhost:11434") -> str | None:
    """Check if Ollama version supports Gemma 4.

    Returns the version string, or None if unavailable.
    """
    from agenthandover_worker.ollama import OllamaClient

    try:
        return OllamaClient(ollama_host).version(timeout=5)
    except Exception:
        return None


def ollama_supports_gemma4(ollama_host: str = "http://localhost:11434") -> bool:
    """Check if the running Ollama version supports Gemma 4 models."""
    version = check_ollama_version(ollama_host)
    return bool(version) and version_at_least(
        version, MINIMUM_OLLAMA_VERSION_FOR_GEMMA4,
    )


# ---------------------------------------------------------------------------
# RAM tiers: the one table of which models a Mac runs
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "nomic-embed-text"

# The per-frame annotator is the small resident model on every tier
# (text-first annotation, WP-C3b); from 24 GB a larger model writes Skills
# and behavioural syntheses, where quality matters and calls are rare.
# 16 GB runs qwen3.5:4b for both (WP-F6, measured head-to-head on the test
# Mac); 8 GB keeps it too, pending an 8 GB bench.
TIER_BASIS = (
    "Annotator qwen3.5:4b on every tier. Measured on a 16 GB Mac with Ollama "
    "0.34.2: gemma4:12b-it-qat takes 40+ s per annotated frame (11 s image "
    "prefill + 30 s decode) against a 3 s budget, and does not fit next to "
    "qwen3.5:4b (3.4 GB) once ordinary apps are open. Writing Skills, "
    "qwen3.5:4b ran at 29.5 tok/s against Gemma 4 12B's 12.1 tok/s and "
    "wrote a Skill in 153 s against 230 s, of comparable quality, so 16 GB "
    "Macs run qwen3.5:4b for everything. From 24 GB a Gemma 4 QAT model "
    "writes Skills and synthesis, and the worker loads one model at a time: "
    "annotation first, then Skills."
)

# Download size in GB of the models a tier installs by default (``ollama
# list`` on the test Mac; the 31B figure is the QAT quant's footprint).
MODEL_SIZES_GB: dict[str, float] = {
    "qwen3.5:4b": 3.4,
    "gemma4:12b-it-qat": 7.2,
    "gemma4:31b-it-qat": 18.0,
    EMBEDDING_MODEL: 0.27,
}


@dataclass(frozen=True)
class Tier:
    """One row of the RAM tier table."""

    id: str
    label: str
    subtitle: str
    min_ram_gb: int
    annotation_model: str
    sop_model: str
    # The A/B arm for the SOP model, used when ``[vlm] sop_ab = true``.
    ab_sop_model: str = ""

    @property
    def models(self) -> list[str]:
        """Every model this tier needs pulled, annotation model first."""
        return list(dict.fromkeys(
            [self.annotation_model, self.sop_model, EMBEDDING_MODEL],
        ))


# Ascending by ``min_ram_gb``; the first row takes any Mac.
TIERS: tuple[Tier, ...] = (
    Tier(
        id="standard",
        label="Standard",
        subtitle="Qwen 3.5 4B for everything, works on 8 GB Macs",
        min_ram_gb=0,
        annotation_model="qwen3.5:4b",
        sop_model="qwen3.5:4b",
    ),
    Tier(
        id="recommended",
        label="Recommended",
        subtitle="Qwen 3.5 4B for everything, measured best on 16 GB Macs",
        min_ram_gb=16,
        annotation_model="qwen3.5:4b",
        sop_model="qwen3.5:4b",
    ),
    Tier(
        id="performance",
        label="Performance",
        subtitle="Qwen 3.5 4B watches, Gemma 4 12B QAT writes Skills",
        min_ram_gb=24,
        annotation_model="qwen3.5:4b",
        sop_model="gemma4:12b-it-qat",
    ),
    Tier(
        id="max_quality",
        label="Max Quality",
        subtitle="Qwen 3.5 4B watches, Gemma 4 31B QAT writes Skills",
        min_ram_gb=48,
        annotation_model="qwen3.5:4b",
        sop_model="gemma4:31b-it-qat",
        ab_sop_model="gemma4:26b-a4b-it-qat",
    ),
)


# The tier every Mac can run: the library default of every model setting.
STANDARD_TIER = TIERS[0]


def machine_ram_gb() -> int:
    """Physical memory in whole GiB (what "a 16 GB Mac" reports)."""
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) // (1 << 30)
    except (ValueError, OSError, AttributeError):
        return 0


def get_tier(tier_id: str) -> Tier | None:
    """The tier named *tier_id*, or None."""
    return next((t for t in TIERS if t.id == tier_id), None)


def recommend_tier(ram_gb: int | None = None) -> Tier:
    """The largest tier whose ``min_ram_gb`` fits *ram_gb* (default: this Mac)."""
    ram = machine_ram_gb() if ram_gb is None else ram_gb
    return [t for t in TIERS if t.min_ram_gb <= ram][-1]


def tiers_document() -> dict:
    """The tier table as the JSON object ``tiers.json`` holds."""
    return {
        "schema_version": 1,
        "basis": TIER_BASIS,
        "minimum_ollama_version_for_gemma4": MINIMUM_OLLAMA_VERSION_FOR_GEMMA4,
        "embedding_model": EMBEDDING_MODEL,
        "model_sizes_gb": MODEL_SIZES_GB,
        "tiers": [asdict(t) for t in TIERS],
    }


def tiers_json() -> str:
    """``tiers.json``'s exact text (the committed and the installed copy)."""
    return json.dumps(tiers_document(), indent=2) + "\n"


def vlm_block(tier: Tier) -> str:
    """The managed ``[vlm]`` lines every writer puts in config.toml.

    The installer (through this function), the app's onboarding
    (``ModelTiers.swift``) and ``agenthandover setup``
    (``crates/common/src/tiers.rs``) write these same bytes.
    """
    return (
        "[vlm]\n"
        'mode = "local"\n'
        f'tier = "{tier.id}"\n'
        f'annotation_model = "{tier.annotation_model}"\n'
        f'sop_model = "{tier.sop_model}"\n'
    )


def _main(argv: list[str]) -> int:
    """``write-tiers PATH`` | ``vlm-block TIER_ID``."""
    if len(argv) == 2 and argv[0] == "write-tiers":
        target = Path(argv[1])
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(tiers_json(), encoding="utf-8")
        tmp.replace(target)
        return 0
    if len(argv) == 2 and argv[0] == "vlm-block":
        tier = get_tier(argv[1])
        if tier is None:
            print(f"unknown tier {argv[1]!r}", file=sys.stderr)
            return 2
        sys.stdout.write(vlm_block(tier))
        return 0
    print(
        "usage: python -m agenthandover_worker.model_profiles "
        "write-tiers PATH | vlm-block TIER_ID",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))

