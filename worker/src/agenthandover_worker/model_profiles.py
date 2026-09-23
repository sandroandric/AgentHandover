"""Model profiles — per-model optimal inference settings.

Each model family has different optimal parameters for annotation and SOP
generation. This module provides a single source of truth so the pipeline
automatically uses the right settings regardless of which model the user picks.

The RAM-based model recommendation lives in the app's onboarding tier table
(OnboardingView.swift); the worker only consumes the models it is configured with.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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

    # SOP generation settings (reasoning, structured output)
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
    # QAT 12B — the v0.4.0 default for the 16GB and 24GB tiers.  Settings
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

# Bumped to 0.30.6 in v0.4.0: the QAT Gemma 4 tags (gemma4:12b-it-qat,
# gemma4:31b-it-qat, etc.) require Ollama 0.30.6+ — older Ollama returns
# HTTP 412 ("requires a newer version of Ollama") on pull.
MINIMUM_OLLAMA_VERSION_FOR_GEMMA4 = "0.30.6"


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
    if not version:
        return False

    try:
        # Parse version like "0.20.0-rc1" → (0, 20, 0)
        clean = version.split("-")[0]  # strip -rc1 etc
        parts = [int(p) for p in clean.split(".")]
        min_parts = [int(p) for p in MINIMUM_OLLAMA_VERSION_FOR_GEMMA4.split(".")]
        return parts >= min_parts
    except (ValueError, IndexError):
        return False
