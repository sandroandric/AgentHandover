"""Model profiles — per-model optimal inference settings.

Each model family has different optimal parameters for annotation and SOP
generation. This module provides a single source of truth so the pipeline
automatically uses the right settings regardless of which model the user picks.

VRAM tier detection recommends models based on available unified memory
(Apple Silicon) or dedicated VRAM.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field

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

    # Annotation settings (vision, JSON extraction)
    ann_system: str = ""
    ann_temperature: float = 0.3
    ann_num_predict: int = 1500
    ann_num_ctx: int = 8192
    ann_top_k: int = 40
    ann_top_p: float = 0.95
    ann_presence_penalty: float = 0.0
    ann_think: bool | str = False  # False, True, "low", "medium", "high"
    ann_image_before_text: bool = False  # Gemma wants image before text

    # SOP generation settings (reasoning, structured output)
    sop_system: str = ""
    sop_temperature: float = 0.3
    sop_num_predict: int = 6000
    sop_num_ctx: int = 16384
    sop_top_k: int = 40
    sop_top_p: float = 0.95
    sop_presence_penalty: float = 0.0
    sop_think: bool | str = False

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
        ann_num_predict=1000,  # lower to avoid JSON overflow
        ann_num_ctx=8192,
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
        ann_num_predict=1000,
        ann_num_ctx=8192,
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
        ann_temperature=0.3,
        ann_num_predict=1500,
        ann_num_ctx=8192,
        ann_top_k=64,
        ann_top_p=0.95,
        ann_presence_penalty=1.0,
        ann_think=False,
        ann_image_before_text=True,
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
        ann_temperature=0.3,
        ann_num_predict=1500,
        ann_num_ctx=8192,
        ann_top_k=64,
        ann_top_p=0.95,
        ann_presence_penalty=1.0,
        ann_think=False,
        ann_image_before_text=True,
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
        ann_temperature=0.3,
        ann_num_predict=1500,
        ann_num_ctx=8192,
        ann_top_k=64,
        ann_top_p=0.95,
        ann_presence_penalty=1.0,
        ann_think=False,
        ann_image_before_text=True,
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
        ann_temperature=0.3,
        ann_num_predict=1500,
        ann_num_ctx=8192,
        ann_top_k=64,
        ann_top_p=0.95,
        ann_presence_penalty=0.0,  # E2B overflows with penalty
        ann_think=False,
        ann_image_before_text=True,
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
        ann_temperature=0.3,
        ann_num_predict=1500,
        ann_num_ctx=8192,
        ann_top_k=64,
        ann_top_p=0.95,
        ann_presence_penalty=1.0,
        ann_think=False,
        ann_image_before_text=True,
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
        ann_temperature=0.3,
        ann_num_predict=1500,
        ann_num_ctx=8192,
        ann_top_k=64,
        ann_top_p=0.95,
        ann_presence_penalty=1.0,
        ann_think=False,
        ann_image_before_text=True,
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
        ann_temperature=0.3,
        ann_num_predict=1500,
        ann_num_ctx=8192,
        ann_top_k=64,
        ann_top_p=0.95,
        ann_presence_penalty=1.0,
        ann_think=False,
        ann_image_before_text=True,
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
# VRAM tier detection and model recommendation
# ---------------------------------------------------------------------------

@dataclass
class VRAMTier:
    """A recommended model configuration for a VRAM range."""

    name: str
    min_ram_gb: int
    annotation_model: str
    sop_model: str
    total_disk_gb: float
    description: str


# Aligned to Apple Silicon SKUs
VRAM_TIERS: list[VRAMTier] = [
    VRAMTier(
        name="standard",
        min_ram_gb=0,
        annotation_model="qwen3.5:2b",
        sop_model="qwen3.5:4b",
        total_disk_gb=6.1,
        description="Qwen 3.5 (2B + 4B) — works on 8GB machines",
    ),
    VRAMTier(
        name="recommended",
        min_ram_gb=16,
        annotation_model="gemma4",
        sop_model="gemma4",
        total_disk_gb=9.6,
        description="Gemma 4 E4B — best balance of speed and quality",
    ),
    VRAMTier(
        name="performance",
        min_ram_gb=24,
        annotation_model="gemma4:e4b-it-q8_0",
        sop_model="gemma4:e4b-it-q8_0",
        total_disk_gb=12.0,
        description="Gemma 4 E4B Q8 — higher precision, better extraction",
    ),
    VRAMTier(
        name="max_quality",
        min_ram_gb=48,
        annotation_model="gemma4:31b",
        sop_model="gemma4:31b",
        total_disk_gb=20.0,
        description="Gemma 4 31B — maximum quality, frontier intelligence",
    ),
    VRAMTier(
        name="ultra",
        min_ram_gb=96,
        annotation_model="gemma4:31b-it-q8_0",
        sop_model="gemma4:31b-it-q8_0",
        total_disk_gb=34.0,
        description="Gemma 4 31B Q8 — unquantized quality for M4 Ultra/Max",
    ),
]


def detect_system_ram_gb() -> int:
    """Detect total system RAM in GB (unified memory on Apple Silicon)."""
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True, timeout=5,
        ).strip()
        return int(output) // (1024 ** 3)
    except Exception:
        logger.debug("Failed to detect system RAM", exc_info=True)
        return 0


def recommend_tier(ram_gb: int | None = None) -> VRAMTier:
    """Recommend a model tier based on available RAM.

    If ram_gb is None, auto-detects from the system.
    """
    if ram_gb is None:
        ram_gb = detect_system_ram_gb()

    # Pick the highest tier that fits
    best = VRAM_TIERS[0]
    for tier in VRAM_TIERS:
        if ram_gb >= tier.min_ram_gb:
            best = tier

    return best


def log_recommendation() -> VRAMTier:
    """Detect RAM, recommend tier, and log the recommendation."""
    ram = detect_system_ram_gb()
    tier = recommend_tier(ram)
    logger.info(
        "System RAM: %dGB — recommended tier: %s (%s)",
        ram, tier.name, tier.description,
    )
    if tier.annotation_model != tier.sop_model:
        logger.info(
            "  Annotation: %s, SOP generation: %s",
            tier.annotation_model, tier.sop_model,
        )
    else:
        logger.info("  Model: %s (single model for both)", tier.annotation_model)
    return tier


# ---------------------------------------------------------------------------
# Ollama version check
# ---------------------------------------------------------------------------

MINIMUM_OLLAMA_VERSION_FOR_GEMMA4 = "0.20.0"


def check_ollama_version(ollama_host: str = "http://localhost:11434") -> str | None:
    """Check if Ollama version supports Gemma 4.

    Returns the version string, or None if unavailable.
    """
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{ollama_host}/api/version",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            import json
            data = json.loads(resp.read())
            return data.get("version", "")
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
