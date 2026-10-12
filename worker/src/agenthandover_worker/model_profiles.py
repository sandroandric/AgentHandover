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
import re
import shutil
import sys
import tempfile
import tomllib
from collections.abc import Collection, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
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
    # Qwen 3.5 4B is the SOP model on the Standard (8 GB) and Recommended
    # (16 GB) tiers, so those tiers' generation phase runs at 24,576; Gemma 4 12B
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
    # --- Qwen 3.5 family ---
    # The Standard (8 GB) tier's annotator (WP-F8).  The Qwen card warns
    # that the 2B is prone to thinking loops, so ``think`` is off on every
    # call type and the annotation output stays capped at 300 tokens (the
    # compact schema's answers are ~120).  These are the settings the 8 GB
    # measurement ran with (the focus_dailynews annotation prompts replayed
    # on the test Mac: 10/10 valid annotations).  The SOP settings apply
    # only when the scheduler falls back to the resident annotator.
    "qwen3.5:2b-q4_K_M": ModelProfile(
        model="qwen3.5:2b-q4_K_M",
        family="qwen3.5",
        ann_think=False,
        ann_num_predict=300,
        ann_num_ctx=4096,
        sop_num_predict=12000,
        sop_num_ctx=24576,
        sop_temperature=0.3,
        sop_think=False,
        diff_num_predict=500,
        diff_think=False,
    ),
    # The generic profile of any other Qwen tag (``get_profile``'s fallback).
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

# EmbeddingGemma 2 (:data:`MEMORY_MODEL`) runs from Ollama 0.40.0, which
# added multimodal ``/api/embed`` inputs (verified on the test Mac with
# Ollama 0.40.3).  Older Ollama: Memory is unavailable; the text paths stay
# on nomic-embed-text.
MINIMUM_OLLAMA_VERSION_FOR_MEMORY = "0.40.0"


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

# Memory embeds every kept frame (screenshot + "app — title") with Google's
# EmbeddingGemma 2 (text and image inputs; Ollama >=
# :data:`MINIMUM_OLLAMA_VERSION_FOR_MEMORY`).
MEMORY_MODEL = "embeddinggemma-2:440m"


@dataclass(frozen=True)
class EmbeddingThresholds:
    """The cosine thresholds the worker applies to one embedding model's
    vectors (each is meaningful only for the model it was measured on)."""

    # Task segmenter: similarity to a cluster centroid for "same task".
    segmenter: float
    # Session linker: cosine at or above which two intents link, and the
    # band just below it where an LLM verdict decides.
    linker_cos: float
    linker_band: float
    # SOP dedup: a semantic match, and the lower edge of the family range
    # ``[dedup_family, dedup)``.
    dedup: float
    dedup_family: float
    # Procedure matcher: a match counts as supporting evidence.
    matcher: float
    # Skill search (MCP ``search_procedures`` and the query API's
    # ``/search``): the default lowest score returned.
    min_score: float


@dataclass(frozen=True)
class EmbeddingProfile:
    """How the worker embeds with one model: the vector size it stores, the
    task prefix for each kind of text, and the thresholds measured on it
    (None until calibrated)."""

    model: str
    dim: int
    doc_prefix: str
    query_prefix: str
    cluster_prefix: str
    thresholds: EmbeddingThresholds | None


# One row per embedding model the worker can use.  nomic-embed-text keeps
# the numbers the readers hold today (task_segmenter, session_linker,
# sop_dedup, procedure_matcher, mcp_server, query_api, and vector_kb's
# prefixes); WP-E2 points the readers at this table.  EmbeddingGemma 2's
# prefixes are its model card's; it is stored truncated to 256 dimensions
# (Matryoshka; measured near-lossless against 768), and its thresholds come
# from the calibration recordings (WP-E3), never guessed.
EMBEDDING_PROFILES: dict[str, EmbeddingProfile] = {
    EMBEDDING_MODEL: EmbeddingProfile(
        model=EMBEDDING_MODEL,
        dim=768,
        doc_prefix="search_document: ",
        query_prefix="search_query: ",
        cluster_prefix="clustering: ",
        thresholds=EmbeddingThresholds(
            segmenter=0.83,
            linker_cos=0.89,
            linker_band=0.08,
            dedup=0.70,
            dedup_family=0.60,
            matcher=0.50,
            min_score=0.3,
        ),
    ),
    MEMORY_MODEL: EmbeddingProfile(
        model=MEMORY_MODEL,
        dim=256,
        doc_prefix="title: none | text: ",
        query_prefix="task: search result | query: ",
        cluster_prefix="task: clustering | query: ",
        thresholds=None,
    ),
}

# The per-frame annotator is the small resident model on every tier
# (text-first annotation, WP-C3b); a larger model writes Skills and
# behavioural syntheses, where quality matters and calls are rare.  16 GB
# runs qwen3.5:4b for both (WP-F6, measured head-to-head on the test Mac).
# 8 GB annotates with qwen3.5:2b-q4_K_M and writes Skills with qwen3.5:4b
# (WP-F8, measured 2026-09-23/24 on the 16 GB test Mac replaying the
# focus_dailynews prompts): the 2B's annotations were valid 10/10 with the
# right app 10/10 and location 6/10 against qwen3.5:4b, at ~55 tok/s;
# writing the Skill, qwen3.5:4b got 18 steps right against 10 for
# qwen3.5:2b and 13 for granite4.2:3b (rejected as annotator: no vision).
TIER_BASIS = (
    "Measured on a 16 GB Mac with Ollama "
    "0.34.2: gemma4:12b-it-qat takes 40+ s per annotated frame (11 s image "
    "prefill + 30 s decode) against a 3 s budget, and does not fit next to "
    "qwen3.5:4b (3.4 GB) once ordinary apps are open. Writing Skills, "
    "qwen3.5:4b ran at 29.5 tok/s against Gemma 4 12B's 12.1 tok/s and "
    "wrote a Skill in 153 s against 230 s, of comparable quality, so 16 GB "
    "Macs run qwen3.5:4b for everything. 8 GB Macs annotate with "
    "qwen3.5:2b-q4_K_M (1.9 GB, ~55 tok/s, valid annotations 10/10 on the "
    "focus_dailynews prompts), resident all day in ~2.3 GB instead of "
    "~3.8 GB, and write Skills with qwen3.5:4b (18 correct steps against 10 "
    "for qwen3.5:2b). From 24 GB a Gemma 4 QAT model writes Skills and "
    "synthesis. Where the two models differ the worker loads one at a time: "
    "annotation first, then Skills."
)

# Download size in GB of the models a tier installs by default (``ollama
# list`` on the test Mac; the 31B figure is the QAT quant's footprint).
MODEL_SIZES_GB: dict[str, float] = {
    "qwen3.5:2b-q4_K_M": 1.9,
    "qwen3.5:4b": 3.4,
    "gemma4:12b-it-qat": 7.2,
    "gemma4:31b-it-qat": 18.0,
    EMBEDDING_MODEL: 0.27,
    MEMORY_MODEL: 0.71,
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
        subtitle="Qwen 3.5 2B watches, Qwen 3.5 4B writes Skills, works on 8 GB Macs",
        min_ram_gb=0,
        annotation_model="qwen3.5:2b-q4_K_M",
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

# Each tier's earlier default ``(annotation_model, sop_model)`` pairs: the
# models a shipped writer (the installer's ``vlm-block``, onboarding's
# ``ModelTiers.vlmBlock``, ``agenthandover setup --tier``) could have put
# in a config's managed ``[vlm]`` block under that ``tier`` id, and that
# the tier no longer runs.  Explicit models win over the tier, so such a
# config keeps them after an upgrade until :func:`migrate_vlm_block`
# moves it to the tier's current pair.  Add a pair here whenever a tier's
# default changes; never remove one.
#
# Sources (``git log -p -- worker/src/agenthandover_worker/model_profiles.py
# resources/tiers.json`` from WP-C6, which introduced ``TIERS`` and the
# ``tier`` key; no earlier release wrote a ``tier`` key):
#   recommended  qwen3.5:4b + gemma4:12b-it-qat   the default from d770865
#                (WP-C6; written by the installer 2ce4c15, setup 85e9cbb and
#                onboarding 012b97f, merged in 2461285) until dfaa35d (WP-F6)
#                moved the 16 GB tier to qwen3.5:4b for both.
#   standard     qwen3.5:4b + qwen3.5:4b          the default from d770865
#                (WP-C6; written by the same installer, setup and
#                onboarding commits, and the installer's fresh-config block
#                since 2ce4c15) until WP-F8 moved the 8 GB tier's annotator
#                to qwen3.5:2b-q4_K_M.
# performance and max_quality have kept their WP-C6 pair.
PREVIOUS_TIER_DEFAULTS: dict[str, tuple[tuple[str, str], ...]] = {
    "standard": (("qwen3.5:4b", "qwen3.5:4b"),),
    "recommended": (("qwen3.5:4b", "gemma4:12b-it-qat"),),
}

# The ``[vlm]`` models v0.4.0 wrote, which name no tier: its onboarding
# (``git show v0.4.0:app/AgentHandoverApp/Sources/AgentHandoverApp/Views/
# OnboardingView.swift``, ``ModelTier`` and ``saveCustomModelConfig``)
# appended ``annotation_model`` and ``sop_model`` only, with no ``tier``
# or ``mode`` key; its installer wrote no ``[vlm]`` block.  Each pair
# lists the tiers it stood for; :func:`untiered_default_tier` picks one
# by RAM.  v0.4.0's Recommended (16 GB) and Performance (24 GB) tiers
# wrote the same Gemma 4 12B pair.
UNTIERED_DEFAULTS: tuple[tuple[tuple[str, str], tuple[str, ...]], ...] = (
    (("qwen3.5:2b", "qwen3.5:4b"), ("standard",)),
    (("gemma4:12b-it-qat", "gemma4:12b-it-qat"), ("recommended", "performance")),
    (("gemma4:31b-it-qat", "gemma4:31b-it-qat"), ("max_quality",)),
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


def untiered_default_tier(pair: tuple[str, str], ram_gb: int | None = None) -> Tier | None:
    """The tier a tier-less ``[vlm]`` pair v0.4.0 wrote now means
    (:data:`UNTIERED_DEFAULTS`), or None for any other pair.  Of the pair's
    tiers, the largest whose ``min_ram_gb`` fits *ram_gb* (default: this
    Mac), as :func:`recommend_tier` picks; the smallest when none fits.
    The CLI's doctor makes the same pick from ``tiers.json``."""
    ids = next((ids for p, ids in UNTIERED_DEFAULTS if p == pair), None)
    if ids is None:
        return None
    candidates = [t for t in TIERS if t.id in ids]
    ram = machine_ram_gb() if ram_gb is None else ram_gb
    fitting = [t for t in candidates if t.min_ram_gb <= ram]
    return fitting[-1] if fitting else candidates[0]


def tiers_document() -> dict:
    """The tier table as the JSON object ``tiers.json`` holds."""
    return {
        "schema_version": 1,
        "basis": TIER_BASIS,
        "minimum_ollama_version_for_gemma4": MINIMUM_OLLAMA_VERSION_FOR_GEMMA4,
        "minimum_ollama_version_for_memory": MINIMUM_OLLAMA_VERSION_FOR_MEMORY,
        "embedding_model": EMBEDDING_MODEL,
        "memory_model": MEMORY_MODEL,
        "model_sizes_gb": MODEL_SIZES_GB,
        "tiers": [
            {
                **asdict(t),
                "previous_defaults": [
                    {"annotation_model": ann, "sop_model": sop}
                    for ann, sop in PREVIOUS_TIER_DEFAULTS.get(t.id, ())
                ],
            }
            for t in TIERS
        ],
        "untiered_defaults": [
            {"annotation_model": ann, "sop_model": sop, "tiers": list(ids)}
            for (ann, sop), ids in UNTIERED_DEFAULTS
        ],
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


# The keys of ``[vlm]`` a tier choice writes, in order (:func:`vlm_block`).
MANAGED_VLM_KEYS = ("mode", "tier", "annotation_model", "sop_model")


def _config_lines(config: str) -> list[str]:
    """*config* split after every ``\\n``, as Rust's ``split_inclusive``:
    the last piece has no terminator and is dropped when empty."""
    parts = config.split("\n")
    lines = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def _managed_key(stripped_line: str) -> str | None:
    """The managed key a stripped ``[vlm]`` line sets, or None."""
    key, sep, _ = stripped_line.partition("=")
    key = key.strip()
    return key if sep and key in MANAGED_VLM_KEYS else None


def apply_vlm_block(config: str, tier: Tier) -> str:
    """*config* with *tier* chosen: the managed keys of ``[vlm]`` replaced
    by :func:`vlm_block` (right after the header, other ``[vlm]`` keys and
    every other section kept byte for byte), or the block appended when
    there is no ``[vlm]`` section.

    The same edit as ``agenthandover setup`` (``tiers::apply_vlm_block``)
    and onboarding (``ModelTiers.applyVLMBlock``); the shared fixture
    ``tests/fixtures/vlm_block`` pins the three byte for byte.
    """
    block = vlm_block(tier)
    lines = _config_lines(config)
    header = next((i for i, line in enumerate(lines) if line.strip() == "[vlm]"), None)
    if header is None:
        out = config
        if out and not out.endswith("\n"):
            out += "\n"
        return out + "\n" + block
    out = "".join(lines[:header]) + block
    in_section = True
    for line in lines[header + 1:]:
        stripped = line.strip()
        if stripped.startswith("["):
            in_section = False
        if not (in_section and _managed_key(stripped)):
            out += line
    return out


# The ``[vlm]`` keys of cloud mode, removed in 0.5.0 (``mode`` stays: the
# tier writers put ``mode = "local"`` there).
REMOTE_VLM_KEYS = ("mode", "provider", "model", "api_key_env")


def remote_to_local(config_text: str) -> tuple[str, str | None]:
    """*config_text* with a cloud-mode ``[vlm]`` block made local, and the
    reason; ``(config_text, None)`` when there is nothing to change.

    Cloud mode (``mode = "remote"`` with ``provider``, ``model`` and
    ``api_key_env``) was removed in 0.5.0.  It never changed which local
    models annotate and write Skills, so the block keeps its ``tier`` and
    model keys: the lines of the four cloud keys go, ``mode = "local"``
    goes right after the ``[vlm]`` header, and every other line and
    section is kept byte for byte.  The result must read, as TOML, as the
    old ``[vlm]`` minus the cloud keys plus ``mode = "local"`` with every
    other section the same; otherwise nothing changes (doctor then says
    the config is still in cloud mode).
    """
    lines = _config_lines(config_text)
    header = next((i for i, line in enumerate(lines) if line.strip() == "[vlm]"), None)
    if header is None:
        return config_text, None
    try:
        before = tomllib.loads(config_text)
    except tomllib.TOMLDecodeError:
        return config_text, None
    vlm = before.get("vlm")
    if not isinstance(vlm, dict) or vlm.get("mode", "local") == "local":
        return config_text, None

    end = next(
        (i for i in range(header + 1, len(lines)) if lines[i].strip().startswith("[")),
        len(lines),
    )

    def cloud_key(line: str) -> bool:
        key, sep, _ = line.strip().partition("=")
        return bool(sep) and key.strip() in REMOTE_VLM_KEYS

    newline = "\r\n" if lines[header].endswith("\r\n") else "\n"
    header_line = lines[header] if lines[header].endswith("\n") else lines[header] + newline
    new_text = "".join(
        lines[:header]
        + [header_line, f'mode = "local"{newline}']
        + [line for line in lines[header + 1:end] if not cloud_key(line)]
        + lines[end:]
    )
    try:
        after = tomllib.loads(new_text)
    except tomllib.TOMLDecodeError:
        return config_text, None
    expected = {
        **before,
        "vlm": {
            "mode": "local",
            **{k: v for k, v in vlm.items() if k not in REMOTE_VLM_KEYS},
        },
    }
    if after != expected:
        return config_text, None
    cloud = ", ".join(
        f"{key} {vlm[key]}" for key in ("provider", "model") if isinstance(vlm.get(key), str)
    )
    return new_text, (
        f'cloud mode was removed in 0.5.0: mode "{vlm.get("mode")}"'
        + (f" ({cloud})" if cloud else "")
        + ' -> "local"; the local models keep running as before'
    )


def migrate_vlm_block(
    config_text: str,
    served: Collection[str] | None = None,
    ram_gb: int | None = None,
) -> tuple[str, bool, str]:
    """Upgrade config.toml's ``[vlm]`` block: a cloud-mode block becomes
    local (:func:`remote_to_local`), then a block still on earlier default
    models moves to its tier's current ones (:func:`_tier_target`), but
    only when Ollama serves every model of the new pair: *served* is the
    names ``/api/tags`` lists, or None when Ollama was not asked (the
    installer), which keeps the models.  Switching to a model that is not
    pulled would stop processing.  Returns ``(new_text, changed, reason)``.
    """
    text, remote_reason = remote_to_local(config_text)
    tier, _, reason = _tier_target(text, ram_gb)
    new_text, changed = text, False
    if tier is not None:
        models = target_models(tier)
        if served is None:
            reason += (
                f"; the switch waits until Ollama serves {', '.join(models)} "
                "(the app offers the download; or: agenthandover setup --tier "
                f"{tier.id})"
            )
        elif missing := missing_models(models, served):
            reason += (
                f"; the switch waits until {', '.join(missing)} "
                f"({download_label(missing)}) {'is' if len(missing) == 1 else 'are'} "
                "downloaded (the app offers it; "
                f"or: agenthandover setup --tier {tier.id})"
            )
        else:
            new_text, changed = apply_vlm_block(text, tier), True
    if remote_reason is None:
        return new_text, changed, reason
    return new_text, True, remote_reason + (f"; {reason}" if tier is not None else "")


def target_models(tier: Tier) -> list[str]:
    """The models *tier*'s pair runs, annotation model first, no repeats."""
    return list(dict.fromkeys([tier.annotation_model, tier.sop_model]))


def missing_models(models: Iterable[str], served: Collection[str]) -> list[str]:
    """The *models* not in *served* (``/api/tags`` names; a bare model
    name means its ``:latest`` tag, as ``OllamaClient.has_model``)."""
    return [
        m for m in models
        if m not in served and (":" in m or f"{m}:latest" not in served)
    ]


def download_label(models: Iterable[str]) -> str:
    """"~3.4 GB": the download of *models* by :data:`MODEL_SIZES_GB`."""
    return f"~{sum(MODEL_SIZES_GB.get(m, 0.0) for m in models):.1f} GB"


def _ollama_host(config_text: str) -> str:
    """``[vlm] ollama_host`` of *config_text*, else Ollama's default."""
    from agenthandover_worker.ollama import DEFAULT_HOST

    try:
        vlm = tomllib.loads(config_text).get("vlm")
    except tomllib.TOMLDecodeError:
        return DEFAULT_HOST
    host = vlm.get("ollama_host") if isinstance(vlm, dict) else None
    return host if isinstance(host, str) and host else DEFAULT_HOST


def served_models(config_text: str) -> list[str] | None:
    """The models Ollama serves at *config_text*'s ``[vlm] ollama_host``
    (``/api/tags``), or None when it does not answer."""
    from agenthandover_worker.ollama import OllamaClient

    try:
        return OllamaClient(_ollama_host(config_text)).tags(timeout=3)
    except Exception:
        return None


def upgrade_plan(config_text: str, ram_gb: int | None = None) -> tuple[dict | None, str]:
    """The model switch :func:`migrate_vlm_block` makes once the models
    are pulled, as ``(plan, reason)``; ``(None, reason)`` when there is
    none.  Read only.  The plan: ``tier`` and ``label`` (the tier it moves
    to), ``current`` and ``target`` (``annotation_model`` and
    ``sop_model``), ``models`` (the target pair's models) and ``sizes_gb``
    (their download sizes from :data:`MODEL_SIZES_GB`).
    """
    text, _ = remote_to_local(config_text)
    tier, pair, reason = _tier_target(text, ram_gb)
    if tier is None or pair is None:
        return None, reason
    models = target_models(tier)
    return {
        "tier": tier.id,
        "label": tier.label,
        "current": {"annotation_model": pair[0], "sop_model": pair[1]},
        "target": {"annotation_model": tier.annotation_model, "sop_model": tier.sop_model},
        "models": models,
        "sizes_gb": {m: MODEL_SIZES_GB[m] for m in models if m in MODEL_SIZES_GB},
    }, reason


def _tier_target(
    config_text: str, ram_gb: int | None = None,
) -> tuple[Tier | None, tuple[str, str] | None, str]:
    """The tier a ``[vlm]`` block on earlier default models moves to, the
    block's ``(annotation_model, sop_model)`` and why; ``(None, None,
    reason)`` when it stays.

    A block moves only when a writer left it and the user has not
    customised it:

    * ``mode`` is ``"local"`` or absent;
    * with a ``tier`` key: a known tier, and the two models are one of
      its :data:`PREVIOUS_TIER_DEFAULTS` pairs (not its current pair);
    * without one: the two models are a pair v0.4.0 wrote
      (:data:`UNTIERED_DEFAULTS`), and the tier comes from RAM
      (:func:`untiered_default_tier`);
    * each managed key appears at most once, exactly as the writers put it
      (``key = "value"``, no comment or other quoting), and ``tomllib``
      reads the same values.

    The move is :func:`apply_vlm_block` with the tier (what ``agenthandover
    setup --tier`` writes): other ``[vlm]`` keys and every other section
    are kept byte for byte.
    """
    def keep(reason: str) -> tuple[None, None, str]:
        return None, None, reason

    lines = _config_lines(config_text)
    header = next((i for i, line in enumerate(lines) if line.strip() == "[vlm]"), None)
    if header is None:
        return keep("no [vlm] section")

    values: dict[str, str] = {}
    for line in lines[header + 1:]:
        stripped = line.strip()
        if stripped.startswith("["):
            break
        key = _managed_key(stripped)
        if key is None:
            continue
        if key in values:
            return keep(f"[vlm] sets {key} more than once")
        match = re.fullmatch(rf'{key} = "([^"\\]*)"', stripped)
        if match is None:
            return keep(f"[vlm] {key} was edited by hand ({stripped})")
        values[key] = match.group(1)

    try:
        parsed = tomllib.loads(config_text)
    except tomllib.TOMLDecodeError as exc:
        return keep(f"config.toml is not valid TOML ({exc})")
    vlm = parsed.get("vlm")
    if not isinstance(vlm, dict) or any(
        vlm.get(key) != values.get(key) for key in MANAGED_VLM_KEYS
    ):
        return keep("[vlm] reads differently as TOML than line by line")

    mode = values.get("mode", "local")
    if mode != "local":
        return keep(f'[vlm] mode is "{mode}"')
    ann, sop = values.get("annotation_model"), values.get("sop_model")
    tier_id = values.get("tier")
    if tier_id is None:
        if ann is None or sop is None:
            return keep("[vlm] names no tier")
        tier = untiered_default_tier((ann, sop), ram_gb)
        if tier is None:
            return keep(
                f"[vlm] names no tier and {ann} + {sop} is not a default of "
                "0.4.0 (set by hand)"
            )
        why = f"{ann} + {sop} was 0.4.0's default"
    else:
        tier = get_tier(tier_id)
        if tier is None:
            return keep(f'unknown tier "{tier_id}"')
        if ann is None or sop is None:
            return keep("[vlm] does not pin both models (the tier supplies the rest)")
        if (ann, sop) == (tier.annotation_model, tier.sop_model):
            return keep(f'already the current models of tier "{tier.id}"')
        if (ann, sop) not in PREVIOUS_TIER_DEFAULTS.get(tier.id, ()):
            return keep(
                f'{ann} + {sop} is not an earlier default of tier "{tier.id}" '
                "(set by hand)"
            )
        why = f"{ann} + {sop} was the tier's earlier default"

    changes = ", ".join(
        f"{key} {old} -> {new}"
        for key, old, new in zip(
            ("annotation_model", "sop_model"), (ann, sop), (tier.annotation_model, tier.sop_model),
        )
        if old != new
    ) or "the tier key added"
    return tier, (ann, sop), (
        f'tier "{tier.id}": {changes} ({why}; it now runs '
        f"{tier.annotation_model} + {tier.sop_model})"
    )


def migrate_vlm_file(
    path: Path,
    now: datetime | None = None,
    served: Collection[str] | None = None,
    ram_gb: int | None = None,
) -> tuple[bool, str, Path | None]:
    """Apply :func:`migrate_vlm_block` (with *served* and *ram_gb*) to the
    config file at *path*.

    Only when it changes something: first a timestamped copy of the old
    file (``config.toml.<YYYYmmdd-HHMMSS>.bak`` beside it), then the new
    text written atomically (temporary file in the same directory, then
    ``os.replace``) with the old file's permissions.  A symlinked config
    is written through to its target.  Returns ``(changed, reason,
    backup_path)``.
    """
    target = Path(path).resolve()
    original = target.read_bytes()
    new_text, changed, reason = migrate_vlm_block(original.decode("utf-8"), served, ram_gb)
    if not changed:
        return False, reason, None
    return True, reason, _replace_with_backup(target, original, new_text, now)


def _replace_with_backup(
    target: Path, original: bytes, new_text: str, now: datetime | None,
) -> Path:
    """Write *new_text* over the config file *target* (resolved) whose
    bytes were *original*: first a timestamped copy of them
    (``config.toml.<YYYYmmdd-HHMMSS>.bak`` beside it), then the new text
    atomically (temporary file in the same directory, then
    ``os.replace``) with the old file's permissions.  Returns the backup's
    path."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    backup = target.with_name(f"{target.name}.{stamp}.bak")
    n = 1
    while True:
        try:
            with open(backup, "xb") as fh:
                fh.write(original)
            break
        except FileExistsError:
            n += 1
            backup = target.with_name(f"{target.name}.{stamp}-{n}.bak")
    shutil.copymode(target, backup)

    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(new_text.encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
        shutil.copymode(target, tmp_name)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return backup


# What the upgrade offer of Memory (the app, after updating) reads: the
# file beside the worker's status files, a JSON object.  The embedding
# migration sets ``wanted_visual_search`` to true when the config it
# migrated had ``[embedding] image_embeddings = true`` (the removed
# "Visual search" option); other keys are kept.
MEMORY_OFFER_FILE = "memory-offer.json"


def drop_image_embeddings(config_text: str) -> tuple[str, bool | None]:
    """*config_text* without ``[embedding] image_embeddings`` (the SigLIP
    "Visual search" option, removed in 0.6.0), and whether it was on;
    ``(config_text, None)`` when there is nothing to remove or the edit
    cannot be made safely.

    The key's line goes, with the comment lines right above it when they
    begin with "# Image embeddings" (the installer's note on it); every
    other line and section is kept byte for byte.  The result must read,
    as TOML, as the old file minus that one key; otherwise nothing
    changes.  "On" is what the config loader read: the value as a bool.
    """
    lines = _config_lines(config_text)
    header = next(
        (i for i, line in enumerate(lines) if line.strip() == "[embedding]"), None,
    )
    if header is None:
        return config_text, None
    try:
        before = tomllib.loads(config_text)
    except tomllib.TOMLDecodeError:
        return config_text, None
    embedding = before.get("embedding")
    if not isinstance(embedding, dict) or "image_embeddings" not in embedding:
        return config_text, None

    end = next(
        (i for i in range(header + 1, len(lines)) if lines[i].strip().startswith("[")),
        len(lines),
    )
    drop: set[int] = set()
    for i in range(header + 1, end):
        key, sep, _ = lines[i].strip().partition("=")
        if not (sep and key.strip() == "image_embeddings"):
            continue
        drop.add(i)
        first = i
        while first - 1 > header and lines[first - 1].lstrip().startswith("#"):
            first -= 1
        if lines[first].lstrip().lower().startswith("# image embeddings"):
            drop.update(range(first, i))
    new_text = "".join(line for i, line in enumerate(lines) if i not in drop)
    try:
        after = tomllib.loads(new_text)
    except tomllib.TOMLDecodeError:
        return config_text, None
    expected = {
        **before,
        "embedding": {k: v for k, v in embedding.items() if k != "image_embeddings"},
    }
    if after != expected:
        return config_text, None
    from agenthandover_worker.config import _coerce

    return new_text, _coerce(embedding["image_embeddings"], False) is True


def _record_wanted_visual_search(state_dir: Path) -> None:
    """Set ``wanted_visual_search`` in :data:`MEMORY_OFFER_FILE` under
    *state_dir* (atomic write; other keys of an existing object kept)."""
    path = Path(state_dir) / MEMORY_OFFER_FILE
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    if not isinstance(state, dict):
        state = {}
    state["wanted_visual_search"] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def migrate_embedding_file(
    path: Path, state_dir: Path, now: datetime | None = None,
) -> tuple[bool, bool, Path | None]:
    """Remove ``[embedding] image_embeddings`` from the config file at
    *path* (:func:`drop_image_embeddings`).  When it was on, first record
    that in :data:`MEMORY_OFFER_FILE` under *state_dir*, so the Memory
    offer can say the user wanted visual search; then the config is
    rewritten after a timestamped backup, as :func:`migrate_vlm_file`
    does.  Nothing happens when the key is absent (so a second run changes
    nothing).  Returns ``(changed, wanted_visual_search, backup_path)``.
    """
    target = Path(path).resolve()
    original = target.read_bytes()
    new_text, wanted = drop_image_embeddings(original.decode("utf-8"))
    if wanted is None:
        return False, False, None
    if wanted:
        _record_wanted_visual_search(state_dir)
    return True, wanted, _replace_with_backup(target, original, new_text, now)


# ``upgrade-apply``'s exit status when it refuses: Ollama does not answer
# or does not serve every model of the new pair.
EXIT_MODELS_MISSING = 3


def _upgrade_apply(target: Path) -> int:
    """``upgrade-apply``: the model switch, only when Ollama serves every
    model of the new pair (asked at ``[vlm] ollama_host``)."""
    text = target.read_text(encoding="utf-8")
    plan, reason = upgrade_plan(text)
    if plan is None:
        print(f"No model switch for [vlm] in {target}: {reason}")
        return 0
    served = served_models(text)
    if served is None:
        print(
            f"Kept [vlm] in {target}: Ollama does not answer at {_ollama_host(text)}",
            file=sys.stderr,
        )
        return EXIT_MODELS_MISSING
    missing = missing_models(plan["models"], served)
    if missing:
        print(
            f"Kept [vlm] in {target}: {', '.join(missing)} not downloaded",
            file=sys.stderr,
        )
        return EXIT_MODELS_MISSING
    changed, reason, backup = migrate_vlm_file(target, served=served)
    if changed:
        print(f"Migrated [vlm] in {target}: {reason}; backup: {backup}")
    else:
        print(f"Kept [vlm] in {target} as it is: {reason}")
    return 0


def _main(argv: list[str]) -> int:
    """``write-tiers PATH`` | ``vlm-block TIER_ID`` | ``migrate-vlm PATH`` |
    ``upgrade-plan PATH`` | ``upgrade-apply PATH``.

    ``migrate-vlm`` is the installer's: it asks no Ollama, so it makes a
    cloud block local and leaves a model switch to the worker and the
    app.  ``upgrade-plan`` prints :func:`upgrade_plan` as JSON
    (``{"plan": ..., "reason": ...}``); ``upgrade-apply`` makes the switch
    when Ollama serves the models, else exits :data:`EXIT_MODELS_MISSING`.
    """
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
    if len(argv) == 2 and argv[0] in ("migrate-vlm", "upgrade-plan", "upgrade-apply"):
        command, target = argv[0], Path(argv[1])
        if not target.exists():
            if command == "upgrade-plan":
                print(json.dumps({"plan": None, "reason": f"no {target}"}))
            else:
                print(f"No {target}; no [vlm] models to migrate")
            return 0
        try:
            if command == "upgrade-plan":
                plan, reason = upgrade_plan(target.read_text(encoding="utf-8"))
                print(json.dumps({"plan": plan, "reason": reason}))
                return 0
            if command == "upgrade-apply":
                return _upgrade_apply(target)
            changed, reason, backup = migrate_vlm_file(target)
        except (OSError, UnicodeDecodeError) as exc:
            print(f"could not migrate the [vlm] models in {target}: {exc}", file=sys.stderr)
            return 1
        if changed:
            print(f"Migrated [vlm] in {target}: {reason}; backup: {backup}")
        else:
            print(f"Kept [vlm] in {target} as it is: {reason}")
        return 0
    print(
        "usage: python -m agenthandover_worker.model_profiles "
        "write-tiers PATH | vlm-block TIER_ID | migrate-vlm PATH | "
        "upgrade-plan PATH | upgrade-apply PATH",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
