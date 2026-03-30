"""Style and voice analyzer for user-produced text.

Uses the local LLM (Qwen) to analyze writing patterns, tone, and
personality from user-produced text samples.  Populates the procedure's
``voice_profile`` and ``content_samples`` fields.

The LLM sees the actual text and understands nuance — sarcasm,
enthusiasm, domain vocabulary, cultural tone — that regex heuristics
miss entirely.  Fires when enough samples accumulate (3+).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agenthandover_worker.llm_reasoning import LLMReasoner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM-driven style analysis
# ---------------------------------------------------------------------------

_STYLE_PROMPT = """\
Analyze the writing style of the following text samples from the same user.
These are real examples of text they produced while working.

TEXT SAMPLES:
{samples}

Respond with a JSON object describing their writing voice:

{{
  "formality": "<formal | neutral | casual>",
  "tone": "<e.g. friendly, matter-of-fact, enthusiastic, dry, empathetic, professional>",
  "sentence_style": "<e.g. short and punchy, long and detailed, mixed, fragmented>",
  "vocabulary": "<e.g. simple and direct, technical, varied and rich, colloquial>",
  "personality_markers": ["<list 2-4 distinctive traits, e.g. uses humor, asks rhetorical questions, hedges with maybe/perhaps, exclamation marks, emoji>"],
  "sample_phrases": ["<2-3 short phrases that are most characteristic of their style>"],
  "would_say": "<a single example sentence this person would naturally write>",
  "would_never_say": "<a single example sentence that would feel wrong coming from this person>"
}}

Rules:
- Base everything on the actual text samples, not assumptions
- If the samples are too short or generic to characterize, set tone to "insufficient_data"
- Be specific — "casual" is not enough, say "casual with dry humor and emoji"
- personality_markers should be things an agent could actually replicate

Respond with ONLY the JSON object."""


def analyze_style(
    texts: list[str],
    llm_reasoner: "LLMReasoner | None" = None,
) -> dict:
    """Analyze a collection of user-produced texts.

    Uses LLM when available, returns empty dict if no reasoner
    or insufficient text.
    """
    if not texts:
        return {}

    combined = " ".join(texts)
    if len(combined) < 50:
        return {}

    # Need the LLM for real analysis
    if llm_reasoner is None:
        logger.debug("No LLM reasoner — style analysis skipped")
        return {}

    # Format samples for the prompt
    sample_text = ""
    for i, t in enumerate(texts[:10], 1):
        sample_text += f"\n[{i}] {t[:500]}"

    prompt = _STYLE_PROMPT.format(samples=sample_text)

    try:
        result = llm_reasoner.reason_json(
            prompt,
            caller="style_analyzer",
            think=False,
        )
        if result.success and result.value and isinstance(result.value, dict):
            profile = result.value
            # Add metadata
            profile["word_count_analyzed"] = len(combined.split())
            profile["sample_count"] = len(texts)
            # Derive style_confidence from sample count
            if len(texts) >= 10:
                profile["style_confidence"] = "high"
            elif len(texts) >= 3:
                profile["style_confidence"] = "moderate"
            else:
                profile["style_confidence"] = "low"
            return profile
    except Exception:
        logger.debug("LLM style analysis failed", exc_info=True)

    return {}


def extract_content_samples(
    texts: list[str],
    max_samples: int = 5,
    min_length: int = 20,
    max_length: int = 500,
) -> list[dict]:
    """Select representative text samples from user-produced content.

    Picks diverse samples that best represent the user's writing style.
    """
    if not texts:
        return []

    candidates = [t for t in texts if len(t) >= min_length]
    if not candidates:
        return []

    # Sort by length (prefer medium-length — most representative)
    candidates.sort(key=lambda t: abs(len(t) - 150))

    samples = []
    seen_prefixes: set[str] = set()

    for text in candidates:
        if len(samples) >= max_samples:
            break
        prefix = text[:30].lower()
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        samples.append({
            "text": text[:max_length],
            "length": len(text),
        })

    return samples


def analyze_procedure_style(
    procedure: dict,
    llm_reasoner: "LLMReasoner | None" = None,
) -> tuple[dict, list[dict]]:
    """Extract style profile from a procedure's evidence.

    Reads content_produced and analyzes writing patterns via LLM.
    """
    evidence = procedure.get("evidence", {})
    extracted = evidence.get("extracted_evidence", {})
    content_items = extracted.get("content_produced", [])

    texts = []
    for item in content_items:
        full = item.get("full_value", "")
        if full and len(full) > 10:
            texts.append(full)
        else:
            preview = item.get("value_preview", "")
            if preview and len(preview) > 10:
                texts.append(preview)

    voice_profile = analyze_style(texts, llm_reasoner=llm_reasoner)
    content_samples = extract_content_samples(texts)

    # Merge with existing voice_profile if present
    existing_vp = procedure.get("voice_profile", {})
    if existing_vp and voice_profile:
        voice_profile = merge_voice_profiles(existing_vp, voice_profile)

    return voice_profile, content_samples


def merge_voice_profiles(existing: dict, new: dict) -> dict:
    """Merge two voice profiles, strengthening confidence over sessions.

    The LLM profile from the latest analysis takes precedence for
    qualitative fields (tone, formality, personality_markers).
    Cumulative fields (sample_count, word_count) are summed.
    """
    if not existing:
        return new
    if not new:
        return existing

    merged = dict(new)  # Start with latest LLM analysis

    # Cumulative counts
    merged["word_count_analyzed"] = (
        existing.get("word_count_analyzed", 0) +
        new.get("word_count_analyzed", 0)
    )
    total_samples = (
        existing.get("sample_count", 0) +
        new.get("sample_count", 0)
    )
    merged["sample_count"] = total_samples

    # Merge personality markers (union, deduplicated)
    old_markers = set(existing.get("personality_markers", []))
    new_markers = set(new.get("personality_markers", []))
    merged["personality_markers"] = list(old_markers | new_markers)[:6]

    # Merge sample phrases (keep latest + best from old)
    old_phrases = existing.get("sample_phrases", [])
    new_phrases = new.get("sample_phrases", [])
    merged["sample_phrases"] = (new_phrases + old_phrases)[:4]

    # Style confidence from cumulative sample count
    if total_samples >= 10:
        merged["style_confidence"] = "high"
    elif total_samples >= 3:
        merged["style_confidence"] = "moderate"
    else:
        merged["style_confidence"] = "low"

    return merged


def aggregate_user_style(
    procedures: list[dict],
    llm_reasoner: "LLMReasoner | None" = None,
) -> dict:
    """Build a user-level style profile from all procedures.

    Aggregates voice_profiles across all procedures.
    If an LLM is available and there are enough profiles, asks
    it to synthesize a holistic user voice description.
    """
    all_profiles = []
    for proc in procedures:
        vp = proc.get("voice_profile", {})
        if vp and vp.get("sample_count", 0) > 0:
            all_profiles.append(vp)

    if not all_profiles:
        return {}

    # Merge all profiles
    result = all_profiles[0]
    for vp in all_profiles[1:]:
        result = merge_voice_profiles(result, vp)

    # Per-context breakdown
    contexts = []
    for proc in procedures:
        vp = proc.get("voice_profile", {})
        if vp and vp.get("formality"):
            contexts.append({
                "procedure": proc.get("id", ""),
                "formality": vp.get("formality", "neutral"),
                "tone": vp.get("tone", ""),
                "sample_count": vp.get("sample_count", 0),
            })

    if contexts:
        result["per_workflow"] = contexts

    return result
