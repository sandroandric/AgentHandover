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


_MIN_COMBINED_TEXT_FOR_STYLE = 30


def analyze_style(
    texts: list[str],
    llm_reasoner: "LLMReasoner | None" = None,
) -> dict:
    """Analyze a collection of user-produced texts.

    Uses LLM when available, returns empty dict if no reasoner
    or insufficient text.

    Threshold is 30 chars combined (down from 50 in earlier versions).
    Real workflows often produce terse content that's still
    voice-revealing — a casual comment or one-line message reflects
    style.  We log at INFO level when skipping due to insufficient
    content so operators can see why a Skill ended up voice-less.
    """
    if not texts:
        return {}

    combined = " ".join(texts)
    if len(combined) < _MIN_COMBINED_TEXT_FOR_STYLE:
        logger.info(
            "Style analysis skipped: only %d chars across %d samples "
            "(need at least %d)",
            len(combined), len(texts), _MIN_COMBINED_TEXT_FOR_STYLE,
        )
        return {}

    # Need the LLM for real analysis
    if llm_reasoner is None:
        logger.info(
            "Style analysis skipped for %d samples (%d chars): no LLM "
            "reasoner available",
            len(texts), len(combined),
        )
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


def _collect_user_authored_texts(procedure: dict) -> list[str]:
    """Pull user-authored text from every place it can land in a procedure.

    Voice analysis was previously gated on
    ``evidence.extracted_evidence.content_produced``, which is populated
    by the EvidenceExtractor's daily pass — that pass only runs against
    observations 12+ days old (targeting the 14-day expiry window).  So
    a freshly-generated procedure had ``content_produced = []`` until
    almost two weeks later, and ``analyze_style`` returned an empty
    voice profile every time on every Skill.  All 24 historical Skills
    show ``voice_profile: {}`` because of this delay.

    This collector reads from EVERY source that's available at procedure
    write time, not only the delayed extraction:

    - ``evidence.extracted_evidence.content_produced`` (fast path when
      the daily extractor has already run for older procedures)
    - ``evidence.clipboard_events[].text`` (the user's own copy/paste
      content captured live during the recording)
    - per-step ``parameters.input`` and top-level ``input`` fields (text
      the user typed during the recording — composes, prompts, search
      queries, comments)
    - ``content_samples`` already on the procedure from prior runs

    Only includes strings that are at least 10 characters and look like
    natural language (not single tokens, URLs, or boolean flags).
    """
    texts: list[str] = []

    def _add(text: object) -> None:
        if not isinstance(text, str):
            return
        stripped = text.strip()
        if len(stripped) < 10:
            return
        # Reject obvious non-prose values (URLs, IDs, JSON-looking blobs).
        if stripped.startswith(("http://", "https://", "{", "[")):
            return
        texts.append(stripped)

    evidence = procedure.get("evidence", {}) or {}
    extracted = evidence.get("extracted_evidence", {}) or {}

    for item in extracted.get("content_produced", []):
        if not isinstance(item, dict):
            continue
        full = item.get("full_value", "")
        if full and isinstance(full, str) and len(full) > 10:
            _add(full)
        else:
            _add(item.get("value_preview", ""))

    for event in evidence.get("clipboard_events", []):
        if isinstance(event, dict):
            _add(event.get("text"))
            _add(event.get("preview"))

    for step in procedure.get("steps", []):
        if not isinstance(step, dict):
            continue
        _add(step.get("input"))
        _add(step.get("description"))
        params = step.get("parameters")
        if isinstance(params, dict):
            _add(params.get("input"))

    for sample in procedure.get("content_samples", []) or []:
        if isinstance(sample, dict):
            _add(sample.get("text"))
        elif isinstance(sample, str):
            _add(sample)

    # De-duplicate while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def analyze_procedure_style(
    procedure: dict,
    llm_reasoner: "LLMReasoner | None" = None,
) -> tuple[dict, list[dict]]:
    """Extract style profile from a procedure's text content.

    Pulls source texts from every place they can live in a procedure
    (clipboard events, typed step inputs, content samples, and the
    delayed ``extracted_evidence.content_produced``) so style analysis
    can run at procedure write time instead of waiting 12+ days for the
    evidence extractor to populate ``content_produced``.  See
    ``_collect_user_authored_texts`` for the rationale.
    """
    texts = _collect_user_authored_texts(procedure)

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
