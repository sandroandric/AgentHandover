"""Tests for the style_analyzer module."""

from __future__ import annotations

from unittest.mock import MagicMock
from dataclasses import dataclass

import pytest

from agenthandover_worker.style_analyzer import (
    analyze_style,
    extract_content_samples,
    analyze_procedure_style,
    merge_voice_profiles,
    aggregate_user_style,
)


# ---------------------------------------------------------------------------
# Mock LLM reasoner
# ---------------------------------------------------------------------------

@dataclass
class MockResult:
    success: bool = True
    value: dict | None = None
    abstained: bool = False


def make_mock_reasoner(response: dict) -> MagicMock:
    """Create a mock LLMReasoner that returns the given dict."""
    reasoner = MagicMock()
    reasoner.reason_json.return_value = MockResult(
        success=True,
        value=response,
    )
    return reasoner


CASUAL_PROFILE = {
    "formality": "casual",
    "tone": "friendly and enthusiastic",
    "sentence_style": "short and punchy",
    "vocabulary": "simple and direct",
    "personality_markers": ["uses emoji", "exclamation marks", "informal contractions"],
    "sample_phrases": ["great point!", "def try that"],
    "would_say": "Hey love this approach!",
    "would_never_say": "Upon further consideration, the methodology appears sound.",
}

FORMAL_PROFILE = {
    "formality": "formal",
    "tone": "professional and measured",
    "sentence_style": "long and detailed",
    "vocabulary": "technical and precise",
    "personality_markers": ["hedging language", "passive voice", "no contractions"],
    "sample_phrases": ["it appears that", "upon review"],
    "would_say": "The analysis suggests a viable approach.",
    "would_never_say": "lol yeah lets do it!!",
}


# ---------------------------------------------------------------------------
# analyze_style
# ---------------------------------------------------------------------------

class TestAnalyzeStyle:

    def test_empty_input(self):
        assert analyze_style([]) == {}

    def test_too_short(self):
        assert analyze_style(["hi"]) == {}

    def test_no_reasoner_returns_empty(self):
        texts = ["This is a long enough text sample for analysis purposes."]
        result = analyze_style(texts, llm_reasoner=None)
        assert result == {}

    def test_with_reasoner_returns_profile(self):
        reasoner = make_mock_reasoner(CASUAL_PROFILE)
        texts = [
            "Hey great point! I'm totally on board with this approach lol",
            "Yeah let's def do it, we can't wait any longer!!",
        ]
        result = analyze_style(texts, llm_reasoner=reasoner)
        assert result["formality"] == "casual"
        assert result["tone"] == "friendly and enthusiastic"
        assert result["sample_count"] == 2
        assert result["word_count_analyzed"] > 0
        assert result["style_confidence"] == "low"  # only 2 samples

    def test_confidence_moderate(self):
        reasoner = make_mock_reasoner(CASUAL_PROFILE)
        texts = [f"Sample text number {i} with enough content." for i in range(5)]
        result = analyze_style(texts, llm_reasoner=reasoner)
        assert result["style_confidence"] == "moderate"

    def test_confidence_high(self):
        reasoner = make_mock_reasoner(CASUAL_PROFILE)
        texts = [f"Sample text number {i} with enough content." for i in range(12)]
        result = analyze_style(texts, llm_reasoner=reasoner)
        assert result["style_confidence"] == "high"

    def test_llm_failure_returns_empty(self):
        reasoner = MagicMock()
        reasoner.reason_json.return_value = MockResult(success=False, value=None)
        texts = ["Enough text for analysis here in this sample."]
        result = analyze_style(texts, llm_reasoner=reasoner)
        assert result == {}


# ---------------------------------------------------------------------------
# extract_content_samples
# ---------------------------------------------------------------------------

class TestExtractContentSamples:

    def test_empty(self):
        assert extract_content_samples([]) == []

    def test_too_short_filtered(self):
        assert extract_content_samples(["hi", "ok", "yes"]) == []

    def test_returns_samples(self):
        texts = [
            "This is a medium length text sample for testing purposes.",
            "Another sample text that should be included in the results.",
        ]
        samples = extract_content_samples(texts, max_samples=2)
        assert len(samples) <= 2
        assert all("text" in s and "length" in s for s in samples)

    def test_deduplicates(self):
        texts = [
            "Same prefix text with different endings one.",
            "Same prefix text with different endings two.",
        ]
        samples = extract_content_samples(texts)
        assert len(samples) == 1

    def test_respects_max_samples(self):
        texts = [f"Unique text sample number {i} for testing." for i in range(20)]
        samples = extract_content_samples(texts, max_samples=3)
        assert len(samples) <= 3


# ---------------------------------------------------------------------------
# analyze_procedure_style
# ---------------------------------------------------------------------------

class TestAnalyzeProcedureStyle:

    def test_empty_procedure(self):
        vp, cs = analyze_procedure_style({})
        assert vp == {}
        assert cs == []

    def test_with_content_and_reasoner(self):
        reasoner = make_mock_reasoner(CASUAL_PROFILE)
        proc = {
            "evidence": {
                "extracted_evidence": {
                    "content_produced": [
                        {"type": "text_input", "full_value": "Hey this is a great idea! Let's do it :)"},
                        {"type": "text_input", "full_value": "Totally agree, we should move fast on this one!"},
                        {"type": "text_input", "full_value": "Love the approach, shipping it today!"},
                    ],
                },
            },
        }
        vp, cs = analyze_procedure_style(proc, llm_reasoner=reasoner)
        assert vp["formality"] == "casual"
        assert vp["tone"] == "friendly and enthusiastic"
        assert len(cs) > 0

    def test_no_reasoner_returns_empty(self):
        proc = {
            "evidence": {
                "extracted_evidence": {
                    "content_produced": [
                        {"type": "text_input", "full_value": "Some text here for analysis."},
                    ],
                },
            },
        }
        vp, cs = analyze_procedure_style(proc, llm_reasoner=None)
        assert vp == {}

    def test_merges_with_existing(self):
        reasoner = make_mock_reasoner(CASUAL_PROFILE)
        proc = {
            "voice_profile": {
                "formality": "casual",
                "tone": "friendly",
                "sample_count": 10,
                "word_count_analyzed": 200,
                "personality_markers": ["uses humor"],
                "sample_phrases": ["old phrase"],
                "style_confidence": "moderate",
            },
            "evidence": {
                "extracted_evidence": {
                    "content_produced": [
                        {"type": "text_input", "full_value": "Another casual reply here! Love it :) Really great stuff."},
                        {"type": "text_input", "full_value": "Yeah totally agree, shipping it now!"},
                        {"type": "text_input", "full_value": "Great work on this one, keep it up!"},
                    ],
                },
            },
        }
        vp, _ = analyze_procedure_style(proc, llm_reasoner=reasoner)
        # Cumulative
        assert vp["word_count_analyzed"] > 200
        assert vp["sample_count"] > 10
        # Merged markers
        assert "uses humor" in vp["personality_markers"]


# ---------------------------------------------------------------------------
# merge_voice_profiles
# ---------------------------------------------------------------------------

class TestMergeVoiceProfiles:

    def test_merge_empty(self):
        assert merge_voice_profiles({}, CASUAL_PROFILE) == CASUAL_PROFILE
        assert merge_voice_profiles(CASUAL_PROFILE, {}) == CASUAL_PROFILE

    def test_cumulative_counts(self):
        old = {"sample_count": 10, "word_count_analyzed": 200, "personality_markers": ["humor"], "sample_phrases": ["old"]}
        new = {"sample_count": 5, "word_count_analyzed": 100, "personality_markers": ["emoji"], "sample_phrases": ["new"]}
        merged = merge_voice_profiles(old, new)
        assert merged["sample_count"] == 15
        assert merged["word_count_analyzed"] == 300
        assert "humor" in merged["personality_markers"]
        assert "emoji" in merged["personality_markers"]

    def test_confidence_levels(self):
        low = merge_voice_profiles(
            {"sample_count": 1, "word_count_analyzed": 10, "personality_markers": [], "sample_phrases": []},
            {"sample_count": 1, "word_count_analyzed": 10, "personality_markers": [], "sample_phrases": []},
        )
        assert low["style_confidence"] == "low"

        high = merge_voice_profiles(
            {"sample_count": 8, "word_count_analyzed": 500, "personality_markers": [], "sample_phrases": []},
            {"sample_count": 5, "word_count_analyzed": 300, "personality_markers": [], "sample_phrases": []},
        )
        assert high["style_confidence"] == "high"


# ---------------------------------------------------------------------------
# aggregate_user_style
# ---------------------------------------------------------------------------

class TestAggregateUserStyle:

    def test_empty(self):
        assert aggregate_user_style([]) == {}

    def test_single_procedure(self):
        procs = [{"id": "test", "voice_profile": {**CASUAL_PROFILE, "sample_count": 5, "word_count_analyzed": 100}}]
        result = aggregate_user_style(procs)
        assert result["formality"] == "casual"

    def test_multiple_procedures_with_contexts(self):
        procs = [
            {"id": "reddit", "voice_profile": {**CASUAL_PROFILE, "sample_count": 10, "word_count_analyzed": 200}},
            {"id": "email", "voice_profile": {**FORMAL_PROFILE, "sample_count": 8, "word_count_analyzed": 300}},
        ]
        result = aggregate_user_style(procs)
        assert "per_workflow" in result
        assert len(result["per_workflow"]) == 2

    def test_skips_empty_profiles(self):
        procs = [
            {"id": "no-text", "voice_profile": {}},
            {"id": "has-text", "voice_profile": {**CASUAL_PROFILE, "sample_count": 3, "word_count_analyzed": 50}},
        ]
        result = aggregate_user_style(procs)
        assert result.get("sample_count") == 3
