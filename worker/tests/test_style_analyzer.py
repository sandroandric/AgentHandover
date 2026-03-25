"""Tests for the style_analyzer module."""

from __future__ import annotations

import pytest

from agenthandover_worker.style_analyzer import (
    analyze_style,
    extract_content_samples,
    analyze_procedure_style,
    merge_voice_profiles,
    aggregate_user_style,
)


# ---------------------------------------------------------------------------
# analyze_style
# ---------------------------------------------------------------------------

class TestAnalyzeStyle:

    def test_empty_input(self):
        assert analyze_style([]) == {}

    def test_too_short(self):
        assert analyze_style(["hi"]) == {}

    def test_casual_text(self):
        texts = [
            "hey great point! I'm totally on board with this approach lol",
            "yeah let's def do it, we can't wait any longer!!",
            "I've been thinking about this and it's a no brainer :)",
        ]
        result = analyze_style(texts)
        assert result["formality"] == "casual"
        assert result["formality_score"] < 0
        assert result["sample_count"] == 3
        assert result["word_count_analyzed"] > 0

    def test_formal_text(self):
        texts = [
            "The proposal was reviewed by the committee and subsequently approved.",
            "It appears to be the case that the implementation could be improved.",
            "The results were analyzed by the research team and published accordingly.",
        ]
        result = analyze_style(texts)
        assert result["formality"] in ("formal", "neutral")
        assert result["formality_score"] >= -0.3

    def test_emoji_detection(self):
        result = analyze_style(["great work on this one :) really happy with it lol"])
        assert result["uses_emoji"] is True

    def test_no_emoji(self):
        result = analyze_style(["The quarterly report has been submitted for review."])
        assert result["uses_emoji"] is False

    def test_sentence_length(self):
        result = analyze_style(["Short. Very short. Tiny sentences. Yes. Ok. Done. Fine."])
        assert result["avg_sentence_length"] < 8

    def test_vocabulary_richness(self):
        # Diverse vocabulary
        result = analyze_style([
            "The magnificent cathedral towered above the ancient cobblestone streets "
            "while curious tourists photographed elaborate architectural details."
        ])
        assert result["vocabulary_richness"] > 0.5

    def test_exclamation_rate(self):
        result = analyze_style(["Wow this is great! Amazing work! Incredible results! So good overall!"])
        assert result["exclamation_rate"] > 0


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
            "A third piece of content that demonstrates style.",
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
        assert len(samples) == 1  # same prefix → deduped

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

    def test_with_content_produced(self):
        proc = {
            "evidence": {
                "extracted_evidence": {
                    "content_produced": [
                        {"type": "text_input", "full_value": "Hey this is a great idea! Let's do it :)"},
                        {"type": "text_input", "full_value": "Totally agree, we should move fast on this one!"},
                    ],
                },
            },
        }
        vp, cs = analyze_procedure_style(proc)
        assert vp.get("formality") in ("casual", "neutral")
        assert len(cs) > 0

    def test_uses_preview_fallback(self):
        proc = {
            "evidence": {
                "extracted_evidence": {
                    "content_produced": [
                        {"type": "text_input", "value_preview": "This is a preview text that should be analyzed for style and tone characteristics."},
                        {"type": "text_input", "value_preview": "Another preview text that adds enough content for the analyzer to work with properly."},
                    ],
                },
            },
        }
        vp, cs = analyze_procedure_style(proc)
        assert vp.get("word_count_analyzed", 0) > 0

    def test_merges_with_existing(self):
        proc = {
            "voice_profile": {
                "formality": "casual",
                "formality_score": -0.5,
                "word_count_analyzed": 100,
                "sample_count": 10,
                "uses_emoji": True,
                "avg_sentence_length": 8.0,
                "vocabulary_richness": 0.6,
                "exclamation_rate": 0.3,
                "question_rate": 0.1,
                "caps_rate": 0.01,
            },
            "evidence": {
                "extracted_evidence": {
                    "content_produced": [
                        {"type": "text_input", "full_value": "Another casual reply here with some more text to make it long enough! Love it :) Really great stuff here."},
                        {"type": "text_input", "full_value": "Yeah totally agree with this approach, we should definitely try it out soon! Can't wait :D"},
                    ],
                },
            },
        }
        vp, _ = analyze_procedure_style(proc)
        # Should merge, cumulative word count
        assert vp.get("word_count_analyzed", 0) > 100


# ---------------------------------------------------------------------------
# merge_voice_profiles
# ---------------------------------------------------------------------------

class TestMergeVoiceProfiles:

    def test_merge_empty(self):
        assert merge_voice_profiles({}, {"formality": "casual"}) == {"formality": "casual"}
        assert merge_voice_profiles({"formality": "formal"}, {}) == {"formality": "formal"}

    def test_weighted_merge(self):
        old = {
            "formality_score": -0.5,
            "avg_sentence_length": 8.0,
            "vocabulary_richness": 0.6,
            "exclamation_rate": 0.3,
            "question_rate": 0.1,
            "caps_rate": 0.01,
            "word_count_analyzed": 200,
            "sample_count": 10,
            "uses_emoji": True,
        }
        new = {
            "formality_score": -0.4,
            "avg_sentence_length": 10.0,
            "vocabulary_richness": 0.65,
            "exclamation_rate": 0.2,
            "question_rate": 0.15,
            "caps_rate": 0.02,
            "word_count_analyzed": 100,
            "sample_count": 5,
            "uses_emoji": False,
        }
        merged = merge_voice_profiles(old, new)
        assert merged["word_count_analyzed"] == 300
        assert merged["sample_count"] == 15
        assert merged["uses_emoji"] is True  # OR
        assert merged["style_confidence"] == "moderate"  # 15 samples

    def test_confidence_levels(self):
        low = merge_voice_profiles(
            {"word_count_analyzed": 10, "sample_count": 2, "formality_score": 0, "avg_sentence_length": 10, "vocabulary_richness": 0.5, "exclamation_rate": 0, "question_rate": 0, "caps_rate": 0, "uses_emoji": False},
            {"word_count_analyzed": 10, "sample_count": 2, "formality_score": 0, "avg_sentence_length": 10, "vocabulary_richness": 0.5, "exclamation_rate": 0, "question_rate": 0, "caps_rate": 0, "uses_emoji": False},
        )
        assert low["style_confidence"] == "low"

        high = merge_voice_profiles(
            {"word_count_analyzed": 500, "sample_count": 15, "formality_score": 0, "avg_sentence_length": 10, "vocabulary_richness": 0.5, "exclamation_rate": 0, "question_rate": 0, "caps_rate": 0, "uses_emoji": False},
            {"word_count_analyzed": 500, "sample_count": 10, "formality_score": 0, "avg_sentence_length": 10, "vocabulary_richness": 0.5, "exclamation_rate": 0, "question_rate": 0, "caps_rate": 0, "uses_emoji": False},
        )
        assert high["style_confidence"] == "high"


# ---------------------------------------------------------------------------
# aggregate_user_style
# ---------------------------------------------------------------------------

class TestAggregateUserStyle:

    def test_empty(self):
        assert aggregate_user_style([]) == {}

    def test_single_procedure(self):
        procs = [{
            "id": "test",
            "voice_profile": {
                "formality": "casual",
                "formality_score": -0.4,
                "word_count_analyzed": 100,
                "sample_count": 5,
                "avg_sentence_length": 8.0,
                "vocabulary_richness": 0.6,
                "exclamation_rate": 0.2,
                "question_rate": 0.1,
                "caps_rate": 0.01,
                "uses_emoji": True,
            },
        }]
        result = aggregate_user_style(procs)
        assert result["formality"] == "casual"

    def test_multiple_procedures(self):
        procs = [
            {"id": "reddit", "voice_profile": {"formality": "casual", "formality_score": -0.5, "word_count_analyzed": 200, "sample_count": 10, "avg_sentence_length": 8, "vocabulary_richness": 0.6, "exclamation_rate": 0.3, "question_rate": 0.1, "caps_rate": 0.01, "uses_emoji": True}},
            {"id": "email", "voice_profile": {"formality": "formal", "formality_score": 0.4, "word_count_analyzed": 150, "sample_count": 8, "avg_sentence_length": 18, "vocabulary_richness": 0.7, "exclamation_rate": 0.05, "question_rate": 0.2, "caps_rate": 0.0, "uses_emoji": False}},
        ]
        result = aggregate_user_style(procs)
        assert "per_workflow" in result
        assert len(result["per_workflow"]) == 2
        assert result["sample_count"] == 18

    def test_skips_empty_profiles(self):
        procs = [
            {"id": "no-text", "voice_profile": {}},
            {"id": "has-text", "voice_profile": {"formality": "neutral", "formality_score": 0, "word_count_analyzed": 50, "sample_count": 3, "avg_sentence_length": 12, "vocabulary_richness": 0.5, "exclamation_rate": 0.1, "question_rate": 0.1, "caps_rate": 0, "uses_emoji": False}},
        ]
        result = aggregate_user_style(procs)
        assert result.get("word_count_analyzed") == 50
