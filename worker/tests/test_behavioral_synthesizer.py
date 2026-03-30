"""Tests for behavioral_synthesizer.py."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agenthandover_worker.behavioral_synthesizer import (
    BehavioralInsights,
    BehavioralSynthesizer,
    SynthesizerConfig,
)


def _make_procedure(**overrides):
    proc = {
        "id": "test-proc",
        "title": "Test Workflow",
        "description": "A test workflow",
        "steps": [
            {"step_id": "step_1", "action": "Open browser", "app": "Chrome"},
            {"step_id": "step_2", "action": "Navigate to site", "app": "Chrome"},
        ],
        "evidence": {
            "total_observations": 5,
            "observations": [{"date": "2026-03-10"}, {"date": "2026-03-11"}],
        },
        "constraints": {"guardrails": []},
    }
    proc.update(overrides)
    return proc


def _make_procedure_with_evidence():
    proc = _make_procedure()
    proc["evidence"]["extracted_evidence"] = {
        "content_produced": [
            {"type": "clipboard", "content_types": ["text/plain"], "byte_size": 256},
            {"type": "text_input", "field": "reply", "value_preview": "Great point about marketing..."},
        ],
        "selection_signals": [
            {"location": "https://reddit.com/r/startups/post1", "avg_dwell_seconds": 45.0, "visit_count": 2, "engagement": "high"},
            {"location": "https://reddit.com/r/startups/post2", "avg_dwell_seconds": 3.0, "visit_count": 1, "engagement": "low"},
        ],
        "url_patterns": [
            {"url": "https://reddit.com/r/startups", "visit_count": 5, "domain": "reddit.com"},
        ],
        "timing_patterns": {
            "total_duration_seconds": 900.0,
            "significant_pauses": 2,
            "avg_gap_seconds": 15.0,
        },
    }
    return proc


def _make_observations(count=3):
    return [
        [{"action": "open browser"}, {"action": "navigate"}]
        for _ in range(count)
    ]


class TestBuildPrompt:

    def test_includes_extracted_evidence(self):
        synth = BehavioralSynthesizer(SynthesizerConfig(min_observations=1))
        proc = _make_procedure_with_evidence()
        prompt = synth._build_prompt(proc, _make_observations())
        assert "EXTRACTED EVIDENCE" in prompt
        assert "CONTENT PRODUCED BY USER" in prompt
        assert "Great point about marketing" in prompt
        assert "ENGAGEMENT SIGNALS" in prompt
        assert "45s avg dwell" in prompt
        assert "high engagement" in prompt
        assert "URL PATTERNS" in prompt
        assert "reddit.com/r/startups" in prompt
        assert "TIMING" in prompt
        assert "15 min total" in prompt

    def test_without_evidence_graceful(self):
        synth = BehavioralSynthesizer(SynthesizerConfig(min_observations=1))
        proc = _make_procedure()  # no extracted_evidence
        prompt = synth._build_prompt(proc, _make_observations())
        assert "no extracted evidence available" in prompt

    def test_empty_evidence_sections(self):
        synth = BehavioralSynthesizer(SynthesizerConfig(min_observations=1))
        proc = _make_procedure()
        proc["evidence"]["extracted_evidence"] = {
            "content_produced": [],
            "selection_signals": [],
            "url_patterns": [],
            "timing_patterns": {},
        }
        prompt = synth._build_prompt(proc, _make_observations())
        # Empty sections should not appear
        assert "CONTENT PRODUCED BY USER" not in prompt


class TestSynthesize:

    def test_mocked_vlm_returns_insights(self):
        vlm_response = json.dumps({
            "strategy": "Daily community marketing on Reddit",
            "selection_criteria": [{"criterion": "Posts with 10+ comments", "examples": [], "confidence": 0.8}],
            "content_templates": [],
            "decision_branches": [],
            "guardrails": ["Never reply to promotional posts"],
            "timing": {"avg_duration_minutes": 15},
            "confidence": 0.85,
        })
        synth = BehavioralSynthesizer(SynthesizerConfig(min_observations=1))
        with patch(
            "agenthandover_worker.behavioral_synthesizer._call_ollama_text",
            return_value=(vlm_response, 5.0),
        ):
            insights = synth.synthesize(
                "test-proc", _make_procedure_with_evidence(),
                _make_observations(3),
            )
        assert insights.strategy == "Daily community marketing on Reddit"
        assert len(insights.guardrails) == 1
        assert insights.confidence == 0.85

    def test_below_min_observations_returns_empty(self):
        synth = BehavioralSynthesizer(SynthesizerConfig(min_observations=3))
        insights = synth.synthesize(
            "test-proc", _make_procedure(), _make_observations(2),
        )
        assert insights.strategy == ""

    def test_should_synthesize_true_when_enough_observations(self):
        synth = BehavioralSynthesizer(SynthesizerConfig(min_observations=3))
        proc = _make_procedure()
        proc["evidence"]["total_observations"] = 5
        assert synth.should_synthesize(proc) is True

    def test_should_synthesize_false_when_too_few(self):
        synth = BehavioralSynthesizer(SynthesizerConfig(min_observations=3))
        proc = _make_procedure()
        proc["evidence"]["total_observations"] = 2
        assert synth.should_synthesize(proc) is False

    def test_should_synthesize_false_when_recently_done(self):
        synth = BehavioralSynthesizer(SynthesizerConfig(min_observations=3, re_synthesis_interval=3))
        proc = _make_procedure()
        proc["evidence"]["total_observations"] = 5
        proc["last_synthesized"] = "2026-03-17T00:00:00Z"
        proc["_obs_at_last_synthesis"] = 5
        assert synth.should_synthesize(proc) is False


class TestMergeInsights:

    def test_merge_updates_procedure(self):
        synth = BehavioralSynthesizer()
        proc = _make_procedure()
        insights = BehavioralInsights(
            strategy="Test strategy",
            guardrails=["Never do X"],
            confidence=0.9,
        )
        updated = synth.merge_insights_into_procedure(proc, insights)
        assert updated["strategy"] == "Test strategy"
        assert "Never do X" in updated["constraints"]["guardrails"]
        assert updated["behavioral_confidence"] == 0.9
        assert updated["last_synthesized"] is not None

    def test_merge_does_not_mutate_original(self):
        synth = BehavioralSynthesizer()
        proc = _make_procedure()
        insights = BehavioralInsights(strategy="Test")
        updated = synth.merge_insights_into_procedure(proc, insights)
        assert proc.get("strategy") is None
        assert updated["strategy"] == "Test"
