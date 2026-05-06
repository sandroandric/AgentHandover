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


def _make_observations_with_apps():
    """Observations that carry app names, for cold-start context tests."""
    return [
        [
            {"action": "focus", "app": "Visual Studio Code"},
            {"action": "switch", "app": "Google Chrome"},
            {"action": "switch", "app": "Terminal"},
        ],
        [
            {"action": "focus", "app": "Visual Studio Code"},
            {"action": "switch", "app": "Google Chrome"},
        ],
    ]


class _FakeKB:
    """Minimal KB stub that returns a preset profile."""

    def __init__(self, profile: dict):
        self._profile = profile

    def get_profile(self) -> dict:
        return self._profile


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


class TestUserContext:
    """Tests for user profile injection into the synthesis prompt."""

    def test_no_kb_no_apps_uses_generic_placeholder(self):
        """With no KB and no app signals, the prompt shows the generic fallback."""
        synth = BehavioralSynthesizer(SynthesizerConfig(min_observations=1))
        prompt = synth._build_prompt(_make_procedure(), _make_observations())
        assert "no user profile yet" in prompt
        assert "first-time user" in prompt

    def test_cold_start_derives_apps_from_observations(self):
        """With no KB but apps in observations, cold-start context kicks in."""
        synth = BehavioralSynthesizer(SynthesizerConfig(min_observations=1))
        prompt = synth._build_prompt(
            _make_procedure(), _make_observations_with_apps(),
        )
        assert "first-time user" in prompt
        assert "Visual Studio Code" in prompt
        assert "Google Chrome" in prompt
        assert "Observed apps in this workflow" in prompt

    def test_kb_profile_renders_full_block(self):
        """With a populated KB, the full USER PROFILE block is injected."""
        profile = {
            "tools": {
                "browser": "Google Chrome",
                "editor": "Visual Studio Code",
                "terminal": "Terminal",
                "primary_apps": [
                    {"app": "Google Chrome", "total_minutes": 1200, "days_used": 20},
                    {"app": "Visual Studio Code", "total_minutes": 900, "days_used": 18},
                    {"app": "Terminal", "total_minutes": 300, "days_used": 15},
                ],
            },
            "accounts": [
                {"service": "github", "frequency": "daily"},
                {"service": "vercel", "frequency": "weekly"},
            ],
            "working_hours": {
                "typical_start": "09:00",
                "typical_end": "18:00",
                "avg_active_hours": 7.5,
                "weekend_active": True,
            },
            "communication_style": {
                "primary_channels": ["Slack", "Mail"],
                "avg_comm_minutes_per_day": 30,
            },
            "writing_style": {"formality": "casual", "confidence": 0.72},
        }
        synth = BehavioralSynthesizer(
            SynthesizerConfig(min_observations=1),
            knowledge_base=_FakeKB(profile),
        )
        prompt = synth._build_prompt(
            _make_procedure(), _make_observations_with_apps(),
        )
        assert "USER PROFILE (who this user is):" in prompt
        assert "Google Chrome" in prompt
        assert "editor=Visual Studio Code" in prompt
        assert "github (daily)" in prompt
        assert "09:00-18:00" in prompt
        assert "Slack" in prompt
        assert "casual" in prompt
        # Role hint should classify this as dev given github + editor
        assert "software developer" in prompt.lower()

    def test_empty_kb_profile_falls_back_to_cold_start(self):
        """A KB that returns empty-defaults profile should fall through."""
        synth = BehavioralSynthesizer(
            SynthesizerConfig(min_observations=1),
            knowledge_base=_FakeKB({
                "tools": {},
                "working_hours": {},
                "accounts": [],
                "communication_style": {},
            }),
        )
        prompt = synth._build_prompt(
            _make_procedure(), _make_observations_with_apps(),
        )
        # No rich profile — cold-start derived apps should appear
        assert "first-time user" in prompt
        assert "Visual Studio Code" in prompt

    def test_kb_failure_is_graceful(self):
        """If KB.get_profile() throws, we still get a prompt."""
        class _BrokenKB:
            def get_profile(self):
                raise RuntimeError("boom")

        synth = BehavioralSynthesizer(
            SynthesizerConfig(min_observations=1),
            knowledge_base=_BrokenKB(),
        )
        prompt = synth._build_prompt(
            _make_procedure(), _make_observations_with_apps(),
        )
        # Falls back to cold-start
        assert "first-time user" in prompt
        assert "Visual Studio Code" in prompt

    def test_role_hint_designer(self):
        """Figma-heavy profile classifies as designer."""
        profile = {
            "tools": {
                "browser": "Google Chrome",
                "primary_apps": [
                    {"app": "Figma", "total_minutes": 2000, "days_used": 25},
                    {"app": "Google Chrome", "total_minutes": 500, "days_used": 20},
                ],
            },
            "accounts": [{"service": "figma", "frequency": "daily"}],
        }
        synth = BehavioralSynthesizer(
            SynthesizerConfig(min_observations=1),
            knowledge_base=_FakeKB(profile),
        )
        prompt = synth._build_prompt(_make_procedure(), _make_observations())
        assert "designer" in prompt.lower()

    def test_role_hint_founder_when_all_signals(self):
        """Dev + design + PM tools classify as founder/generalist."""
        profile = {
            "tools": {
                "editor": "Visual Studio Code",
                "primary_apps": [
                    {"app": "Figma", "total_minutes": 500, "days_used": 10},
                    {"app": "Visual Studio Code", "total_minutes": 800, "days_used": 15},
                ],
            },
            "accounts": [
                {"service": "github", "frequency": "daily"},
                {"service": "figma", "frequency": "weekly"},
                {"service": "notion", "frequency": "daily"},
            ],
        }
        synth = BehavioralSynthesizer(
            SynthesizerConfig(min_observations=1),
            knowledge_base=_FakeKB(profile),
        )
        prompt = synth._build_prompt(_make_procedure(), _make_observations())
        assert "founder" in prompt.lower() or "generalist" in prompt.lower()


class TestSynthesize:

    def test_mocked_vlm_returns_insights(self):
        vlm_response = json.dumps({
            "goal": "Sends daily marketing replies on /r/startups to drive product signups",
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
        assert insights.goal == "Sends daily marketing replies on /r/startups to drive product signups"
        assert insights.strategy == "Daily community marketing on Reddit"
        assert len(insights.guardrails) == 1
        assert insights.confidence == 0.85

    def test_goal_field_is_parsed_and_stripped(self):
        """The goal field must be extracted from VLM JSON and stripped."""
        vlm_response = json.dumps({
            "goal": "  Sends a daily HN digest email to self at sandro@sandric.co  ",
            "strategy": "Copy HN top stories, paste into Gmail compose, send",
            "confidence": 0.9,
        })
        synth = BehavioralSynthesizer(SynthesizerConfig(min_observations=1))
        with patch(
            "agenthandover_worker.behavioral_synthesizer._call_ollama_text",
            return_value=(vlm_response, 1.0),
        ):
            insights = synth.synthesize(
                "daily-email", _make_procedure(), _make_observations(1),
            )
        assert insights.goal == "Sends a daily HN digest email to self at sandro@sandric.co"
        assert "sandro@sandric.co" in insights.goal

    def test_prompt_requires_concrete_goal(self):
        """The synthesis prompt body must ask for a concrete goal with \
examples of what acceptable vs abstract answers look like."""
        from agenthandover_worker.behavioral_synthesizer import (
            BEHAVIORAL_SYNTHESIS_PROMPT,
        )
        assert '"goal":' in BEHAVIORAL_SYNTHESIS_PROMPT
        assert "FOR WHOM" in BEHAVIORAL_SYNTHESIS_PROMPT
        assert "TRIGGER/CADENCE" in BEHAVIORAL_SYNTHESIS_PROMPT
        # The prompt should include the daily-HN example as a concrete goal
        assert "Hacker News" in BEHAVIORAL_SYNTHESIS_PROMPT
        # And explicitly reject abstract phrasings
        assert "research-to-communication cycle" in BEHAVIORAL_SYNTHESIS_PROMPT

    def test_merge_insights_writes_goal_to_procedure(self):
        synth = BehavioralSynthesizer()
        proc = _make_procedure()
        insights = BehavioralInsights(
            goal="Sends a daily HN digest email to self",
            strategy="Copy-paste workflow",
            confidence=0.9,
        )
        updated = synth.merge_insights_into_procedure(proc, insights)
        assert updated["goal"] == "Sends a daily HN digest email to self"
        assert updated["strategy"] == "Copy-paste workflow"
        # Original procedure must NOT be mutated
        assert "goal" not in proc


class TestTimelineEvidence:
    """Tests for the per-frame evidence formatter that surfaces verbatim
    text (emails, URLs, typed text) into the synthesizer prompt.

    Historical bug (caught 2026-04-10): the synthesizer asked the model
    for a concrete goal with verbatim text but the prompt template never
    actually formatted observation content into the prompt — len() and
    cold-start app extraction were the only consumers.  The model
    rightly returned 'Unclear — recipient not visible'.
    """

    @staticmethod
    def _make_rich_frame(**kwargs) -> dict:
        base = {
            "action": "composing email",
            "app": "Gmail",
            "location": "mail.google.com",
            "target": "mail.google.com",
        }
        base.update(kwargs)
        return base

    def test_empty_observations_returns_empty(self):
        assert BehavioralSynthesizer._format_timeline_evidence([]) == ""
        assert BehavioralSynthesizer._format_timeline_evidence([[]]) == ""

    def test_renders_email_addresses_verbatim(self):
        obs = [[
            self._make_rich_frame(
                email_addresses=["sandro@sandric.co"],
                typed_text="updates",
            )
        ]]
        out = BehavioralSynthesizer._format_timeline_evidence(obs)
        assert "sandro@sandric.co" in out
        assert "TIMELINE EVIDENCE" in out
        assert "verbatim" in out
        assert 'typed="updates"' in out

    def test_renders_urls_and_values(self):
        obs = [[
            self._make_rich_frame(
                urls=["https://news.ycombinator.com"],
                visible_values=["Top 10 HN stories", "by time"],
            )
        ]]
        out = BehavioralSynthesizer._format_timeline_evidence(obs)
        assert "news.ycombinator.com" in out
        assert "Top 10 HN stories" in out

    def test_renders_clipboard_preview(self):
        obs = [[
            self._make_rich_frame(
                clipboard_preview="Top story 1\nTop story 2\nTop story 3",
            )
        ]]
        out = BehavioralSynthesizer._format_timeline_evidence(obs)
        assert 'COPIED="Top story 1' in out

    def test_caps_at_max_frames(self):
        # Build 100 frames, expect a truncation marker
        obs = [[
            self._make_rich_frame(typed_text=f"frame {i}")
            for i in range(100)
        ]]
        out = BehavioralSynthesizer._format_timeline_evidence(obs)
        assert "more frames truncated" in out

    def test_skips_frames_with_no_useful_fields(self):
        obs = [[
            {"app": "", "location": "", "action": ""},
            self._make_rich_frame(email_addresses=["x@y.com"]),
        ]]
        out = BehavioralSynthesizer._format_timeline_evidence(obs)
        # Frame without any useful fields should not produce a line
        assert "frame 1:" not in out
        assert "frame 2:" in out
        assert "x@y.com" in out

    def test_build_prompt_includes_timeline_evidence(self):
        """End-to-end: rich observations must appear in the formatted prompt."""
        synth = BehavioralSynthesizer(SynthesizerConfig(min_observations=1))
        rich_obs = [[
            {
                "action": "compose email",
                "app": "Gmail",
                "location": "mail.google.com",
                "email_addresses": ["sandro@sandric.co"],
                "typed_text": "daily updates email",
            },
            {
                "action": "scan top stories",
                "app": "Comet",
                "location": "https://news.ycombinator.com",
                "urls": ["https://news.ycombinator.com"],
                "visible_values": ["Story 1", "Story 2"],
            },
        ]]
        prompt = synth._build_prompt(_make_procedure(), rich_obs)
        # The verbatim grounding the synthesizer prompt asks for must
        # actually appear in the prompt body.
        assert "sandro@sandric.co" in prompt
        assert "news.ycombinator.com" in prompt
        assert "daily updates email" in prompt
        assert "Story 1" in prompt

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

    def test_merge_does_not_set_last_synthesized_on_empty_insights(self):
        """``last_synthesized`` is only set when something substantive was extracted.

        Regression test for v0.2.x bug: marketing-stats-email had
        ``last_synthesized`` populated despite empty strategy and empty
        guardrails, falsely signalling that synthesis succeeded.  After
        the fix, an empty BehavioralInsights() leaves the procedure
        unchanged with no timestamp.
        """
        synth = BehavioralSynthesizer()
        proc = _make_procedure()
        empty = BehavioralInsights()  # all defaults
        updated = synth.merge_insights_into_procedure(proc, empty)
        assert "last_synthesized" not in updated
        assert "_obs_at_last_synthesis" not in updated

    def test_merge_sets_last_synthesized_on_goal_only(self):
        """A goal alone is enough to consider synthesis successful."""
        synth = BehavioralSynthesizer()
        proc = _make_procedure()
        insights = BehavioralInsights(goal="Send weekly report email")
        updated = synth.merge_insights_into_procedure(proc, insights)
        assert "last_synthesized" in updated
        assert updated["goal"] == "Send weekly report email"


class TestParseInsightsValidation:
    """Tests for the EmptyInsightsError validation in _parse_insights."""

    def test_parse_raises_on_completely_empty_dict(self):
        from agenthandover_worker.behavioral_synthesizer import (
            EmptyInsightsError,
        )
        with pytest.raises(EmptyInsightsError):
            BehavioralSynthesizer._parse_insights({})

    def test_parse_raises_on_all_empty_fields(self):
        from agenthandover_worker.behavioral_synthesizer import (
            EmptyInsightsError,
        )
        with pytest.raises(EmptyInsightsError):
            BehavioralSynthesizer._parse_insights({
                "goal": "",
                "strategy": "",
                "guardrails": [],
                "selection_criteria": [],
            })

    def test_parse_accepts_goal_only(self):
        insights = BehavioralSynthesizer._parse_insights({"goal": "test goal"})
        assert insights.goal == "test goal"
        assert insights.strategy == ""

    def test_parse_accepts_strategy_only(self):
        insights = BehavioralSynthesizer._parse_insights(
            {"strategy": "test strategy"}
        )
        assert insights.strategy == "test strategy"

    def test_parse_accepts_guardrails_only(self):
        """Guardrails alone are enough — they are real extracted constraints."""
        insights = BehavioralSynthesizer._parse_insights(
            {"guardrails": ["never auto-send"]}
        )
        assert insights.guardrails == ["never auto-send"]

    def test_parse_accepts_selection_criteria_only(self):
        insights = BehavioralSynthesizer._parse_insights({
            "selection_criteria": [{"criterion": "B2B SaaS founders"}],
        })
        assert len(insights.selection_criteria) == 1

    def test_parse_raises_on_decision_branches_only(self):
        """Decision branches alone are insufficient — they describe HOW but
        not WHY/WHAT (no goal/strategy)."""
        from agenthandover_worker.behavioral_synthesizer import (
            EmptyInsightsError,
        )
        with pytest.raises(EmptyInsightsError):
            BehavioralSynthesizer._parse_insights({
                "decision_branches": [{"condition": "if X then Y"}],
            })

    def test_parse_raises_on_non_dict_input(self):
        from agenthandover_worker.behavioral_synthesizer import (
            EmptyInsightsError,
        )
        with pytest.raises(EmptyInsightsError):
            BehavioralSynthesizer._parse_insights("not a dict")  # type: ignore
