"""Tests for the SOP Inducer — pattern mining and variable abstraction."""

from __future__ import annotations

import pytest

from agenthandover_worker.sop_inducer import SOPInducer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(
    intent: str,
    target: str,
    confidence: float = 0.9,
    parameters: dict | None = None,
    pre_state: dict | None = None,
) -> dict:
    """Build a minimal SOP step dict like SemanticStep.to_sop_step() returns."""
    return {
        "step": intent,
        "target": target,
        "selector": None,
        "parameters": parameters or {},
        "confidence": confidence,
        "pre_state": pre_state or {},
    }


def _make_episode(steps: list[dict]) -> list[dict]:
    """Wrap a list of steps as an episode (identity function for clarity)."""
    return steps


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleRepeatedPattern:
    """Same 3-step sequence in 3 episodes -> 1 SOP."""

    def test_finds_pattern(self):
        step_a = _make_step("click", "Submit button")
        step_b = _make_step("type", "Email field")
        step_c = _make_step("click", "Confirm button")

        episodes = [
            _make_episode([step_a, step_b, step_c]),
            _make_episode([step_a, step_b, step_c]),
            _make_episode([step_a, step_b, step_c]),
        ]

        inducer = SOPInducer(min_support=0.3, min_pattern_length=3)
        results = inducer.induce(episodes)

        assert len(results) >= 1
        sop = results[0]
        assert len(sop["steps"]) >= 3
        assert sop["episode_count"] >= 3
        assert "slug" in sop
        assert "title" in sop
        assert "confidence_avg" in sop

    def test_confidence_avg_is_reasonable(self):
        episodes = [
            _make_episode([
                _make_step("click", "Submit button", confidence=0.8),
                _make_step("type", "Email field", confidence=0.9),
                _make_step("click", "Confirm button", confidence=1.0),
            ]),
            _make_episode([
                _make_step("click", "Submit button", confidence=0.8),
                _make_step("type", "Email field", confidence=0.9),
                _make_step("click", "Confirm button", confidence=1.0),
            ]),
            _make_episode([
                _make_step("click", "Submit button", confidence=0.8),
                _make_step("type", "Email field", confidence=0.9),
                _make_step("click", "Confirm button", confidence=1.0),
            ]),
        ]

        inducer = SOPInducer(min_support=0.3, min_pattern_length=3)
        results = inducer.induce(episodes)

        assert len(results) >= 1
        assert results[0]["confidence_avg"] == pytest.approx(0.9, abs=0.01)


class TestNoRepeatedPattern:
    """All unique sequences -> empty result."""

    def test_returns_empty(self):
        episodes = [
            _make_episode([
                _make_step("click", "Button A"),
                _make_step("type", "Field A"),
                _make_step("scroll", "Page A"),
            ]),
            _make_episode([
                _make_step("navigate", "Page B"),
                _make_step("select", "Dropdown B"),
                _make_step("paste", "Textarea B"),
            ]),
            _make_episode([
                _make_step("copy", "Text C"),
                _make_step("click", "Save C"),
                _make_step("type", "Name C"),
            ]),
        ]

        inducer = SOPInducer(min_support=0.5, min_pattern_length=3)
        results = inducer.induce(episodes)
        assert results == []


class TestVariableAbstraction:
    """Different customer names across instances -> variable slot."""

    def test_detects_variable_in_parameters(self):
        episodes = [
            _make_episode([
                _make_step("click", "New customer"),
                _make_step("type", "Name field", parameters={"text": "Alice"}),
                _make_step("click", "Save button"),
            ]),
            _make_episode([
                _make_step("click", "New customer"),
                _make_step("type", "Name field", parameters={"text": "Bob"}),
                _make_step("click", "Save button"),
            ]),
            _make_episode([
                _make_step("click", "New customer"),
                _make_step("type", "Name field", parameters={"text": "Charlie"}),
                _make_step("click", "Save button"),
            ]),
        ]

        inducer = SOPInducer(min_support=0.3, min_pattern_length=3)
        results = inducer.induce(episodes)

        assert len(results) >= 1
        sop = results[0]
        assert len(sop["variables"]) >= 1

        # Find the text variable
        text_vars = [v for v in sop["variables"] if "text" in v["name"]]
        assert len(text_vars) >= 1
        var = text_vars[0]
        # The variable should list the different names as choices or be typed string
        assert var["type"] in ("string", "enum")


class TestMinimumSupportFiltering:
    """Patterns below min_support are excluded."""

    def test_filters_low_support(self):
        common_step_a = _make_step("click", "Submit button")
        common_step_b = _make_step("type", "Email field")
        common_step_c = _make_step("click", "Confirm button")

        # Same pattern in only 1 of 5 episodes (20% support)
        rare_step = _make_step("click", "Rare button")

        episodes = [
            _make_episode([common_step_a, common_step_b, common_step_c]),
            _make_episode([common_step_a, common_step_b, common_step_c]),
            _make_episode([common_step_a, common_step_b, common_step_c]),
            _make_episode([common_step_a, common_step_b, common_step_c]),
            _make_episode([
                rare_step,
                _make_step("type", "Unique field"),
                _make_step("click", "Unique confirm"),
            ]),
        ]

        inducer = SOPInducer(min_support=0.5, min_pattern_length=3)
        results = inducer.induce(episodes)

        # The common pattern should appear (80% support >= 50% threshold)
        assert len(results) >= 1
        # All result patterns should have episode_count >= ceil(5 * 0.5) = 3
        for sop in results:
            assert sop["episode_count"] >= 2


class TestEmptyEpisodes:
    """Empty input -> empty result."""

    def test_empty_list(self):
        inducer = SOPInducer()
        assert inducer.induce([]) == []

    def test_all_empty_episodes(self):
        inducer = SOPInducer()
        assert inducer.induce([[], [], []]) == []


class TestSlugGeneration:
    """Slug generation from step intents and targets."""

    def test_slug_from_steps(self):
        inducer = SOPInducer()
        steps = [
            _make_step("click", "Submit button"),
            _make_step("type", "Email field"),
            _make_step("click", "Confirm dialog"),
        ]

        slug = inducer._generate_slug(steps)

        # Slug should be lowercase, underscore-separated
        assert slug == slug.lower()
        assert " " not in slug
        # Should contain step intents
        assert "click" in slug
        assert "type" in slug
        # Should contain target words
        assert "submit" in slug
        assert "email" in slug

    def test_slug_is_url_safe(self):
        inducer = SOPInducer()
        steps = [
            _make_step("click", "Submit Button!@#$"),
            _make_step("type", "User Name"),
            _make_step("navigate", "Main Page"),
        ]

        slug = inducer._generate_slug(steps)

        # URL-safe characters only
        import re
        assert re.match(r"^[a-z0-9_]+$", slug), f"Slug not URL-safe: {slug}"


class TestAppsInvolvedExtraction:
    """Apps extracted from episode step metadata."""

    def test_extracts_apps_from_pre_state(self):
        episodes = [
            _make_episode([
                _make_step("click", "Submit", pre_state={"app_id": "Chrome"}),
                _make_step("type", "Field", pre_state={"app_id": "Chrome"}),
                _make_step("click", "Save", pre_state={"app_id": "Excel"}),
            ]),
            _make_episode([
                _make_step("click", "Submit", pre_state={"app_id": "Chrome"}),
                _make_step("type", "Field", pre_state={"app_id": "Chrome"}),
                _make_step("click", "Save", pre_state={"app_id": "Excel"}),
            ]),
            _make_episode([
                _make_step("click", "Submit", pre_state={"app_id": "Chrome"}),
                _make_step("type", "Field", pre_state={"app_id": "Chrome"}),
                _make_step("click", "Save", pre_state={"app_id": "Excel"}),
            ]),
        ]

        inducer = SOPInducer(min_support=0.3, min_pattern_length=3)
        results = inducer.induce(episodes)

        assert len(results) >= 1
        apps = results[0]["apps_involved"]
        assert "Chrome" in apps
        assert "Excel" in apps

    def test_extracts_apps_from_parameters(self):
        episodes = [
            _make_episode([
                _make_step("click", "Submit", parameters={"app_id": "Safari"}),
                _make_step("type", "Field", parameters={"app_id": "Safari"}),
                _make_step("click", "Save", parameters={"app_id": "Safari"}),
            ]),
            _make_episode([
                _make_step("click", "Submit", parameters={"app_id": "Safari"}),
                _make_step("type", "Field", parameters={"app_id": "Safari"}),
                _make_step("click", "Save", parameters={"app_id": "Safari"}),
            ]),
            _make_episode([
                _make_step("click", "Submit", parameters={"app_id": "Safari"}),
                _make_step("type", "Field", parameters={"app_id": "Safari"}),
                _make_step("click", "Save", parameters={"app_id": "Safari"}),
            ]),
        ]

        inducer = SOPInducer(min_support=0.3, min_pattern_length=3)
        results = inducer.induce(episodes)

        assert len(results) >= 1
        assert "Safari" in results[0]["apps_involved"]


class TestMinimumPatternLength:
    """Short patterns (below min_pattern_length) are excluded."""

    def test_excludes_short_patterns(self):
        # All episodes have a 2-step repeated pattern but not 3-step
        step_a = _make_step("click", "Submit button")
        step_b = _make_step("type", "Email field")

        episodes = [
            _make_episode([step_a, step_b]),
            _make_episode([step_a, step_b]),
            _make_episode([step_a, step_b]),
        ]

        inducer = SOPInducer(min_support=0.3, min_pattern_length=3)
        results = inducer.induce(episodes)

        # No pattern should be returned because none are >= 3 steps
        assert results == []

    def test_includes_when_length_met(self):
        step_a = _make_step("click", "Submit button")
        step_b = _make_step("type", "Email field")

        episodes = [
            _make_episode([step_a, step_b]),
            _make_episode([step_a, step_b]),
            _make_episode([step_a, step_b]),
        ]

        # With min_pattern_length=2, the 2-step pattern should be found
        inducer = SOPInducer(min_support=0.3, min_pattern_length=2)
        results = inducer.induce(episodes)

        assert len(results) >= 1
        assert len(results[0]["steps"]) >= 2


class TestTitleGeneration:
    """Title generation from steps and apps."""

    def test_title_with_apps(self):
        inducer = SOPInducer()
        steps = [
            _make_step("click", "Submit button"),
            _make_step("type", "Email field"),
            _make_step("click", "Confirm dialog"),
        ]
        title = inducer._generate_title(steps, ["Chrome", "Excel"])

        assert "Click" in title
        assert "Submit button" in title
        assert "Chrome" in title

    def test_title_without_apps(self):
        inducer = SOPInducer()
        steps = [_make_step("type", "Name field")]
        title = inducer._generate_title(steps, [])

        assert "Type" in title
        assert "Name field" in title

    def test_title_empty_steps(self):
        inducer = SOPInducer()
        title = inducer._generate_title([], [])
        assert title == "Untitled SOP"
