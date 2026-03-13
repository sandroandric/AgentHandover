"""Tests for the decision extractor module."""

from __future__ import annotations

from pathlib import Path

import pytest

from oc_apprentice_worker.decision_extractor import (
    DecisionExtractor,
    DecisionRule,
    DecisionSet,
)
from oc_apprentice_worker.knowledge_base import KnowledgeBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_observation(steps: list[dict], context: dict | None = None) -> dict:
    """Create an observation dict."""
    return {"steps": steps, "context": context or {}}


def make_step(
    action: str, step_id: str | None = None, input_val: str = ""
) -> dict:
    """Create a step dict."""
    return {
        "step_id": step_id or "step_1",
        "action": action,
        "input": input_val,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    """Create a KnowledgeBase rooted in a temp directory."""
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    return kb


@pytest.fixture()
def extractor(kb: KnowledgeBase) -> DecisionExtractor:
    """Create a DecisionExtractor with a temp KB."""
    return DecisionExtractor(kb)


# ---------------------------------------------------------------------------
# extract_decisions — basic cases
# ---------------------------------------------------------------------------


class TestExtractDecisionsBasic:
    """Basic extraction scenarios."""

    def test_none_observations_returns_empty(
        self, extractor: DecisionExtractor
    ) -> None:
        result = extractor.extract_decisions("test-slug", observations=None)
        assert result == []

    def test_empty_observations_returns_empty(
        self, extractor: DecisionExtractor
    ) -> None:
        result = extractor.extract_decisions("test-slug", observations=[])
        assert result == []

    def test_single_observation_returns_empty(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [make_observation([make_step("click")])]
        result = extractor.extract_decisions("test-slug", observations=obs)
        assert result == []

    def test_identical_observations_no_decisions(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([make_step("click", "step_1")]),
            make_observation([make_step("click", "step_1")]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        assert result == []

    def test_identical_three_observations_no_decisions(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([make_step("click", "step_1")]),
            make_observation([make_step("click", "step_1")]),
            make_observation([make_step("click", "step_1")]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        assert result == []


# ---------------------------------------------------------------------------
# extract_decisions — action variation
# ---------------------------------------------------------------------------


class TestActionVariation:
    """Tests for action variation detection."""

    def test_two_observations_different_action(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([make_step("click", "step_1")]),
            make_observation([make_step("type", "step_1")]),
        ]
        result = extractor.extract_decisions("buy-domain", observations=obs)
        assert len(result) == 1
        ds = result[0]
        assert ds.procedure_slug == "buy-domain"
        assert ds.applies_to_step == "step_1"
        assert len(ds.rules) == 2
        assert ds.inferred_from_observations == 2

    def test_three_observations_majority_action(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([make_step("click", "step_1")]),
            make_observation([make_step("click", "step_1")]),
            make_observation([make_step("type", "step_1")]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        assert len(result) == 1
        ds = result[0]
        # "click" has 2/3 observations, "type" has 1/3
        click_rule = next(r for r in ds.rules if r.action == "click")
        type_rule = next(r for r in ds.rules if r.action == "type")
        assert click_rule.observed_count == 2
        assert type_rule.observed_count == 1

    def test_three_distinct_actions(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([make_step("click", "step_1")]),
            make_observation([make_step("type", "step_1")]),
            make_observation([make_step("scroll", "step_1")]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        assert len(result) == 1
        assert len(result[0].rules) == 3

    def test_action_variation_uses_step_id_from_step(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([make_step("click", "login_step")]),
            make_observation([make_step("type", "login_step")]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        assert result[0].applies_to_step == "login_step"

    def test_variation_only_in_second_step(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([
                make_step("click", "step_1"),
                make_step("approve", "step_2"),
            ]),
            make_observation([
                make_step("click", "step_1"),
                make_step("reject", "step_2"),
            ]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        assert len(result) == 1
        assert result[0].applies_to_step == "step_2"


# ---------------------------------------------------------------------------
# extract_decisions — input variation
# ---------------------------------------------------------------------------


class TestInputVariation:
    """Tests for input variation detection."""

    def test_different_inputs_same_action(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([
                make_step("type", "step_1", input_val="hello")
            ]),
            make_observation([
                make_step("type", "step_1", input_val="world")
            ]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        assert len(result) == 1
        ds = result[0]
        assert ds.applies_to_step == "step_1"
        assert len(ds.rules) == 2
        # Check the input rules
        actions_in_rules = {r.action for r in ds.rules}
        assert "use input 'hello'" in actions_in_rules
        assert "use input 'world'" in actions_in_rules

    def test_input_variation_conditions(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([
                make_step("type", "step_1", input_val="foo")
            ]),
            make_observation([
                make_step("type", "step_1", input_val="bar")
            ]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        ds = result[0]
        conditions = {r.condition for r in ds.rules}
        assert "input_value == 'foo'" in conditions
        assert "input_value == 'bar'" in conditions

    def test_empty_input_not_counted(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([make_step("type", "step_1", input_val="foo")]),
            make_observation([make_step("type", "step_1", input_val="")]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        # Only one input ("foo"), and actions are the same ("type") -> no
        # variation since both actions are identical and there is only one
        # non-empty input value.
        assert result == []


# ---------------------------------------------------------------------------
# extract_decisions — multiple variation points
# ---------------------------------------------------------------------------


class TestMultipleVariationPoints:
    """Tests with variations in multiple steps."""

    def test_two_steps_with_variation(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([
                make_step("click", "step_1"),
                make_step("approve", "step_2"),
            ]),
            make_observation([
                make_step("type", "step_1"),
                make_step("reject", "step_2"),
            ]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        assert len(result) == 2
        step_ids = {ds.applies_to_step for ds in result}
        assert step_ids == {"step_1", "step_2"}

    def test_first_step_identical_second_varies(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([
                make_step("navigate", "step_1"),
                make_step("buy", "step_2"),
                make_step("confirm", "step_3"),
            ]),
            make_observation([
                make_step("navigate", "step_1"),
                make_step("skip", "step_2"),
                make_step("confirm", "step_3"),
            ]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        assert len(result) == 1
        assert result[0].applies_to_step == "step_2"


# ---------------------------------------------------------------------------
# extract_decisions — confidence calculation
# ---------------------------------------------------------------------------


class TestConfidenceCalculation:
    """Tests for confidence scoring."""

    def test_even_split_confidence(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([make_step("click", "step_1")]),
            make_observation([make_step("type", "step_1")]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        # 1/2 = 0.5
        assert result[0].confidence == 0.5

    def test_dominant_action_confidence(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([make_step("click", "step_1")]),
            make_observation([make_step("click", "step_1")]),
            make_observation([make_step("click", "step_1")]),
            make_observation([make_step("type", "step_1")]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        # 3/4 = 0.75
        assert result[0].confidence == 0.75

    def test_three_way_split_confidence(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([make_step("click", "step_1")]),
            make_observation([make_step("type", "step_1")]),
            make_observation([make_step("scroll", "step_1")]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        # 1/3 ≈ 0.333
        assert result[0].confidence == pytest.approx(0.333, abs=0.001)


# ---------------------------------------------------------------------------
# extract_decisions — condition inference
# ---------------------------------------------------------------------------


class TestConditionInference:
    """Tests for rule condition inference."""

    def test_two_actions_if_else_condition(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([make_step("approve", "step_1")]),
            make_observation([make_step("reject", "step_1")]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        rules = result[0].rules
        conditions = {r.condition for r in rules}
        # For 2 actions: "when choosing X over Y"
        assert any("when choosing" in c for c in conditions)

    def test_three_actions_generic_condition(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([make_step("approve", "step_1")]),
            make_observation([make_step("reject", "step_1")]),
            make_observation([make_step("defer", "step_1")]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        rules = result[0].rules
        # For 3+ actions: "when action is X"
        conditions = {r.condition for r in rules}
        assert any("when action is" in c for c in conditions)

    def test_two_actions_condition_references_both(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([make_step("buy", "step_1")]),
            make_observation([make_step("skip", "step_1")]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        buy_rule = next(r for r in result[0].rules if r.action == "buy")
        # Condition should reference "skip" as the other option
        assert "skip" in buy_rule.condition


# ---------------------------------------------------------------------------
# extract_decisions — edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_observation_with_no_steps(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([]),
            make_observation([make_step("click", "step_1")]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        assert result == []

    def test_shorter_observation_skips_missing_steps(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([
                make_step("click", "step_1"),
                make_step("type", "step_2"),
            ]),
            make_observation([
                make_step("click", "step_1"),
                # no step_2
            ]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        # step_1 is same in both, step_2 only present in one observation
        # but since second observation has no step_2, it's skipped
        assert result == []

    def test_default_step_id_when_missing(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([{"action": "click", "input": ""}]),
            make_observation([{"action": "type", "input": ""}]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        assert len(result) == 1
        assert result[0].applies_to_step == "step_1"

    def test_whitespace_actions_normalized(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([make_step("  click  ", "step_1")]),
            make_observation([make_step("click", "step_1")]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        # After strip().lower() both become "click" — identical
        assert result == []

    def test_case_insensitive_action_match(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation([make_step("Click", "step_1")]),
            make_observation([make_step("CLICK", "step_1")]),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        # Both lowercase to "click"
        assert result == []

    def test_observations_with_context(
        self, extractor: DecisionExtractor
    ) -> None:
        obs = [
            make_observation(
                [make_step("buy", "step_1")],
                context={"price": 10},
            ),
            make_observation(
                [make_step("skip", "step_1")],
                context={"price": 100},
            ),
        ]
        result = extractor.extract_decisions("test-slug", observations=obs)
        assert len(result) == 1
        assert result[0].inferred_from_observations == 2


# ---------------------------------------------------------------------------
# save_decisions
# ---------------------------------------------------------------------------


class TestSaveDecisions:
    """Tests for persisting decisions to the knowledge base."""

    def test_save_decisions_creates_file(
        self, extractor: DecisionExtractor, kb: KnowledgeBase
    ) -> None:
        ds = DecisionSet(
            procedure_slug="test-slug",
            applies_to_step="step_1",
            rules=[
                DecisionRule(
                    condition="always",
                    action="click",
                    observed_count=3,
                    description="test",
                )
            ],
            confidence=0.9,
            inferred_from_observations=3,
        )
        extractor.save_decisions([ds])
        decisions = kb.get_decisions()
        assert len(decisions["decision_sets"]) == 1
        assert decisions["decision_sets"][0]["procedure_slug"] == "test-slug"

    def test_save_decisions_preserves_rules(
        self, extractor: DecisionExtractor, kb: KnowledgeBase
    ) -> None:
        ds = DecisionSet(
            procedure_slug="test-slug",
            applies_to_step="step_1",
            rules=[
                DecisionRule(
                    condition="price < 50",
                    action="buy",
                    observed_count=2,
                    description="buy when cheap",
                ),
                DecisionRule(
                    condition="price >= 50",
                    action="skip",
                    observed_count=1,
                    description="skip when expensive",
                ),
            ],
            confidence=0.67,
            inferred_from_observations=3,
        )
        extractor.save_decisions([ds])
        decisions = kb.get_decisions()
        rules = decisions["decision_sets"][0]["rules"]
        assert len(rules) == 2
        assert rules[0]["condition"] == "price < 50"
        assert rules[0]["action"] == "buy"
        assert rules[1]["observed_count"] == 1

    def test_save_decisions_updates_existing(
        self, extractor: DecisionExtractor, kb: KnowledgeBase
    ) -> None:
        ds1 = DecisionSet(
            procedure_slug="test-slug",
            applies_to_step="step_1",
            rules=[
                DecisionRule("old", "click", 1, "old rule")
            ],
            confidence=0.5,
            inferred_from_observations=2,
        )
        extractor.save_decisions([ds1])

        ds2 = DecisionSet(
            procedure_slug="test-slug",
            applies_to_step="step_1",
            rules=[
                DecisionRule("new", "type", 3, "new rule")
            ],
            confidence=0.8,
            inferred_from_observations=5,
        )
        extractor.save_decisions([ds2])

        decisions = kb.get_decisions()
        # Should have updated, not duplicated
        assert len(decisions["decision_sets"]) == 1
        assert decisions["decision_sets"][0]["confidence"] == 0.8
        assert decisions["decision_sets"][0]["rules"][0]["condition"] == "new"

    def test_save_decisions_appends_different_step(
        self, extractor: DecisionExtractor, kb: KnowledgeBase
    ) -> None:
        ds1 = DecisionSet(
            procedure_slug="test-slug",
            applies_to_step="step_1",
            rules=[DecisionRule("c1", "a1", 1, "d1")],
            confidence=0.5,
            inferred_from_observations=2,
        )
        ds2 = DecisionSet(
            procedure_slug="test-slug",
            applies_to_step="step_2",
            rules=[DecisionRule("c2", "a2", 2, "d2")],
            confidence=0.6,
            inferred_from_observations=2,
        )
        extractor.save_decisions([ds1])
        extractor.save_decisions([ds2])

        decisions = kb.get_decisions()
        assert len(decisions["decision_sets"]) == 2

    def test_save_decisions_appends_different_slug(
        self, extractor: DecisionExtractor, kb: KnowledgeBase
    ) -> None:
        ds1 = DecisionSet(
            procedure_slug="slug-a",
            applies_to_step="step_1",
            rules=[DecisionRule("c1", "a1", 1, "d1")],
            confidence=0.5,
            inferred_from_observations=2,
        )
        ds2 = DecisionSet(
            procedure_slug="slug-b",
            applies_to_step="step_1",
            rules=[DecisionRule("c2", "a2", 2, "d2")],
            confidence=0.6,
            inferred_from_observations=3,
        )
        extractor.save_decisions([ds1, ds2])

        decisions = kb.get_decisions()
        assert len(decisions["decision_sets"]) == 2
        slugs = {d["procedure_slug"] for d in decisions["decision_sets"]}
        assert slugs == {"slug-a", "slug-b"}

    def test_save_empty_list(
        self, extractor: DecisionExtractor, kb: KnowledgeBase
    ) -> None:
        extractor.save_decisions([])
        decisions = kb.get_decisions()
        assert decisions["decision_sets"] == []

    def test_save_decisions_sets_updated_at(
        self, extractor: DecisionExtractor, kb: KnowledgeBase
    ) -> None:
        ds = DecisionSet(
            procedure_slug="test-slug",
            applies_to_step="step_1",
            rules=[DecisionRule("c", "a", 1, "d")],
            confidence=0.5,
            inferred_from_observations=2,
        )
        extractor.save_decisions([ds])
        decisions = kb.get_decisions()
        assert decisions["updated_at"] is not None


# ---------------------------------------------------------------------------
# Dataclass checks
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Verify dataclass field defaults and structure."""

    def test_decision_rule_fields(self) -> None:
        r = DecisionRule(
            condition="x > 5",
            action="approve",
            observed_count=3,
            description="test",
        )
        assert r.condition == "x > 5"
        assert r.action == "approve"
        assert r.observed_count == 3
        assert r.description == "test"

    def test_decision_set_fields(self) -> None:
        ds = DecisionSet(
            procedure_slug="slug",
            applies_to_step="step_1",
            rules=[],
            confidence=0.75,
            inferred_from_observations=4,
        )
        assert ds.procedure_slug == "slug"
        assert ds.applies_to_step == "step_1"
        assert ds.rules == []
        assert ds.confidence == 0.75
        assert ds.inferred_from_observations == 4
