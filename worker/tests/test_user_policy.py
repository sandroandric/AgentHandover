"""Tests for the user policy module.

Covers CRUD operations, pattern matching semantics, rule priority,
persistence round-trips, and edge cases.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.user_policy import PolicyAction, PolicyRule, UserPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def kb(tmp_path):
    """Temporary knowledge base for policy persistence."""
    kb = KnowledgeBase(root=tmp_path)
    kb.ensure_structure()
    return kb


# ===================================================================
# TestCRUD — 6 tests
# ===================================================================

class TestCRUD:
    """Basic create / read / delete operations on policy rules."""

    def test_add_rule_appends(self, kb):
        policy = UserPolicy(kb)
        rule = PolicyRule(rule_type="app", pattern="VS Code", action=PolicyAction.ALWAYS_INCLUDE)
        policy.add_rule(rule)
        assert len(policy.load_rules()) == 1
        assert policy.load_rules()[0].pattern == "VS Code"

    def test_remove_rule_by_index(self, kb):
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(rule_type="app", pattern="A", action=PolicyAction.IGNORE))
        policy.add_rule(PolicyRule(rule_type="app", pattern="B", action=PolicyAction.IGNORE))
        policy.remove_rule(0)
        rules = policy.load_rules()
        assert len(rules) == 1
        assert rules[0].pattern == "B"

    def test_remove_rule_invalid_index_raises(self, kb):
        policy = UserPolicy(kb)
        with pytest.raises(IndexError):
            policy.remove_rule(0)

    def test_load_rules_returns_copy(self, kb):
        """Modifying the returned list should not affect internal state."""
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(rule_type="app", pattern="X", action=PolicyAction.IGNORE))
        rules = policy.load_rules()
        rules.clear()
        assert len(policy.load_rules()) == 1

    def test_fresh_kb_has_empty_rules(self, kb):
        policy = UserPolicy(kb)
        assert policy.load_rules() == []

    def test_multiple_add_calls_accumulate(self, kb):
        policy = UserPolicy(kb)
        for i in range(5):
            policy.add_rule(PolicyRule(
                rule_type="app", pattern=f"App{i}", action=PolicyAction.IGNORE,
            ))
        assert len(policy.load_rules()) == 5


# ===================================================================
# TestMatching — 8 tests
# ===================================================================

class TestMatching:
    """Pattern matching via UserPolicy.check()."""

    def test_app_pattern_exact_match(self, kb):
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(rule_type="app", pattern="VS Code", action=PolicyAction.IGNORE))
        result = policy.check(app="VS Code")
        assert result is not None
        assert result.pattern == "VS Code"

    def test_url_pattern_glob_match(self, kb):
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(rule_type="url", pattern="*youtube*", action=PolicyAction.IGNORE))
        result = policy.check(url="https://youtube.com/watch?v=abc")
        assert result is not None

    def test_browser_profile_pattern_match(self, kb):
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(
            rule_type="browser_profile", pattern="Work*",
            action=PolicyAction.ALWAYS_INCLUDE,
        ))
        result = policy.check(browser_profile="Work Profile")
        assert result is not None

    def test_no_match_returns_none(self, kb):
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(rule_type="app", pattern="Firefox", action=PolicyAction.IGNORE))
        result = policy.check(app="Chrome")
        assert result is None

    def test_case_insensitive_match(self, kb):
        """fnmatch matching should be case-insensitive."""
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(rule_type="app", pattern="vs code", action=PolicyAction.IGNORE))
        result = policy.check(app="VS Code")
        assert result is not None

    def test_wildcard_matches_everything(self, kb):
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(rule_type="app", pattern="*", action=PolicyAction.IGNORE))
        result = policy.check(app="AnyAppAtAll")
        assert result is not None

    def test_partial_glob_match(self, kb):
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(rule_type="app", pattern="Chrome*", action=PolicyAction.IGNORE))
        result = policy.check(app="Chrome Beta")
        assert result is not None

    def test_source_rule_type_matches_url(self, kb):
        """'source' rule_type matches against the url parameter."""
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(
            rule_type="source", pattern="*github.com*",
            action=PolicyAction.ALWAYS_INCLUDE,
        ))
        result = policy.check(url="https://github.com/user/repo")
        assert result is not None


# ===================================================================
# TestPriority — 3 tests
# ===================================================================

class TestPriority:
    """First-match-wins rule ordering."""

    def test_first_matching_rule_wins(self, kb):
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(
            rule_type="url", pattern="*youtube*", action=PolicyAction.IGNORE,
        ))
        policy.add_rule(PolicyRule(
            rule_type="url", pattern="*youtube*", action=PolicyAction.ALWAYS_INCLUDE,
        ))
        result = policy.check(url="https://youtube.com")
        assert result is not None
        assert result.action == PolicyAction.IGNORE

    def test_rule_order_matters(self, kb):
        """Swapping the order changes which rule wins."""
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(
            rule_type="url", pattern="*youtube*", action=PolicyAction.ALWAYS_INCLUDE,
        ))
        policy.add_rule(PolicyRule(
            rule_type="url", pattern="*youtube*", action=PolicyAction.IGNORE,
        ))
        result = policy.check(url="https://youtube.com")
        assert result is not None
        assert result.action == PolicyAction.ALWAYS_INCLUDE

    def test_non_matching_rules_skipped(self, kb):
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(
            rule_type="app", pattern="Firefox", action=PolicyAction.IGNORE,
        ))
        policy.add_rule(PolicyRule(
            rule_type="app", pattern="Chrome", action=PolicyAction.ALWAYS_INCLUDE,
        ))
        result = policy.check(app="Chrome")
        assert result is not None
        assert result.action == PolicyAction.ALWAYS_INCLUDE


# ===================================================================
# TestPersistence — 3 tests
# ===================================================================

class TestPersistence:
    """Rules survive save/load cycles via the knowledge base."""

    def test_rules_survive_reload(self, kb):
        """Create a policy, add rules, create a new instance — rules persist."""
        p1 = UserPolicy(kb)
        p1.add_rule(PolicyRule(rule_type="app", pattern="VS Code", action=PolicyAction.ALWAYS_INCLUDE))
        p1.add_rule(PolicyRule(rule_type="url", pattern="*slack*", action=PolicyAction.IGNORE))

        p2 = UserPolicy(kb)
        rules = p2.load_rules()
        assert len(rules) == 2
        assert rules[0].pattern == "VS Code"
        assert rules[1].pattern == "*slack*"

    def test_atomic_write_produces_valid_json(self, kb):
        """After saving, the file on disk is valid JSON."""
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(
            rule_type="app", pattern="Test", action=PolicyAction.IGNORE,
        ))
        path = kb.root / "policy.json"
        assert path.is_file()
        with open(path) as f:
            data = json.load(f)
        assert "rules" in data
        assert len(data["rules"]) == 1

    def test_missing_policy_json_returns_empty(self, kb):
        """First-run scenario: no policy.json file means empty rules."""
        policy = UserPolicy(kb)
        assert policy.load_rules() == []


# ===================================================================
# TestEdgeCases — 5 tests
# ===================================================================

class TestEdgeCases:
    """Boundary and degenerate inputs."""

    def test_empty_pattern_no_crash(self, kb):
        """An empty pattern should not cause errors (fnmatch handles it)."""
        policy = UserPolicy(kb)
        policy.add_rule(PolicyRule(rule_type="app", pattern="", action=PolicyAction.IGNORE))
        # Empty pattern matches empty app, not non-empty app
        result = policy.check(app="VS Code")
        assert result is None

    def test_classify_as_with_value(self, kb):
        """CLASSIFY_AS rule stores its target value."""
        rule = PolicyRule(
            rule_type="app", pattern="*",
            action=PolicyAction.CLASSIFY_AS,
            value="research",
        )
        policy = UserPolicy(kb)
        policy.add_rule(rule)
        loaded = policy.load_rules()
        assert loaded[0].action == PolicyAction.CLASSIFY_AS
        assert loaded[0].value == "research"

    def test_corrupted_json_handled_gracefully(self, kb):
        """If policy.json contains invalid JSON, rules should be empty,
        not raise an exception."""
        path = kb.root / "policy.json"
        path.write_text("{corrupted json!!! not valid")
        policy = UserPolicy(kb)
        assert policy.load_rules() == []

    def test_policy_action_enum_values_are_strings(self):
        """PolicyAction members should be usable as plain strings."""
        assert PolicyAction.IGNORE == "ignore"
        assert PolicyAction.NEVER_LEARN == "never_learn"
        assert PolicyAction.CLASSIFY_AS == "classify_as"
        assert PolicyAction.ALWAYS_INCLUDE == "always_include"

    def test_asdict_roundtrip_preserves_fields(self, kb):
        """asdict on PolicyRule should produce a dict that can reconstruct the rule."""
        original = PolicyRule(
            rule_type="url",
            pattern="*github*",
            action=PolicyAction.ALWAYS_INCLUDE,
            value="",
            comment="Track all GitHub activity",
        )
        d = asdict(original)
        reconstructed = PolicyRule(
            rule_type=d["rule_type"],
            pattern=d["pattern"],
            action=PolicyAction(d["action"]),
            value=d["value"],
            comment=d["comment"],
        )
        assert reconstructed == original
