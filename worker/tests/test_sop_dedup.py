"""Tests for SOP Deduplication — fingerprint, matching, merging, registry.

Tests cover:
- Fingerprint computation (apps, domains, action verbs)
- App normalization (PIDs, bundle IDs, arrow notation)
- Domain extraction (URLs, preconditions)
- Action verb canonicalization (synonyms)
- Jaccard similarity
- Fingerprint similarity (weighted)
- Matching (threshold, best match selection)
- Merging (episode accumulation, step selection, variable union)
- Registry (load/save, cumulative dedup)
- Real-world scenarios (exact repeat, evolved workflow, different tasks)
- LLM-based step conflict resolution
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agenthandover_worker.sop_dedup import (
    _extract_action_verbs,
    _extract_apps,
    _extract_domains,
    _jaccard,
    _normalize_app,
    _url_to_domain,
    _merge_variables,
    _resolve_conflict_with_llm,
    compute_fingerprint,
    deduplicate_templates,
    find_matching_sop,
    fingerprint_similarity,
    load_registry,
    merge_sops,
    save_registry,
)
from agenthandover_worker.llm_reasoning import LLMReasoner, ReasoningResult


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _amazon_sop(**overrides) -> dict:
    """Search Amazon SOP."""
    sop = {
        "slug": "search-product-amazon",
        "title": "Search for a Product on Amazon",
        "source": "v2_focus_recording",
        "steps": [
            {"step": "Open Amazon homepage", "parameters": {"app": "Chrome", "location": "https://www.amazon.com"}},
            {"step": "Enter search query", "parameters": {"app": "Chrome", "input": "{{query}}"}},
            {"step": "Click search button", "parameters": {"app": "Chrome"}},
            {"step": "Review search results", "parameters": {"app": "Chrome", "verify": "Results visible"}},
        ],
        "variables": [{"name": "query", "type": "string", "example": "wireless earbuds"}],
        "confidence_avg": 0.85,
        "episode_count": 1,
        "apps_involved": ["Chrome"],
        "preconditions": ["Browser is open"],
        "task_description": "Search for a product on Amazon.",
        "execution_overview": {"when_to_use": "When shopping on Amazon"},
    }
    sop.update(overrides)
    return sop


def _amazon_sop_v2(**overrides) -> dict:
    """Same Amazon search, slightly different VLM wording."""
    sop = {
        "slug": "search-amazon-products",
        "title": "Search Amazon for Products",
        "source": "v2_focus_recording",
        "steps": [
            {"step": "Navigate to Amazon", "parameters": {"app": "Chrome", "location": "https://www.amazon.com"}},
            {"step": "Type search term", "parameters": {"app": "Chrome", "input": "{{search_term}}"}},
            {"step": "Press search button", "parameters": {"app": "Chrome"}},
            {"step": "Check results page", "parameters": {"app": "Chrome", "verify": "Products displayed"}},
        ],
        "variables": [{"name": "search_term", "type": "string", "example": "keyboard"}],
        "confidence_avg": 0.80,
        "episode_count": 1,
        "apps_involved": ["Chrome"],
        "preconditions": ["Amazon.com accessible"],
        "task_description": "Search Amazon for products.",
        "execution_overview": {"when_to_use": "When you need to find products on Amazon"},
    }
    sop.update(overrides)
    return sop


def _deploy_sop(**overrides) -> dict:
    """Deploy to staging SOP — different from Amazon."""
    sop = {
        "slug": "deploy-feature-staging",
        "title": "Deploy Feature to Staging",
        "source": "v2_focus_recording",
        "steps": [
            {"step": "Review code changes", "parameters": {"app": "VS Code"}},
            {"step": "Run unit tests", "parameters": {"app": "VS Code → Terminal", "input": "pytest"}},
            {"step": "Commit changes", "parameters": {"app": "VS Code → Terminal", "input": "git commit"}},
            {"step": "Push to remote", "parameters": {"app": "VS Code → Terminal", "input": "git push"}},
        ],
        "variables": [{"name": "commit_message", "type": "string"}],
        "confidence_avg": 0.82,
        "episode_count": 1,
        "apps_involved": ["VS Code"],
        "preconditions": ["Repository cloned"],
        "task_description": "Deploy a feature to staging.",
        "execution_overview": {},
    }
    sop.update(overrides)
    return sop


def _ebay_sop(**overrides) -> dict:
    """Search eBay — similar structure to Amazon but different site."""
    sop = {
        "slug": "search-product-ebay",
        "title": "Search for a Product on eBay",
        "source": "v2_focus_recording",
        "steps": [
            {"step": "Open eBay homepage", "parameters": {"app": "Chrome", "location": "https://www.ebay.com"}},
            {"step": "Enter search query", "parameters": {"app": "Chrome", "input": "{{query}}"}},
            {"step": "Click search button", "parameters": {"app": "Chrome"}},
            {"step": "Review listings", "parameters": {"app": "Chrome"}},
        ],
        "variables": [{"name": "query", "type": "string"}],
        "confidence_avg": 0.78,
        "episode_count": 1,
        "apps_involved": ["Chrome"],
        "preconditions": [],
        "task_description": "Search eBay for products.",
        "execution_overview": {},
    }
    sop.update(overrides)
    return sop


# ------------------------------------------------------------------
# Tests: App normalization
# ------------------------------------------------------------------


class TestNormalizeApp:
    def test_lowercase(self):
        assert _normalize_app("Chrome") == "chrome"

    def test_strip_pid(self):
        assert _normalize_app("Visual Studio Code in pid:1184:Visual Studio Code") == "visual studio code"

    def test_strip_bundle_id(self):
        assert _normalize_app("com.chrome.Chrome") == "chrome"

    def test_strip_arrow(self):
        assert _normalize_app("VS Code → Terminal") == "vs code"

    def test_plain_app(self):
        assert _normalize_app("Finder") == "finder"


# ------------------------------------------------------------------
# Tests: Domain extraction
# ------------------------------------------------------------------


class TestDomainExtraction:
    def test_url_to_domain_https(self):
        assert _url_to_domain("https://www.amazon.com/search?q=test") == "amazon.com"

    def test_url_to_domain_http(self):
        assert _url_to_domain("http://example.org/page") == "example.org"

    def test_url_to_domain_strips_www(self):
        assert _url_to_domain("https://www.google.com") == "google.com"

    def test_url_to_domain_no_url(self):
        assert _url_to_domain("left sidebar") == ""

    def test_url_to_domain_file_path(self):
        assert _url_to_domain("~/project/src/main.py") == ""

    def test_url_to_domain_empty(self):
        assert _url_to_domain("") == ""

    def test_extract_domains_from_steps(self):
        sop = _amazon_sop()
        domains = _extract_domains(sop)
        assert "amazon.com" in domains

    def test_extract_domains_from_preconditions(self):
        sop = _amazon_sop(preconditions=["url_open:https://expenses.example.com"])
        domains = _extract_domains(sop)
        assert "expenses.example.com" in domains


# ------------------------------------------------------------------
# Tests: Action verb extraction
# ------------------------------------------------------------------


class TestActionVerbs:
    def test_basic_extraction(self):
        sop = _amazon_sop()
        verbs = _extract_action_verbs(sop)
        assert "open" in verbs
        assert "enter" in verbs
        assert "click" in verbs
        assert "review" in verbs

    def test_synonym_mapping(self):
        """navigate -> open, type -> enter, press -> click."""
        sop = _amazon_sop_v2()  # Uses Navigate/Type/Press/Check
        verbs = _extract_action_verbs(sop)
        assert "open" in verbs      # Navigate -> open
        assert "enter" in verbs     # Type -> enter
        assert "click" in verbs     # Press -> click
        assert "verify" in verbs    # Check -> verify

    def test_empty_steps(self):
        sop = _amazon_sop(steps=[])
        verbs = _extract_action_verbs(sop)
        assert verbs == set()


# ------------------------------------------------------------------
# Tests: Fingerprint computation
# ------------------------------------------------------------------


class TestFingerprint:
    def test_compute_basic(self):
        sop = _amazon_sop()
        fp = compute_fingerprint(sop)
        assert "chrome" in fp["apps"]
        assert "amazon.com" in fp["domains"]
        assert len(fp["action_verbs"]) > 0

    def test_fingerprint_deterministic(self):
        """Same SOP always produces same fingerprint."""
        sop = _amazon_sop()
        fp1 = compute_fingerprint(sop)
        fp2 = compute_fingerprint(sop)
        assert fp1 == fp2


# ------------------------------------------------------------------
# Tests: Similarity
# ------------------------------------------------------------------


class TestSimilarity:
    def test_jaccard_identical(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_jaccard_partial(self):
        assert _jaccard({"a", "b", "c"}, {"a", "b"}) == pytest.approx(2 / 3)

    def test_jaccard_both_empty(self):
        assert _jaccard(set(), set()) == 1.0

    def test_same_sop_different_wording(self):
        """Two Amazon SOPs with different VLM wording should be similar."""
        fp1 = compute_fingerprint(_amazon_sop())
        fp2 = compute_fingerprint(_amazon_sop_v2())
        sim = fingerprint_similarity(fp1, fp2)
        # Same app (chrome), same domain (amazon.com), same action verbs
        # (open, enter, click, review/verify are mapped to same canonicals)
        assert sim >= 0.70, f"Expected >= 0.70, got {sim}"

    def test_different_tasks_low_similarity(self):
        """Amazon search vs Deploy should be low similarity."""
        fp1 = compute_fingerprint(_amazon_sop())
        fp2 = compute_fingerprint(_deploy_sop())
        sim = fingerprint_similarity(fp1, fp2)
        assert sim < 0.50, f"Expected < 0.50, got {sim}"

    def test_same_structure_different_site(self):
        """Amazon vs eBay — same structure but different domain."""
        fp1 = compute_fingerprint(_amazon_sop())
        fp2 = compute_fingerprint(_ebay_sop())
        sim = fingerprint_similarity(fp1, fp2)
        # Same app + same verbs, but different domain
        # Should be below threshold (domains weight = 0.35)
        assert sim < 0.80, f"Expected < 0.80, got {sim}"

    def test_no_domains_redistribution(self):
        """Desktop-only workflows (no URLs) redistribute domain weight."""
        fp1 = {"apps": ["vs code"], "domains": [], "action_verbs": ["open", "enter"]}
        fp2 = {"apps": ["vs code"], "domains": [], "action_verbs": ["open", "enter"]}
        sim = fingerprint_similarity(fp1, fp2)
        assert sim == 1.0


# ------------------------------------------------------------------
# Tests: Matching
# ------------------------------------------------------------------


class TestMatching:
    def test_match_found(self):
        """Second Amazon recording matches first."""
        existing = [_amazon_sop()]
        new = _amazon_sop_v2()
        idx = find_matching_sop(new, existing)
        assert idx == 0

    def test_no_match_different_task(self):
        """Deploy SOP doesn't match Amazon SOP."""
        existing = [_amazon_sop()]
        new = _deploy_sop()
        idx = find_matching_sop(new, existing)
        assert idx is None

    def test_no_match_empty_registry(self):
        idx = find_matching_sop(_amazon_sop(), [])
        assert idx is None

    def test_best_match_selected(self):
        """When multiple candidates exist, best match is selected."""
        existing = [_deploy_sop(), _amazon_sop(), _ebay_sop()]
        new = _amazon_sop_v2()
        idx = find_matching_sop(new, existing)
        assert idx == 1  # Amazon SOP is best match

    def test_threshold_respected(self):
        """High threshold prevents loose matches."""
        existing = [_amazon_sop()]
        new = _ebay_sop()
        # With very high threshold, Amazon vs eBay shouldn't match
        idx = find_matching_sop(new, existing, threshold=0.95)
        assert idx is None


# ------------------------------------------------------------------
# Tests: Merging
# ------------------------------------------------------------------


class TestMerging:
    def test_episode_count_accumulates(self):
        merged = merge_sops(_amazon_sop(episode_count=2), _amazon_sop_v2(episode_count=1))
        assert merged["episode_count"] == 3

    def test_slug_preserved(self):
        """Existing slug is kept (stable identity)."""
        merged = merge_sops(_amazon_sop(), _amazon_sop_v2())
        assert merged["slug"] == "search-product-amazon"

    def test_title_updated(self):
        """Latest title is kept."""
        merged = merge_sops(_amazon_sop(), _amazon_sop_v2())
        assert merged["title"] == "Search Amazon for Products"

    def test_steps_more_kept(self):
        """Version with more steps is kept."""
        short = _amazon_sop(steps=[{"step": "Open", "parameters": {}}])
        long = _amazon_sop_v2()  # 4 steps
        merged = merge_sops(short, long)
        assert len(merged["steps"]) == 4

    def test_steps_existing_kept_when_more(self):
        """If existing has more steps, keep existing."""
        long = _amazon_sop()  # 4 steps
        short = _amazon_sop_v2(steps=[{"step": "Open", "parameters": {}}])
        merged = merge_sops(long, short)
        assert len(merged["steps"]) == 4

    def test_variables_union(self):
        """Variables from both are merged by name."""
        merged = merge_sops(_amazon_sop(), _amazon_sop_v2())
        var_names = {v["name"] for v in merged["variables"]}
        assert "query" in var_names
        assert "search_term" in var_names

    def test_apps_union(self):
        """Apps from both are combined."""
        sop1 = _amazon_sop(apps_involved=["Chrome"])
        sop2 = _amazon_sop_v2(apps_involved=["Chrome", "Firefox"])
        merged = merge_sops(sop1, sop2)
        assert "Chrome" in merged["apps_involved"]
        assert "Firefox" in merged["apps_involved"]

    def test_preconditions_deduped(self):
        """Preconditions are unioned and deduped."""
        sop1 = _amazon_sop(preconditions=["Browser is open", "Logged in"])
        sop2 = _amazon_sop_v2(preconditions=["Browser is open", "Amazon accessible"])
        merged = merge_sops(sop1, sop2)
        assert merged["preconditions"].count("Browser is open") == 1
        assert "Logged in" in merged["preconditions"]
        assert "Amazon accessible" in merged["preconditions"]

    def test_merge_count_tracked(self):
        merged = merge_sops(_amazon_sop(), _amazon_sop_v2())
        assert merged["_merge_count"] == 1
        merged2 = merge_sops(merged, _amazon_sop())
        assert merged2["_merge_count"] == 2

    def test_fingerprint_updated(self):
        merged = merge_sops(_amazon_sop(), _amazon_sop_v2())
        assert "_fingerprint" in merged

    def test_task_description_updated(self):
        merged = merge_sops(_amazon_sop(), _amazon_sop_v2())
        assert merged["task_description"] == "Search Amazon for products."


# ------------------------------------------------------------------
# Tests: Variable merging
# ------------------------------------------------------------------


class TestVariableMerging:
    def test_union_by_name(self):
        existing = [{"name": "query", "type": "string", "example": "earbuds"}]
        new = [{"name": "category", "type": "string", "example": "Electronics"}]
        result = _merge_variables(existing, new)
        names = {v["name"] for v in result}
        assert names == {"query", "category"}

    def test_richer_definition_wins(self):
        existing = [{"name": "query", "type": "string", "example": "", "description": ""}]
        new = [{"name": "query", "type": "string", "example": "headphones", "description": "Search term"}]
        result = _merge_variables(existing, new)
        assert result[0]["example"] == "headphones"

    def test_empty_lists(self):
        assert _merge_variables([], []) == []


# ------------------------------------------------------------------
# Tests: Registry persistence
# ------------------------------------------------------------------


class TestRegistry:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def test_save_and_load(self):
        sops = [_amazon_sop(), _deploy_sop()]
        save_registry(self.tmpdir, sops)

        loaded = load_registry(self.tmpdir)
        assert len(loaded) == 2
        slugs = {s["slug"] for s in loaded}
        assert "search-product-amazon" in slugs
        assert "deploy-feature-staging" in slugs

    def test_timeline_stripped(self):
        """_timeline is stripped when saving to save space."""
        sop = _amazon_sop()
        sop["_timeline"] = [{"large": "data" * 1000}]
        save_registry(self.tmpdir, [sop])

        loaded = load_registry(self.tmpdir)
        assert "_timeline" not in loaded[0]

    def test_fingerprint_persisted(self):
        save_registry(self.tmpdir, [_amazon_sop()])
        loaded = load_registry(self.tmpdir)
        assert "_fingerprint" in loaded[0]

    def test_load_empty(self):
        assert load_registry(self.tmpdir) == []

    def test_load_corrupt(self):
        """Corrupt JSON returns empty list."""
        (self.tmpdir / "sop-registry.json").write_text("not json{{{")
        assert load_registry(self.tmpdir) == []


# ------------------------------------------------------------------
# Tests: Full dedup pipeline
# ------------------------------------------------------------------


class TestDeduplicateTemplates:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def test_first_sop_added(self):
        """First SOP always added to registry."""
        result = deduplicate_templates([_amazon_sop()], self.tmpdir)
        assert len(result) == 1

        registry = load_registry(self.tmpdir)
        assert len(registry) == 1

    def test_duplicate_merged(self):
        """Second recording of same task merges into first."""
        # First recording
        deduplicate_templates([_amazon_sop()], self.tmpdir)

        # Second recording (different VLM wording)
        result = deduplicate_templates([_amazon_sop_v2()], self.tmpdir)
        assert len(result) == 1
        assert result[0]["slug"] == "search-product-amazon"  # Kept original slug
        assert result[0]["episode_count"] == 2  # Accumulated

        # Registry should still have 1 entry
        registry = load_registry(self.tmpdir)
        assert len(registry) == 1

    def test_different_task_not_merged(self):
        """Different task creates new entry."""
        deduplicate_templates([_amazon_sop()], self.tmpdir)
        result = deduplicate_templates([_deploy_sop()], self.tmpdir)
        assert len(result) == 1
        assert result[0]["slug"] == "deploy-feature-staging"

        registry = load_registry(self.tmpdir)
        assert len(registry) == 2

    def test_triple_merge(self):
        """Three recordings of same task all merge correctly."""
        deduplicate_templates([_amazon_sop(episode_count=1)], self.tmpdir)
        deduplicate_templates([_amazon_sop_v2(episode_count=1)], self.tmpdir)
        result = deduplicate_templates(
            [_amazon_sop(slug="find-items-amazon", episode_count=1)],
            self.tmpdir,
        )

        assert len(result) == 1
        assert result[0]["episode_count"] == 3
        assert result[0]["slug"] == "search-product-amazon"

        registry = load_registry(self.tmpdir)
        assert len(registry) == 1

    def test_batch_dedup(self):
        """Multiple SOPs in one batch are deduped individually."""
        result = deduplicate_templates(
            [_amazon_sop(), _deploy_sop()],
            self.tmpdir,
        )
        assert len(result) == 2

        registry = load_registry(self.tmpdir)
        assert len(registry) == 2


# ------------------------------------------------------------------
# Tests: Real-world scenarios
# ------------------------------------------------------------------


class TestRealWorldScenarios:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def test_exact_repeat(self):
        """User records same task identically twice."""
        sop1 = _amazon_sop(episode_count=1)
        sop2 = _amazon_sop(episode_count=1)  # Exact same

        deduplicate_templates([sop1], self.tmpdir)
        result = deduplicate_templates([sop2], self.tmpdir)

        assert len(result) == 1
        assert result[0]["episode_count"] == 2

    def test_evolved_workflow(self):
        """User adds a new step to their workflow."""
        sop1 = _amazon_sop(episode_count=1)

        sop2 = _amazon_sop_v2(episode_count=1)
        sop2["steps"].append(
            {"step": "Add to cart", "parameters": {"app": "Chrome"}},
        )

        deduplicate_templates([sop1], self.tmpdir)
        result = deduplicate_templates([sop2], self.tmpdir)

        assert len(result) == 1
        assert result[0]["episode_count"] == 2
        assert len(result[0]["steps"]) == 5  # Kept longer version

    def test_amazon_vs_ebay_separate(self):
        """Amazon and eBay are similar but stay separate."""
        deduplicate_templates([_amazon_sop()], self.tmpdir)
        result = deduplicate_templates([_ebay_sop()], self.tmpdir)

        registry = load_registry(self.tmpdir)
        # Should have 2 separate SOPs (different domains)
        assert len(registry) == 2

    def test_variables_discovered_over_time(self):
        """New variables discovered in second recording are added."""
        sop1 = _amazon_sop(variables=[
            {"name": "query", "type": "string", "example": "earbuds"},
        ])
        sop2 = _amazon_sop_v2(variables=[
            {"name": "search_term", "type": "string", "example": "keyboard"},
            {"name": "category", "type": "string", "example": "Electronics"},
        ])

        deduplicate_templates([sop1], self.tmpdir)
        result = deduplicate_templates([sop2], self.tmpdir)

        var_names = {v["name"] for v in result[0]["variables"]}
        assert var_names == {"query", "search_term", "category"}


# ------------------------------------------------------------------
# Tests: LLM-based merge conflict resolution
# ------------------------------------------------------------------


class TestLLMMerge:
    """Tests for LLM-assisted step conflict resolution during merge."""

    def test_merge_conflict_calls_llm(self):
        """When steps differ and counts are within 1, LLM resolves."""
        reasoner = LLMReasoner()
        reasoner.reason_json = MagicMock(return_value=ReasoningResult(
            value={"keep": "B", "reason": "Version B is more specific"},
            success=True,
        ))

        sop_a = _amazon_sop(steps=[
            {"step": "Open website", "parameters": {"app": "Chrome"}},
            {"step": "Enter query", "parameters": {"app": "Chrome"}},
        ])
        sop_b = _amazon_sop_v2(steps=[
            {"step": "Open Amazon homepage", "parameters": {"app": "Chrome"}},
            {"step": "Type search term", "parameters": {"app": "Chrome"}},
        ])

        merged = merge_sops(sop_a, sop_b, llm_reasoner=reasoner)

        # LLM was called for each conflicting step pair
        assert reasoner.reason_json.call_count == 2
        # LLM chose "B" so both steps should be from sop_b
        assert merged["steps"][0]["step"] == "Open Amazon homepage"
        assert merged["steps"][1]["step"] == "Type search term"

    def test_merge_no_conflict_skips_llm(self):
        """When existing has many more steps (diff >= 2), LLM is skipped."""
        reasoner = LLMReasoner()
        reasoner.reason_json = MagicMock()

        sop_long = _amazon_sop(steps=[
            {"step": "Open website", "parameters": {}},
            {"step": "Enter query", "parameters": {}},
            {"step": "Click search", "parameters": {}},
            {"step": "Review results", "parameters": {}},
        ])
        sop_short = _amazon_sop_v2(steps=[
            {"step": "Open site", "parameters": {}},
            {"step": "Search", "parameters": {}},
        ])

        merged = merge_sops(sop_long, sop_short, llm_reasoner=reasoner)

        # Diff is 2, so heuristic path is used (keep longer = existing)
        reasoner.reason_json.assert_not_called()
        assert len(merged["steps"]) == 4

    def test_merge_llm_keeps_both(self):
        """When LLM says 'both', step A gets step B as an alternative."""
        reasoner = LLMReasoner()
        reasoner.reason_json = MagicMock(return_value=ReasoningResult(
            value={"keep": "both", "reason": "Both approaches are valid"},
            success=True,
        ))

        sop_a = _amazon_sop(steps=[
            {"step": "Click the search icon", "parameters": {"app": "Chrome"}},
        ])
        sop_b = _amazon_sop_v2(steps=[
            {"step": "Press Enter to search", "parameters": {"app": "Chrome"}},
        ])

        merged = merge_sops(sop_a, sop_b, llm_reasoner=reasoner)

        reasoner.reason_json.assert_called_once()
        # Step A is kept with B as alternative
        assert merged["steps"][0]["step"] == "Click the search icon"
        assert "alternatives" in merged["steps"][0]
        alt_steps = [
            a.get("step", a.get("action", ""))
            for a in merged["steps"][0]["alternatives"]
        ]
        assert "Press Enter to search" in alt_steps

    def test_merge_llm_failure_uses_heuristic(self):
        """When LLM fails, merge falls back to heuristic (keep new)."""
        reasoner = LLMReasoner()
        reasoner.reason_json = MagicMock(return_value=ReasoningResult(
            value=None,
            success=False,
            error="Ollama not reachable",
        ))

        sop_a = _amazon_sop(steps=[
            {"step": "Open website", "parameters": {"app": "Chrome"}},
            {"step": "Enter query", "parameters": {"app": "Chrome"}},
        ])
        sop_b = _amazon_sop_v2(steps=[
            {"step": "Navigate to Amazon", "parameters": {"app": "Chrome"}},
            {"step": "Type search term", "parameters": {"app": "Chrome"}},
        ])

        merged = merge_sops(sop_a, sop_b, llm_reasoner=reasoner)

        # LLM was attempted but failed — falls back to keeping new (sop_b)
        reasoner.reason_json.assert_called()
        assert merged["steps"][0]["step"] == "Navigate to Amazon"
        assert merged["steps"][1]["step"] == "Type search term"
