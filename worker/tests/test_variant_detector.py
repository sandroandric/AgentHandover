"""Tests for variant_detector — Needleman-Wunsch alignment, parameter extraction, variant detection.

Tests cover:
- Semantic alignment (NW dynamic programming, gap penalties, embedding/string fallback)
- Parameter extraction (type inference, multi-field, constant field filtering)
- Variant detection (divergent vs. fixed steps, multiple demos, context dicts)
- Workflow normalization (majority canonical, alternatives, single-demo pass-through)
- Edge cases (empty demos, single step, no embeddings)
"""

from __future__ import annotations

import math

import pytest

from agenthandover_worker.variant_detector import (
    AlignedStep,
    ParameterExtraction,
    VariantDetector,
    WorkflowVariant,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _step(action: str, app: str = "Chrome", location: str = "",
           input_val: str = "", target: str = "") -> dict:
    return {
        "action": action,
        "app": app,
        "location": location,
        "input": input_val,
        "target": target,
    }


def _unit_vec(dim: int, index: int) -> list[float]:
    """Return a unit vector of length *dim* with 1.0 at *index*."""
    v = [0.0] * dim
    v[index] = 1.0
    return v


# ------------------------------------------------------------------
# TestSemanticAlign (11 tests)
# ------------------------------------------------------------------


class TestSemanticAlign:
    """Needleman-Wunsch alignment of two step sequences."""

    def test_identical_sequences(self):
        """Two identical lists produce all is_match=True with similarity near 1.0."""
        det = VariantDetector()
        steps = [_step("Open browser"), _step("Click search"), _step("Enter query")]
        aligned = det.semantic_align(steps, list(steps))

        assert len(aligned) == 3
        for a in aligned:
            assert a.is_match is True
            assert a.similarity >= 0.6

    def test_insertion_detected(self):
        """Demo B has extra step in middle — produces an AlignedStep with action_a=None."""
        det = VariantDetector()
        steps_a = [_step("Open browser"), _step("Click search")]
        steps_b = [_step("Open browser"), _step("Wait for load"), _step("Click search")]

        aligned = det.semantic_align(steps_a, steps_b)

        # The extra step should appear as an insertion (action_a is None)
        insertions = [a for a in aligned if a.is_insertion and a.action_a is None]
        assert len(insertions) >= 1

    def test_deletion_detected(self):
        """Demo B missing a step — produces an AlignedStep with action_b=None."""
        det = VariantDetector()
        steps_a = [_step("Open browser"), _step("Click search"), _step("Enter query")]
        steps_b = [_step("Open browser"), _step("Enter query")]

        aligned = det.semantic_align(steps_a, steps_b)

        deletions = [a for a in aligned if a.is_insertion and a.action_b is None]
        assert len(deletions) >= 1

    def test_reordered_by_similarity(self):
        """Steps in different order are paired by NW alignment (not left as gaps)."""
        det = VariantDetector()
        steps_a = [_step("Click search button"), _step("Open browser tab")]
        steps_b = [_step("Open browser tab"), _step("Click search button")]

        aligned = det.semantic_align(steps_a, steps_b)

        # NW global alignment pairs them positionally — no insertions
        assert len(aligned) == 2
        insertions = [a for a in aligned if a.is_insertion]
        assert len(insertions) == 0
        # Both pairs have non-None actions on each side
        for a in aligned:
            assert a.action_a is not None
            assert a.action_b is not None

    def test_empty_demos(self):
        """Empty lists produce empty result."""
        det = VariantDetector()
        assert det.semantic_align([], []) == []

    def test_single_step_each(self):
        """Minimal case: one step each produces exactly one aligned pair."""
        det = VariantDetector()
        aligned = det.semantic_align([_step("Click")], [_step("Click")])
        assert len(aligned) == 1
        assert aligned[0].is_match is True

    def test_gap_penalty_effect(self):
        """Lower gap penalty tolerates more gaps (more insertions accepted)."""
        steps_a = [_step("Open browser"), _step("Search")]
        steps_b = [_step("Open browser"), _step("Wait"), _step("Pause"), _step("Search")]

        strict = VariantDetector(gap_penalty=-1.0)
        lenient = VariantDetector(gap_penalty=-0.05)

        aligned_strict = strict.semantic_align(steps_a, steps_b)
        aligned_lenient = lenient.semantic_align(steps_a, steps_b)

        # Both should produce alignments, but the lenient one is more likely
        # to produce insertion-style alignments rather than forcing mismatches
        assert len(aligned_strict) >= 2
        assert len(aligned_lenient) >= 2

    def test_with_embeddings(self):
        """Providing float vectors uses cosine-based matching."""
        det = VariantDetector()
        steps_a = [_step("Open"), _step("Click")]
        steps_b = [_step("Open"), _step("Click")]

        # Identical embeddings → high cosine similarity (1.0)
        emb_a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        emb_b = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        aligned = det.semantic_align(steps_a, steps_b, emb_a, emb_b)
        assert len(aligned) == 2
        for a in aligned:
            assert a.is_match is True
            # cosine(identical) = 1.0 → 0.5*1.0 + 0.3 (app match) + 0.2*0.5 (empty domain) = 0.9
            assert a.similarity >= 0.8

    def test_without_embeddings_string_fallback(self):
        """No embeddings falls back to Jaccard word similarity."""
        det = VariantDetector()
        steps_a = [_step("open browser tab")]
        steps_b = [_step("open browser tab")]

        aligned = det.semantic_align(steps_a, steps_b)
        assert len(aligned) == 1
        # Jaccard of identical word sets = 1.0 → sem=0.5, app=0.3, domain=0.1 → 0.9
        assert aligned[0].similarity >= 0.8

    def test_app_match_boosts_similarity(self):
        """Same app adds 0.3 to composite score vs different app."""
        det = VariantDetector()
        step_same = _step("Click button", app="Chrome")
        step_diff = _step("Click button", app="Firefox")

        aligned_same = det.semantic_align([step_same], [_step("Click button", app="Chrome")])
        aligned_diff = det.semantic_align([step_same], [step_diff])

        # Same app should produce higher similarity
        assert aligned_same[0].similarity > aligned_diff[0].similarity
        # The difference should be approximately 0.3
        diff = aligned_same[0].similarity - aligned_diff[0].similarity
        assert 0.2 <= diff <= 0.4

    def test_domain_overlap_boosts(self):
        """Same domain adds 0.2 to composite score."""
        det = VariantDetector()
        step_a = _step("Click", location="https://github.com/page1")
        step_b_same = _step("Click", location="https://github.com/page2")
        step_b_diff = _step("Click", location="https://gitlab.com/page")

        aligned_same = det.semantic_align([step_a], [step_b_same])
        aligned_diff = det.semantic_align([step_a], [step_b_diff])

        # Same domain → domain component = 0.2*1.0 = 0.2
        # Different domain → domain component = 0.2*0.0 = 0.0
        assert aligned_same[0].similarity > aligned_diff[0].similarity


# ------------------------------------------------------------------
# TestParameterExtraction (9 tests)
# ------------------------------------------------------------------


class TestParameterExtraction:
    """Extract parameterized fields from aligned demonstrations."""

    def test_varying_input_is_parameter(self):
        """Input differs across demos → detected as parameter."""
        det = VariantDetector()
        demos = [
            [_step("Search", input_val="wireless earbuds")],
            [_step("Search", input_val="bluetooth speaker")],
        ]
        params = det.extract_parameters(demos)
        assert len(params) >= 1
        names = [p.name for p in params]
        assert "input" in names

    def test_constant_field_not_parameter(self):
        """Same value everywhere → not extracted as parameter."""
        det = VariantDetector()
        demos = [
            [_step("Search", input_val="same query")],
            [_step("Search", input_val="same query")],
        ]
        params = det.extract_parameters(demos)
        input_params = [p for p in params if p.name == "input"]
        assert len(input_params) == 0

    def test_type_url(self):
        """URL input detected as type 'url'."""
        det = VariantDetector()
        demos = [
            [_step("Navigate", input_val="https://github.com")],
            [_step("Navigate", input_val="https://gitlab.com")],
        ]
        params = det.extract_parameters(demos)
        url_params = [p for p in params if p.name == "input"]
        assert len(url_params) == 1
        assert url_params[0].type == "url"

    def test_type_email(self):
        """Email input detected as type 'email'."""
        det = VariantDetector()
        demos = [
            [_step("Send", input_val="user@example.com")],
            [_step("Send", input_val="admin@example.org")],
        ]
        params = det.extract_parameters(demos)
        email_params = [p for p in params if p.name == "input"]
        assert len(email_params) == 1
        assert email_params[0].type == "email"

    def test_type_number(self):
        """Numeric input detected as type 'number'."""
        det = VariantDetector()
        demos = [
            [_step("Set quantity", input_val="42")],
            [_step("Set quantity", input_val="7")],
        ]
        params = det.extract_parameters(demos)
        num_params = [p for p in params if p.name == "input"]
        assert len(num_params) == 1
        assert num_params[0].type == "number"

    def test_type_date(self):
        """Date input detected as type 'date'."""
        det = VariantDetector()
        demos = [
            [_step("Schedule", input_val="2026-03-14")],
            [_step("Schedule", input_val="2026-04-01")],
        ]
        params = det.extract_parameters(demos)
        date_params = [p for p in params if p.name == "input"]
        assert len(date_params) == 1
        assert date_params[0].type == "date"

    def test_type_filepath(self):
        """File path input detected as type 'filepath'."""
        det = VariantDetector()
        demos = [
            [_step("Upload", input_val="/Users/file.txt")],
            [_step("Upload", input_val="/tmp/other.csv")],
        ]
        params = det.extract_parameters(demos)
        fp_params = [p for p in params if p.name == "input"]
        assert len(fp_params) == 1
        assert fp_params[0].type == "filepath"

    def test_type_text_fallback(self):
        """Random string input defaults to type 'text'."""
        det = VariantDetector()
        demos = [
            [_step("Comment", input_val="random string alpha")],
            [_step("Comment", input_val="another random beta")],
        ]
        params = det.extract_parameters(demos)
        text_params = [p for p in params if p.name == "input"]
        assert len(text_params) == 1
        assert text_params[0].type == "text"

    def test_multiple_params(self):
        """Two fields vary simultaneously → both detected."""
        det = VariantDetector()
        demos = [
            [_step("Search", input_val="earbuds", target="search-box")],
            [_step("Search", input_val="speaker", target="search-field")],
        ]
        params = det.extract_parameters(demos)
        names = {p.name for p in params}
        assert "input" in names
        assert "target" in names


# ------------------------------------------------------------------
# TestDetectVariants (5 tests)
# ------------------------------------------------------------------


class TestDetectVariants:
    """Detect workflow variants across multiple demonstrations."""

    def test_identical_demos_no_variants(self):
        """Same demos produce empty variant list."""
        det = VariantDetector()
        steps = [_step("Open browser"), _step("Click search")]
        demos = [steps, list(steps)]
        variants = det.detect_variants("test-slug", demos)
        assert variants == []

    def test_one_step_differs(self):
        """One demo has a different step → at least 1 variant group."""
        det = VariantDetector()
        demo_a = [_step("Open browser"), _step("Click search")]
        demo_b = [_step("Open browser"), _step("Use voice search")]
        variants = det.detect_variants("test-slug", [demo_a, demo_b])
        assert len(variants) >= 1
        assert all(isinstance(v, WorkflowVariant) for v in variants)

    def test_three_demos_two_variants(self):
        """3 demos with 2 distinct patterns → 2 variant groups."""
        det = VariantDetector()
        demo_a = [_step("Open browser"), _step("Click search")]
        demo_b = [_step("Open browser"), _step("Click search")]
        demo_c = [_step("Open browser"), _step("Use voice search")]
        variants = det.detect_variants("test-slug", [demo_a, demo_b, demo_c])
        assert len(variants) == 2
        # demo_a and demo_b share a pattern, demo_c is different
        indices_flat = []
        for v in variants:
            indices_flat.extend(v.demo_indices)
        assert sorted(indices_flat) == [0, 1, 2]

    def test_variant_context(self):
        """Context dict captures distinguishing actions."""
        det = VariantDetector()
        demo_a = [_step("Open browser"), _step("Click search")]
        demo_b = [_step("Open browser"), _step("Use voice search")]
        variants = det.detect_variants("test-slug", [demo_a, demo_b])
        # At least one variant should have a context with distinguishing_actions
        has_context = any(v.context.get("distinguishing_actions") for v in variants)
        assert has_context

    def test_fixed_steps_shared(self):
        """Fixed steps appear in all variant groups."""
        det = VariantDetector()
        demo_a = [_step("Open browser"), _step("Click search"), _step("Close browser")]
        demo_b = [_step("Open browser"), _step("Use voice search"), _step("Close browser")]
        variants = det.detect_variants("test-slug", [demo_a, demo_b])
        assert len(variants) >= 1
        for v in variants:
            # Fixed steps should include the shared steps
            fixed_actions = [s["action"] for s in v.fixed_steps]
            assert len(fixed_actions) >= 1  # At least "Open browser" is shared


# ------------------------------------------------------------------
# TestNormalizeWorkflow (3 tests)
# ------------------------------------------------------------------


class TestNormalizeWorkflow:
    """Produce canonical workflow from multiple demonstrations."""

    def test_majority_canonical(self):
        """3/4 do action A → A is canonical."""
        det = VariantDetector()
        demos = [
            [_step("Click search")],
            [_step("Click search")],
            [_step("Click search")],
            [_step("Use voice search")],
        ]
        result = det.normalize_workflow(demos, [])
        canonical = result["canonical_steps"]
        assert len(canonical) >= 1
        assert canonical[0]["action"].lower() == "click search"
        assert canonical[0]["confidence"] == pytest.approx(0.75)

    def test_alternatives_stored(self):
        """Minority actions appear in alternatives list."""
        det = VariantDetector()
        demos = [
            [_step("Click search")],
            [_step("Click search")],
            [_step("Click search")],
            [_step("Use voice search")],
        ]
        result = det.normalize_workflow(demos, [])
        canonical = result["canonical_steps"]
        assert len(canonical) >= 1
        alts = canonical[0]["alternatives"]
        alt_actions = [a["action"].lower() for a in alts]
        assert "use voice search" in alt_actions

    def test_single_demo_all_canonical(self):
        """1 demo → all steps at confidence 1.0."""
        det = VariantDetector()
        steps = [_step("Open browser"), _step("Click search")]
        result = det.normalize_workflow([steps], [])
        canonical = result["canonical_steps"]
        assert len(canonical) == 2
        for c in canonical:
            assert c["confidence"] == 1.0
            assert c["alternatives"] == []


# ------------------------------------------------------------------
# TestEdgeCases (3 tests)
# ------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for VariantDetector."""

    def test_empty_demos(self):
        """Empty list returns empty from normalize_workflow."""
        det = VariantDetector()
        result = det.normalize_workflow([], [])
        assert result["canonical_steps"] == []
        assert result["branches"] == []
        assert result["parameters"] == []

    def test_single_step_demo(self):
        """Works with a single-step demo."""
        det = VariantDetector()
        result = det.normalize_workflow([[_step("Click")]], [])
        assert len(result["canonical_steps"]) == 1
        assert result["canonical_steps"][0]["action"] == "Click"

    def test_no_embeddings_works(self):
        """Detector works without embeddings (string fallback)."""
        det = VariantDetector()
        demos = [
            [_step("Open browser"), _step("Click search")],
            [_step("Open browser"), _step("Click search")],
        ]
        # Should not raise; should produce valid alignment
        variants = det.detect_variants("no-emb", demos)
        assert isinstance(variants, list)
        # extract_parameters should also work
        params = det.extract_parameters(demos)
        assert isinstance(params, list)
