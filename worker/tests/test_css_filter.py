"""Tests for agenthandover_worker.css_filter.

Covers CSS rot detection for all framework patterns, semantic class
preservation, full selector cleaning, stability ranking, and best-selector
selection.
"""

from __future__ import annotations

from agenthandover_worker.css_filter import CSSRotFilter


# ------------------------------------------------------------------
# 1. Emotion classes detected
# ------------------------------------------------------------------


class TestEmotionClassesDetected:
    def test_emotion_short(self) -> None:
        """css-abc123 → detected as rot."""
        f = CSSRotFilter()
        assert f.is_css_rot("css-abc123") is True

    def test_emotion_long(self) -> None:
        """css-1a2b3c4d → detected as rot."""
        f = CSSRotFilter()
        assert f.is_css_rot("css-1a2b3c4d") is True

    def test_emotion_minimal(self) -> None:
        """css-a → detected as rot."""
        f = CSSRotFilter()
        assert f.is_css_rot("css-a") is True


# ------------------------------------------------------------------
# 2. styled-components classes detected
# ------------------------------------------------------------------


class TestStyledComponentsDetected:
    def test_sc_mixed_case(self) -> None:
        """sc-bdfBjE → detected as rot."""
        f = CSSRotFilter()
        assert f.is_css_rot("sc-bdfBjE") is True

    def test_sc_lowercase(self) -> None:
        """sc-abcdef → detected as rot."""
        f = CSSRotFilter()
        assert f.is_css_rot("sc-abcdef") is True

    def test_sc_uppercase(self) -> None:
        """sc-ABCDEF → detected as rot."""
        f = CSSRotFilter()
        assert f.is_css_rot("sc-ABCDEF") is True


# ------------------------------------------------------------------
# 3. CSS Modules classes detected
# ------------------------------------------------------------------


class TestCSSModulesDetected:
    def test_css_modules_standard(self) -> None:
        """styles_container__abc12 → detected as rot."""
        f = CSSRotFilter()
        assert f.is_css_rot("styles_container__abc12") is True

    def test_css_modules_different_prefix(self) -> None:
        """Header_wrapper__x9z → detected as rot."""
        f = CSSRotFilter()
        assert f.is_css_rot("Header_wrapper__x9z") is True


# ------------------------------------------------------------------
# 4. Generic hash classes detected
# ------------------------------------------------------------------


class TestGenericHashDetected:
    def test_generic_hash_short(self) -> None:
        """abc-1a2b3c → detected as rot."""
        f = CSSRotFilter()
        assert f.is_css_rot("abc-1a2b3c") is True

    def test_generic_hash_single_prefix(self) -> None:
        """a-1x2y3z4w → detected as rot."""
        f = CSSRotFilter()
        assert f.is_css_rot("a-1x2y3z4w") is True

    def test_generic_hash_four_prefix(self) -> None:
        """abcd-0aef1234 → detected as rot."""
        f = CSSRotFilter()
        assert f.is_css_rot("abcd-0aef1234") is True


# ------------------------------------------------------------------
# 5. Semantic classes kept
# ------------------------------------------------------------------


class TestSemanticClassesKept:
    def test_common_semantic_classes(self) -> None:
        """btn, header, nav, sidebar, container → NOT rot."""
        f = CSSRotFilter()
        for cls in ["btn", "header", "nav", "sidebar", "container", "main-content"]:
            assert f.is_css_rot(cls) is False, f"{cls!r} should not be flagged as rot"

    def test_tailwind_classes_kept(self) -> None:
        """Tailwind utility classes like flex, mt-4 → NOT rot."""
        f = CSSRotFilter()
        # Tailwind classes should not be flagged as rot.
        # "mt-4" doesn't match the generic hash pattern because "4" alone
        # is too short (pattern requires 4-12 chars after the dash).
        assert f.is_css_rot("flex") is False
        assert f.is_css_rot("mt-4") is False
        assert f.is_css_rot("text-center") is False

    def test_bem_classes_kept(self) -> None:
        """BEM classes like block__element--modifier → NOT rot."""
        f = CSSRotFilter()
        # BEM uses double underscores and double dashes
        assert f.is_css_rot("block__element--modifier") is False
        assert f.is_css_rot("card__title") is False

    def test_empty_string(self) -> None:
        """Empty class name → NOT rot."""
        f = CSSRotFilter()
        assert f.is_css_rot("") is False

    def test_strip_preserves_semantic(self) -> None:
        """strip_rot_classes keeps semantic, removes rot."""
        f = CSSRotFilter()
        class_list = ["btn", "css-abc123", "primary", "sc-bdfBjE", "sidebar"]
        result = f.strip_rot_classes(class_list)
        assert result == ["btn", "primary", "sidebar"]


# ------------------------------------------------------------------
# 6. Full selector cleaning
# ------------------------------------------------------------------


class TestCleanSelector:
    def test_clean_simple_selector(self) -> None:
        """Removes rot class from a simple selector."""
        f = CSSRotFilter()
        selector = "div.container.css-1a2b3c"
        cleaned = f.clean_selector(selector)
        assert "css-1a2b3c" not in cleaned
        assert "div" in cleaned
        assert "container" in cleaned

    def test_clean_nested_selector(self) -> None:
        """Removes rot from a nested descendant selector."""
        f = CSSRotFilter()
        selector = "div.wrapper.sc-abcXYZ > button.btn.css-xyz789"
        cleaned = f.clean_selector(selector)
        assert "sc-abcXYZ" not in cleaned
        assert "css-xyz789" not in cleaned
        assert "wrapper" in cleaned
        assert "btn" in cleaned

    def test_clean_preserves_id(self) -> None:
        """IDs are preserved during cleaning."""
        f = CSSRotFilter()
        selector = "div#main.css-abc123"
        cleaned = f.clean_selector(selector)
        assert "#main" in cleaned
        assert "css-abc123" not in cleaned

    def test_clean_empty_selector(self) -> None:
        """Empty selector → empty result."""
        f = CSSRotFilter()
        assert f.clean_selector("") == ""

    def test_clean_preserves_attribute_selectors(self) -> None:
        """Attribute selectors like [data-testid='x'] are preserved."""
        f = CSSRotFilter()
        selector = "button.css-abc123[aria-label='Submit']"
        cleaned = f.clean_selector(selector)
        assert "[aria-label='Submit']" in cleaned
        assert "css-abc123" not in cleaned


# ------------------------------------------------------------------
# 7. Selector stability ranking
# ------------------------------------------------------------------


class TestSelectorStabilityRanking:
    def test_aria_most_stable(self) -> None:
        """aria_label has rank 0 (most stable)."""
        f = CSSRotFilter()
        assert f.rank_selector_stability("aria_label") == 0

    def test_visible_text_second(self) -> None:
        """visible_text has rank 1."""
        f = CSSRotFilter()
        assert f.rank_selector_stability("visible_text") == 1

    def test_role_heading_third(self) -> None:
        """role_heading has rank 2."""
        f = CSSRotFilter()
        assert f.rank_selector_stability("role_heading") == 2

    def test_test_id_fourth(self) -> None:
        """test_id has rank 3."""
        f = CSSRotFilter()
        assert f.rank_selector_stability("test_id") == 3

    def test_vision_bbox_least_stable(self) -> None:
        """vision_bbox has rank 5 (least stable of known types)."""
        f = CSSRotFilter()
        assert f.rank_selector_stability("vision_bbox") == 5

    def test_unknown_type_worst(self) -> None:
        """Unknown types get worst rank."""
        f = CSSRotFilter()
        rank = f.rank_selector_stability("some_unknown_type")
        assert rank > f.rank_selector_stability("vision_bbox")

    def test_full_ordering(self) -> None:
        """Complete ordering: ARIA > text > role > testid > semantic > bbox."""
        f = CSSRotFilter()
        ranks = [
            f.rank_selector_stability("aria_label"),
            f.rank_selector_stability("visible_text"),
            f.rank_selector_stability("role_heading"),
            f.rank_selector_stability("test_id"),
            f.rank_selector_stability("semantic_class"),
            f.rank_selector_stability("vision_bbox"),
        ]
        assert ranks == sorted(ranks)
        # Each is unique
        assert len(set(ranks)) == len(ranks)


# ------------------------------------------------------------------
# 8. Best stable selector selection
# ------------------------------------------------------------------


class TestBestStableSelector:
    def test_picks_aria_over_inner_text(self) -> None:
        """Given both ARIA and text candidates, picks ARIA."""
        f = CSSRotFilter()
        candidates = [
            {"type": "visible_text", "value": "Submit", "selector": "text='submit'"},
            {"type": "aria_label", "value": "Submit review", "selector": "[aria-label='Submit review']"},
        ]
        best = f.best_stable_selector(candidates)
        assert best is not None
        assert best["type"] == "aria_label"

    def test_picks_text_over_bbox(self) -> None:
        """Given text and bbox, picks text."""
        f = CSSRotFilter()
        candidates = [
            {"type": "vision_bbox", "value": "450,320", "selector": "bbox(450,320)"},
            {"type": "visible_text", "value": "Click me", "selector": "text='click me'"},
        ]
        best = f.best_stable_selector(candidates)
        assert best is not None
        assert best["type"] == "visible_text"

    def test_picks_test_id_over_bbox(self) -> None:
        """Given test_id and bbox, picks test_id."""
        f = CSSRotFilter()
        candidates = [
            {"type": "vision_bbox", "value": "100,200", "selector": "bbox(100,200)"},
            {"type": "test_id", "value": "submit-btn", "selector": "[data-testid='submit-btn']"},
        ]
        best = f.best_stable_selector(candidates)
        assert best is not None
        assert best["type"] == "test_id"

    def test_empty_candidates(self) -> None:
        """Empty candidate list → None."""
        f = CSSRotFilter()
        assert f.best_stable_selector([]) is None

    def test_single_candidate(self) -> None:
        """Single candidate is returned regardless of type."""
        f = CSSRotFilter()
        candidates = [
            {"type": "vision_bbox", "value": "50,60", "selector": "bbox(50,60)"},
        ]
        best = f.best_stable_selector(candidates)
        assert best is not None
        assert best["type"] == "vision_bbox"

    def test_all_types_priority_selection(self) -> None:
        """With all types present, ARIA wins."""
        f = CSSRotFilter()
        candidates = [
            {"type": "vision_bbox", "value": "x", "selector": "bbox"},
            {"type": "semantic_class", "value": "y", "selector": ".btn"},
            {"type": "test_id", "value": "z", "selector": "[data-testid]"},
            {"type": "role_heading", "value": "w", "selector": "role"},
            {"type": "visible_text", "value": "v", "selector": "text"},
            {"type": "aria_label", "value": "u", "selector": "[aria-label]"},
        ]
        best = f.best_stable_selector(candidates)
        assert best is not None
        assert best["type"] == "aria_label"


# ------------------------------------------------------------------
# 9. Vue scoped detection
# ------------------------------------------------------------------


class TestVueScopedDetected:
    def test_vue_scoped_attribute(self) -> None:
        """data-v-abcdef12 → detected as rot."""
        f = CSSRotFilter()
        assert f.is_css_rot("data-v-abcdef12") is True

    def test_vue_scoped_short(self) -> None:
        """data-v-0a1b → detected as rot."""
        f = CSSRotFilter()
        assert f.is_css_rot("data-v-0a1b") is True
