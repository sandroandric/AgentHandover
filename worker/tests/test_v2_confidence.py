"""Tests for v2 SOP-level confidence scoring.

Tests cover:
- _score_demo_count: scaling from 0 demos through 5+
- _score_step_consistency: single demo, multi-demo, partial match
- _score_annotation_quality: empty, partial, full annotations
- _score_variable_detection: focus mode, passive mode, varying values
- compute_v2_confidence: full integration with weights + focus bonus
- V2ConfidenceBreakdown: dataclass fields and reasons
"""

from __future__ import annotations

import pytest

from agenthandover_worker.confidence import (
    V2ConfidenceBreakdown,
    _FOCUS_BONUS,
    _W_ANNOTATION_QUALITY,
    _W_DEMO_COUNT,
    _W_STEP_CONSISTENCY,
    _W_VARIABLE_DETECTION,
    _score_annotation_quality,
    _score_demo_count,
    _score_step_consistency,
    _score_variable_detection,
    compute_v2_confidence,
)


# =====================================================================
# _score_demo_count
# =====================================================================

class TestScoreDemoCount:
    def test_zero_demos(self):
        assert _score_demo_count(0) == 0.0

    def test_one_demo(self):
        assert _score_demo_count(1) == 0.50

    def test_two_demos(self):
        assert _score_demo_count(2) == 0.70

    def test_three_demos(self):
        assert _score_demo_count(3) == 0.90

    def test_four_demos(self):
        score = _score_demo_count(4)
        assert score == pytest.approx(0.925, abs=0.001)

    def test_ten_demos_capped_at_one(self):
        score = _score_demo_count(10)
        assert score <= 1.0

    def test_negative_demos(self):
        assert _score_demo_count(-1) == 0.0


# =====================================================================
# _score_step_consistency
# =====================================================================

class TestScoreStepConsistency:
    def test_empty_steps(self):
        assert _score_step_consistency([], None) == 0.0

    def test_single_demo_returns_one(self):
        steps = [{"step": "click button"}]
        assert _score_step_consistency(steps, None) == 1.0

    def test_single_demo_list_returns_one(self):
        steps = [{"step": "type email"}]
        demos = [[{"annotation": {}, "diff": {}}]]
        assert _score_step_consistency(steps, demos) == 1.0

    def test_multi_demo_all_consistent(self):
        steps = [{"step": "navigate home"}]
        frame = {
            "annotation": {
                "task_context": {"what_doing": "navigate home page"},
            },
            "diff": {},
        }
        demos = [[frame], [frame], [frame]]
        score = _score_step_consistency(steps, demos)
        assert score == 1.0

    def test_multi_demo_partial_consistency(self):
        steps = [
            {"step": "navigate home"},
            {"step": "fill form"},
        ]
        frame_a = {
            "annotation": {
                "task_context": {"what_doing": "navigate to home page"},
            },
            "diff": {},
        }
        frame_b = {
            "annotation": {
                "task_context": {"what_doing": "fill out the form"},
            },
            "diff": {},
        }
        frame_c = {
            "annotation": {
                "task_context": {"what_doing": "reading documentation"},
            },
            "diff": {},
        }
        # Demo 1 has both steps, demo 2 only has navigate
        demos = [
            [frame_a, frame_b],
            [frame_a, frame_c],
        ]
        score = _score_step_consistency(steps, demos)
        # "navigate" matches both demos, "fill" only matches demo 1
        assert 0.0 < score < 1.0

    def test_no_matching_steps(self):
        steps = [{"step": "deploy to production"}]
        frame = {
            "annotation": {
                "task_context": {"what_doing": "reading email"},
            },
            "diff": {},
        }
        demos = [[frame], [frame]]
        score = _score_step_consistency(steps, demos)
        assert score == 0.0


# =====================================================================
# _score_annotation_quality
# =====================================================================

class TestScoreAnnotationQuality:
    def test_none_annotations(self):
        assert _score_annotation_quality(None) == 0.5

    def test_empty_list(self):
        assert _score_annotation_quality([]) == 0.5

    def test_full_annotation(self):
        ann = {
            "app": "Chrome",
            "location": "https://example.com",
            "visible_content": {"headings": ["Title"]},
            "ui_state": {"active_element": "input"},
            "task_context": {"what_doing": "filling form"},
        }
        score = _score_annotation_quality([ann])
        assert score == 1.0

    def test_partial_annotation(self):
        ann = {
            "app": "Chrome",
            "location": "https://example.com",
            # Missing visible_content, ui_state, task_context
        }
        score = _score_annotation_quality([ann])
        assert 0.0 < score < 1.0
        assert score == pytest.approx(2 / 5, abs=0.01)

    def test_empty_dict_not_counted(self):
        ann = {
            "app": "Chrome",
            "location": "",
            "visible_content": {},
            "ui_state": {},
            "task_context": {},
        }
        score = _score_annotation_quality([ann])
        # Only "app" is non-empty; location is empty string (falsy),
        # others are empty dicts
        assert score == pytest.approx(1 / 5, abs=0.01)

    def test_multiple_annotations_averaged(self):
        full = {
            "app": "Chrome",
            "location": "https://example.com",
            "visible_content": {"h": ["x"]},
            "ui_state": {"a": "b"},
            "task_context": {"what_doing": "x"},
        }
        empty = {"app": "Chrome"}
        score = _score_annotation_quality([full, empty])
        # (5/5 + 1/5) / 2 = 0.6
        assert score == pytest.approx(0.6, abs=0.01)

    def test_non_dict_item_skipped(self):
        score = _score_annotation_quality(["not_a_dict", None])
        # All items skipped → falls back to conservative 0.5
        # Actually 0 counted / 2 = 0.0
        assert score == pytest.approx(0.0, abs=0.01)


# =====================================================================
# _score_variable_detection
# =====================================================================

class TestScoreVariableDetection:
    def test_focus_with_variables(self):
        variables = [{"name": "amount"}, {"name": "category"}]
        score = _score_variable_detection(variables, None)
        assert score >= 0.5

    def test_focus_no_variables(self):
        score = _score_variable_detection([], None)
        assert score == 0.3

    def test_passive_no_varying_values_no_vars(self):
        # No inputs in diffs → no varying values → 0.8 (correct: task is constant)
        demos = [
            [{"annotation": {}, "diff": {}}],
            [{"annotation": {}, "diff": {}}],
        ]
        score = _score_variable_detection([], demos)
        assert score == 0.8

    def test_passive_all_detected(self):
        # Two demos with different "Amount" values → should detect {{amount}}
        demos = [
            [{"annotation": {}, "diff": {"inputs": [{"field": "Amount", "value": "50"}]}}],
            [{"annotation": {}, "diff": {"inputs": [{"field": "Amount", "value": "75"}]}}],
        ]
        variables = [{"name": "amount"}]
        score = _score_variable_detection(variables, demos)
        assert score == 1.0

    def test_passive_partial_detection(self):
        demos = [
            [{"annotation": {}, "diff": {"inputs": [
                {"field": "Amount", "value": "50"},
                {"field": "Category", "value": "Travel"},
            ]}}],
            [{"annotation": {}, "diff": {"inputs": [
                {"field": "Amount", "value": "75"},
                {"field": "Category", "value": "Food"},
            ]}}],
        ]
        # Only amount detected, not category
        variables = [{"name": "amount"}]
        score = _score_variable_detection(variables, demos)
        assert score == pytest.approx(0.5, abs=0.01)


# =====================================================================
# compute_v2_confidence — integration
# =====================================================================

class TestComputeV2Confidence:
    def test_basic_focus_sop(self):
        sop = {
            "episode_count": 1,
            "steps": [{"step": "click button"}],
            "variables": [{"name": "url"}],
        }
        result = compute_v2_confidence(sop, is_focus=True)
        assert isinstance(result, V2ConfidenceBreakdown)
        assert result.focus_bonus == _FOCUS_BONUS
        assert result.total > 0.0
        assert result.total <= 1.0
        assert "focus_recording_bonus" in result.reasons

    def test_passive_two_demos(self):
        sop = {
            "episode_count": 2,
            "steps": [{"step": "submit form"}],
            "variables": [],
        }
        result = compute_v2_confidence(sop, is_focus=False)
        assert result.focus_bonus == 0.0
        assert result.demo_count_score == pytest.approx(0.70 * _W_DEMO_COUNT, abs=0.01)
        assert "demos=2" in result.reasons

    def test_passive_three_demos_higher(self):
        sop_2 = {"episode_count": 2, "steps": [{"step": "x"}], "variables": []}
        sop_3 = {"episode_count": 3, "steps": [{"step": "x"}], "variables": []}
        r2 = compute_v2_confidence(sop_2, is_focus=False)
        r3 = compute_v2_confidence(sop_3, is_focus=False)
        assert r3.total > r2.total

    def test_weights_sum_to_one(self):
        total = _W_DEMO_COUNT + _W_STEP_CONSISTENCY + _W_ANNOTATION_QUALITY + _W_VARIABLE_DETECTION
        assert total == pytest.approx(1.0)

    def test_total_capped_at_one(self):
        sop = {
            "episode_count": 10,
            "steps": [{"step": "x"}],
            "variables": [{"name": "x"}, {"name": "y"}, {"name": "z"}],
        }
        result = compute_v2_confidence(sop, is_focus=True)
        assert result.total <= 1.0

    def test_with_demonstrations_data(self):
        frame = {
            "annotation": {
                "app": "Chrome",
                "location": "https://example.com",
                "visible_content": {"headings": ["Title"]},
                "ui_state": {"active": "input"},
                "task_context": {"what_doing": "filling form"},
            },
            "diff": {"inputs": [{"field": "name", "value": "Alice"}]},
        }
        frame2 = {
            "annotation": {
                "app": "Chrome",
                "location": "https://example.com",
                "visible_content": {"headings": ["Title"]},
                "ui_state": {"active": "input"},
                "task_context": {"what_doing": "filling form"},
            },
            "diff": {"inputs": [{"field": "name", "value": "Bob"}]},
        }
        sop = {
            "episode_count": 2,
            "steps": [{"step": "filling form"}],
            "variables": [{"name": "name"}],
        }
        result = compute_v2_confidence(
            sop,
            demonstrations=[[frame], [frame2]],
            annotations=[frame["annotation"], frame2["annotation"]],
            is_focus=False,
        )
        assert result.total > 0.0
        assert result.annotation_quality_score > 0.0
        assert "high_annotation_quality" in result.reasons

    def test_empty_sop(self):
        sop = {"episode_count": 0, "steps": [], "variables": []}
        result = compute_v2_confidence(sop)
        assert result.total >= 0.0

    def test_reasons_populated(self):
        sop = {
            "episode_count": 1,
            "steps": [{"step": "click"}],
            "variables": [],
        }
        result = compute_v2_confidence(sop, is_focus=True)
        assert len(result.reasons) > 0
        assert any("demos=" in r for r in result.reasons)

    def test_breakdown_fields_rounded(self):
        sop = {
            "episode_count": 2,
            "steps": [{"step": "click"}],
            "variables": [],
        }
        result = compute_v2_confidence(sop)
        # All scores should be rounded to 4 decimal places
        assert result.total == round(result.total, 4)
        assert result.demo_count_score == round(result.demo_count_score, 4)
