"""Tests for the v2 VLM-based SOP generator."""

from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock

from oc_apprentice_worker.sop_generator import (
    SOPGenerator,
    SOPGeneratorConfig,
    GeneratedSOP,
    _format_timeline_entry,
    _build_focus_prompt,
    _build_passive_prompt,
    _parse_sop_response,
    _vlm_sop_to_template,
    _generate_slug,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_annotation(
    app="Google Chrome",
    location="https://example.com",
    what_doing="Filling out a form",
    is_workflow=True,
    headings=None,
    labels=None,
    values=None,
):
    return {
        "app": app,
        "location": location,
        "visible_content": {
            "headings": headings or ["Example Page"],
            "labels": labels or ["Name", "Email", "Submit"],
            "values": values or ["John Doe", "john@example.com"],
        },
        "ui_state": {
            "active_element": "Email field",
            "modals_or_popups": "none",
            "scroll_position": "top",
        },
        "task_context": {
            "what_doing": what_doing,
            "likely_next": "Click submit button",
            "is_workflow": is_workflow,
        },
    }


def _make_diff(
    diff_type="action",
    actions=None,
    inputs=None,
    step_description="User typed email address",
):
    if diff_type == "action":
        return {
            "diff_type": "action",
            "actions": actions or ["Typed 'john@example.com' in Email field"],
            "inputs": inputs or [
                {"field": "Email", "value": "john@example.com"},
            ],
            "step_description": step_description,
        }
    elif diff_type == "app_switch":
        return {
            "diff_type": "app_switch",
            "from_app": "Finder",
            "to_app": "Google Chrome",
        }
    elif diff_type == "no_change":
        return {"diff_type": "no_change"}
    return {"diff_type": diff_type}


def _make_timeline(n=3):
    """Build a simple timeline with n frames."""
    timeline = []
    for i in range(n):
        timeline.append({
            "annotation": _make_annotation(
                what_doing=f"Step {i + 1} of the task",
                values=[f"value_{i}"],
            ),
            "diff": _make_diff(
                step_description=f"Action for step {i + 1}",
            ) if i > 0 else None,
            "timestamp": f"2026-03-03T10:00:{i:02d}Z",
            "app": "Google Chrome",
            "event_id": f"evt-{i}",
        })
    return timeline


def _make_vlm_sop_json():
    """Build a valid VLM SOP response JSON."""
    return {
        "title": "File Expense Report",
        "description": "Submit an expense report for a business expense.",
        "when_to_use": "After incurring a business expense that needs reimbursement.",
        "prerequisites": ["Expensify account access", "Receipt photo"],
        "steps": [
            {
                "step_number": 1,
                "action": "Navigate to Expensify new report page",
                "app": "Google Chrome",
                "location": "https://expensify.com/report/new",
                "input": "",
                "verify": "New Expense Report form is displayed",
            },
            {
                "step_number": 2,
                "action": "Enter expense title",
                "app": "Google Chrome",
                "location": "https://expensify.com/report/new",
                "input": "{{expense_title}}",
                "verify": "Title field shows the entered text",
            },
            {
                "step_number": 3,
                "action": "Enter expense amount",
                "app": "Google Chrome",
                "location": "https://expensify.com/report/new",
                "input": "{{amount}}",
                "verify": "Amount field is populated",
            },
        ],
        "success_criteria": ["Report submitted", "Confirmation page shown"],
        "variables": [
            {
                "name": "expense_title",
                "description": "Title of the expense",
                "example": "Uber ride to airport",
            },
            {
                "name": "amount",
                "description": "Dollar amount of the expense",
                "example": "$24.50",
            },
        ],
        "common_errors": [
            "Missing required fields → fill in all mandatory fields",
            "Receipt not attached → attach before submitting",
        ],
        "apps_involved": ["Google Chrome"],
    }


# ---------------------------------------------------------------------------
# TestFormatTimelineEntry
# ---------------------------------------------------------------------------

class TestFormatTimelineEntry:
    def test_basic_entry(self):
        text = _format_timeline_entry(
            0,
            _make_annotation(),
            None,
            "2026-03-03T10:00:00Z",
        )
        assert "Frame 1" in text
        assert "Google Chrome" in text
        assert "example.com" in text
        assert "Filling out a form" in text

    def test_with_action_diff(self):
        text = _format_timeline_entry(
            1,
            _make_annotation(),
            _make_diff(diff_type="action"),
            "2026-03-03T10:00:01Z",
        )
        assert "Actions since previous frame" in text
        assert "john@example.com" in text
        assert "Inputs:" in text

    def test_with_app_switch_diff(self):
        text = _format_timeline_entry(
            1,
            _make_annotation(),
            _make_diff(diff_type="app_switch"),
            "2026-03-03T10:00:01Z",
        )
        assert "Switched from Finder to Google Chrome" in text

    def test_with_no_change_diff(self):
        text = _format_timeline_entry(
            1,
            _make_annotation(),
            _make_diff(diff_type="no_change"),
            "2026-03-03T10:00:01Z",
        )
        assert "No visible change" in text

    def test_truncates_timestamp(self):
        text = _format_timeline_entry(
            0,
            _make_annotation(),
            None,
            "2026-03-03T14:30:45Z",
        )
        assert "14:30:45" in text

    def test_visible_content(self):
        text = _format_timeline_entry(
            0,
            _make_annotation(headings=["Dashboard"], labels=["Save", "Cancel"]),
            None,
            "2026-03-03T10:00:00Z",
        )
        assert "Headings: Dashboard" in text
        assert "Labels: Save, Cancel" in text

    def test_empty_annotation(self):
        """Empty annotation should not crash."""
        text = _format_timeline_entry(0, {}, None, "2026-03-03T10:00:00Z")
        assert "Frame 1" in text


# ---------------------------------------------------------------------------
# TestBuildFocusPrompt
# ---------------------------------------------------------------------------

class TestBuildFocusPrompt:
    def test_includes_title(self):
        timeline = _make_timeline(2)
        prompt = _build_focus_prompt("Expense Report", timeline)
        assert "Expense Report" in prompt

    def test_includes_frame_count(self):
        timeline = _make_timeline(5)
        prompt = _build_focus_prompt("Task", timeline)
        assert "5 frames" in prompt

    def test_includes_all_frames(self):
        timeline = _make_timeline(3)
        prompt = _build_focus_prompt("Task", timeline)
        assert "Frame 1" in prompt
        assert "Frame 2" in prompt
        assert "Frame 3" in prompt

    def test_empty_timeline(self):
        prompt = _build_focus_prompt("Task", [])
        assert "0 frames" in prompt


# ---------------------------------------------------------------------------
# TestBuildPassivePrompt
# ---------------------------------------------------------------------------

class TestBuildPassivePrompt:
    def test_includes_demo_count(self):
        demos = [_make_timeline(3), _make_timeline(4)]
        prompt = _build_passive_prompt(demos)
        assert "2 times" in prompt

    def test_includes_all_demonstrations(self):
        demos = [_make_timeline(2), _make_timeline(2)]
        prompt = _build_passive_prompt(demos)
        assert "Demonstration 1" in prompt
        assert "Demonstration 2" in prompt


# ---------------------------------------------------------------------------
# TestParseSopResponse
# ---------------------------------------------------------------------------

class TestParseSopResponse:
    def test_valid_json(self):
        sop = _make_vlm_sop_json()
        result = _parse_sop_response(json.dumps(sop))
        assert result is not None
        assert result["title"] == "File Expense Report"
        assert len(result["steps"]) == 3

    def test_with_markdown_fences(self):
        sop = _make_vlm_sop_json()
        raw = f"```json\n{json.dumps(sop)}\n```"
        result = _parse_sop_response(raw)
        assert result is not None
        assert result["title"] == "File Expense Report"

    def test_with_thinking_tags(self):
        sop = _make_vlm_sop_json()
        raw = f"<think>I need to analyze...</think>\n{json.dumps(sop)}"
        result = _parse_sop_response(raw)
        assert result is not None
        assert result["title"] == "File Expense Report"

    def test_with_preamble_text(self):
        sop = _make_vlm_sop_json()
        raw = f"Here is the SOP:\n{json.dumps(sop)}"
        result = _parse_sop_response(raw)
        assert result is not None

    def test_missing_title(self):
        result = _parse_sop_response('{"steps": []}')
        assert result is None

    def test_missing_steps(self):
        result = _parse_sop_response('{"title": "Test"}')
        assert result is None

    def test_invalid_json(self):
        result = _parse_sop_response("not json at all")
        assert result is None

    def test_empty_string(self):
        result = _parse_sop_response("")
        assert result is None

    def test_non_dict(self):
        result = _parse_sop_response('[1, 2, 3]')
        assert result is None


# ---------------------------------------------------------------------------
# TestVlmSopToTemplate
# ---------------------------------------------------------------------------

class TestVlmSopToTemplate:
    def test_focus_mode(self):
        vlm_sop = _make_vlm_sop_json()
        template = _vlm_sop_to_template(vlm_sop, mode="focus")
        assert template["source"] == "v2_focus_recording"
        # v2 confidence scoring is dynamic (computed from multi-signal formula)
        assert 0.0 < template["confidence_avg"] <= 1.0
        assert "confidence_breakdown" in template
        assert template["confidence_breakdown"]["focus_bonus"] == 0.10
        assert template["episode_count"] == 1
        assert template["abs_support"] == 1

    def test_passive_mode(self):
        vlm_sop = _make_vlm_sop_json()
        template = _vlm_sop_to_template(vlm_sop, mode="passive")
        assert template["source"] == "v2_passive_discovery"
        # v2 confidence scoring is dynamic
        assert 0.0 < template["confidence_avg"] <= 1.0
        assert "confidence_breakdown" in template
        assert template["confidence_breakdown"]["focus_bonus"] == 0.0
        assert template["abs_support"] == 2

    def test_title_override(self):
        vlm_sop = _make_vlm_sop_json()
        template = _vlm_sop_to_template(
            vlm_sop, title_override="My Custom Title"
        )
        assert template["title"] == "My Custom Title"

    def test_steps_conversion(self):
        vlm_sop = _make_vlm_sop_json()
        template = _vlm_sop_to_template(vlm_sop)
        assert len(template["steps"]) == 3
        step = template["steps"][0]
        assert step["step"] == "Navigate to Expensify new report page"
        assert step["parameters"]["app"] == "Google Chrome"
        assert step["parameters"]["location"] == "https://expensify.com/report/new"
        assert step["selector"] is None  # v2 SOPs are semantic

    def test_variables_conversion(self):
        vlm_sop = _make_vlm_sop_json()
        template = _vlm_sop_to_template(vlm_sop)
        assert len(template["variables"]) == 2
        assert template["variables"][0]["name"] == "expense_title"
        assert template["variables"][0]["example"] == "Uber ride to airport"

    def test_apps_involved(self):
        vlm_sop = _make_vlm_sop_json()
        template = _vlm_sop_to_template(vlm_sop)
        assert "Google Chrome" in template["apps_involved"]

    def test_apps_extracted_from_steps_if_missing(self):
        vlm_sop = _make_vlm_sop_json()
        vlm_sop.pop("apps_involved")
        template = _vlm_sop_to_template(vlm_sop)
        assert "Google Chrome" in template["apps_involved"]

    def test_task_description(self):
        vlm_sop = _make_vlm_sop_json()
        template = _vlm_sop_to_template(vlm_sop)
        assert "expense report" in template["task_description"].lower()

    def test_execution_overview(self):
        vlm_sop = _make_vlm_sop_json()
        template = _vlm_sop_to_template(vlm_sop)
        eo = template["execution_overview"]
        assert "when_to_use" in eo
        assert "success_criteria" in eo
        assert "common_errors" in eo
        assert "prerequisites" in eo

    def test_slug_generation(self):
        vlm_sop = _make_vlm_sop_json()
        template = _vlm_sop_to_template(vlm_sop)
        assert template["slug"] == "file-expense-report"


# ---------------------------------------------------------------------------
# TestGenerateSlug
# ---------------------------------------------------------------------------

class TestGenerateSlug:
    def test_basic(self):
        assert _generate_slug("File Expense Report") == "file-expense-report"

    def test_special_characters(self):
        assert _generate_slug("Create Bug Report (GitHub)") == "create-bug-report-github"

    def test_unicode(self):
        slug = _generate_slug("Über Report")
        assert "uber" in slug or "ber" in slug  # NFKD decomposition

    def test_empty(self):
        assert _generate_slug("") == "untitled"

    def test_truncation(self):
        long_title = "A" * 200
        slug = _generate_slug(long_title)
        assert len(slug) <= 80


# ---------------------------------------------------------------------------
# TestSOPGenerator (with mocked VLM)
# ---------------------------------------------------------------------------

class TestSOPGeneratorFocus:
    def test_generate_focus_success(self):
        """Successful focus SOP generation with mocked VLM."""
        vlm_sop = _make_vlm_sop_json()
        generator = SOPGenerator(SOPGeneratorConfig())

        with patch(
            "oc_apprentice_worker.sop_generator._call_ollama",
            return_value=(json.dumps(vlm_sop), 72.0),
        ):
            result = generator.generate_from_focus(
                _make_timeline(5), "File Expense Report"
            )

        assert result.success
        assert result.sop["title"] == "File Expense Report"
        assert len(result.sop["steps"]) == 3
        assert result.inference_time_seconds == 72.0
        assert result.sop["source"] == "v2_focus_recording"

    def test_generate_focus_empty_timeline(self):
        generator = SOPGenerator()
        result = generator.generate_from_focus([], "Empty Task")
        assert not result.success
        assert "Empty timeline" in result.error

    def test_generate_focus_no_annotations(self):
        """Timeline with no meaningful annotations."""
        timeline = [
            {"annotation": None, "diff": None, "timestamp": "t", "app": "x"}
        ]
        generator = SOPGenerator()
        result = generator.generate_from_focus(timeline, "Bad Task")
        assert not result.success
        assert "No annotated frames" in result.error

    def test_generate_focus_vlm_connection_error(self):
        generator = SOPGenerator()
        with patch(
            "oc_apprentice_worker.sop_generator._call_ollama",
            side_effect=ConnectionError("Ollama not running"),
        ):
            result = generator.generate_from_focus(
                _make_timeline(3), "Task"
            )
        assert not result.success
        assert "connection failed" in result.error.lower()

    def test_generate_focus_invalid_json_retry(self):
        """First VLM call returns invalid JSON, retry succeeds."""
        vlm_sop = _make_vlm_sop_json()
        call_count = {"n": 0}

        def mock_call(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ("This is not JSON", 10.0)
            return (json.dumps(vlm_sop), 20.0)

        generator = SOPGenerator()
        with patch(
            "oc_apprentice_worker.sop_generator._call_ollama",
            side_effect=mock_call,
        ):
            result = generator.generate_from_focus(
                _make_timeline(3), "Retry Task"
            )

        assert result.success
        assert result.inference_time_seconds == 30.0  # 10 + 20
        assert call_count["n"] == 2

    def test_generate_focus_both_attempts_fail(self):
        generator = SOPGenerator()
        with patch(
            "oc_apprentice_worker.sop_generator._call_ollama",
            return_value=("garbage output", 5.0),
        ):
            result = generator.generate_from_focus(
                _make_timeline(3), "Bad Task"
            )
        assert not result.success
        assert "Failed to parse" in result.error


class TestSOPGeneratorPassive:
    def test_generate_passive_success(self):
        vlm_sop = _make_vlm_sop_json()
        generator = SOPGenerator()
        demos = [_make_timeline(4), _make_timeline(5)]

        with patch(
            "oc_apprentice_worker.sop_generator._call_ollama",
            return_value=(json.dumps(vlm_sop), 100.0),
        ):
            result = generator.generate_from_passive(demos, "Expense Report")

        assert result.success
        assert result.sop["episode_count"] == 2
        assert result.sop["source"] == "v2_passive_discovery"
        # v2 confidence is dynamic — just verify it's reasonable
        assert 0.0 < result.sop["confidence_avg"] <= 1.0
        assert "confidence_breakdown" in result.sop

    def test_generate_passive_3_demos_higher_confidence(self):
        vlm_sop = _make_vlm_sop_json()
        generator = SOPGenerator()
        demos_2 = [_make_timeline(3)] * 2
        demos_3 = [_make_timeline(3)] * 3

        with patch(
            "oc_apprentice_worker.sop_generator._call_ollama",
            return_value=(json.dumps(vlm_sop), 100.0),
        ):
            result_2 = generator.generate_from_passive(demos_2, "Task2")

        with patch(
            "oc_apprentice_worker.sop_generator._call_ollama",
            return_value=(json.dumps(vlm_sop), 100.0),
        ):
            result_3 = generator.generate_from_passive(demos_3, "Task3")

        assert result_3.success
        # 3 demos should have higher demo_count component than 2 demos
        assert result_3.sop["confidence_avg"] > result_2.sop["confidence_avg"]
        assert result_3.sop["abs_support"] == 3

    def test_generate_passive_too_few_demos(self):
        generator = SOPGenerator()
        result = generator.generate_from_passive(
            [_make_timeline(3)], "Single Demo"
        )
        assert not result.success
        assert "at least 2" in result.error.lower()

    def test_generate_passive_vlm_failure(self):
        generator = SOPGenerator()
        demos = [_make_timeline(3)] * 2
        with patch(
            "oc_apprentice_worker.sop_generator._call_ollama",
            side_effect=RuntimeError("out of memory"),
        ):
            result = generator.generate_from_passive(demos, "Task")
        assert not result.success


# ---------------------------------------------------------------------------
# TestSOPGeneratorConfig
# ---------------------------------------------------------------------------

class TestSOPGeneratorConfig:
    def test_defaults(self):
        cfg = SOPGeneratorConfig()
        assert cfg.model == "qwen3.5:4b"
        assert cfg.num_predict == 8000
        assert cfg.timeout == 180.0

    def test_custom(self):
        cfg = SOPGeneratorConfig(model="custom:1b", num_predict=4000)
        assert cfg.model == "custom:1b"
        assert cfg.num_predict == 4000
