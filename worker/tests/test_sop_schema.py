"""Tests for sop_schema — versioned JSON export and validation."""

from __future__ import annotations

import json

import pytest

from oc_apprentice_worker.sop_schema import (
    SOP_SCHEMA_VERSION,
    _ACCEPTED_VERSIONS,
    sop_to_json,
    validate_sop_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_template(
    slug: str = "test_workflow",
    title: str = "Test Workflow",
    confidence_avg: float = 0.88,
    steps: list[dict] | None = None,
    variables: list[dict] | None = None,
    apps: list[str] | None = None,
    preconditions: list[str] | None = None,
    postconditions: list[str] | None = None,
    exceptions_seen: list[str] | None = None,
    tags: list[str] | None = None,
) -> dict:
    if steps is None:
        steps = [
            {
                "step": "click",
                "target": "Submit button",
                "selector": "[aria-label='Submit']",
                "parameters": {"text": "Hello"},
                "confidence": 0.9,
            },
            {
                "step": "type",
                "target": "Email field",
                "selector": "#email",
                "parameters": {"text": "user@example.com"},
                "confidence": 0.85,
            },
        ]
    if variables is None:
        variables = []
    if apps is None:
        apps = ["Chrome"]
    return {
        "slug": slug,
        "title": title,
        "steps": steps,
        "variables": variables,
        "confidence_avg": confidence_avg,
        "episode_count": 5,
        "apps_involved": apps,
        "preconditions": preconditions or [],
        "postconditions": postconditions or [],
        "exceptions_seen": exceptions_seen or [],
        "tags": tags or [],
    }


# ---------------------------------------------------------------------------
# sop_to_json — conversion
# ---------------------------------------------------------------------------


class TestSopToJsonTopLevel:
    """Top-level fields match the YAML frontmatter structure."""

    def test_schema_version(self):
        result = sop_to_json(_sample_template())
        assert result["schema_version"] == SOP_SCHEMA_VERSION
        assert result["schema_version"] == "1.1.0"

    def test_slug_and_title(self):
        result = sop_to_json(_sample_template(slug="my_slug", title="My Title"))
        assert result["slug"] == "my_slug"
        assert result["title"] == "My Title"

    def test_confidence_fields(self):
        result = sop_to_json(_sample_template(confidence_avg=0.92))
        assert result["confidence_avg"] == 0.92
        assert result["confidence_summary"] == "high"

    def test_confidence_summary_medium(self):
        result = sop_to_json(_sample_template(confidence_avg=0.70))
        assert result["confidence_summary"] == "medium"

    def test_confidence_summary_low(self):
        result = sop_to_json(_sample_template(confidence_avg=0.40))
        assert result["confidence_summary"] == "low"

    def test_episode_count(self):
        result = sop_to_json(_sample_template())
        assert result["episode_count"] == 5

    def test_evidence_window_default(self):
        result = sop_to_json(_sample_template())
        assert result["evidence_window"] == "last_30_days"

    def test_apps_involved(self):
        result = sop_to_json(_sample_template(apps=["Chrome", "Excel"]))
        assert result["apps_involved"] == ["Chrome", "Excel"]

    def test_generated_at_is_iso(self):
        result = sop_to_json(_sample_template())
        assert "generated_at" in result
        # Should be parseable as ISO 8601
        assert "T" in result["generated_at"]

    def test_metadata_generator(self):
        result = sop_to_json(_sample_template())
        assert result["metadata"]["generator"] == "openmimic"
        assert "generator_version" in result["metadata"]

    def test_preconditions(self):
        result = sop_to_json(_sample_template(preconditions=["app_open:Chrome"]))
        assert result["preconditions"] == ["app_open:Chrome"]

    def test_postconditions(self):
        result = sop_to_json(_sample_template(postconditions=["final_action:click"]))
        assert result["postconditions"] == ["final_action:click"]

    def test_exceptions_seen(self):
        result = sop_to_json(_sample_template(exceptions_seen=["error:timeout"]))
        assert result["exceptions_seen"] == ["error:timeout"]

    def test_tags(self):
        result = sop_to_json(_sample_template(tags=["billing", "customer"]))
        assert result["tags"] == ["billing", "customer"]


class TestSopToJsonDefaults:
    """Default values when template has missing keys."""

    def test_empty_template(self):
        result = sop_to_json({})
        assert result["slug"] == "unknown"
        assert result["title"] == "Untitled"
        assert result["description"] == ""
        assert result["confidence_avg"] == 0.0
        assert result["confidence_summary"] == "low"
        assert result["episode_count"] == 0
        assert result["apps_involved"] == []
        assert result["steps"] == []
        assert result["variables"] == []
        assert result["preconditions"] == []
        assert result["postconditions"] == []
        assert result["exceptions_seen"] == []
        assert result["tags"] == []


class TestSopToJsonSteps:
    """Step serialization preserves all fields."""

    def test_step_fields(self):
        result = sop_to_json(_sample_template())
        step = result["steps"][0]
        assert step["index"] == 0
        assert step["action"] == "click"
        assert step["target"] == "Submit button"
        assert step["selector"] == "[aria-label='Submit']"
        assert step["parameters"] == {"text": "Hello"}
        assert step["confidence"] == 0.9

    def test_step_pre_state_included(self):
        template = _sample_template(steps=[
            {
                "step": "click",
                "target": "Button",
                "confidence": 0.8,
                "pre_state": {"app": "Chrome", "url": "https://example.com"},
            },
        ])
        result = sop_to_json(template)
        assert result["steps"][0]["pre_state"]["app"] == "Chrome"

    def test_step_pre_state_omitted_when_empty(self):
        template = _sample_template(steps=[
            {"step": "click", "target": "Button", "confidence": 0.8},
        ])
        result = sop_to_json(template)
        assert "pre_state" not in result["steps"][0]

    def test_step_uses_action_key_fallback(self):
        template = _sample_template(steps=[
            {"action": "navigate", "target": "Home", "confidence": 0.7},
        ])
        result = sop_to_json(template)
        assert result["steps"][0]["action"] == "navigate"


class TestSopToJsonVariables:
    """Variable serialization with all type-specific fields."""

    def test_basic_string_variable(self):
        template = _sample_template(variables=[
            {"name": "customer_name", "type": "string", "example": "Alice"},
        ])
        result = sop_to_json(template)
        var = result["variables"][0]
        assert var["name"] == "customer_name"
        assert var["type"] == "string"
        assert var["example"] == "Alice"

    def test_number_variable_with_range(self):
        template = _sample_template(variables=[
            {"name": "amount", "type": "number", "example": "42", "min": 1.0, "max": 100.0},
        ])
        result = sop_to_json(template)
        var = result["variables"][0]
        assert var["type"] == "number"
        assert var["min"] == 1.0
        assert var["max"] == 100.0

    def test_enum_variable_with_choices(self):
        template = _sample_template(variables=[
            {"name": "status", "type": "enum", "example": "active", "choices": ["active", "inactive"]},
        ])
        result = sop_to_json(template)
        var = result["variables"][0]
        assert var["type"] == "enum"
        assert var["choices"] == ["active", "inactive"]

    def test_variable_description(self):
        template = _sample_template(variables=[
            {"name": "query", "type": "string", "description": "Search term"},
        ])
        result = sop_to_json(template)
        var = result["variables"][0]
        assert var["description"] == "Search term"

    def test_variable_omits_missing_optional_fields(self):
        template = _sample_template(variables=[
            {"name": "x", "type": "string"},
        ])
        result = sop_to_json(template)
        var = result["variables"][0]
        assert "example" not in var
        assert "description" not in var
        assert "min" not in var
        assert "max" not in var
        assert "choices" not in var


class TestSopToJsonSerialization:
    """Output is JSON-serializable."""

    def test_json_serializable(self):
        result = sop_to_json(_sample_template())
        serialized = json.dumps(result, default=str)
        roundtripped = json.loads(serialized)
        assert roundtripped["schema_version"] == SOP_SCHEMA_VERSION
        assert roundtripped["slug"] == "test_workflow"

    def test_confidence_avg_rounded(self):
        result = sop_to_json(_sample_template(confidence_avg=0.876543))
        assert result["confidence_avg"] == 0.8765


# ---------------------------------------------------------------------------
# validate_sop_json — validation
# ---------------------------------------------------------------------------


class TestValidateSopJsonValid:
    """Valid data passes validation."""

    def test_valid_sop_passes(self):
        data = sop_to_json(_sample_template())
        errors = validate_sop_json(data)
        assert errors == []

    def test_minimal_valid(self):
        data = {
            "schema_version": SOP_SCHEMA_VERSION,
            "slug": "test",
            "title": "Test",
            "steps": [],
        }
        assert validate_sop_json(data) == []


class TestValidateSopJsonMissingFields:
    """Missing required fields produce errors."""

    def test_missing_schema_version(self):
        data = {"slug": "x", "title": "x", "steps": []}
        errors = validate_sop_json(data)
        assert any("schema_version" in e for e in errors)

    def test_missing_slug(self):
        data = {"schema_version": SOP_SCHEMA_VERSION, "title": "x", "steps": []}
        errors = validate_sop_json(data)
        assert any("slug" in e for e in errors)

    def test_missing_title(self):
        data = {"schema_version": SOP_SCHEMA_VERSION, "slug": "x", "steps": []}
        errors = validate_sop_json(data)
        assert any("title" in e for e in errors)

    def test_missing_steps(self):
        data = {"schema_version": SOP_SCHEMA_VERSION, "slug": "x", "title": "x"}
        errors = validate_sop_json(data)
        assert any("steps" in e for e in errors)

    def test_all_missing(self):
        errors = validate_sop_json({})
        assert len(errors) == 4  # schema_version, slug, title, steps


class TestValidateSopJsonTypeChecks:
    """Type constraints on sub-structures."""

    def test_steps_not_a_list(self):
        data = {"schema_version": SOP_SCHEMA_VERSION, "slug": "x", "title": "x", "steps": "bad"}
        errors = validate_sop_json(data)
        assert any("steps" in e and "list" in e for e in errors)

    def test_step_missing_action(self):
        data = {
            "schema_version": SOP_SCHEMA_VERSION,
            "slug": "x",
            "title": "x",
            "steps": [{"target": "Button"}],
        }
        errors = validate_sop_json(data)
        assert any("steps[0]" in e and "action" in e for e in errors)

    def test_step_not_a_dict(self):
        data = {
            "schema_version": SOP_SCHEMA_VERSION,
            "slug": "x",
            "title": "x",
            "steps": ["not_a_dict"],
        }
        errors = validate_sop_json(data)
        assert any("steps[0]" in e and "dict" in e for e in errors)

    def test_variables_not_a_list(self):
        data = {
            "schema_version": SOP_SCHEMA_VERSION,
            "slug": "x",
            "title": "x",
            "steps": [],
            "variables": "bad",
        }
        errors = validate_sop_json(data)
        assert any("variables" in e and "list" in e for e in errors)

    def test_variable_missing_name(self):
        data = {
            "schema_version": SOP_SCHEMA_VERSION,
            "slug": "x",
            "title": "x",
            "steps": [],
            "variables": [{"type": "string"}],
        }
        errors = validate_sop_json(data)
        assert any("variables[0]" in e and "name" in e for e in errors)

    def test_variable_missing_type(self):
        data = {
            "schema_version": SOP_SCHEMA_VERSION,
            "slug": "x",
            "title": "x",
            "steps": [],
            "variables": [{"name": "x"}],
        }
        errors = validate_sop_json(data)
        assert any("variables[0]" in e and "type" in e for e in errors)

    def test_variable_not_a_dict(self):
        data = {
            "schema_version": SOP_SCHEMA_VERSION,
            "slug": "x",
            "title": "x",
            "steps": [],
            "variables": [42],
        }
        errors = validate_sop_json(data)
        assert any("variables[0]" in e and "dict" in e for e in errors)


class TestValidateSopJsonListFields:
    """List fields must be lists when present."""

    @pytest.mark.parametrize("field", [
        "apps_involved",
        "preconditions",
        "postconditions",
        "exceptions_seen",
        "tags",
    ])
    def test_list_field_not_a_list(self, field: str):
        data = {
            "schema_version": SOP_SCHEMA_VERSION,
            "slug": "x",
            "title": "x",
            "steps": [],
            field: "not_a_list",
        }
        errors = validate_sop_json(data)
        assert any(field in e and "list" in e for e in errors)


class TestValidateSopJsonSchemaVersion:
    """Schema version compatibility."""

    def test_wrong_version(self):
        data = {
            "schema_version": "99.0.0",
            "slug": "x",
            "title": "x",
            "steps": [],
        }
        errors = validate_sop_json(data)
        assert any("Unsupported schema version" in e for e in errors)

    def test_version_1_0_0_accepted(self):
        data = {
            "schema_version": "1.0.0",
            "slug": "x",
            "title": "x",
            "steps": [],
        }
        errors = validate_sop_json(data)
        assert not any("Unsupported schema version" in e for e in errors)

    def test_version_1_1_0_accepted(self):
        data = {
            "schema_version": "1.1.0",
            "slug": "x",
            "title": "x",
            "steps": [],
        }
        errors = validate_sop_json(data)
        assert not any("Unsupported schema version" in e for e in errors)

    def test_accepted_versions_contains_both(self):
        assert "1.0.0" in _ACCEPTED_VERSIONS
        assert "1.1.0" in _ACCEPTED_VERSIONS


class TestValidateSopJsonConfidenceSummary:
    """Confidence summary must be a known label."""

    def test_invalid_confidence_summary(self):
        data = {
            "schema_version": SOP_SCHEMA_VERSION,
            "slug": "x",
            "title": "x",
            "steps": [],
            "confidence_summary": "very_high",
        }
        errors = validate_sop_json(data)
        assert any("confidence_summary" in e for e in errors)

    @pytest.mark.parametrize("label", ["high", "medium", "low"])
    def test_valid_confidence_summary(self, label: str):
        data = {
            "schema_version": SOP_SCHEMA_VERSION,
            "slug": "x",
            "title": "x",
            "steps": [],
            "confidence_summary": label,
        }
        errors = validate_sop_json(data)
        assert not any("confidence_summary" in e for e in errors)


class TestFrontmatterParity:
    """JSON output contains every field present in YAML frontmatter."""

    def test_all_frontmatter_fields_present(self):
        template = _sample_template(
            preconditions=["app_open:Chrome"],
            postconditions=["final_action:save"],
            exceptions_seen=["error:timeout"],
            tags=["billing"],
        )
        result = sop_to_json(template)
        # Every field from _build_frontmatter has a JSON counterpart
        assert "schema_version" in result          # sop_version
        assert "slug" in result                    # sop_slug
        assert "title" in result                   # sop_title
        assert "generated_at" in result            # generated_at
        assert "evidence_window" in result         # evidence_window
        assert "confidence_summary" in result      # confidence_summary
        assert "confidence_avg" in result          # confidence_score_avg
        assert "apps_involved" in result           # apps_involved
        assert "variables" in result               # input_variables
        assert "preconditions" in result           # preconditions
        assert "postconditions" in result          # postconditions
        assert "exceptions_seen" in result         # exceptions_seen
        assert "tags" in result                    # tags
        assert "steps" in result                   # (body content)
        assert "metadata" in result                # generator info


# ---------------------------------------------------------------------------
# LLM-enhanced fields (schema 1.1.0)
# ---------------------------------------------------------------------------


class TestSopToJsonEnhancedFields:
    """LLM-enhanced task_description and execution_overview in JSON export."""

    def test_task_description_included(self):
        template = _sample_template()
        template["task_description"] = "This workflow submits a form."
        result = sop_to_json(template)
        assert result["task_description"] == "This workflow submits a form."

    def test_execution_overview_included(self):
        template = _sample_template()
        template["execution_overview"] = {
            "goal": "Submit the contact form",
            "prerequisites": "Browser open",
        }
        result = sop_to_json(template)
        assert result["execution_overview"]["goal"] == "Submit the contact form"
        assert result["execution_overview"]["prerequisites"] == "Browser open"

    def test_enhanced_fields_omitted_when_absent(self):
        result = sop_to_json(_sample_template())
        assert "task_description" not in result
        assert "execution_overview" not in result

    def test_empty_task_description_omitted(self):
        template = _sample_template()
        template["task_description"] = ""
        result = sop_to_json(template)
        # Falsy string should not be included
        assert "task_description" not in result

    def test_empty_execution_overview_omitted(self):
        template = _sample_template()
        template["execution_overview"] = {}
        result = sop_to_json(template)
        assert "execution_overview" not in result

    def test_non_dict_execution_overview_omitted(self):
        template = _sample_template()
        template["execution_overview"] = "not a dict"
        result = sop_to_json(template)
        assert "execution_overview" not in result

    def test_both_enhanced_fields_together(self):
        template = _sample_template()
        template["task_description"] = "Fills and submits form."
        template["execution_overview"] = {
            "goal": "Submit form",
            "typical_duration": "30 seconds",
        }
        result = sop_to_json(template)
        assert "task_description" in result
        assert "execution_overview" in result

    def test_enhanced_json_serializable(self):
        template = _sample_template()
        template["task_description"] = "Test desc"
        template["execution_overview"] = {"goal": "Test"}
        result = sop_to_json(template)
        serialized = json.dumps(result, default=str)
        roundtripped = json.loads(serialized)
        assert roundtripped["task_description"] == "Test desc"


class TestValidateSopJsonEnhancedFields:
    """Validation of LLM-enhanced optional fields."""

    def test_valid_task_description(self):
        data = {
            "schema_version": "1.1.0",
            "slug": "x",
            "title": "x",
            "steps": [],
            "task_description": "A valid description.",
        }
        errors = validate_sop_json(data)
        assert not any("task_description" in e for e in errors)

    def test_invalid_task_description_type(self):
        data = {
            "schema_version": "1.1.0",
            "slug": "x",
            "title": "x",
            "steps": [],
            "task_description": 42,
        }
        errors = validate_sop_json(data)
        assert any("task_description" in e for e in errors)

    def test_valid_execution_overview(self):
        data = {
            "schema_version": "1.1.0",
            "slug": "x",
            "title": "x",
            "steps": [],
            "execution_overview": {"goal": "Test goal", "typical_duration": "1 min"},
        }
        errors = validate_sop_json(data)
        assert not any("execution_overview" in e for e in errors)

    def test_invalid_execution_overview_type(self):
        data = {
            "schema_version": "1.1.0",
            "slug": "x",
            "title": "x",
            "steps": [],
            "execution_overview": "not a dict",
        }
        errors = validate_sop_json(data)
        assert any("execution_overview" in e for e in errors)

    def test_execution_overview_non_string_value(self):
        data = {
            "schema_version": "1.1.0",
            "slug": "x",
            "title": "x",
            "steps": [],
            "execution_overview": {"goal": 42},
        }
        errors = validate_sop_json(data)
        assert any("execution_overview" in e and "goal" in e for e in errors)

    def test_enhanced_fields_optional(self):
        """Enhanced fields are optional — absence is not an error."""
        data = {
            "schema_version": "1.1.0",
            "slug": "x",
            "title": "x",
            "steps": [],
        }
        errors = validate_sop_json(data)
        assert errors == []
