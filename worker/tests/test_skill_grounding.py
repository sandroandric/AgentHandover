"""WP-Q1 pins: what a Skill says is grounded in what was recorded.

From the live run of 2026-09-23 (qwen3.5:4b for every role): a typing
session in TextEdit became "Copy ... / Paste ..." steps with no clipboard
event recorded, a frame diff read TextEdit's font toolbar ("Halvetica
Ragular") as an edit, and the Skill used placeholders it never declared.
"""

from __future__ import annotations

from agenthandover_worker.procedure_schema import sop_to_procedure, validate_procedure
from agenthandover_worker.sop_generator import _vlm_sop_to_template
from agenthandover_worker.sop_linter import lint_sop


# ---------------------------------------------------------------------------
# Variables: every placeholder is declared, every declared variable is used
# ---------------------------------------------------------------------------


def _sop(steps: list[dict], variables: list[dict]) -> dict:
    return {"title": "Look up the IANA reserved domains page and note it",
            "steps": steps, "variables": variables}


def test_a_variable_wired_into_a_description_is_kept() -> None:
    """The run logged "Wired 4 variable substitution(s)" and shipped
    ``variables: []``: the wiring wrote into descriptions, which the
    unused-variable pass did not read, so it dropped every variable."""
    template = _vlm_sop_to_template(_sop(
        steps=[{"action": "Open TextEdit", "app": "TextEdit",
                "description": "Write the note about Reserved domains per RFC 2606.",
                "verify": "The note shows the text"}],
        variables=[{"name": "source_note", "example": "Reserved domains per RFC 2606",
                    "description": "What the note says"}],
    ))
    assert "{{source_note}}" in template["steps"][0]["description"]
    assert [v["name"] for v in template["variables"]] == ["source_note"]
    assert template["variables"][0]["example"] == "Reserved domains per RFC 2606"
    assert not [i for i in lint_sop(template).issues if "source_note" in i.message]
    assert validate_procedure(sop_to_procedure(template)) == []


def test_a_placeholder_the_model_did_not_declare_is_declared() -> None:
    template = _vlm_sop_to_template(_sop(
        steps=[{"action": "Type {{iana_reserved_domains_text}} into the document",
                "app": "TextEdit", "description": "The note body.", "verify": "Text shows"}],
        variables=[{"name": "unused_thing", "example": "never used here"}],
    ))
    assert [v["name"] for v in template["variables"]] == ["iana_reserved_domains_text"]
    declared = template["variables"][0]
    assert declared["example"] == ""  # not seen in the recording: unknown
    assert "iana reserved domains text" in declared["description"]
    assert lint_sop(template).valid


def test_the_wiring_reaches_the_step_text() -> None:
    """The internal template's action field is "step": the wiring pass
    looked for "action" only and never templated the step itself."""
    template = _vlm_sop_to_template(_sop(
        steps=[{"action": "Search for example.com", "app": "Safari",
                "description": "Find the page.", "verify": "Results show"}],
        variables=[{"name": "domain", "example": "example.com"}],
    ))
    assert template["steps"][0]["step"] == "Search for {{domain}}"
    assert [v["name"] for v in template["variables"]] == ["domain"]


def test_lint_and_validation_fail_on_an_undeclared_placeholder() -> None:
    proc = {
        "schema_version": "3.2.0", "id": "p", "title": "P",
        "steps": [
            {"action": "Paste", "verify": "The note reads {{source_note}}"},
            {"action": "Save", "description": "Keep {{ file_name }}", "verify": "Saved"},
        ],
        "inputs": [],
    }
    errors = validate_procedure(proc)
    assert any("{{source_note}}" in e and "{{file_name}}" in e for e in errors), errors
    assert {i.field for i in lint_sop(proc).errors} >= {
        "variables.source_note", "variables.file_name",
    }
    proc["inputs"] = [{"name": "source_note", "type": "text"},
                      {"name": "file_name", "type": "text"}]
    assert validate_procedure(proc) == []
    assert not [i for i in lint_sop(proc).errors if i.field.startswith("variables.")]
