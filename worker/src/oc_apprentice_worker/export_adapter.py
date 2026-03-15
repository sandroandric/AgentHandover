"""Abstract base class for SOP export adapters.

Defines the contract that all SOP export adapters must implement.
This allows OpenMimic to support multiple output targets (OpenClaw,
generic filesystem, future cloud backends, etc.) through a pluggable
adapter pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


def procedure_to_sop_template(procedure: dict) -> dict:
    """Reverse-map a v3 procedure dict to an SOP template dict.

    This allows export adapters that only understand SOP templates to
    render v3 procedures via their existing write_sop() method.
    """
    steps = []
    for s in procedure.get("steps", []):
        step = {
            "step": s.get("action", ""),
            "target": s.get("target", ""),
            "app": s.get("app", ""),
            "location": s.get("location", ""),
            "input": s.get("input", ""),
            "verify": s.get("verify", ""),
            "confidence": s.get("confidence", 0.0),
            "parameters": s.get("parameters", {}),
        }
        if s.get("selector"):
            step["selector"] = s["selector"]
        if s.get("pre_state"):
            step["pre_state"] = s["pre_state"]
        steps.append(step)

    # Reverse-map inputs to variables
    variables = []
    for inp in procedure.get("inputs", []):
        var = {
            "name": inp.get("name", ""),
            "type": inp.get("type", "string"),
        }
        if inp.get("description"):
            var["description"] = inp["description"]
        if inp.get("example"):
            var["example"] = inp["example"]
        if inp.get("default") is not None:
            var["default"] = inp["default"]
        variables.append(var)

    return {
        "slug": procedure.get("id", "unknown"),
        "title": procedure.get("title", "Untitled"),
        "short_title": procedure.get("short_title", ""),
        "description": procedure.get("description", ""),
        "tags": procedure.get("tags", []),
        "steps": steps,
        "variables": variables,
        "confidence_avg": procedure.get("confidence_avg", 0.0),
        "episode_count": procedure.get("episode_count", 0),
        "apps_involved": procedure.get("apps_involved", []),
        "preconditions": procedure.get("preconditions", []),
        "postconditions": procedure.get("postconditions", []),
        "exceptions_seen": procedure.get("exceptions_seen", []),
        "source": procedure.get("source", "unknown"),
        "task_description": procedure.get("task_description", ""),
        "execution_overview": procedure.get("execution_overview", {}),
        "outcome": procedure.get("outcome", ""),
        "when_to_use": procedure.get("when_to_use", ""),
        "evidence_window": procedure.get("evidence_window", "last_30_days"),
    }


class SOPExportAdapter(ABC):
    """Abstract base for SOP export adapters.

    All adapters must implement these methods to write SOPs,
    metadata, and provide directory information.
    """

    @abstractmethod
    def write_sop(self, sop_template: dict) -> Path:
        """Write a single SOP and return the path to the written file."""
        ...

    def write_procedure(self, procedure: dict) -> Path:
        """Write a v3 procedure to the export target.

        Default implementation reverse-maps to SOP template format and
        delegates to write_sop(). Subclasses may override for richer
        v3-aware rendering.
        """
        sop_template = procedure_to_sop_template(procedure)
        return self.write_sop(sop_template)

    @abstractmethod
    def write_all_sops(self, sop_templates: list[dict]) -> list[Path]:
        """Write multiple SOPs and return paths to all written files."""
        ...

    @abstractmethod
    def write_metadata(self, metadata_type: str, data: dict) -> Path:
        """Write a metadata file and return its path."""
        ...

    @abstractmethod
    def get_sops_dir(self) -> Path:
        """Return the directory where SOPs are stored."""
        ...

    @abstractmethod
    def list_sops(self) -> list[dict]:
        """List all SOPs with summary info (slug, title, path, confidence)."""
        ...
