"""Knowledge base export adapter — always-on adapter that writes
procedures to ~/.agenthandover/knowledge/.

Implements the SOPExportAdapter interface so it can be used as an
additional export target alongside OpenClaw, SKILL.md, etc.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agenthandover_worker.export_adapter import SOPExportAdapter
from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.procedure_schema import sop_to_procedure, validate_procedure

logger = logging.getLogger(__name__)


class KnowledgeBaseExportAdapter(SOPExportAdapter):
    """Always-on adapter that writes procedures to the knowledge base."""

    def __init__(self, kb: KnowledgeBase) -> None:
        self._kb = kb

    def write_sop(self, sop_template: dict) -> Path:
        """Convert SOP template to v3 procedure and save to KB."""
        procedure = sop_to_procedure(sop_template)
        errors = validate_procedure(procedure)
        if errors:
            logger.warning(
                "Procedure '%s' has validation issues: %s",
                procedure.get("id", "unknown"),
                errors,
            )
        return self._kb.save_procedure(procedure)

    def write_all_sops(self, sop_templates: list[dict]) -> list[Path]:
        """Convert and save multiple SOPs."""
        paths = []
        for template in sop_templates:
            path = self.write_sop(template)
            paths.append(path)
        return paths

    def write_metadata(self, metadata_type: str, data: dict) -> Path:
        """Write metadata to the knowledge base.

        Maps metadata_type to the appropriate KB method:
        - "profile" → update_profile
        - "triggers" → update_triggers
        - "constraints" → update_constraints
        - anything else → save as context
        """
        if metadata_type == "profile":
            self._kb.update_profile(data)
            return self._kb.root / "profile.json"
        elif metadata_type == "triggers":
            self._kb.update_triggers(data)
            return self._kb.root / "triggers.json"
        elif metadata_type == "constraints":
            self._kb.update_constraints(data)
            return self._kb.root / "constraints.json"
        elif metadata_type == "decisions":
            self._kb.update_decisions(data)
            return self._kb.root / "decisions.json"
        else:
            self._kb.update_context(metadata_type, data)
            return self._kb.root / "context" / f"{metadata_type}.json"

    def get_sops_dir(self) -> Path:
        """Return the procedures directory."""
        return self._kb.root / "procedures"

    def list_sops(self) -> list[dict]:
        """List all procedures with summary info."""
        procedures = self._kb.list_procedures()
        return [
            {
                "slug": p.get("id", ""),
                "title": p.get("title", ""),
                "path": str(self._kb.root / "procedures" / f"{p.get('id', '')}.json"),
                "confidence": p.get("confidence_avg", 0.0),
            }
            for p in procedures
        ]
