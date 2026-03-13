"""Procedure writer — converts SOP templates to v3 procedures in the knowledge base.

Orchestrates:
1. SOP template → v3 procedure conversion (procedure_schema.py)
2. Evidence linking (evidence_tracker.py)
3. Knowledge base persistence (knowledge_base.py)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from oc_apprentice_worker.evidence_tracker import EvidenceTracker, ObservationEvidence
from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.procedure_schema import (
    sop_to_procedure,
    validate_procedure,
)

logger = logging.getLogger(__name__)


class ProcedureWriter:
    """Convert SOP templates → v3 procedures → knowledge base."""

    def __init__(
        self,
        kb: KnowledgeBase,
        evidence: EvidenceTracker,
    ) -> None:
        self._kb = kb
        self._evidence = evidence

    def write_procedure(
        self,
        sop_template: dict,
        source: str,
        source_id: str,
        *,
        event_count: int = 0,
        duration_minutes: int = 0,
    ) -> Path:
        """Convert an SOP template to a v3 procedure and save it.

        Args:
            sop_template: Internal SOP template dict (from sop_generator).
            source: Origin of the SOP — "focus" or "passive".
            source_id: Unique identifier for the session/segment.
            event_count: Number of events in the source observation.
            duration_minutes: Duration of the source observation.

        Returns:
            Path to the written procedure JSON file.
        """
        # Ensure source is set on the template
        sop_template.setdefault("source", source)

        # Convert to v3 procedure
        procedure = sop_to_procedure(sop_template)

        # Validate
        errors = validate_procedure(procedure)
        if errors:
            logger.warning(
                "Procedure '%s' has validation issues: %s",
                procedure.get("id", "unknown"),
                errors,
            )

        # Save to knowledge base
        path = self._kb.save_procedure(procedure)

        # Add evidence from this observation
        slug = procedure["id"]
        self._evidence.add_observation(
            slug,
            ObservationEvidence(
                date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                type=source,
                duration_minutes=duration_minutes,
                session_id=source_id,
                event_count=event_count,
            ),
        )

        logger.info(
            "Wrote procedure '%s' from %s session %s",
            slug, source, source_id,
        )
        return path

    def update_procedure(
        self,
        slug: str,
        new_template: dict,
        *,
        source: str = "passive",
        source_id: str = "",
        event_count: int = 0,
        duration_minutes: int = 0,
    ) -> Path:
        """Update an existing procedure with a new SOP template.

        Merges the new template's data while preserving existing
        evidence, staleness history, and constraints.

        Returns the path to the updated procedure file.
        """
        existing = self._kb.get_procedure(slug)

        # Convert new template to v3
        new_template.setdefault("source", source)
        new_proc = sop_to_procedure(new_template)

        if existing is not None:
            # Preserve accumulated data from the existing procedure
            new_proc["evidence"] = existing.get("evidence", new_proc["evidence"])
            new_proc["constraints"] = existing.get("constraints", new_proc["constraints"])
            new_proc["recurrence"] = existing.get("recurrence", new_proc["recurrence"])

            # Merge staleness: keep history, update last_observed
            old_staleness = existing.get("staleness", {})
            new_proc["staleness"]["confidence_trend"] = old_staleness.get(
                "confidence_trend", []
            )
            # Append new confidence to trend
            new_conf = new_proc.get("confidence_avg", 0.0)
            if new_conf > 0:
                new_proc["staleness"]["confidence_trend"].append(
                    round(new_conf, 4)
                )
            new_proc["staleness"]["last_observed"] = (
                datetime.now(timezone.utc).isoformat()
            )
            new_proc["staleness"]["drift_signals"] = old_staleness.get(
                "drift_signals", []
            )

            # Preserve branches from existing
            new_proc["branches"] = existing.get("branches", [])
            new_proc["expected_outcomes"] = existing.get("expected_outcomes", [])

        # Save
        path = self._kb.save_procedure(new_proc)

        # Add evidence
        if source_id:
            self._evidence.add_observation(
                slug,
                ObservationEvidence(
                    date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    type=source,
                    duration_minutes=duration_minutes,
                    session_id=source_id,
                    event_count=event_count,
                ),
            )

        logger.info("Updated procedure '%s'", slug)
        return path
