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
        lifecycle_manager=None,
    ) -> None:
        self._kb = kb
        self._evidence = evidence
        self._lifecycle = lifecycle_manager

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

        # Enrich environment from observation data
        self._enrich_environment(procedure, sop_template)

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

        # Auto-transition: observed -> draft on SOP generation
        if self._lifecycle is not None:
            try:
                from oc_apprentice_worker.lifecycle_manager import ProcedureLifecycle
                current = self._lifecycle.get_state(slug)
                if current == ProcedureLifecycle.OBSERVED:
                    self._lifecycle.transition(
                        slug, ProcedureLifecycle.DRAFT,
                        trigger="sop_generated",
                        actor="system",
                        reason=f"SOP generated from {source} session {source_id}",
                    )
            except Exception:
                logger.debug("Lifecycle transition failed for %s", slug, exc_info=True)

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

        # Enrich environment
        self._enrich_environment(new_proc, new_template)

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

            # Preserve lifecycle state and history
            new_proc["lifecycle_state"] = existing.get("lifecycle_state", "observed")
            new_proc["lifecycle_history"] = existing.get("lifecycle_history", [])
            new_proc["compiled_outputs"] = existing.get("compiled_outputs", {})

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

    # ------------------------------------------------------------------
    # Environment enrichment
    # ------------------------------------------------------------------

    def _enrich_environment(self, procedure: dict, sop_template: dict) -> None:
        """Enrich the procedure's environment field from observation data.

        Extracts:
        - required_apps from steps' app fields and apps_involved
        - URL patterns from steps' location fields
        - Account hints from URL patterns (e.g., github.com -> github account)
        """
        env = procedure.setdefault("environment", {
            "required_apps": [],
            "accounts": [],
            "setup_actions": [],
        })

        # Collect apps from steps
        apps = set(env.get("required_apps", []))
        urls: set[str] = set()

        for step in sop_template.get("steps", []):
            app = step.get("app", "") or step.get("parameters", {}).get("app", "")
            if app:
                apps.add(app)
            location = step.get("location", "") or step.get("target", "")
            if location and ("http://" in location or "https://" in location):
                urls.add(location)

        for app in sop_template.get("apps_involved", []):
            apps.add(app)

        env["required_apps"] = sorted(apps)

        # Extract account hints from URLs
        accounts: list[dict] = env.get("accounts", [])
        existing_services = {a.get("service") for a in accounts}

        _SERVICE_PATTERNS = {
            "github.com": "github",
            "gitlab.com": "gitlab",
            "slack.com": "slack",
            "notion.so": "notion",
            "figma.com": "figma",
            "stripe.com": "stripe",
            "aws.amazon.com": "aws",
            "console.cloud.google.com": "gcp",
            "portal.azure.com": "azure",
            "trello.com": "trello",
            "jira": "jira",
            "linear.app": "linear",
        }

        for url in urls:
            for pattern, service in _SERVICE_PATTERNS.items():
                if pattern in url and service not in existing_services:
                    account: dict = {"service": service, "identity": "", "url_pattern": url}
                    # Detect environment (prod/staging/test) from URL
                    url_lower = url.lower()
                    if "staging" in url_lower or "stg" in url_lower:
                        account["environment"] = "staging"
                    elif "test" in url_lower or "sandbox" in url_lower:
                        account["environment"] = "test"
                    elif "localhost" in url_lower or "127.0.0.1" in url_lower:
                        account["environment"] = "local"
                    else:
                        account["environment"] = "production"
                    accounts.append(account)
                    existing_services.add(service)

        env["accounts"] = accounts

    # ------------------------------------------------------------------
    # Cross-procedure composition
    # ------------------------------------------------------------------

    def enrich_chains(self, slug: str) -> None:
        """Detect and enrich chain relationships for a procedure.

        Scans daily summaries for co-occurrence patterns (procedure A
        commonly followed by procedure B) and updates the chain field.
        """
        proc = self._kb.get_procedure(slug)
        if proc is None:
            return

        chain = proc.get("chain", {
            "depends_on": [],
            "followed_by": [],
            "co_occurrence_count": 0,
            "can_compose": False,
        })

        # Scan recent daily summaries for task sequences
        dates = self._kb.list_daily_summaries(limit=14)
        sequence_counts: dict[str, int] = {}

        for date_str in dates:
            summary = self._kb.get_daily_summary(date_str)
            if summary is None:
                continue

            # Look for procedures_observed list in daily summary
            observed = summary.get("procedures_observed", [])
            if slug not in observed:
                continue

            # Find what comes after this slug
            try:
                idx = observed.index(slug)
                if idx + 1 < len(observed):
                    next_slug = observed[idx + 1]
                    sequence_counts[next_slug] = sequence_counts.get(next_slug, 0) + 1
            except (ValueError, IndexError):
                pass

        # Update followed_by with procedures that co-occur at least twice
        followed_by = []
        for next_slug, count in sorted(sequence_counts.items(), key=lambda x: -x[1]):
            if count >= 2:
                followed_by.append(next_slug)

        chain["followed_by"] = followed_by
        chain["co_occurrence_count"] = sum(sequence_counts.values())
        chain["can_compose"] = len(followed_by) > 0

        proc["chain"] = chain
        self._kb.save_procedure(proc)
