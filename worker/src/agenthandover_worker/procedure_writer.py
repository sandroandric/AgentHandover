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

from agenthandover_worker.evidence_tracker import EvidenceTracker, ObservationEvidence
from agenthandover_worker.inference_runner import model_work
from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.procedure_schema import (
    sop_to_procedure,
    validate_procedure,
)

logger = logging.getLogger(__name__)


# What an agent run writes onto a procedure (``skill_improver``).
_EXECUTION_FIELDS = ("execution_stats", "execution_confidence", "observed_alternatives")


def carry_run_history(new_proc: dict, existing: dict) -> dict:
    """Carry the agent-run record and staleness history of *existing* over.

    The agent-run fields (``execution_stats``, ``execution_confidence``,
    ``observed_alternatives``), ``staleness.last_confirmed``, the drift
    signals and the confidence trend (with *new_proc*'s confidence
    appended when it differs from the last entry).  Mutates and returns
    *new_proc*.
    """
    if new_proc is existing:
        return new_proc
    for key in _EXECUTION_FIELDS:
        if key in existing:
            new_proc[key] = existing[key]
    old_staleness = existing.get("staleness") or {}
    staleness = new_proc.setdefault("staleness", {})
    trend = list(old_staleness.get("confidence_trend", []))
    new_conf = round(new_proc.get("confidence_avg", 0.0), 4)
    if new_conf > 0 and (not trend or trend[-1] != new_conf):
        trend.append(new_conf)
    staleness["confidence_trend"] = trend
    staleness["drift_signals"] = old_staleness.get("drift_signals", [])
    staleness["last_confirmed"] = old_staleness.get("last_confirmed")
    return new_proc


def merge_existing_procedure(
    new_proc: dict, existing: dict, *, keep_lifecycle: bool = True,
) -> dict:
    """Carry what accumulated on *existing* over to its regeneration *new_proc*.

    Kept from *existing*: everything :func:`carry_run_history` keeps, plus
    evidence, constraints, recurrence, the learning ledger (``knowledge``:
    entries other learners wrote survive, observation entries are
    re-derived on save), branches, expected outcomes,
    compiled outputs, the lifecycle history and, unless *keep_lifecycle*
    is false, the lifecycle state.  Everything else comes from
    *new_proc*.  Mutates and returns *new_proc*.
    """
    if new_proc is existing:
        return new_proc
    for key in ("evidence", "constraints", "recurrence", "knowledge"):
        if key in existing:
            new_proc[key] = existing[key]
    new_proc["branches"] = existing.get("branches", [])
    new_proc["expected_outcomes"] = existing.get("expected_outcomes", [])
    new_proc["compiled_outputs"] = existing.get("compiled_outputs", {})
    new_proc["lifecycle_history"] = existing.get("lifecycle_history", [])
    if keep_lifecycle:
        new_proc["lifecycle_state"] = existing.get("lifecycle_state", "observed")
    carry_run_history(new_proc, existing)
    new_proc.setdefault("staleness", {})["last_observed"] = datetime.now(timezone.utc).isoformat()
    return new_proc


class ProcedureWriter:
    """Convert SOP templates → v3 procedures → knowledge base."""

    def __init__(
        self,
        kb: KnowledgeBase,
        evidence: EvidenceTracker,
        lifecycle_manager=None,
        vector_kb=None,
        llm_reasoner=None,
    ) -> None:
        self._kb = kb
        self._evidence = evidence
        self._lifecycle = lifecycle_manager
        self._vector_kb = vector_kb
        self._llm_reasoner = llm_reasoner

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

        # Pre-populate from evidence (before synthesis data exists)
        self._pre_populate_from_evidence(procedure, sop_template)

        # Validate
        errors = validate_procedure(procedure)
        if errors:
            logger.warning(
                "Procedure '%s' has validation issues: %s",
                procedure.get("id", "unknown"),
                errors,
            )

        # A regeneration keeps what accumulated on the existing procedure
        # (evidence, trust, lifecycle, agent runs).
        existing = self._kb.get_procedure(procedure.get("id", ""))
        if existing is not None:
            merge_existing_procedure(procedure, existing)

        # Save to knowledge base
        path = self._kb.save_procedure(procedure)

        # Embed it for semantic search and learn the user's writing style
        # from it: model calls, so they run on the inference thread (an
        # approval on the control thread queues them, inference_runner).
        slug = procedure["id"]
        model_work(f"enrich-procedure:{slug}", lambda: self._enrich(slug))

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
                from agenthandover_worker.lifecycle_manager import ProcedureLifecycle
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
    # Vector KB embedding
    # ------------------------------------------------------------------

    def _embed_procedure(self, procedure: dict) -> None:
        """Embed procedure content in vector KB for semantic search."""
        if self._vector_kb is None:
            return
        try:
            slug = procedure.get("id", "")
            title = procedure.get("title", "")
            desc = procedure.get("description", "")
            strategy = procedure.get("strategy") or ""
            steps = procedure.get("steps", [])
            step_summary = "; ".join(
                s.get("action", s.get("description", ""))
                for s in steps[:15]
                if isinstance(s, dict)
            )
            text = f"{title} | {desc} | {step_summary} | {strategy}"
            self._vector_kb.upsert("procedure", slug, text[:3000])
        except Exception:
            logger.debug("Procedure embedding failed for %s", procedure.get("id"), exc_info=True)

    def _enrich(self, slug: str) -> None:
        """Embed the saved procedure *slug* and analyze its writing style."""
        procedure = self._kb.get_procedure(slug)
        if procedure is None:
            return
        self._embed_procedure(procedure)
        self._analyze_style(procedure)

    def _analyze_style(self, procedure: dict) -> None:
        """Analyze user writing style via LLM and populate voice_profile + content_samples.

        The result goes into the procedure as saved *after* the model call:
        the control thread may have changed it meanwhile (an approval moves
        its lifecycle), and saving the copy read before the call would undo
        that."""
        try:
            from agenthandover_worker.style_analyzer import analyze_procedure_style
            voice_profile, content_samples = analyze_procedure_style(
                procedure, llm_reasoner=self._llm_reasoner,
            )
            if voice_profile or content_samples:
                procedure = self._kb.get_procedure(procedure.get("id", "")) or procedure
            if voice_profile:
                existing_vp = procedure.get("voice_profile", {})
                existing_vp.update(voice_profile)
                procedure["voice_profile"] = existing_vp
            if content_samples:
                procedure["content_samples"] = content_samples
            # Re-save with style data
            if voice_profile or content_samples:
                self._kb.save_procedure(procedure)
                logger.debug(
                    "Style analysis for '%s': %s, %d samples",
                    procedure.get("id"),
                    voice_profile.get("formality", "unknown"),
                    len(content_samples),
                )
        except Exception:
            logger.debug("Style analysis failed for %s", procedure.get("id"), exc_info=True)

    # ------------------------------------------------------------------
    # Evidence pre-population
    # ------------------------------------------------------------------

    def _pre_populate_from_evidence(
        self, procedure: dict, sop_template: dict,
    ) -> None:
        """Pre-populate procedure fields from extracted evidence.

        Only populates fields that are currently empty — does not
        overwrite data from behavioral synthesis.
        """
        evidence = sop_template.get("_extracted_evidence", {})
        if not evidence:
            return

        # Workflow rhythm: avg_duration_minutes from timing
        timing = evidence.get("timing_patterns", {})
        total_dur = timing.get("total_duration_seconds", 0)
        if total_dur > 0:
            rhythm = procedure.setdefault("workflow_rhythm", {})
            if not rhythm.get("avg_duration_minutes"):
                rhythm["avg_duration_minutes"] = round(total_dur / 60, 1)

        # Build evidence summary string
        summary_parts: list[str] = []

        selection = evidence.get("selection_signals", {})
        high_engagement = selection.get("high_engagement_locations", [])
        if high_engagement:
            summary_parts.append(
                f"High engagement at {len(high_engagement)} location(s)"
            )

        content = evidence.get("content_produced", {})
        content_count = content.get("count", 0)
        content_types = content.get("types", [])
        if content_count > 0:
            type_str = ", ".join(content_types[:5]) if content_types else "unknown"
            summary_parts.append(
                f"{content_count} content item(s) produced ({type_str})"
            )

        url_patterns = evidence.get("url_patterns", {})
        domains = url_patterns.get("domains", [])
        if domains:
            summary_parts.append(
                f"Domains: {', '.join(domains[:5])}"
            )

        if summary_parts:
            procedure["evidence_summary"] = "; ".join(summary_parts)

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
