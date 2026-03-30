"""Procedure matcher — matches task segments to known procedures via fingerprint similarity.

Uses the structural fingerprint machinery from ``sop_dedup`` (apps, URL
domains, action verbs with Jaccard similarity) to find existing procedures
that correspond to newly observed task segments or continuity spans.

The match threshold (default 0.50) is intentionally lower than dedup's 0.70
because partial matches are still valuable as supporting evidence for
continuity detection — a segment that partially overlaps a known procedure
is a signal, even if it's not a full duplicate.

Integration points:
- Called by the continuity detector to link segments to known SOPs
- Reads procedures from ``KnowledgeBase``
- Reuses ``compute_fingerprint`` / ``fingerprint_similarity`` from ``sop_dedup``
- Converts procedures to SOP template format via ``procedure_to_sop_template``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agenthandover_worker.export_adapter import procedure_to_sop_template
from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.sop_dedup import (
    _normalize_app,
    _url_to_domain,
    compute_fingerprint,
    fingerprint_similarity,
)

if TYPE_CHECKING:
    from agenthandover_worker.continuity_tracker import ContinuitySpan
    from agenthandover_worker.task_segmenter import TaskSegment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verb normalization map (mirrors sop_dedup._extract_action_verbs local map)
# ---------------------------------------------------------------------------

_VERB_MAP: dict[str, str] = {
    "navigate": "open",
    "go": "open",
    "visit": "open",
    "browse": "open",
    "launch": "open",
    "type": "enter",
    "input": "enter",
    "fill": "enter",
    "write": "enter",
    "press": "click",
    "tap": "click",
    "hit": "click",
    "submit": "click",
    "check": "verify",
    "confirm": "verify",
    "validate": "verify",
    "inspect": "review",
    "examine": "review",
    "look": "review",
    "read": "review",
    "choose": "select",
    "pick": "select",
    "filter": "select",
    "wait": "wait",
    "pause": "wait",
}


# ---------------------------------------------------------------------------
# ProcedureMatcher
# ---------------------------------------------------------------------------


class ProcedureMatcher:
    """Match task segments and continuity spans to known procedures.

    Uses structural fingerprint similarity (apps, domains, action verbs)
    from sop_dedup.py. Threshold is 0.50 (lower than dedup's 0.70) to
    catch partial matches as supporting evidence for continuity.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        match_threshold: float = 0.50,
        vector_kb=None,
    ) -> None:
        self._kb = kb
        self._match_threshold = match_threshold
        self._vector_kb = vector_kb
        self._proc_fingerprints: list[tuple[str, dict]] | None = None  # cache

    def match_segment(self, segment: TaskSegment) -> list[tuple[str, float]]:
        """Find procedure candidates for a single segment.

        Tries vector similarity first (semantic match), falls back to
        structural fingerprints.  Returns [(slug, similarity)] sorted
        descending, filtered to only matches above threshold.
        """
        # Try vector search first — catches semantic matches that
        # structural fingerprints miss ("deploy staging" = "push to stage")
        if self._vector_kb is not None:
            try:
                query = segment.task_label or ""
                if not query and segment.frames:
                    query = segment.frames[0].what_doing or segment.frames[0].app
                if query:
                    results = self._vector_kb.search(
                        query,
                        top_k=5,
                        source_types=["procedure"],
                        min_score=self._match_threshold,
                    )
                    if results:
                        return [(r.source_id, r.score) for r in results]
            except Exception:
                pass  # fall through to fingerprint matching

        if self._proc_fingerprints is None:
            self._refresh_procedure_fingerprints()

        seg_fp = self._segment_to_fingerprint(segment)
        return self._match_fingerprint(seg_fp)

    def match_span(self, span: ContinuitySpan) -> list[tuple[str, float]]:
        """Find procedure candidates for a continuity span.

        Aggregates fingerprint data across the span's metadata.  Since
        spans store segment_ids (not full segments), this method uses
        the span's ``apps_involved`` and ``goal_summary`` to build a
        fingerprint.

        Returns [(slug, similarity)] sorted descending by similarity,
        filtered to only include matches above the threshold.
        """
        if self._proc_fingerprints is None:
            self._refresh_procedure_fingerprints()

        span_fp = self._span_to_fingerprint(span)
        return self._match_fingerprint(span_fp)

    def invalidate_cache(self) -> None:
        """Force a refresh of the procedure fingerprint cache.

        Call this after procedures are added, updated, or deleted in the
        knowledge base to ensure subsequent matches use current data.
        """
        self._proc_fingerprints = None
        logger.debug("Procedure fingerprint cache invalidated")

    # ------------------------------------------------------------------
    # Fingerprint builders
    # ------------------------------------------------------------------

    def _segment_to_fingerprint(self, segment: TaskSegment) -> dict:
        """Build a sop_dedup-compatible fingerprint from segment data.

        Components:
        - ``apps``: from ``segment.apps_involved``, normalized via
          ``_normalize_app``
        - ``domains``: from each frame's ``location`` field, extracted
          via ``_url_to_domain``
        - ``action_verbs``: first word of each frame's ``what_doing``,
          normalized via the verb map
        """
        # Apps: normalize each app name
        apps: set[str] = set()
        for app in segment.apps_involved:
            normalized = _normalize_app(app)
            if normalized:
                apps.add(normalized)

        # Domains: extract from frame locations
        domains: set[str] = set()
        for frame in segment.frames:
            if frame.location:
                domain = _url_to_domain(frame.location)
                if domain:
                    domains.add(domain)

        # Action verbs: first word of each frame.what_doing, normalized
        action_verbs: set[str] = set()
        for frame in segment.frames:
            if frame.what_doing:
                words = frame.what_doing.strip().split()
                if words:
                    first_word = words[0].lower()
                    canonical = _VERB_MAP.get(first_word, first_word)
                    action_verbs.add(canonical)

        return {
            "apps": sorted(apps),
            "domains": sorted(domains),
            "action_verbs": sorted(action_verbs),
        }

    def _span_to_fingerprint(self, span: ContinuitySpan) -> dict:
        """Build a fingerprint from a span's aggregated data.

        Components:
        - ``apps``: from ``span.apps_involved``, normalized
        - ``domains``: empty (spans don't store frame-level locations)
        - ``action_verbs``: first word of ``span.goal_summary``, normalized
        """
        # Apps: normalize each app name
        apps: set[str] = set()
        for app in span.apps_involved:
            normalized = _normalize_app(app)
            if normalized:
                apps.add(normalized)

        # Domains: not available at span level
        domains: set[str] = set()

        # Action verbs: extract from goal_summary
        action_verbs: set[str] = set()
        if span.goal_summary:
            words = span.goal_summary.strip().split()
            if words:
                first_word = words[0].lower()
                canonical = _VERB_MAP.get(first_word, first_word)
                action_verbs.add(canonical)

        return {
            "apps": sorted(apps),
            "domains": sorted(domains),
            "action_verbs": sorted(action_verbs),
        }

    # ------------------------------------------------------------------
    # Procedure fingerprint cache
    # ------------------------------------------------------------------

    def _refresh_procedure_fingerprints(self) -> None:
        """Reload procedures from KB and compute fingerprints.

        For each procedure, converts to SOP template format via
        ``procedure_to_sop_template()``, then computes a structural
        fingerprint via ``compute_fingerprint()``.

        Caches as a list of ``(slug, fingerprint)`` tuples.
        """
        procedures = self._kb.list_procedures()
        fingerprints: list[tuple[str, dict]] = []

        for proc in procedures:
            slug = proc.get("id", proc.get("slug", "unknown"))
            try:
                sop_template = procedure_to_sop_template(proc)
                fp = compute_fingerprint(sop_template)
                fingerprints.append((slug, fp))
            except Exception:
                logger.warning(
                    "Failed to compute fingerprint for procedure '%s'",
                    slug,
                    exc_info=True,
                )

        self._proc_fingerprints = fingerprints
        logger.debug(
            "Refreshed procedure fingerprints: %d procedures loaded",
            len(fingerprints),
        )

    # ------------------------------------------------------------------
    # Internal matching
    # ------------------------------------------------------------------

    def _match_fingerprint(
        self, target_fp: dict,
    ) -> list[tuple[str, float]]:
        """Compare a target fingerprint against all cached procedure fingerprints.

        Returns [(slug, similarity)] sorted descending by similarity,
        filtered to only include matches at or above ``self._match_threshold``.
        """
        assert self._proc_fingerprints is not None  # noqa: S101

        matches: list[tuple[str, float]] = []
        for slug, proc_fp in self._proc_fingerprints:
            score = fingerprint_similarity(target_fp, proc_fp)
            if score >= self._match_threshold:
                matches.append((slug, score))

        # Sort descending by similarity score
        matches.sort(key=lambda m: m[1], reverse=True)
        return matches
