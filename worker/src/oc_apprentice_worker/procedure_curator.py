"""Curation orchestrator for OpenMimic procedures (Phase 5).

Aggregates signals from staleness detection, trust advising, lifecycle
management, and structural deduplication to build a unified curation
queue.  Provides merge, promote, demote, and archive operations that
coordinate all subsystems consistently.

The curator never auto-acts — it surfaces recommendations for the human
operator and executes only when explicitly asked.

Decisions are persisted at ``{kb_root}/observations/curation_decisions.json``
so dismissed merges and drift signals are not re-surfaced.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from itertools import combinations

from oc_apprentice_worker.export_adapter import procedure_to_sop_template
from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.lifecycle_manager import (
    LifecycleManager,
    ProcedureLifecycle,
    InvalidTransitionError,
)
from oc_apprentice_worker.sop_dedup import (
    compute_fingerprint,
    detect_procedure_family,
    fingerprint_similarity,
    merge_sops,
)
from oc_apprentice_worker.staleness_detector import StalenessDetector
from oc_apprentice_worker.trust_advisor import TrustAdvisor

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass
class MergeCandidate:
    """Two procedures that may be duplicates worth merging."""

    slug_a: str
    slug_b: str
    similarity: float
    reason: str  # "fingerprint_overlap", "same_goal", "variant_family"
    explanation: str
    shared_apps: list[str] = field(default_factory=list)


@dataclass
class UpgradeCandidate:
    """A procedure eligible for lifecycle promotion."""

    slug: str
    current_state: str
    proposed_state: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class DriftReport:
    """A single drift signal observed on a procedure."""

    slug: str
    drift_type: str  # "step_failure", "confidence_drift", "url_changed", "new_step", "last_observed_old"
    severity: str  # "high", "medium", "low"
    detail: str
    first_seen: str = ""


@dataclass
class ProcedureFamily:
    """A group of procedures that are variants of the same workflow."""

    family_id: str
    canonical_slug: str
    variant_slugs: list[str] = field(default_factory=list)
    shared_apps: list[str] = field(default_factory=list)
    shared_steps_count: int = 0


@dataclass
class CurationItem:
    """A single item in the curation queue."""

    item_type: str  # "merge", "upgrade", "stale", "trust", "family", "drift"
    priority: float  # 0.0–1.0, higher = more urgent
    slug: str
    title: str
    explanation: str
    data: dict = field(default_factory=dict)


@dataclass
class CurationDecision:
    """Record of a curation action taken."""

    action: str  # "merge", "promote", "demote", "archive", "dismiss_merge", "dismiss_drift"
    slug: str
    actor: str
    timestamp: str
    details: dict = field(default_factory=dict)


@dataclass
class CurationSummary:
    """Summary of a full curation run."""

    merge_candidates: int = 0
    upgrade_candidates: int = 0
    stale_procedures: int = 0
    trust_suggestions: int = 0
    families: int = 0
    drift_reports: int = 0
    total_queue_items: int = 0


# ------------------------------------------------------------------
# Curator
# ------------------------------------------------------------------


class ProcedureCurator:
    """Curation orchestrator — aggregates all quality signals into an
    actionable queue and executes merge/promote/demote/archive operations.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        staleness_detector: StalenessDetector,
        trust_advisor: TrustAdvisor,
        lifecycle_manager: LifecycleManager,
        evidence_normalizer=None,
    ) -> None:
        self._kb = kb
        self._staleness = staleness_detector
        self._trust = trust_advisor
        self._lifecycle = lifecycle_manager
        self._evidence_normalizer = evidence_normalizer

        # Persisted curation decisions
        self._dismissed_merges: set[frozenset[str]] = set()
        self._dismissed_drift: list[dict] = []
        self._decisions: list[dict] = []

        self._load_decisions()

    # ------------------------------------------------------------------
    # Detection: merge candidates
    # ------------------------------------------------------------------

    def detect_merge_candidates(self) -> list[MergeCandidate]:
        """Find procedure pairs with similarity in [0.70, 0.95) that
        could be merged.  Filters out previously dismissed pairs.
        """
        procedures = self._kb.list_procedures()
        if len(procedures) < 2:
            return []

        # Convert to SOP templates and compute fingerprints
        templates: list[tuple[str, dict, dict]] = []
        for proc in procedures:
            slug = proc.get("id", proc.get("slug", ""))
            if not slug:
                continue
            tpl = procedure_to_sop_template(proc)
            fp = compute_fingerprint(tpl)
            templates.append((slug, tpl, fp))

        candidates: list[MergeCandidate] = []

        for (slug_a, tpl_a, fp_a), (slug_b, tpl_b, fp_b) in combinations(templates, 2):
            sim = fingerprint_similarity(fp_a, fp_b)

            # Only consider the merge-worthy range
            if sim < 0.70 or sim >= 0.95:
                continue

            # Check dismissed
            pair_key = frozenset({slug_a, slug_b})
            if pair_key in self._dismissed_merges:
                continue

            # Compute shared apps
            apps_a = set(tpl_a.get("apps_involved", []))
            apps_b = set(tpl_b.get("apps_involved", []))
            shared_apps = sorted(apps_a & apps_b)

            # Determine reason
            verbs_a = set(fp_a.get("action_verbs", []))
            verbs_b = set(fp_b.get("action_verbs", []))
            apps_fp_a = set(fp_a.get("apps", []))
            apps_fp_b = set(fp_b.get("apps", []))
            domains_a = set(fp_a.get("domains", []))
            domains_b = set(fp_b.get("domains", []))

            if apps_fp_a == apps_fp_b and verbs_a == verbs_b and domains_a == domains_b:
                reason = "fingerprint_overlap"
            elif verbs_a == verbs_b:
                reason = "same_goal"
            else:
                reason = "variant_family"

            explanation = (
                f"These procedures share {len(shared_apps)} apps "
                f"({', '.join(shared_apps)}) and have {sim:.0%} structural similarity"
            )

            candidates.append(MergeCandidate(
                slug_a=slug_a,
                slug_b=slug_b,
                similarity=sim,
                reason=reason,
                explanation=explanation,
                shared_apps=shared_apps,
            ))

        return candidates

    # ------------------------------------------------------------------
    # Detection: upgrade candidates
    # ------------------------------------------------------------------

    def detect_upgrade_candidates(self) -> list[UpgradeCandidate]:
        """Find procedures eligible for lifecycle promotion based on
        evidence quality, episode count, and freshness.
        """
        procedures = self._kb.list_procedures()
        candidates: list[UpgradeCandidate] = []

        # Filter out dismissed lifecycle upgrades
        dismissed_upgrade_slugs: set[str] = set()
        for d in self._dismissed_drift:
            if d.get("drift_type") == "__lifecycle_upgrade_dismissed__":
                dismissed_upgrade_slugs.add(d.get("slug", ""))

        for proc in procedures:
            slug = proc.get("id", proc.get("slug", ""))
            if not slug:
                continue

            if slug in dismissed_upgrade_slugs:
                continue

            lifecycle_str = proc.get("lifecycle_state", "observed")

            # Skip stale and archived — not eligible for promotion
            if lifecycle_str in ("stale", "archived"):
                continue

            episode_count = proc.get("episode_count", 0)
            confidence_avg = proc.get("confidence_avg", 0.0)
            evidence = proc.get("evidence", {})
            staleness_data = proc.get("staleness", {})
            contradictions = evidence.get("contradictions", [])

            # Compute freshness
            freshness = self._staleness.compute_freshness_score(proc)

            reasons: list[str] = []
            proposed: str | None = None

            if lifecycle_str == "observed":
                if episode_count >= 3 and confidence_avg >= 0.65:
                    proposed = "draft"
                    reasons.append(f"Episode count {episode_count} >= 3")
                    reasons.append(f"Confidence {confidence_avg:.2f} >= 0.65")

            elif lifecycle_str == "draft":
                if (
                    episode_count >= 5
                    and confidence_avg >= 0.75
                    and freshness >= 0.7
                    and not contradictions
                ):
                    proposed = "reviewed"
                    reasons.append(f"Episode count {episode_count} >= 5")
                    reasons.append(f"Confidence {confidence_avg:.2f} >= 0.75")
                    reasons.append(f"Freshness {freshness:.2f} >= 0.70")
                    reasons.append("No contradictions")

            elif lifecycle_str == "verified":
                last_confirmed = staleness_data.get("last_confirmed")
                if last_confirmed:
                    try:
                        lc_dt = datetime.fromisoformat(
                            last_confirmed.replace("Z", "+00:00")
                        )
                        days_since = (datetime.now(timezone.utc) - lc_dt).days
                        if days_since <= 7:
                            proposed = "agent_ready"
                            reasons.append(f"Last confirmed {days_since} days ago (<= 7)")
                    except (ValueError, TypeError):
                        pass

            if proposed is None:
                continue

            # Validate the transition is actually allowed
            try:
                target = ProcedureLifecycle(proposed)
            except ValueError:
                continue

            if not self._lifecycle.can_transition(slug, target):
                continue

            candidates.append(UpgradeCandidate(
                slug=slug,
                current_state=lifecycle_str,
                proposed_state=proposed,
                reasons=reasons,
            ))

        return candidates

    # ------------------------------------------------------------------
    # Detection: drift
    # ------------------------------------------------------------------

    def detect_drift(self, slug: str) -> list[DriftReport]:
        """Detect drift signals for a single procedure.

        Reads from staleness.drift_signals and staleness signals.
        Classifies severity and filters dismissed signals.
        """
        proc = self._kb.get_procedure(slug)
        if proc is None:
            return []

        staleness_data = proc.get("staleness", {})
        drift_signals = staleness_data.get("drift_signals", [])

        reports: list[DriftReport] = []

        for ds in drift_signals:
            if not isinstance(ds, dict):
                continue

            drift_type = ds.get("type", "unknown")
            detail = ds.get("detail", "")
            first_seen = ds.get("first_seen", "")

            # Check dismissed
            if self._is_drift_dismissed(slug, drift_type):
                continue

            severity = self._classify_drift_severity(drift_type)

            reports.append(DriftReport(
                slug=slug,
                drift_type=drift_type,
                severity=severity,
                detail=detail,
                first_seen=first_seen,
            ))

        # Also check staleness report signals
        staleness_report = self._staleness.check_procedure(slug, proc)
        for signal in staleness_report.signals:
            # Avoid duplicates with drift_signals already processed
            already_covered = any(r.drift_type == signal.type for r in reports)
            if already_covered:
                continue

            if self._is_drift_dismissed(slug, signal.type):
                continue

            severity = self._classify_drift_severity(signal.type)

            reports.append(DriftReport(
                slug=slug,
                drift_type=signal.type,
                severity=severity,
                detail=signal.detail,
                first_seen=signal.first_seen,
            ))

        return reports

    def detect_all_drift(self) -> dict[str, list[DriftReport]]:
        """Detect drift for all procedures. Returns dict keyed by slug."""
        result: dict[str, list[DriftReport]] = {}
        procedures = self._kb.list_procedures()
        for proc in procedures:
            slug = proc.get("id", proc.get("slug", ""))
            if not slug:
                continue
            drift = self.detect_drift(slug)
            if drift:
                result[slug] = drift
        return result

    # ------------------------------------------------------------------
    # Detection: families
    # ------------------------------------------------------------------

    def build_families(self) -> list[ProcedureFamily]:
        """Build variant families from all procedures using structural
        fingerprint similarity.
        """
        procedures = self._kb.list_procedures()
        if len(procedures) < 2:
            return []

        # Convert to SOP templates for detect_procedure_family
        templates: list[dict] = []
        proc_map: dict[str, dict] = {}
        for proc in procedures:
            slug = proc.get("id", proc.get("slug", ""))
            if not slug:
                continue
            tpl = procedure_to_sop_template(proc)
            templates.append(tpl)
            proc_map[slug] = proc

        raw_families = detect_procedure_family(templates)

        families: list[ProcedureFamily] = []
        for fam in raw_families:
            canonical_slug = fam["canonical_slug"]
            variant_slugs = fam["variant_slugs"]
            shared_apps = fam.get("shared_apps", [])

            # Enrich with shared_steps_count: count common action verbs
            canonical_proc = proc_map.get(canonical_slug, {})
            canonical_verbs = self._extract_action_verbs(canonical_proc)

            total_shared = 0
            for v_slug in variant_slugs:
                v_proc = proc_map.get(v_slug, {})
                v_verbs = self._extract_action_verbs(v_proc)
                total_shared += len(canonical_verbs & v_verbs)

            families.append(ProcedureFamily(
                family_id=fam["family_id"],
                canonical_slug=canonical_slug,
                variant_slugs=variant_slugs,
                shared_apps=shared_apps,
                shared_steps_count=total_shared,
            ))

        return families

    # ------------------------------------------------------------------
    # Curation queue
    # ------------------------------------------------------------------

    def build_curation_queue(self) -> list[CurationItem]:
        """Aggregate all detection sources into a prioritized curation queue."""
        items: list[CurationItem] = []

        # 1. Merge candidates
        for mc in self.detect_merge_candidates():
            priority = 0.7 if mc.similarity > 0.80 else 0.5
            items.append(CurationItem(
                item_type="merge",
                priority=priority,
                slug=f"{mc.slug_a}+{mc.slug_b}",
                title=f"Merge: {mc.slug_a} + {mc.slug_b}",
                explanation=mc.explanation,
                data=asdict(mc),
            ))

        # 2. Upgrade candidates
        for uc in self.detect_upgrade_candidates():
            items.append(CurationItem(
                item_type="upgrade",
                priority=0.6,
                slug=uc.slug,
                title=f"Upgrade: {uc.slug} → {uc.proposed_state}",
                explanation=f"Eligible for promotion: {'; '.join(uc.reasons)}",
                data=asdict(uc),
            ))

        # 3. Stale procedures
        stale_reports: list = []
        try:
            stale_reports = self._staleness.check_all()
        except Exception:
            logger.debug("Staleness check failed in curation queue", exc_info=True)
        for sr in stale_reports:
            if sr.status == "current":
                continue
            lifecycle = self._lifecycle.get_state(sr.slug)
            priority = 1.0 if lifecycle == ProcedureLifecycle.AGENT_READY else 0.4
            items.append(CurationItem(
                item_type="stale",
                priority=priority,
                slug=sr.slug,
                title=f"Stale: {sr.slug} ({sr.status})",
                explanation=f"Recommended action: {sr.recommended_action}",
                data={
                    "status": sr.status,
                    "recommended_action": sr.recommended_action,
                    "signal_count": len(sr.signals),
                },
            ))

        # 4. Trust suggestions
        trust_suggestions = self._trust.get_suggestions()
        for ts in trust_suggestions:
            if ts.dismissed or ts.accepted:
                continue
            items.append(CurationItem(
                item_type="trust",
                priority=0.8,
                slug=ts.procedure_slug,
                title=f"Trust: {ts.procedure_slug} → {ts.suggested_level}",
                explanation=ts.reason,
                data={
                    "current_level": ts.current_level,
                    "suggested_level": ts.suggested_level,
                    "evidence": ts.evidence,
                },
            ))

        # 5. Families with >= 2 variants
        for fam in self.build_families():
            if len(fam.variant_slugs) < 2:
                continue
            items.append(CurationItem(
                item_type="family",
                priority=0.4,
                slug=fam.canonical_slug,
                title=f"Family: {fam.canonical_slug} ({len(fam.variant_slugs)} variants)",
                explanation=(
                    f"Variant family with {len(fam.variant_slugs)} variants, "
                    f"{fam.shared_steps_count} shared action verbs"
                ),
                data=asdict(fam),
            ))

        # 6. Drift reports
        all_drift = self.detect_all_drift()
        for slug, drifts in all_drift.items():
            for dr in drifts:
                if dr.severity == "high":
                    priority = 0.5
                elif dr.severity == "medium":
                    priority = 0.3
                else:
                    priority = 0.2
                items.append(CurationItem(
                    item_type="drift",
                    priority=priority,
                    slug=slug,
                    title=f"Drift: {slug} ({dr.drift_type})",
                    explanation=dr.detail,
                    data=asdict(dr),
                ))

        # Sort by priority descending
        items.sort(key=lambda it: it.priority, reverse=True)

        return items

    # ------------------------------------------------------------------
    # Execution: merge
    # ------------------------------------------------------------------

    def execute_merge(
        self,
        slug_a: str,
        slug_b: str,
        actor: str = "human",
    ) -> dict:
        """Merge procedure B into A, archive B, and record the decision.

        Returns a result dict with ``success``, ``merged_slug``, ``archived_slug``.
        """
        proc_a = self._kb.get_procedure(slug_a)
        proc_b = self._kb.get_procedure(slug_b)

        if proc_a is None:
            return {"success": False, "error": f"Procedure not found: {slug_a}"}
        if proc_b is None:
            return {"success": False, "error": f"Procedure not found: {slug_b}"}

        # Convert to SOP templates for merging
        tpl_a = procedure_to_sop_template(proc_a)
        tpl_b = procedure_to_sop_template(proc_b)

        # Merge B into A
        if self._evidence_normalizer is not None:
            merged_tpl = merge_sops(tpl_a, tpl_b, self._evidence_normalizer)
        else:
            merged_tpl = merge_sops(tpl_a, tpl_b)

        # Convert merged template back to procedure and preserve A's id
        from oc_apprentice_worker.procedure_schema import sop_to_procedure
        merged_proc = sop_to_procedure(merged_tpl)
        merged_proc["id"] = slug_a
        # Preserve lifecycle state from A
        merged_proc["lifecycle_state"] = proc_a.get("lifecycle_state", "observed")
        merged_proc["lifecycle_history"] = proc_a.get("lifecycle_history", [])

        # Save merged A
        self._kb.save_procedure(merged_proc)

        # Archive B
        try:
            self._lifecycle.transition(
                slug_b,
                ProcedureLifecycle.ARCHIVED,
                trigger="curation_merge",
                actor=actor,
                reason=f"Merged into {slug_a}",
            )
        except InvalidTransitionError:
            # If direct transition to ARCHIVED is not valid from current state,
            # force the state directly
            proc_b["lifecycle_state"] = "archived"
            self._kb.save_procedure(proc_b)

        # Record decision
        self._record_decision(CurationDecision(
            action="merge",
            slug=slug_a,
            actor=actor,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details={"merged_slug": slug_a, "archived_slug": slug_b},
        ))

        return {"merged_slug": slug_a, "archived_slug": slug_b, "success": True}

    # ------------------------------------------------------------------
    # Execution: promote / demote / archive
    # ------------------------------------------------------------------

    def execute_promote(
        self,
        slug: str,
        to_state: str,
        actor: str = "human",
        reason: str = "",
    ) -> dict:
        """Promote a procedure to a higher lifecycle state."""
        try:
            target = ProcedureLifecycle(to_state)
        except ValueError:
            return {"success": False, "error": f"Invalid lifecycle state: {to_state}"}

        try:
            result = self._lifecycle.transition(
                slug, target, trigger="curation_promote", actor=actor, reason=reason,
            )
        except InvalidTransitionError as exc:
            return {"success": False, "error": str(exc)}

        if not result:
            return {"success": False, "error": f"Procedure not found: {slug}"}

        self._record_decision(CurationDecision(
            action="promote",
            slug=slug,
            actor=actor,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details={"to_state": to_state, "reason": reason},
        ))

        return {"success": True, "slug": slug, "new_state": to_state}

    def execute_demote(
        self,
        slug: str,
        to_state: str,
        actor: str = "human",
        reason: str = "",
    ) -> dict:
        """Demote a procedure to a lower lifecycle state."""
        try:
            target = ProcedureLifecycle(to_state)
        except ValueError:
            return {"success": False, "error": f"Invalid lifecycle state: {to_state}"}

        try:
            result = self._lifecycle.transition(
                slug, target, trigger="curation_demote", actor=actor, reason=reason,
            )
        except InvalidTransitionError as exc:
            return {"success": False, "error": str(exc)}

        if not result:
            return {"success": False, "error": f"Procedure not found: {slug}"}

        self._record_decision(CurationDecision(
            action="demote",
            slug=slug,
            actor=actor,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details={"to_state": to_state, "reason": reason},
        ))

        return {"success": True, "slug": slug, "new_state": to_state}

    def execute_archive(
        self,
        slug: str,
        actor: str = "human",
        reason: str = "",
    ) -> dict:
        """Archive a procedure."""
        try:
            result = self._lifecycle.transition(
                slug,
                ProcedureLifecycle.ARCHIVED,
                trigger="curation_archive",
                actor=actor,
                reason=reason,
            )
        except InvalidTransitionError as exc:
            return {"success": False, "error": str(exc)}

        if not result:
            return {"success": False, "error": f"Procedure not found: {slug}"}

        self._record_decision(CurationDecision(
            action="archive",
            slug=slug,
            actor=actor,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details={"reason": reason},
        ))

        return {"success": True, "slug": slug, "new_state": "archived"}

    # ------------------------------------------------------------------
    # Dismiss operations
    # ------------------------------------------------------------------

    def dismiss_merge(self, slug_a: str, slug_b: str) -> None:
        """Dismiss a merge candidate so it is not re-surfaced."""
        self._dismissed_merges.add(frozenset({slug_a, slug_b}))
        self._save_decisions()

    def dismiss_drift(self, slug: str, drift_type: str) -> None:
        """Dismiss a drift signal so it is not re-surfaced."""
        self._dismissed_drift.append({"slug": slug, "drift_type": drift_type})
        self._save_decisions()

    # ------------------------------------------------------------------
    # Full curation run
    # ------------------------------------------------------------------

    def curate(self) -> CurationSummary:
        """Run all detection and return a summary of findings."""
        merge_candidates = self.detect_merge_candidates()
        upgrade_candidates = self.detect_upgrade_candidates()
        staleness_reports = self._staleness.check_all()
        stale_count = sum(1 for sr in staleness_reports if sr.status != "current")
        trust_suggestions = [
            s for s in self._trust.get_suggestions()
            if not s.dismissed and not s.accepted
        ]
        families = self.build_families()
        all_drift = self.detect_all_drift()
        drift_count = sum(len(drifts) for drifts in all_drift.values())

        queue = self.build_curation_queue()

        return CurationSummary(
            merge_candidates=len(merge_candidates),
            upgrade_candidates=len(upgrade_candidates),
            stale_procedures=stale_count,
            trust_suggestions=len(trust_suggestions),
            families=len(families),
            drift_reports=drift_count,
            total_queue_items=len(queue),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_drift_severity(self, drift_type: str) -> str:
        """Classify drift signal severity."""
        if drift_type in ("step_failure", "confidence_drift"):
            return "high"
        if drift_type in ("url_changed", "new_step"):
            return "medium"
        return "low"

    def _is_drift_dismissed(self, slug: str, drift_type: str) -> bool:
        """Check if a drift signal has been dismissed."""
        for d in self._dismissed_drift:
            if d.get("slug") == slug and d.get("drift_type") == drift_type:
                return True
        return False

    def _extract_action_verbs(self, proc: dict) -> set[str]:
        """Extract first-word action verbs from procedure steps."""
        verbs: set[str] = set()
        for step in proc.get("steps", []):
            action = step.get("action", step.get("step", ""))
            if action and action.strip():
                first_word = action.strip().split()[0].lower()
                verbs.add(first_word)
        return verbs

    def _record_decision(self, decision: CurationDecision) -> None:
        """Append a decision and persist."""
        self._decisions.append(asdict(decision))
        self._save_decisions()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_decisions(self) -> None:
        """Load persisted curation decisions from disk."""
        path = self._kb.root / "observations" / "curation_decisions.json"
        if not path.is_file():
            return

        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.debug("Could not load curation decisions", exc_info=True)
            return

        # Restore dismissed merges
        for pair in data.get("dismissed_merges", []):
            if isinstance(pair, list) and len(pair) == 2:
                self._dismissed_merges.add(frozenset(pair))

        # Restore dismissed drift
        self._dismissed_drift = data.get("dismissed_drift", [])

        # Restore decision history
        self._decisions = data.get("decisions", [])

    def _save_decisions(self) -> None:
        """Persist curation decisions to disk."""
        data = {
            "dismissed_merges": [sorted(pair) for pair in self._dismissed_merges],
            "dismissed_drift": self._dismissed_drift,
            "decisions": self._decisions,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self._kb.root / "observations" / "curation_decisions.json"
        self._kb.atomic_write_json(path, data)
