"""Evidence-weighted step normalization for OpenMimic.

Normalizes procedure steps across multiple demonstrations by identifying
canonical actions (the most frequently observed variant at each step
position) and tracking alternatives.  Integrates with the evidence tracker
to weight step confidence by observation count.

Key responsibilities:
- Collapse multiple demonstrations into a single normalized step sequence
- Track canonical vs. alternative actions per position
- Merge new observations into existing procedures with semantic matching
- Group related-but-distinct procedures into variant families
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from oc_apprentice_worker.export_adapter import procedure_to_sop_template
from oc_apprentice_worker.sop_dedup import (
    compute_fingerprint,
    fingerprint_similarity,
)

if TYPE_CHECKING:
    from oc_apprentice_worker.variant_detector import AlignedStep, VariantDetector

logger = logging.getLogger(__name__)

# Similarity thresholds for variant family grouping
_FAMILY_THRESHOLD = 0.60
_MERGE_THRESHOLD = 0.70

# Confidence decay when a step is absent from a new observation
_ABSENCE_DECAY = 0.05


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass
class NormalizedStep:
    """A single step after evidence-weighted normalization.

    Attributes:
        step_id: Positional identifier (e.g. ``"step_1"``).
        canonical_action: The most frequently observed action text.
        canonical_target: The most common target element.
        canonical_app: The most common application context.
        canonical_location: The most common location / URL.
        confidence: ``observation_count / total_observations``.
        observation_count: How many demos included this step.
        alternatives: Other observed action/target/app variants with counts.
        parameters: Extracted parameters observed at this position.
    """

    step_id: str
    canonical_action: str
    canonical_target: str = ""
    canonical_app: str = ""
    canonical_location: str = ""
    confidence: float = 0.0
    observation_count: int = 0
    alternatives: list[dict] = field(default_factory=list)
    parameters: list[dict] = field(default_factory=list)


# ------------------------------------------------------------------
# Normalizer
# ------------------------------------------------------------------


class EvidenceNormalizer:
    """Normalize procedure steps using evidence from multiple demos.

    When a ``VariantDetector`` is provided, step alignment is used to
    correctly pair steps across demonstrations even when they differ in
    count.  Without a detector the normalizer falls back to positional
    alignment (index-based).
    """

    def __init__(self, variant_detector: VariantDetector | None = None) -> None:
        self._detector = variant_detector

    # ----------------------------------------------------------
    # normalize
    # ----------------------------------------------------------

    def normalize(
        self,
        slug: str,
        demos: list[list[dict]],
        alignments: list[list[AlignedStep]] | None = None,
    ) -> list[NormalizedStep]:
        """Collapse multiple demonstrations into normalized steps.

        Args:
            slug: Procedure slug (for logging).
            demos: List of demonstrations.  Each demo is a list of step
                dicts with at least ``action``, ``target``, ``app``, and
                ``location`` keys.
            alignments: Optional pre-computed alignments from a
                ``VariantDetector``.  One alignment list per demo
                (aligned against the first demo).

        Returns:
            Ordered list of ``NormalizedStep`` instances.
        """
        if not demos:
            return []

        # Single demo — return steps at full confidence
        if len(demos) == 1:
            return self._steps_from_single_demo(demos[0])

        # If no alignments supplied, try to compute them via the detector
        if alignments is None and self._detector is not None:
            alignments = self._compute_alignments(demos)

        # If we still lack alignments, fall back to positional matching
        if alignments is None:
            return self._normalize_positional(slug, demos)

        return self._normalize_aligned(slug, demos, alignments)

    # ----------------------------------------------------------
    # merge_with_evidence
    # ----------------------------------------------------------

    def merge_with_evidence(
        self,
        existing_proc: dict,
        new_observation: list[dict],
        alignment: list[AlignedStep] | None = None,
    ) -> dict:
        """Merge a new observation into an existing procedure.

        Unlike ``sop_dedup.merge_sops()`` which uses a "keep more steps"
        strategy, this method performs semantic matching so that each
        aligned step accumulates evidence counts and alternatives.

        Args:
            existing_proc: The current procedure dict (not mutated).
            new_observation: Steps from the new observation.
            alignment: Optional pre-computed alignment between existing
                steps and the new observation.

        Returns:
            A **new** procedure dict with updated steps and evidence.
        """
        proc = _deep_copy_proc(existing_proc)
        existing_steps = proc.get("steps", [])

        # Ensure evidence section exists
        evidence = proc.setdefault("evidence", {
            "observations": [],
            "step_evidence": [],
            "contradictions": [],
            "total_observations": 0,
        })

        # Attempt alignment if not provided
        if alignment is None and self._detector is not None and existing_steps:
            alignment = self._compute_pairwise_alignment(existing_steps, new_observation)

        if alignment is not None:
            proc["steps"] = self._merge_aligned(
                existing_steps, new_observation, alignment,
            )
        else:
            # Fall back to simple append logic (mirrors sop_dedup.merge_sops)
            proc["steps"] = self._merge_simple(existing_steps, new_observation)

        # Bump total observations
        evidence["total_observations"] = evidence.get("total_observations", 0) + 1

        return proc

    # ----------------------------------------------------------
    # compute_step_confidence
    # ----------------------------------------------------------

    @staticmethod
    def compute_step_confidence(step: dict, total_observations: int) -> float:
        """Return the confidence for a step given observation counts.

        Args:
            step: A step dict containing an ``observation_count`` key.
            total_observations: Total number of procedure observations.

        Returns:
            Float in [0.0, 1.0].
        """
        if total_observations <= 0:
            return 0.0
        obs = step.get("observation_count", 0)
        return min(obs / total_observations, 1.0)

    # ----------------------------------------------------------
    # build_variant_family
    # ----------------------------------------------------------

    @staticmethod
    def build_variant_family(
        slug: str,
        related_slugs: list[str],
        procedures: dict[str, dict],
    ) -> dict:
        """Group similar procedures into a variant family.

        Procedures with fingerprint similarity above ``_FAMILY_THRESHOLD``
        (0.60) but below ``_MERGE_THRESHOLD`` (0.70) are considered
        variants of the same task rather than duplicates.

        Args:
            slug: The anchor procedure slug.
            related_slugs: Candidate slugs to compare against.
            procedures: Mapping of slug -> procedure dict.

        Returns:
            A family dict with ``family_id``, ``canonical_slug``,
            ``variant_slugs``, and ``shared_apps``.  Returns an empty
            dict if no family can be formed.
        """
        if not related_slugs or slug not in procedures:
            return {}

        # Build fingerprints
        anchor_proc = procedures[slug]
        anchor_sop = procedure_to_sop_template(anchor_proc)
        anchor_fp = compute_fingerprint(anchor_sop)

        # Collect members that fall in the family band
        members: list[str] = [slug]
        for candidate in related_slugs:
            if candidate == slug or candidate not in procedures:
                continue
            cand_sop = procedure_to_sop_template(procedures[candidate])
            cand_fp = compute_fingerprint(cand_sop)
            sim = fingerprint_similarity(anchor_fp, cand_fp)
            if _FAMILY_THRESHOLD <= sim < _MERGE_THRESHOLD:
                members.append(candidate)

        if len(members) < 2:
            return {}

        # Also check pairwise among candidates to form connected components
        members = _connected_component(members, procedures)

        if len(members) < 2:
            return {}

        # Canonical = highest episode count
        canonical = max(
            members,
            key=lambda s: procedures[s].get("episode_count", 0),
        )
        variant_slugs = [s for s in members if s != canonical]

        # Shared apps across all members
        app_sets = []
        for s in members:
            apps = set(procedures[s].get("apps_involved", []))
            if apps:
                app_sets.append(apps)
        shared_apps = sorted(set.intersection(*app_sets)) if app_sets else []

        family = {
            "family_id": f"family-{canonical}",
            "canonical_slug": canonical,
            "variant_slugs": variant_slugs,
            "shared_apps": shared_apps,
        }
        logger.info(
            "Built variant family '%s' with %d members: %s",
            family["family_id"],
            len(members),
            members,
        )
        return family

    # ==============================================================
    # Private helpers
    # ==============================================================

    def _steps_from_single_demo(self, demo: list[dict]) -> list[NormalizedStep]:
        """Convert a single demo's steps to NormalizedSteps at confidence 1.0."""
        result: list[NormalizedStep] = []
        for i, step in enumerate(demo):
            result.append(NormalizedStep(
                step_id=f"step_{i + 1}",
                canonical_action=step.get("action", ""),
                canonical_target=step.get("target", ""),
                canonical_app=step.get("app", ""),
                canonical_location=step.get("location", ""),
                confidence=1.0,
                observation_count=1,
                parameters=[step.get("parameters", {})] if step.get("parameters") else [],
            ))
        return result

    def _compute_alignments(
        self, demos: list[list[dict]],
    ) -> list[list[AlignedStep]] | None:
        """Align each demo against the first using the variant detector."""
        if self._detector is None or len(demos) < 2:
            return None
        reference = demos[0]
        alignments = []
        for demo in demos[1:]:
            try:
                aligned = self._detector.align(reference, demo)
                alignments.append(aligned)
            except Exception:
                logger.debug(
                    "Alignment failed for demo pair; falling back to positional",
                    exc_info=True,
                )
                return None
        return alignments

    def _compute_pairwise_alignment(
        self,
        existing_steps: list[dict],
        new_steps: list[dict],
    ) -> list[AlignedStep] | None:
        """Align a single pair of step lists."""
        if self._detector is None:
            return None
        try:
            return self._detector.align(existing_steps, new_steps)
        except Exception:
            logger.debug("Pairwise alignment failed", exc_info=True)
            return None

    def _normalize_positional(
        self, slug: str, demos: list[list[dict]],
    ) -> list[NormalizedStep]:
        """Positional (index-based) normalization without alignment."""
        max_len = max(len(d) for d in demos)
        total_demos = len(demos)
        result: list[NormalizedStep] = []

        for pos in range(max_len):
            result.append(self._aggregate_position(pos, demos, total_demos))

        logger.debug(
            "Positional normalization for '%s': %d steps from %d demos",
            slug, len(result), total_demos,
        )
        return result

    def _normalize_aligned(
        self,
        slug: str,
        demos: list[list[dict]],
        alignments: list[list[AlignedStep]],
    ) -> list[NormalizedStep]:
        """Alignment-based normalization using AlignedStep data.

        Each AlignedStep carries ``ref_idx`` (index into reference demo)
        and ``demo_idx`` (index into the compared demo), plus a
        ``similarity`` score.  We collect values per reference position.
        """
        reference = demos[0]
        total_demos = len(demos)

        # Build a mapping: ref_position -> list of step dicts from all demos
        position_map: dict[int, list[dict]] = {}
        for pos, step in enumerate(reference):
            position_map.setdefault(pos, []).append(step)

        for align_idx, aligned_pairs in enumerate(alignments):
            demo = demos[align_idx + 1]
            for ap in aligned_pairs:
                ref_idx = getattr(ap, "ref_idx", None)
                demo_idx = getattr(ap, "demo_idx", None)
                if ref_idx is not None and demo_idx is not None:
                    if 0 <= demo_idx < len(demo):
                        position_map.setdefault(ref_idx, []).append(demo[demo_idx])

        result: list[NormalizedStep] = []
        for pos in sorted(position_map.keys()):
            steps_at_pos = position_map[pos]
            result.append(self._aggregate_from_steps(
                pos, steps_at_pos, total_demos,
            ))

        logger.debug(
            "Aligned normalization for '%s': %d steps from %d demos",
            slug, len(result), total_demos,
        )
        return result

    def _aggregate_position(
        self, pos: int, demos: list[list[dict]], total_demos: int,
    ) -> NormalizedStep:
        """Aggregate step data at a given position across demos."""
        steps_at_pos: list[dict] = []
        for demo in demos:
            if pos < len(demo):
                steps_at_pos.append(demo[pos])
        return self._aggregate_from_steps(pos, steps_at_pos, total_demos)

    @staticmethod
    def _aggregate_from_steps(
        pos: int,
        steps: list[dict],
        total_demos: int,
    ) -> NormalizedStep:
        """Build a NormalizedStep from a collection of step dicts."""
        if not steps:
            return NormalizedStep(
                step_id=f"step_{pos + 1}",
                canonical_action="",
                confidence=0.0,
                observation_count=0,
            )

        observation_count = len(steps)

        # Count actions (normalized: lowercase, stripped)
        action_counter: Counter[str] = Counter()
        target_counter: Counter[str] = Counter()
        app_counter: Counter[str] = Counter()
        location_counter: Counter[str] = Counter()
        all_params: list[dict] = []

        for step in steps:
            action_raw = step.get("action", "")
            action_norm = action_raw.strip().lower()
            if action_norm:
                action_counter[action_norm] += 1

            target = step.get("target", "")
            if target:
                target_counter[target.strip()] += 1

            app = step.get("app", "")
            if app:
                app_counter[app.strip()] += 1

            location = step.get("location", "")
            if location:
                location_counter[location.strip()] += 1

            params = step.get("parameters", {})
            if params:
                all_params.append(params)

        # Most common values are canonical
        canonical_action = action_counter.most_common(1)[0][0] if action_counter else ""
        canonical_target = target_counter.most_common(1)[0][0] if target_counter else ""
        canonical_app = app_counter.most_common(1)[0][0] if app_counter else ""
        canonical_location = location_counter.most_common(1)[0][0] if location_counter else ""

        confidence = observation_count / total_demos if total_demos > 0 else 0.0

        # Build alternatives: actions that differ from canonical
        alternatives: list[dict] = []
        for action_text, count in action_counter.most_common():
            if action_text == canonical_action:
                continue
            alt: dict = {"action": action_text, "observed_count": count}
            # Attach target/app if there's a consistent pairing
            paired_targets: Counter[str] = Counter()
            paired_apps: Counter[str] = Counter()
            for step in steps:
                if step.get("action", "").strip().lower() == action_text:
                    t = step.get("target", "")
                    if t:
                        paired_targets[t.strip()] += 1
                    a = step.get("app", "")
                    if a:
                        paired_apps[a.strip()] += 1
            if paired_targets:
                alt["target"] = paired_targets.most_common(1)[0][0]
            if paired_apps:
                alt["app"] = paired_apps.most_common(1)[0][0]
            alternatives.append(alt)

        return NormalizedStep(
            step_id=f"step_{pos + 1}",
            canonical_action=canonical_action,
            canonical_target=canonical_target,
            canonical_app=canonical_app,
            canonical_location=canonical_location,
            confidence=confidence,
            observation_count=observation_count,
            alternatives=alternatives,
            parameters=all_params,
        )

    # ----------------------------------------------------------
    # Merge helpers
    # ----------------------------------------------------------

    @staticmethod
    def _merge_aligned(
        existing_steps: list[dict],
        new_steps: list[dict],
        alignment: list[AlignedStep],
    ) -> list[dict]:
        """Merge steps using alignment data.

        Returns a new step list — does not mutate the inputs.
        """
        merged: list[dict] = [dict(s) for s in existing_steps]
        matched_existing: set[int] = set()

        for ap in alignment:
            ref_idx = getattr(ap, "ref_idx", None)
            demo_idx = getattr(ap, "demo_idx", None)
            similarity = getattr(ap, "similarity", 0.0)

            if ref_idx is None or demo_idx is None:
                continue
            if demo_idx < 0 or demo_idx >= len(new_steps):
                continue

            new_step = new_steps[demo_idx]

            if ref_idx >= 0 and ref_idx < len(merged):
                matched_existing.add(ref_idx)
                existing_step = merged[ref_idx]

                if similarity >= _MERGE_THRESHOLD:
                    # Match — update observation count, add alternative
                    existing_step["observation_count"] = (
                        existing_step.get("observation_count", 1) + 1
                    )
                    new_action = new_step.get("action", "").strip().lower()
                    existing_action = existing_step.get("action", "").strip().lower()
                    if new_action and new_action != existing_action:
                        alts = existing_step.setdefault("alternatives", [])
                        # Check if this alternative already tracked
                        found = False
                        for alt in alts:
                            if alt.get("action", "").strip().lower() == new_action:
                                alt["observed_count"] = alt.get("observed_count", 1) + 1
                                found = True
                                break
                        if not found:
                            alts.append({
                                "action": new_step.get("action", ""),
                                "target": new_step.get("target", ""),
                                "app": new_step.get("app", ""),
                                "observed_count": 1,
                            })
                else:
                    # Insertion — new step not semantically matching existing
                    variant_step = dict(new_step)
                    variant_step["_variant"] = True
                    variant_step["observation_count"] = 1
                    merged.append(variant_step)
            else:
                # ref_idx out of range — treat as insertion
                variant_step = dict(new_step)
                variant_step["_variant"] = True
                variant_step["observation_count"] = 1
                merged.append(variant_step)

        # Decrease confidence for unmatched existing steps
        for idx in range(len(existing_steps)):
            if idx not in matched_existing and idx < len(merged):
                current_conf = merged[idx].get("confidence", 1.0)
                merged[idx]["confidence"] = max(0.0, current_conf - _ABSENCE_DECAY)

        return merged

    @staticmethod
    def _merge_simple(
        existing_steps: list[dict],
        new_steps: list[dict],
    ) -> list[dict]:
        """Simple merge without alignment — mirrors sop_dedup strategy.

        Keeps the longer step list.  If new is longer or equal, uses new.
        """
        if len(new_steps) >= len(existing_steps):
            return [dict(s) for s in new_steps]
        return [dict(s) for s in existing_steps]


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _deep_copy_proc(proc: dict) -> dict:
    """Create a shallow-ish copy of a procedure dict.

    Copies the top-level dict and the ``steps`` and ``evidence``
    sub-structures so that mutations don't affect the original.
    """
    copy = dict(proc)
    if "steps" in copy:
        copy["steps"] = [dict(s) for s in copy["steps"]]
    if "evidence" in copy:
        ev = dict(copy["evidence"])
        ev["observations"] = list(ev.get("observations", []))
        ev["step_evidence"] = list(ev.get("step_evidence", []))
        ev["contradictions"] = list(ev.get("contradictions", []))
        copy["evidence"] = ev
    if "variants" in copy:
        copy["variants"] = list(copy["variants"])
    return copy


def _connected_component(
    members: list[str],
    procedures: dict[str, dict],
) -> list[str]:
    """Filter members to the connected component containing all of them.

    Two members are connected if their fingerprint similarity is above
    ``_FAMILY_THRESHOLD``.  Returns only the component that contains the
    first member (the anchor).
    """
    if len(members) <= 1:
        return members

    # Build adjacency
    adjacency: dict[str, set[str]] = {s: set() for s in members}
    fp_cache: dict[str, dict] = {}

    for s in members:
        if s not in fp_cache:
            sop = procedure_to_sop_template(procedures[s])
            fp_cache[s] = compute_fingerprint(sop)

    for i, a in enumerate(members):
        for b in members[i + 1:]:
            sim = fingerprint_similarity(fp_cache[a], fp_cache[b])
            if sim >= _FAMILY_THRESHOLD:
                adjacency[a].add(b)
                adjacency[b].add(a)

    # BFS from anchor (first member)
    anchor = members[0]
    visited: set[str] = set()
    queue = [anchor]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                queue.append(neighbor)

    return [s for s in members if s in visited]
