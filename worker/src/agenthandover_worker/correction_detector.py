"""Correction Feedback Loop — detect and apply user corrections to procedures.

When a user corrects agent output (re-edits, undoes, reverts, or supplements),
this module detects the correction pattern from event streams and records it.
Accumulated corrections with sufficient occurrences are applied back to the
procedure in the knowledge base, closing the observe-learn-execute-correct
improvement flywheel.

When an ``LLMReasoner`` is provided, correction patterns can be analyzed
to infer guardrails — rules that prevent the corrected behaviour from
recurring.

Corrections are persisted at ``{kb_root}/observations/corrections.json``.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from agenthandover_worker.event_helpers import (
    extract_app_from_event,
    extract_location,
    extract_what_doing,
    parse_annotation,
    parse_timestamp,
)

if TYPE_CHECKING:
    from agenthandover_worker.llm_reasoning import LLMReasoner

logger = logging.getLogger(__name__)


@dataclass
class Correction:
    """A single detected correction event."""

    correction_id: str
    procedure_slug: str
    execution_id: str
    step_id: str | None
    original_output: str
    corrected_output: str
    correction_type: str  # "edit", "redo", "revert", "supplement"
    detected_at: str
    applied: bool = False


@dataclass
class CorrectionSummary:
    """Aggregated correction statistics for a procedure."""

    procedure_slug: str
    total_corrections: int
    correction_types: dict[str, int]
    most_corrected_steps: list[dict]
    last_correction: str | None


# ---------------------------------------------------------------------------
# Helpers — event field extraction (delegated to event_helpers module)
# ---------------------------------------------------------------------------

# Re-bind to module-private names for backward compatibility with call sites.
_parse_annotation = parse_annotation
_get_app = extract_app_from_event
_get_what_doing = extract_what_doing
_get_location = extract_location


def _parse_timestamp(event: dict) -> datetime | None:
    """Extract and parse the timestamp from an event."""
    ts = event.get("timestamp") or event.get("created_at")
    return parse_timestamp(ts)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class CorrectionDetector:
    """Detect, record, and apply user corrections to procedures.

    Monitors event streams for patterns that indicate a user corrected
    agent output: re-edits, undos, reverts, and supplementary actions.
    Accumulated corrections are applied back to the knowledge base
    procedure when the pattern is strong enough.
    """

    def __init__(
        self,
        knowledge_base,
        llm_reasoner: "LLMReasoner | None" = None,
    ) -> None:
        self._kb = knowledge_base
        self._llm_reasoner = llm_reasoner
        self._corrections: list[Correction] = []
        self._load_corrections()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_correction(
        self,
        events: list[dict],
        execution_id: str | None = None,
        procedure_slug: str = "unknown",
    ) -> list[Correction]:
        """Analyze events to detect corrections.

        Heuristics:
        1. Same-field re-edit: Same app + location edited twice within
           120s with different ``what_doing`` values.
        2. Undo pattern: Events containing "undo", "revert", "fixing",
           or "correcting" keywords in ``what_doing``.
        3. Quick overwrite: Same location changed twice quickly with
           different content.

        Args:
            events: List of event dicts (with scene_annotation_json, etc.).
            execution_id: Optional execution run identifier.
            procedure_slug: Procedure these events relate to.

        Returns:
            List of newly detected Correction objects.
        """
        if not events:
            return []

        exec_id = execution_id or str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        detected: list[Correction] = []

        # Parse annotations and timestamps upfront
        parsed: list[tuple[dict, dict | None, datetime | None, str]] = []
        for event in events:
            ann = _parse_annotation(event)
            ts = _parse_timestamp(event)
            app = _get_app(event)
            parsed.append((event, ann, ts, app))

        # Heuristic 1 & 3: re-edit / quick overwrite in same app+location
        for i, (ev_i, ann_i, ts_i, app_i) in enumerate(parsed):
            if ann_i is None or ts_i is None:
                continue
            what_i = _get_what_doing(ann_i)
            loc_i = _get_location(ann_i)
            if not what_i:
                continue

            for j in range(i + 1, len(parsed)):
                ev_j, ann_j, ts_j, app_j = parsed[j]
                if ann_j is None or ts_j is None:
                    continue
                what_j = _get_what_doing(ann_j)
                loc_j = _get_location(ann_j)
                if not what_j:
                    continue

                # Must be same app and same location
                if app_i != app_j or loc_i != loc_j:
                    continue
                # Must be different what_doing (otherwise it is the same action)
                if what_i.strip().lower() == what_j.strip().lower():
                    continue

                delta = abs((ts_j - ts_i).total_seconds())
                if delta > 120:
                    continue

                # Determine type
                revert_keywords = ("undo", "revert", "fixing", "correcting")
                if any(kw in what_j.lower() for kw in revert_keywords):
                    ctype = "revert"
                else:
                    ctype = "edit"

                correction = Correction(
                    correction_id=str(uuid.uuid4()),
                    procedure_slug=procedure_slug,
                    execution_id=exec_id,
                    step_id=None,
                    original_output=what_i,
                    corrected_output=what_j,
                    correction_type=ctype,
                    detected_at=now_iso,
                )
                detected.append(correction)
                break  # Only match the first re-edit per origin event

        # Heuristic 2: standalone undo/revert keywords (not already caught)
        existing_ids = {c.correction_id for c in detected}
        for ev, ann, ts, app in parsed:
            if ann is None:
                continue
            what = _get_what_doing(ann)
            if not what:
                continue
            what_lower = what.lower()
            revert_keywords = ("undo", "revert", "fixing", "correcting")
            if any(kw in what_lower for kw in revert_keywords):
                # Check that we haven't already created a correction for
                # this exact event via heuristic 1
                already = any(
                    c.corrected_output == what for c in detected
                )
                if already:
                    continue
                correction = Correction(
                    correction_id=str(uuid.uuid4()),
                    procedure_slug=procedure_slug,
                    execution_id=exec_id,
                    step_id=None,
                    original_output="",
                    corrected_output=what,
                    correction_type="revert",
                    detected_at=now_iso,
                )
                detected.append(correction)

        return detected

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_correction(self, correction: Correction) -> None:
        """Record a correction and persist to disk."""
        self._corrections.append(correction)
        self._save_corrections()
        logger.info(
            "Recorded correction %s for %s (type=%s)",
            correction.correction_id,
            correction.procedure_slug,
            correction.correction_type,
        )

    # ------------------------------------------------------------------
    # Applying corrections to procedures
    # ------------------------------------------------------------------

    def apply_corrections(
        self, procedure_slug: str, min_occurrences: int = 2
    ) -> dict:
        """Apply accumulated corrections to a procedure if patterns are strong.

        Groups corrections by step_id (or by corrected_output when step_id
        is None).  If the same correction pattern appears >= min_occurrences
        times, the corresponding procedure step is updated.

        Args:
            procedure_slug: The procedure to update.
            min_occurrences: Minimum times a correction pattern must appear.

        Returns:
            Dict with keys: ``applied``, ``skipped``, ``updates``.
        """
        result: dict = {"applied": 0, "skipped": 0, "updates": []}

        relevant = [
            c for c in self._corrections
            if c.procedure_slug == procedure_slug and not c.applied
        ]
        if not relevant:
            return result

        procedure = self._kb.get_procedure(procedure_slug)
        if procedure is None:
            result["skipped"] = len(relevant)
            return result

        # Group corrections by step_id (or corrected_output as key)
        groups: dict[str, list[Correction]] = {}
        for c in relevant:
            key = c.step_id or c.corrected_output
            groups.setdefault(key, []).append(c)

        modified = False
        steps = procedure.get("steps", [])

        for key, corrections in groups.items():
            if len(corrections) < min_occurrences:
                result["skipped"] += len(corrections)
                continue

            # Find the most common corrected output
            output_counts: dict[str, int] = {}
            for c in corrections:
                output_counts[c.corrected_output] = (
                    output_counts.get(c.corrected_output, 0) + 1
                )
            best_output = max(output_counts, key=output_counts.get)  # type: ignore[arg-type]

            # Try to find and update the matching step
            updated_step = False
            for step in steps:
                sid = step.get("step_id", "")
                if sid == key:
                    step["action"] = best_output
                    updated_step = True
                    break

            # If no step_id match, try to match by original_output in action
            if not updated_step:
                for c in corrections:
                    if c.original_output:
                        for step in steps:
                            if step.get("action", "") == c.original_output:
                                step["action"] = best_output
                                updated_step = True
                                break
                    if updated_step:
                        break

            if updated_step:
                for c in corrections:
                    c.applied = True
                result["applied"] += len(corrections)
                result["updates"].append({
                    "key": key,
                    "new_action": best_output,
                    "count": len(corrections),
                })
                modified = True
            else:
                result["skipped"] += len(corrections)

        if modified:
            self._kb.save_procedure(procedure)
            self._save_corrections()
            logger.info(
                "Applied %d corrections to %s",
                result["applied"],
                procedure_slug,
            )

        return result

    # ------------------------------------------------------------------
    # LLM-based guardrail learning
    # ------------------------------------------------------------------

    def analyze_correction_patterns(
        self,
        procedure_slug: str,
        min_corrections: int = 3,
    ) -> list[dict]:
        """Analyze repeated corrections to infer guardrails via LLM.

        Groups corrections for *procedure_slug* by step_id (or by
        corrected_output when step_id is ``None``).  For groups with
        >= *min_corrections* entries, asks the LLM to identify the
        underlying rule or guardrail.

        Returns a list of guardrail dicts with keys:
        ``guardrail``, ``improved_condition``, ``confidence``.
        """
        if self._llm_reasoner is None:
            return []

        corrections = self.get_corrections(procedure_slug)
        if not corrections:
            return []

        # Group by step_id (or by corrected_output as key)
        groups: dict[str, list[Correction]] = {}
        for c in corrections:
            key = c.step_id or c.corrected_output
            groups.setdefault(key, []).append(c)

        guardrails: list[dict] = []
        for key, group in groups.items():
            if len(group) < min_corrections:
                continue

            from agenthandover_worker.llm_reasoning import sanitize_user_data
            originals = [sanitize_user_data(c.original_output) for c in group if c.original_output]
            corrected = [sanitize_user_data(c.corrected_output) for c in group]

            prompt = (
                f"The user corrected this procedure step {len(group)} times. "
                f"Original outputs: {'; '.join(originals)}. "
                f"Corrected outputs: {'; '.join(corrected)}. "
                f"What rule or guardrail does this suggest? "
                f'Respond with JSON: {{"guardrail": "...", '
                f'"improved_condition": "...", "confidence": 0.0-1.0}}. '
                f"If you cannot determine a pattern, respond with "
                f"INSUFFICIENT_EVIDENCE."
            )

            try:
                result = self._llm_reasoner.reason_json(
                    prompt=prompt,
                    caller="correction_detector.analyze_correction_patterns",
                )
                if result.success and not result.abstained and result.value:
                    guardrails.append(result.value)
            except Exception:
                logger.debug(
                    "LLM guardrail analysis failed for key '%s'",
                    key,
                    exc_info=True,
                )

        return guardrails

    def apply_guardrails_to_procedure(
        self,
        slug: str,
        guardrails: list[dict],
    ) -> bool:
        """Append guardrail strings to a procedure's constraints.

        Loads the procedure from the KB, appends each guardrail's
        ``guardrail`` value to ``procedure["constraints"]["guardrails"]``,
        and saves.

        Returns ``True`` if any guardrails were applied.
        """
        if not guardrails:
            return False

        procedure = self._kb.get_procedure(slug)
        if procedure is None:
            return False

        constraints = procedure.setdefault("constraints", {})
        existing_guardrails = constraints.setdefault("guardrails", [])

        applied = False
        for g in guardrails:
            guardrail_text = g.get("guardrail", "")
            if guardrail_text and guardrail_text not in existing_guardrails:
                existing_guardrails.append(guardrail_text)
                applied = True

        if applied:
            self._kb.save_procedure(procedure)
            logger.info(
                "Applied %d guardrails to procedure '%s'",
                len(guardrails),
                slug,
            )

        return applied

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_corrections(
        self, procedure_slug: str | None = None
    ) -> list[Correction]:
        """Get corrections, optionally filtered by procedure."""
        if procedure_slug is None:
            return list(self._corrections)
        return [
            c for c in self._corrections
            if c.procedure_slug == procedure_slug
        ]

    def get_summary(self, procedure_slug: str) -> CorrectionSummary:
        """Get correction summary for a procedure."""
        relevant = self.get_corrections(procedure_slug)

        type_counts: dict[str, int] = {}
        step_counts: dict[str, int] = {}
        last_at: str | None = None

        for c in relevant:
            type_counts[c.correction_type] = (
                type_counts.get(c.correction_type, 0) + 1
            )
            step_key = c.step_id or "(unknown)"
            step_counts[step_key] = step_counts.get(step_key, 0) + 1
            if last_at is None or c.detected_at > last_at:
                last_at = c.detected_at

        # Sort steps by correction count descending
        most_corrected = sorted(
            [
                {"step_id": k, "count": v}
                for k, v in step_counts.items()
            ],
            key=lambda x: x["count"],
            reverse=True,
        )

        return CorrectionSummary(
            procedure_slug=procedure_slug,
            total_corrections=len(relevant),
            correction_types=type_counts,
            most_corrected_steps=most_corrected,
            last_correction=last_at,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_corrections(self) -> None:
        """Load from {kb_root}/observations/corrections.json."""
        path = self._kb.root / "observations" / "corrections.json"
        if not path.is_file():
            self._corrections = []
            return
        try:
            with open(path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                self._corrections = []
                return
            raw_list = data.get("corrections", [])
            self._corrections = []
            for item in raw_list:
                if not isinstance(item, dict):
                    continue
                self._corrections.append(
                    Correction(
                        correction_id=item.get("correction_id", ""),
                        procedure_slug=item.get("procedure_slug", ""),
                        execution_id=item.get("execution_id", ""),
                        step_id=item.get("step_id"),
                        original_output=item.get("original_output", ""),
                        corrected_output=item.get("corrected_output", ""),
                        correction_type=item.get("correction_type", "edit"),
                        detected_at=item.get("detected_at", ""),
                        applied=item.get("applied", False),
                    )
                )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load corrections: %s", exc)
            self._corrections = []

    def _save_corrections(self) -> None:
        """Persist to {kb_root}/observations/corrections.json."""
        path = self._kb.root / "observations" / "corrections.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "corrections": [
                {
                    "correction_id": c.correction_id,
                    "procedure_slug": c.procedure_slug,
                    "execution_id": c.execution_id,
                    "step_id": c.step_id,
                    "original_output": c.original_output,
                    "corrected_output": c.corrected_output,
                    "correction_type": c.correction_type,
                    "detected_at": c.detected_at,
                    "applied": c.applied,
                }
                for c in self._corrections
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        # Use KB's atomic write for crash safety
        self._kb.atomic_write_json(path, data)
        logger.debug("Saved %d corrections", len(self._corrections))
