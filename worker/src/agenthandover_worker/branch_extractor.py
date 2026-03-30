"""Branch/exception extraction from multi-observation comparison.

Compares multiple demonstrations (observed runs) of the same procedure to
find divergence points — steps where the user took different actions in
different runs.  Each divergence is classified as a pre-condition branch,
data-dependent branch, or error-recovery branch, and can be merged back
into the procedure stored in the knowledge base.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agenthandover_worker.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


@dataclass
class ExtractedBranch:
    """A conditional branch discovered at a step where demonstrations diverge."""

    step_id: str
    condition: str
    paths: list[dict]  # [{condition: str, action: str, observed_count: int}]
    confidence: float
    type: str  # "pre_condition", "data_dependent", "error_recovery"


class BranchExtractor:
    """Extract conditional branches from multi-observation comparison."""

    def __init__(self, kb: KnowledgeBase, variant_detector=None) -> None:
        self._kb = kb
        self._variant_detector = variant_detector

    def extract_branches(
        self,
        slug: str,
        demos: list[list[dict]] | None = None,
    ) -> list[ExtractedBranch]:
        """Extract branches from multiple demonstrations of a procedure.

        Args:
            slug: Procedure slug.
            demos: List of demonstrations.  Each demo is a list of step dicts
                with at least ``action``, ``step_id``, ``app``, ``location``
                keys.  If *None*, loads from procedure evidence.

        Returns:
            List of branches found at divergence points.
        """
        if demos is None:
            proc = self._kb.get_procedure(slug)
            if proc is None or len(
                proc.get("evidence", {}).get("observations", [])
            ) < 2:
                return []
            # Cannot extract demos from evidence alone — need raw step data.
            # This path is used when demos are provided by the caller.
            return []

        if len(demos) < 2:
            return []

        aligned = self._align_demonstrations(demos)
        divergences = self._find_divergence_points(aligned)

        branches: list[ExtractedBranch] = []
        for div in divergences:
            branch_type = self._classify_branch_type(div)
            branches.append(
                ExtractedBranch(
                    step_id=div["step_id"],
                    condition=div["condition"],
                    paths=div["paths"],
                    confidence=div["confidence"],
                    type=branch_type,
                )
            )

        return branches

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def _align_demonstrations(
        self,
        demos: list[list[dict]],
    ) -> list[list[dict | None]]:
        """Align steps across demonstrations using positional alignment.

        Pads shorter demos with *None* to align divergent paths.
        Returns a list of "columns" — each column has one entry per demo.
        """
        if not demos:
            return []

        if self._variant_detector is not None and len(demos) >= 2:
            try:
                return self._semantic_align_demonstrations(demos)
            except Exception:
                logger.debug("Semantic alignment failed, falling back to positional", exc_info=True)
        # Existing positional alignment continues as fallback...

        max_len = max(len(d) for d in demos)

        aligned: list[list[dict | None]] = []
        for step_idx in range(max_len):
            column: list[dict | None] = []
            for demo in demos:
                if step_idx < len(demo):
                    column.append(demo[step_idx])
                else:
                    column.append(None)
            aligned.append(column)

        return aligned

    def _semantic_align_demonstrations(self, demos):
        """Align demos using VariantDetector's semantic alignment."""
        reference = demos[0]
        max_len = max(len(d) for d in demos)
        columns = []
        for pos in range(max_len):
            column = []
            for demo in demos:
                if pos < len(demo):
                    column.append(demo[pos])
                else:
                    column.append(None)
            columns.append(column)

        # Use semantic alignment for the first pair to improve column mapping
        if len(demos) >= 2:
            aligned = self._variant_detector.semantic_align(reference, demos[1])
            # Rebuild columns from alignment
            new_columns = []
            for astep in aligned:
                col = [astep.step_a]
                if len(demos) > 1:
                    col.append(astep.step_b)
                for demo in demos[2:]:
                    # For remaining demos, use positional fallback
                    col.append(demo[astep.position] if astep.position < len(demo) else None)
                new_columns.append(col)
            columns = new_columns

        return columns

    # ------------------------------------------------------------------
    # Divergence detection
    # ------------------------------------------------------------------

    def _find_divergence_points(
        self,
        aligned: list[list[dict | None]],
    ) -> list[dict]:
        """Find points where demonstrations diverge.

        A divergence is where different demos have different actions
        at the same step position.
        """
        divergences: list[dict] = []

        for step_idx, column in enumerate(aligned):
            # Collect non-None entries with their demo indices.
            entries: list[tuple[int, dict]] = [
                (i, e) for i, e in enumerate(column) if e is not None
            ]
            if len(entries) < 2:
                continue

            # Group by normalised action string.
            actions: dict[str, list[int]] = {}
            for demo_idx, entry in entries:
                action = entry.get("action", "").strip().lower()
                if action not in actions:
                    actions[action] = []
                actions[action].append(demo_idx)

            if len(actions) <= 1:
                continue  # All same — no divergence.

            # Build paths from divergent actions.
            paths: list[dict] = []
            for action_text, demo_indices in actions.items():
                rep_entry = column[demo_indices[0]]
                condition = self._infer_condition(
                    rep_entry, column, demo_indices
                )
                paths.append(
                    {
                        "condition": condition,
                        "action": action_text,
                        "observed_count": len(demo_indices),
                    }
                )

            # Try to get step_id from entries, fall back to positional id.
            step_id = f"step_{step_idx + 1}"
            for _, entry in entries:
                if entry and "step_id" in entry:
                    step_id = entry["step_id"]
                    break

            total_entries = len(entries)
            max_path_count = max(p["observed_count"] for p in paths)
            confidence = max_path_count / total_entries

            divergences.append(
                {
                    "step_id": step_id,
                    "step_index": step_idx,
                    "condition": f"Varies at {step_id}",
                    "paths": paths,
                    "confidence": round(confidence, 3),
                }
            )

        return divergences

    # ------------------------------------------------------------------
    # Condition inference
    # ------------------------------------------------------------------

    def _infer_condition(
        self,
        entry: dict | None,
        column: list[dict | None],
        demo_indices: list[int],
    ) -> str:
        """Try to infer what condition leads to this branch path.

        Simple heuristic: look at app, location, or input differences.
        """
        if entry is None:
            return "unknown"

        # Check for precondition hints in the entry
        if isinstance(entry, dict):
            pre = entry.get("pre_state", "")
            if pre:
                return f"when {pre}"
            verify = entry.get("verify", "")
            if verify and "error" not in verify.lower():
                return f"after verifying: {verify}"

        # Check if there's a distinguishing input.
        if entry.get("input"):
            return f"when input is '{entry['input']}'"

        # Check for different apps.
        if entry.get("app"):
            return f"when app is {entry['app']}"

        # Check for error-related keywords.
        action = entry.get("action", "").lower()
        error_keywords = ("error", "fail", "retry", "fix", "correct", "undo")
        if any(kw in action for kw in error_keywords):
            return "on error"

        return "alternative path"

    # ------------------------------------------------------------------
    # Branch classification
    # ------------------------------------------------------------------

    def _classify_branch_type(self, divergence: dict) -> str:
        """Classify the type of branch based on context.

        Types:
        - ``pre_condition``: Branch happens at step 0 (different starting
          conditions).
        - ``error_recovery``: One path has error/retry/fix keywords.
        - ``data_dependent``: Branch depends on data (different inputs/values).
        """
        step_idx = divergence.get("step_index", 0)
        paths = divergence.get("paths", [])

        # Pre-condition: divergence at step 0.
        if step_idx == 0:
            return "pre_condition"

        # Error recovery: any path has error-related action.
        for path in paths:
            action = path.get("action", "").lower()
            if any(
                kw in action
                for kw in ("error", "retry", "fix", "correct", "undo", "fail")
            ):
                return "error_recovery"

        # Default: data-dependent.
        return "data_dependent"

    # ------------------------------------------------------------------
    # Merge into knowledge base
    # ------------------------------------------------------------------

    def merge_branches_into_procedure(
        self,
        slug: str,
        branches: list[ExtractedBranch],
    ) -> None:
        """Merge extracted branches into a procedure in the KB."""
        proc = self._kb.get_procedure(slug)
        if proc is None:
            return

        proc["branches"] = [
            {
                "step_id": b.step_id,
                "condition": b.condition,
                "paths": b.paths,
                "confidence": b.confidence,
                "type": b.type,
            }
            for b in branches
        ]
        self._kb.save_procedure(proc)
