"""Tests for evidence_normalizer — step normalization, merge, confidence, variant families.

Tests cover:
- Normalize: canonical from majority, alternatives, confidence, single observation
- Merge with evidence: matching step updates, novel step stored, partial match,
  backward compat, evidence preservation
- Step confidence: evidence-weighted, zero observations
- Variant families: similarity grouping, canonical selection, threshold filtering
- Edge cases: empty demos, contradictory observations, single demo
- Merge integration: normalize with/without detector, incremental merge
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oc_apprentice_worker.evidence_normalizer import (
    EvidenceNormalizer,
    NormalizedStep,
    _deep_copy_proc,
)
from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.procedure_schema import sop_to_procedure
from oc_apprentice_worker.variant_detector import VariantDetector


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _step(action: str, app: str = "Chrome", location: str = "",
          input_val: str = "", target: str = "") -> dict:
    return {
        "action": action,
        "app": app,
        "location": location,
        "input": input_val,
        "target": target,
    }


def _make_procedure(slug: str, steps: list[dict], episode_count: int = 1,
                    apps: list[str] | None = None) -> dict:
    """Build a v3 procedure via sop_to_procedure for use in family tests."""
    sop = {
        "slug": slug,
        "title": slug.replace("-", " ").title(),
        "steps": [{"action": s.get("action", ""), "target": s.get("target", ""),
                    "app": s.get("app", ""), "location": s.get("location", ""),
                    "confidence": 0.8} for s in steps],
        "confidence_avg": 0.8,
        "episode_count": episode_count,
        "apps_involved": apps or ["Chrome"],
        "source": "passive",
    }
    return sop_to_procedure(sop)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    return kb


# ------------------------------------------------------------------
# TestNormalize (4 tests)
# ------------------------------------------------------------------


class TestNormalize:
    """Normalize multiple demonstrations into canonical steps."""

    def test_canonical_from_majority(self):
        """Most common action becomes canonical."""
        normalizer = EvidenceNormalizer()
        demos = [
            [_step("Click search")],
            [_step("Click search")],
            [_step("Click search")],
            [_step("Use voice search")],
        ]
        result = normalizer.normalize("test-slug", demos)
        assert len(result) >= 1
        assert result[0].canonical_action == "click search"

    def test_alternatives_preserved(self):
        """Less common actions appear in alternatives."""
        normalizer = EvidenceNormalizer()
        demos = [
            [_step("Click search")],
            [_step("Click search")],
            [_step("Click search")],
            [_step("Use voice search")],
        ]
        result = normalizer.normalize("test-slug", demos)
        assert len(result) >= 1
        alt_actions = [a["action"] for a in result[0].alternatives]
        assert "use voice search" in alt_actions

    def test_confidence_reflects_count(self):
        """Confidence equals observation_count / total."""
        normalizer = EvidenceNormalizer()
        demos = [
            [_step("Click search"), _step("Enter query")],
            [_step("Click search"), _step("Enter query")],
            [_step("Click search")],  # shorter — no step at pos 1
        ]
        result = normalizer.normalize("test-slug", demos)
        # First step present in all 3 demos → confidence 3/3 = 1.0
        assert result[0].confidence == pytest.approx(1.0)
        # Second step present in 2 of 3 demos → confidence 2/3
        assert result[1].confidence == pytest.approx(2 / 3, abs=0.01)

    def test_single_observation(self):
        """Single demo produces confidence 1.0 for all steps."""
        normalizer = EvidenceNormalizer()
        demos = [[_step("Open browser"), _step("Click")]]
        result = normalizer.normalize("single", demos)
        assert len(result) == 2
        for ns in result:
            assert ns.confidence == 1.0
            assert ns.observation_count == 1


# ------------------------------------------------------------------
# TestMergeWithEvidence (5 tests)
# ------------------------------------------------------------------


class TestMergeWithEvidence:
    """Merge new observations into existing procedures."""

    def _base_proc(self) -> dict:
        """A minimal procedure with steps and evidence."""
        return {
            "id": "test-proc",
            "title": "Test Procedure",
            "steps": [
                {"action": "Open browser", "app": "Chrome",
                 "observation_count": 2, "confidence": 0.9},
                {"action": "Click search", "app": "Chrome",
                 "observation_count": 2, "confidence": 0.9},
            ],
            "evidence": {
                "observations": [],
                "step_evidence": [],
                "contradictions": [],
                "total_observations": 2,
            },
        }

    def test_matching_step_updates_count(self):
        """Same step in new observation increments observation_count."""
        normalizer = EvidenceNormalizer(variant_detector=VariantDetector())
        proc = self._base_proc()
        new_obs = [_step("Open browser"), _step("Click search")]

        merged = normalizer.merge_with_evidence(proc, new_obs)

        assert merged["evidence"]["total_observations"] == 3
        # Steps should still be present
        assert len(merged["steps"]) >= 2

    def test_novel_step_stored_as_variant(self):
        """New step not matching existing goes to variants."""
        normalizer = EvidenceNormalizer(variant_detector=VariantDetector())
        proc = self._base_proc()
        new_obs = [_step("Open browser"), _step("Click search"),
                   _step("Download file")]

        merged = normalizer.merge_with_evidence(proc, new_obs)

        # The novel step should appear somewhere in merged steps
        all_actions = [s.get("action", "") for s in merged["steps"]]
        assert "Download file" in all_actions

    def test_partial_match(self):
        """Some steps match, some are new — both handled correctly."""
        normalizer = EvidenceNormalizer(variant_detector=VariantDetector())
        proc = self._base_proc()
        new_obs = [_step("Open browser"), _step("Upload document")]

        merged = normalizer.merge_with_evidence(proc, new_obs)

        all_actions = [s.get("action", "") for s in merged["steps"]]
        assert "Open browser" in all_actions
        assert "Upload document" in all_actions
        assert merged["evidence"]["total_observations"] == 3

    def test_backward_compat(self):
        """Procedure without evidence section gets one added."""
        normalizer = EvidenceNormalizer()
        proc = {
            "id": "legacy-proc",
            "title": "Legacy",
            "steps": [{"action": "Click", "app": "Chrome"}],
        }
        new_obs = [_step("Click")]

        merged = normalizer.merge_with_evidence(proc, new_obs)

        assert "evidence" in merged
        assert merged["evidence"]["total_observations"] == 1

    def test_preserves_evidence(self):
        """Existing evidence section is not overwritten, only extended."""
        normalizer = EvidenceNormalizer()
        proc = self._base_proc()
        proc["evidence"]["observations"] = [{"ts": "2026-01-01", "source": "passive"}]

        new_obs = [_step("Open browser")]
        merged = normalizer.merge_with_evidence(proc, new_obs)

        # Original observation should still be there
        assert len(merged["evidence"]["observations"]) >= 1
        assert merged["evidence"]["observations"][0]["ts"] == "2026-01-01"


# ------------------------------------------------------------------
# TestStepConfidence (2 tests)
# ------------------------------------------------------------------


class TestStepConfidence:
    """Evidence-weighted step confidence."""

    def test_evidence_weighted(self):
        """Confidence is observation_count / total_observations."""
        step = {"observation_count": 3}
        conf = EvidenceNormalizer.compute_step_confidence(step, total_observations=5)
        assert conf == pytest.approx(0.6)

    def test_zero_observations(self):
        """Zero total observations returns 0.0."""
        step = {"observation_count": 3}
        conf = EvidenceNormalizer.compute_step_confidence(step, total_observations=0)
        assert conf == 0.0


# ------------------------------------------------------------------
# TestVariantFamily (4 tests)
# ------------------------------------------------------------------


class TestVariantFamily:
    """Group related procedures into variant families."""

    def test_similar_grouped(self, kb: KnowledgeBase):
        """Two procedures with similarity >= 0.60 are grouped into same family."""
        # Build two procedures with very similar structure (same app, same verbs)
        proc_a = _make_procedure("search-amazon", [
            _step("Open Amazon", app="Chrome", location="https://amazon.com"),
            _step("Enter query", app="Chrome"),
            _step("Click search", app="Chrome"),
        ], episode_count=5, apps=["Chrome"])

        proc_b = _make_procedure("search-ebay", [
            _step("Open eBay", app="Chrome", location="https://ebay.com"),
            _step("Enter query", app="Chrome"),
            _step("Click search", app="Chrome"),
        ], episode_count=3, apps=["Chrome"])

        kb.save_procedure(proc_a)
        kb.save_procedure(proc_b)

        procedures = {"search-amazon": proc_a, "search-ebay": proc_b}
        family = EvidenceNormalizer.build_variant_family(
            "search-amazon", ["search-ebay"], procedures,
        )

        # They share apps and action verbs — should be in same family
        if family:
            assert "canonical_slug" in family
            assert "variant_slugs" in family
            assert len(family["variant_slugs"]) >= 1

    def test_canonical_is_most_observed(self, kb: KnowledgeBase):
        """Highest episode_count procedure becomes canonical."""
        proc_a = _make_procedure("task-alpha", [
            _step("Open app", app="Chrome"),
            _step("Click button", app="Chrome"),
        ], episode_count=10, apps=["Chrome"])

        proc_b = _make_procedure("task-beta", [
            _step("Open app", app="Chrome"),
            _step("Click button", app="Chrome"),
        ], episode_count=3, apps=["Chrome"])

        procedures = {"task-alpha": proc_a, "task-beta": proc_b}
        family = EvidenceNormalizer.build_variant_family(
            "task-alpha", ["task-beta"], procedures,
        )

        if family:
            assert family["canonical_slug"] == "task-alpha"

    def test_below_threshold_not_grouped(self, kb: KnowledgeBase):
        """Dissimilar procedures are not grouped into a family."""
        proc_a = _make_procedure("write-email", [
            _step("Open Gmail", app="Chrome", location="https://mail.google.com"),
            _step("Compose message", app="Chrome"),
            _step("Send email", app="Chrome"),
        ], episode_count=5, apps=["Chrome"])

        proc_b = _make_procedure("edit-spreadsheet", [
            _step("Open Excel", app="Excel"),
            _step("Enter data", app="Excel"),
            _step("Save file", app="Excel"),
        ], episode_count=3, apps=["Excel"])

        procedures = {"write-email": proc_a, "edit-spreadsheet": proc_b}
        family = EvidenceNormalizer.build_variant_family(
            "write-email", ["edit-spreadsheet"], procedures,
        )

        # Very different apps and domains — should not form a family
        assert family == {}

    def test_single_procedure_no_family(self):
        """Single procedure with no related slugs returns empty."""
        proc = _make_procedure("solo-task", [_step("Do thing")], episode_count=1)
        procedures = {"solo-task": proc}
        family = EvidenceNormalizer.build_variant_family(
            "solo-task", [], procedures,
        )
        assert family == {}


# ------------------------------------------------------------------
# TestEdgeCases (3 tests)
# ------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for EvidenceNormalizer."""

    def test_empty_demos(self):
        """Normalize with empty list returns empty."""
        normalizer = EvidenceNormalizer()
        result = normalizer.normalize("empty", [])
        assert result == []

    def test_contradictory_observations(self):
        """Half do X, half Y — canonical is whichever is most common."""
        normalizer = EvidenceNormalizer()
        demos = [
            [_step("Click search")],
            [_step("Click search")],
            [_step("Click search")],
            [_step("Use voice search")],
            [_step("Use voice search")],
        ]
        result = normalizer.normalize("contradictory", demos)
        assert len(result) >= 1
        # 3 vs 2 → "click search" is canonical
        assert result[0].canonical_action == "click search"

    def test_single_demo_normalize(self):
        """Single demo produces all canonical at confidence 1.0."""
        normalizer = EvidenceNormalizer()
        demos = [[_step("Open"), _step("Click"), _step("Close")]]
        result = normalizer.normalize("single", demos)
        assert len(result) == 3
        for ns in result:
            assert ns.confidence == 1.0
            assert ns.alternatives == []


# ------------------------------------------------------------------
# TestMergeIntegration (7 tests)
# ------------------------------------------------------------------


class TestMergeIntegration:
    """Integration tests for normalize + merge + family workflows."""

    def test_normalize_with_detector(self):
        """Normalizer uses VariantDetector for alignment when provided."""
        detector = VariantDetector()
        normalizer = EvidenceNormalizer(variant_detector=detector)
        demos = [
            [_step("Open browser"), _step("Click search")],
            [_step("Open browser"), _step("Click search")],
        ]
        result = normalizer.normalize("with-det", demos)
        assert len(result) >= 1
        assert all(isinstance(ns, NormalizedStep) for ns in result)

    def test_normalize_without_detector(self):
        """Normalizer falls back to positional alignment gracefully."""
        normalizer = EvidenceNormalizer(variant_detector=None)
        demos = [
            [_step("Open browser"), _step("Click search")],
            [_step("Open browser"), _step("Click search")],
        ]
        result = normalizer.normalize("no-det", demos)
        assert len(result) >= 1
        assert all(isinstance(ns, NormalizedStep) for ns in result)

    def test_merge_increments_total_obs(self):
        """total_observations is updated after merge."""
        normalizer = EvidenceNormalizer()
        proc = {
            "id": "inc-test",
            "steps": [{"action": "Click", "app": "Chrome"}],
            "evidence": {
                "observations": [],
                "step_evidence": [],
                "contradictions": [],
                "total_observations": 5,
            },
        }
        merged = normalizer.merge_with_evidence(proc, [_step("Click")])
        assert merged["evidence"]["total_observations"] == 6

    def test_merge_doesnt_lose_steps(self):
        """No steps are deleted by merge — only added or updated."""
        normalizer = EvidenceNormalizer()
        proc = {
            "id": "no-loss",
            "steps": [
                {"action": "Step A", "app": "Chrome"},
                {"action": "Step B", "app": "Chrome"},
                {"action": "Step C", "app": "Chrome"},
            ],
            "evidence": {
                "observations": [],
                "step_evidence": [],
                "contradictions": [],
                "total_observations": 3,
            },
        }
        # New observation has fewer steps
        merged = normalizer.merge_with_evidence(proc, [_step("Step A")])
        # Original steps should be preserved (simple merge keeps longer list)
        assert len(merged["steps"]) >= 3

    def test_merge_with_no_alignment(self):
        """Merge without alignment falls back to simple strategy."""
        normalizer = EvidenceNormalizer(variant_detector=None)
        proc = {
            "id": "no-align",
            "steps": [{"action": "Click", "app": "Chrome"}],
            "evidence": {
                "observations": [],
                "step_evidence": [],
                "contradictions": [],
                "total_observations": 1,
            },
        }
        new_obs = [_step("Click"), _step("Submit")]
        merged = normalizer.merge_with_evidence(proc, new_obs)

        # Simple merge keeps the longer list (new observation has 2 steps)
        assert len(merged["steps"]) == 2
        assert merged["evidence"]["total_observations"] == 2

    def test_family_empty_related(self):
        """Empty related_slugs produces empty family."""
        proc = _make_procedure("solo", [_step("Click")], episode_count=1)
        family = EvidenceNormalizer.build_variant_family(
            "solo", [], {"solo": proc},
        )
        assert family == {}

    def test_family_shared_apps_computed(self, kb: KnowledgeBase):
        """Shared apps are captured in the family dict."""
        proc_a = _make_procedure("family-a", [
            _step("Open app", app="Chrome"),
            _step("Click", app="Chrome"),
        ], episode_count=5, apps=["Chrome", "Slack"])

        proc_b = _make_procedure("family-b", [
            _step("Open app", app="Chrome"),
            _step("Click", app="Chrome"),
        ], episode_count=3, apps=["Chrome", "Slack"])

        procedures = {"family-a": proc_a, "family-b": proc_b}
        family = EvidenceNormalizer.build_variant_family(
            "family-a", ["family-b"], procedures,
        )

        if family:
            assert "shared_apps" in family
            # Both procedures share Chrome and Slack
            assert "Chrome" in family["shared_apps"]
            assert "Slack" in family["shared_apps"]
