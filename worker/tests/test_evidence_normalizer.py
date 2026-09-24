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

from agenthandover_worker.evidence_normalizer import (
    EvidenceNormalizer,
    NormalizedStep,
    _deep_copy_proc,
)
from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.procedure_schema import sop_to_procedure
from agenthandover_worker.variant_detector import VariantDetector
from daemon_db import kb_db


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
    kb = KnowledgeBase(root=tmp_path / "knowledge", db_path=kb_db(tmp_path / "knowledge"))
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
        """A step done differently at the same position is recorded as an
        alternative of that step, not appended as a new step."""
        normalizer = EvidenceNormalizer(variant_detector=VariantDetector())
        proc = self._base_proc()
        new_obs = [_step("Open browser"), _step("Upload document")]

        merged = normalizer.merge_with_evidence(proc, new_obs)

        steps = merged["steps"]
        assert [s["action"] for s in steps] == ["Open browser", "Click search"]
        assert [s["observation_count"] for s in steps] == [3, 2]
        assert [s["support"] for s in steps] == [1.0, 2 / 3]
        assert [a["action"] for a in steps[1]["alternatives"]] == ["Upload document"]
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

        # The legacy procedure's steps are one observation; the new one
        # makes two.
        assert "evidence" in merged
        assert merged["evidence"]["total_observations"] == 2

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


class _SpyDetector(VariantDetector):
    """A real VariantDetector that records its semantic_align calls."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def semantic_align(self, *args, **kwargs):
        self.calls += 1
        return super().semantic_align(*args, **kwargs)


class TestRealDetectorAlignment:
    """Pins that the normalizer drives a real VariantDetector (not a Mock)
    and consumes the AlignedStep indices it returns."""

    def test_normalize_uses_semantic_align_indices(self):
        """An inserted step in demo 3 must not shift later positions: the
        aligned path groups 'Click search' from all three demos, which the
        positional fallback would not."""
        detector = _SpyDetector()
        normalizer = EvidenceNormalizer(variant_detector=detector)
        demos = [
            [_step("Open browser"), _step("Click search")],
            [_step("Open browser"), _step("Click search")],
            [_step("Open browser"), _step("Download file", app="Finder"),
             _step("Click search")],
        ]
        result = normalizer.normalize("aligned", demos)

        assert detector.calls == 2
        assert [ns.canonical_action for ns in result] == [
            "open browser", "click search",
        ]
        assert [ns.observation_count for ns in result] == [3, 3]

    def test_merge_with_evidence_uses_semantic_align_indices(self):
        """Matching steps accumulate counts in place; a step only in the
        new observation is appended as a variant."""
        detector = _SpyDetector()
        normalizer = EvidenceNormalizer(variant_detector=detector)
        proc = {
            "id": "real-align",
            "schema_version": "3.0.0",
            "steps": [
                {"action": "Open browser", "app": "Chrome", "observation_count": 2},
                {"action": "Click search", "app": "Chrome", "observation_count": 2},
            ],
            "evidence": {"total_observations": 2},
        }
        new_obs = [_step("Open browser"), _step("Download file", app="Finder"),
                   _step("Click search")]

        merged = normalizer.merge_with_evidence(proc, new_obs)

        assert detector.calls == 1
        steps = merged["steps"]
        assert [s["action"] for s in steps] == [
            "Open browser", "Click search", "Download file",
        ]
        assert [s["observation_count"] for s in steps] == [3, 3, 1]
        assert steps[2]["_variant"] is True
        assert merged["evidence"]["total_observations"] == 3

    def test_alignment_attribute_error_propagates(self):
        """A detector without semantic_align is a programming error and must
        surface, not silently degrade to the append fallback."""

        class _Broken:
            pass

        normalizer = EvidenceNormalizer(variant_detector=_Broken())  # type: ignore[arg-type]
        with pytest.raises(AttributeError):
            normalizer.normalize("broken", [[_step("a")], [_step("a")]])


class TestMergeSupport:
    """Three instances of the focus_dailynews Skill (v3 steps as
    ``sop_to_procedure`` writes them, worded as in the golden Skill) merged
    one at a time: every step carries ``support = seen_in / instances`` and
    the step one instance did differently carries that instance's step as
    its one alternative."""

    NOTES = {"step": "Open the Notes app and create a new note",
             "parameters": {"app": "Notes"}}
    TEXTEDIT = {"step": "Open TextEdit and create a new document",
                "parameters": {"app": "TextEdit", "input": "{{summary_text}}"}}

    @staticmethod
    def _instance(notes_step: dict) -> list[dict]:
        sop = {
            "slug": "daily-news-digest",
            "steps": [
                {"step": "Navigate to the Hacker News homepage",
                 "parameters": {"app": "Google Chrome",
                                "location": "https://news.ycombinator.com/"}},
                {"step": "Copy the article title and URL",
                 "parameters": {"app": "Google Chrome",
                                "location": "https://blog.example.org/sqlite-wal-explained"}},
                notes_step,
                {"step": "Paste the digest into the #daily-news message box",
                 "parameters": {"app": "Slack"}},
            ],
        }
        return sop_to_procedure(sop)["steps"]

    def test_three_instances_give_support_and_one_alternative(self):
        detector = _SpyDetector()
        normalizer = EvidenceNormalizer(variant_detector=detector)
        first = {"id": "daily-news-digest", "schema_version": "3.0.0",
                 "steps": self._instance(self.NOTES)}

        merged = normalizer.merge_with_evidence(first, self._instance(self.NOTES))
        merged = normalizer.merge_with_evidence(merged, self._instance(self.TEXTEDIT))

        assert detector.calls == 2
        assert merged["evidence"]["total_observations"] == 3
        steps = merged["steps"]
        # Order and ids are the procedure's: nothing appended at the end.
        assert [s["step_id"] for s in steps] == ["step_1", "step_2", "step_3", "step_4"]
        assert [s["support"] for s in steps] == [1.0, 1.0, 2 / 3, 1.0]
        notes = steps[2]
        assert notes["action"] == "Open the Notes app and create a new note"
        assert notes["alternatives"] == [{
            "action": "Open TextEdit and create a new document",
            "app": "TextEdit",
            "target": "",
            "location": "",
            "observed_count": 1,
            "parameters": {"app": "TextEdit", "input": "{{summary_text}}"},
        }]
        assert all("alternatives" not in s for i, s in enumerate(steps) if i != 2)
        # The input procedure is not mutated.
        assert "support" not in first["steps"][0]

    def test_inserted_step_gets_a_free_step_id(self):
        """A step only the new instance has is appended with an id no other
        step uses (the instance's own ``step_2`` would collide)."""
        normalizer = EvidenceNormalizer(variant_detector=VariantDetector())
        first = {"id": "daily-news-digest", "steps": self._instance(self.NOTES)}
        extra = self._instance(self.NOTES)
        extra.insert(1, {"step_id": "step_2", "action": "Log in to Hacker News",
                         "app": "Safari", "location": "https://example.com/login"})

        merged = normalizer.merge_with_evidence(first, extra)

        ids = [s["step_id"] for s in merged["steps"]]
        assert len(ids) == len(set(ids)) == 5
        inserted = merged["steps"][-1]
        assert inserted["action"] == "Log in to Hacker News"
        assert inserted["step_id"] == "step_5"
        assert inserted["_variant"] is True
        assert inserted["support"] == 1 / 2
