"""Tests for the branch extractor."""

from __future__ import annotations

from pathlib import Path

import pytest

from oc_apprentice_worker.branch_extractor import BranchExtractor, ExtractedBranch
from oc_apprentice_worker.knowledge_base import KnowledgeBase
from oc_apprentice_worker.procedure_schema import sop_to_procedure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_step(
    action: str,
    step_id: str | None = None,
    app: str = "Chrome",
    location: str = "",
    input_val: str = "",
) -> dict:
    return {
        "step_id": step_id or "step_1",
        "action": action,
        "app": app,
        "location": location,
        "input": input_val,
    }


def _make_procedure(slug: str = "test-proc") -> dict:
    return sop_to_procedure({
        "slug": slug,
        "title": "Test Procedure",
        "steps": [
            {"action": "Open browser", "confidence": 0.9},
            {"action": "Navigate to site", "confidence": 0.85},
            {"action": "Click button", "confidence": 0.8},
        ],
        "confidence_avg": 0.85,
        "apps_involved": ["Chrome"],
        "source": "passive",
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    return kb


@pytest.fixture()
def extractor(kb: KnowledgeBase) -> BranchExtractor:
    return BranchExtractor(kb)


# ---------------------------------------------------------------------------
# ExtractedBranch dataclass
# ---------------------------------------------------------------------------

class TestExtractedBranch:

    def test_dataclass_fields(self) -> None:
        branch = ExtractedBranch(
            step_id="step_3",
            condition="when input is 'foo'",
            paths=[{"condition": "x", "action": "y", "observed_count": 2}],
            confidence=0.8,
            type="data_dependent",
        )
        assert branch.step_id == "step_3"
        assert branch.condition == "when input is 'foo'"
        assert len(branch.paths) == 1
        assert branch.confidence == 0.8
        assert branch.type == "data_dependent"


# ---------------------------------------------------------------------------
# Empty / trivial inputs
# ---------------------------------------------------------------------------

class TestEmptyInputs:

    def test_no_demos_returns_empty(self, extractor: BranchExtractor) -> None:
        assert extractor.extract_branches("nonexistent") == []

    def test_none_demos_no_procedure(self, extractor: BranchExtractor) -> None:
        result = extractor.extract_branches("missing", demos=None)
        assert result == []

    def test_none_demos_procedure_too_few_observations(
        self, kb: KnowledgeBase, extractor: BranchExtractor
    ) -> None:
        proc = _make_procedure()
        proc["evidence"]["observations"] = [{"date": "2026-03-01"}]
        kb.save_procedure(proc)
        result = extractor.extract_branches("test-proc", demos=None)
        assert result == []

    def test_empty_demos_list(self, extractor: BranchExtractor) -> None:
        result = extractor.extract_branches("slug", demos=[])
        assert result == []

    def test_single_demo(self, extractor: BranchExtractor) -> None:
        demo = [make_step("open"), make_step("click")]
        result = extractor.extract_branches("slug", demos=[demo])
        assert result == []


# ---------------------------------------------------------------------------
# Identical demonstrations — no branches
# ---------------------------------------------------------------------------

class TestIdenticalDemos:

    def test_two_identical_demos(self, extractor: BranchExtractor) -> None:
        demo = [
            make_step("open browser", step_id="step_1"),
            make_step("navigate", step_id="step_2"),
            make_step("click submit", step_id="step_3"),
        ]
        result = extractor.extract_branches("slug", demos=[demo, demo])
        assert result == []

    def test_five_identical_demos(self, extractor: BranchExtractor) -> None:
        demo = [make_step("action_a"), make_step("action_b")]
        result = extractor.extract_branches("slug", demos=[demo] * 5)
        assert result == []

    def test_case_insensitive_action_matching(
        self, extractor: BranchExtractor
    ) -> None:
        demo_a = [make_step("Open Browser")]
        demo_b = [make_step("open browser")]
        result = extractor.extract_branches("slug", demos=[demo_a, demo_b])
        assert result == []

    def test_action_whitespace_stripped(
        self, extractor: BranchExtractor
    ) -> None:
        demo_a = [make_step("  click  ")]
        demo_b = [make_step("click")]
        result = extractor.extract_branches("slug", demos=[demo_a, demo_b])
        assert result == []


# ---------------------------------------------------------------------------
# Divergence detection
# ---------------------------------------------------------------------------

class TestDivergenceDetection:

    def test_step_4_varies_across_5_demos(
        self, extractor: BranchExtractor
    ) -> None:
        """Step 4 differs: 3 demos do 'save', 2 demos do 'export'."""
        common_prefix = [
            make_step("open", step_id="step_1"),
            make_step("edit", step_id="step_2"),
            make_step("review", step_id="step_3"),
        ]
        demo_save = common_prefix + [make_step("save", step_id="step_4")]
        demo_export = common_prefix + [make_step("export", step_id="step_4")]
        demos = [demo_save] * 3 + [demo_export] * 2

        branches = extractor.extract_branches("slug", demos=demos)
        assert len(branches) == 1
        b = branches[0]
        assert b.step_id == "step_4"
        assert b.type == "data_dependent"
        assert len(b.paths) == 2
        counts = {p["action"]: p["observed_count"] for p in b.paths}
        assert counts["save"] == 3
        assert counts["export"] == 2

    def test_three_way_divergence(self, extractor: BranchExtractor) -> None:
        """Three different actions at one step."""
        demos = [
            [make_step("a"), make_step("x", step_id="step_2")],
            [make_step("a"), make_step("y", step_id="step_2")],
            [make_step("a"), make_step("z", step_id="step_2")],
        ]
        branches = extractor.extract_branches("slug", demos=demos)
        assert len(branches) == 1
        assert len(branches[0].paths) == 3

    def test_multiple_divergence_points(
        self, extractor: BranchExtractor
    ) -> None:
        """Divergences at step 1 and step 3."""
        demos = [
            [
                make_step("open_chrome", step_id="step_1"),
                make_step("navigate", step_id="step_2"),
                make_step("click_save", step_id="step_3"),
            ],
            [
                make_step("open_firefox", step_id="step_1"),
                make_step("navigate", step_id="step_2"),
                make_step("click_export", step_id="step_3"),
            ],
        ]
        branches = extractor.extract_branches("slug", demos=demos)
        assert len(branches) == 2
        step_ids = {b.step_id for b in branches}
        assert "step_1" in step_ids
        assert "step_3" in step_ids


# ---------------------------------------------------------------------------
# Alignment of different-length demos
# ---------------------------------------------------------------------------

class TestAlignment:

    def test_shorter_demo_padded_with_none(
        self, extractor: BranchExtractor
    ) -> None:
        """Short demo does not cause index error."""
        demo_long = [
            make_step("a", step_id="s1"),
            make_step("b", step_id="s2"),
            make_step("c", step_id="s3"),
        ]
        demo_short = [
            make_step("a", step_id="s1"),
        ]
        # Should not crash; columns 2 and 3 only have one entry each,
        # so they do not create divergences (need >= 2 non-None entries).
        branches = extractor.extract_branches(
            "slug", demos=[demo_long, demo_short]
        )
        # No divergence: step 1 matches, steps 2/3 have only one entry.
        assert branches == []

    def test_different_length_with_divergence(
        self, extractor: BranchExtractor
    ) -> None:
        demo_a = [
            make_step("open", step_id="s1"),
            make_step("click", step_id="s2"),
            make_step("done", step_id="s3"),
        ]
        demo_b = [
            make_step("open", step_id="s1"),
            make_step("type", step_id="s2"),
        ]
        # Step 2 diverges ("click" vs "type").
        branches = extractor.extract_branches(
            "slug", demos=[demo_a, demo_b]
        )
        assert len(branches) == 1
        assert branches[0].step_id == "s2"

    def test_align_empty_demos_returns_empty(
        self, extractor: BranchExtractor
    ) -> None:
        aligned = extractor._align_demonstrations([])
        assert aligned == []

    def test_alignment_column_count(
        self, extractor: BranchExtractor
    ) -> None:
        demos = [
            [make_step("a"), make_step("b")],
            [make_step("a"), make_step("b"), make_step("c")],
        ]
        aligned = extractor._align_demonstrations(demos)
        assert len(aligned) == 3  # max length
        assert aligned[2][0] is None  # first demo padded


# ---------------------------------------------------------------------------
# Confidence calculation
# ---------------------------------------------------------------------------

class TestConfidence:

    def test_even_split_confidence(self, extractor: BranchExtractor) -> None:
        """2 demos, each with a different action — confidence = 0.5."""
        demos = [
            [make_step("x", step_id="s1")],
            [make_step("y", step_id="s1")],
        ]
        branches = extractor.extract_branches("slug", demos=demos)
        assert len(branches) == 1
        assert branches[0].confidence == 0.5

    def test_dominant_path_confidence(
        self, extractor: BranchExtractor
    ) -> None:
        """4 of 5 demos take one path — confidence = 0.8."""
        demos = (
            [[make_step("main_path")]] * 4
            + [[make_step("rare_path")]]
        )
        branches = extractor.extract_branches("slug", demos=demos)
        assert len(branches) == 1
        assert branches[0].confidence == 0.8

    def test_confidence_rounded(self, extractor: BranchExtractor) -> None:
        """Confidence is rounded to 3 decimal places."""
        demos = (
            [[make_step("a")]] * 2
            + [[make_step("b")]] * 1
        )
        branches = extractor.extract_branches("slug", demos=demos)
        assert len(branches) == 1
        assert branches[0].confidence == round(2 / 3, 3)


# ---------------------------------------------------------------------------
# Branch type classification
# ---------------------------------------------------------------------------

class TestBranchClassification:

    def test_pre_condition_at_step_0(
        self, extractor: BranchExtractor
    ) -> None:
        demos = [
            [make_step("open_chrome", step_id="s1")],
            [make_step("open_firefox", step_id="s1")],
        ]
        branches = extractor.extract_branches("slug", demos=demos)
        assert len(branches) == 1
        assert branches[0].type == "pre_condition"

    def test_error_recovery_branch(
        self, extractor: BranchExtractor
    ) -> None:
        """One demo has a 'retry' step — classified as error_recovery."""
        demos = [
            [make_step("open"), make_step("submit", step_id="s2")],
            [make_step("open"), make_step("retry submit", step_id="s2")],
        ]
        branches = extractor.extract_branches("slug", demos=demos)
        assert len(branches) == 1
        assert branches[0].type == "error_recovery"

    def test_error_recovery_fix_keyword(
        self, extractor: BranchExtractor
    ) -> None:
        demos = [
            [make_step("open"), make_step("proceed", step_id="s2")],
            [make_step("open"), make_step("fix typo", step_id="s2")],
        ]
        branches = extractor.extract_branches("slug", demos=demos)
        assert branches[0].type == "error_recovery"

    def test_error_recovery_undo_keyword(
        self, extractor: BranchExtractor
    ) -> None:
        demos = [
            [make_step("open"), make_step("continue", step_id="s2")],
            [make_step("open"), make_step("undo action", step_id="s2")],
        ]
        branches = extractor.extract_branches("slug", demos=demos)
        assert branches[0].type == "error_recovery"

    def test_error_recovery_fail_keyword(
        self, extractor: BranchExtractor
    ) -> None:
        demos = [
            [make_step("open"), make_step("proceed", step_id="s2")],
            [make_step("open"), make_step("handle failure", step_id="s2")],
        ]
        branches = extractor.extract_branches("slug", demos=demos)
        assert branches[0].type == "error_recovery"

    def test_error_recovery_correct_keyword(
        self, extractor: BranchExtractor
    ) -> None:
        demos = [
            [make_step("open"), make_step("submit", step_id="s2")],
            [make_step("open"), make_step("correct entry", step_id="s2")],
        ]
        branches = extractor.extract_branches("slug", demos=demos)
        assert branches[0].type == "error_recovery"

    def test_error_recovery_error_keyword(
        self, extractor: BranchExtractor
    ) -> None:
        demos = [
            [make_step("open"), make_step("proceed", step_id="s2")],
            [make_step("open"), make_step("dismiss error dialog", step_id="s2")],
        ]
        branches = extractor.extract_branches("slug", demos=demos)
        assert branches[0].type == "error_recovery"

    def test_data_dependent_at_middle_step(
        self, extractor: BranchExtractor
    ) -> None:
        demos = [
            [make_step("open"), make_step("save", step_id="s2")],
            [make_step("open"), make_step("export", step_id="s2")],
        ]
        branches = extractor.extract_branches("slug", demos=demos)
        assert branches[0].type == "data_dependent"


# ---------------------------------------------------------------------------
# Condition inference
# ---------------------------------------------------------------------------

class TestConditionInference:

    def test_infer_from_input(self, extractor: BranchExtractor) -> None:
        demos = [
            [make_step("type", input_val="hello")],
            [make_step("click")],
        ]
        branches = extractor.extract_branches("slug", demos=demos)
        assert len(branches) == 1
        # At least one path should have "when input is" condition
        conditions = [p["condition"] for p in branches[0].paths]
        assert any("when input is" in c for c in conditions)

    def test_infer_from_app(self, extractor: BranchExtractor) -> None:
        demos = [
            [make_step("do thing", app="Safari")],
            [make_step("other thing")],  # default app is Chrome
        ]
        branches = extractor.extract_branches("slug", demos=demos)
        conditions = [p["condition"] for p in branches[0].paths]
        assert any("when app is" in c for c in conditions)

    def test_infer_error_condition(self, extractor: BranchExtractor) -> None:
        demos = [
            [make_step("submit", app="")],
            [make_step("retry after error", app="")],
        ]
        branches = extractor.extract_branches("slug", demos=demos)
        conditions = [p["condition"] for p in branches[0].paths]
        assert any("on error" in c for c in conditions)

    def test_infer_unknown_from_none_entry(
        self, extractor: BranchExtractor
    ) -> None:
        result = extractor._infer_condition(None, [], [])
        assert result == "unknown"

    def test_infer_alternative_path_fallback(
        self, extractor: BranchExtractor
    ) -> None:
        # No input, no distinctive app, no error keywords
        entry = make_step("do something", app="", input_val="")
        result = extractor._infer_condition(entry, [], [0])
        assert result == "alternative path"


# ---------------------------------------------------------------------------
# Merge into knowledge base
# ---------------------------------------------------------------------------

class TestMergeBranches:

    def test_merge_saves_branches_to_procedure(
        self, kb: KnowledgeBase, extractor: BranchExtractor
    ) -> None:
        proc = _make_procedure()
        kb.save_procedure(proc)

        branches = [
            ExtractedBranch(
                step_id="step_2",
                condition="Varies at step_2",
                paths=[
                    {"condition": "when app is Chrome", "action": "save", "observed_count": 3},
                    {"condition": "when app is Firefox", "action": "export", "observed_count": 2},
                ],
                confidence=0.6,
                type="data_dependent",
            )
        ]
        extractor.merge_branches_into_procedure("test-proc", branches)

        loaded = kb.get_procedure("test-proc")
        assert loaded is not None
        assert len(loaded["branches"]) == 1
        assert loaded["branches"][0]["step_id"] == "step_2"
        assert loaded["branches"][0]["type"] == "data_dependent"
        assert loaded["branches"][0]["confidence"] == 0.6
        assert len(loaded["branches"][0]["paths"]) == 2

    def test_merge_overwrites_previous_branches(
        self, kb: KnowledgeBase, extractor: BranchExtractor
    ) -> None:
        proc = _make_procedure()
        proc["branches"] = [{"step_id": "old", "condition": "old"}]
        kb.save_procedure(proc)

        branches = [
            ExtractedBranch(
                step_id="step_1",
                condition="new",
                paths=[],
                confidence=0.9,
                type="pre_condition",
            )
        ]
        extractor.merge_branches_into_procedure("test-proc", branches)

        loaded = kb.get_procedure("test-proc")
        assert len(loaded["branches"]) == 1
        assert loaded["branches"][0]["step_id"] == "step_1"

    def test_merge_nonexistent_procedure_is_noop(
        self, extractor: BranchExtractor
    ) -> None:
        branches = [
            ExtractedBranch(
                step_id="s1",
                condition="c",
                paths=[],
                confidence=0.5,
                type="pre_condition",
            )
        ]
        # Should not raise.
        extractor.merge_branches_into_procedure("nonexistent", branches)

    def test_merge_empty_branches_clears_list(
        self, kb: KnowledgeBase, extractor: BranchExtractor
    ) -> None:
        proc = _make_procedure()
        proc["branches"] = [{"step_id": "old"}]
        kb.save_procedure(proc)

        extractor.merge_branches_into_procedure("test-proc", [])

        loaded = kb.get_procedure("test-proc")
        assert loaded["branches"] == []

    def test_merge_multiple_branches(
        self, kb: KnowledgeBase, extractor: BranchExtractor
    ) -> None:
        proc = _make_procedure()
        kb.save_procedure(proc)

        branches = [
            ExtractedBranch(
                step_id="step_1", condition="c1",
                paths=[], confidence=0.7, type="pre_condition",
            ),
            ExtractedBranch(
                step_id="step_3", condition="c2",
                paths=[], confidence=0.8, type="error_recovery",
            ),
        ]
        extractor.merge_branches_into_procedure("test-proc", branches)

        loaded = kb.get_procedure("test-proc")
        assert len(loaded["branches"]) == 2
        types = {b["type"] for b in loaded["branches"]}
        assert types == {"pre_condition", "error_recovery"}


# ---------------------------------------------------------------------------
# Full pipeline (extract + merge)
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_extract_and_merge_round_trip(
        self, kb: KnowledgeBase, extractor: BranchExtractor
    ) -> None:
        proc = _make_procedure()
        kb.save_procedure(proc)

        demos = [
            [
                make_step("open browser", step_id="step_1"),
                make_step("go to site", step_id="step_2"),
                make_step("click save", step_id="step_3"),
            ],
            [
                make_step("open browser", step_id="step_1"),
                make_step("go to site", step_id="step_2"),
                make_step("click export", step_id="step_3"),
            ],
        ]

        branches = extractor.extract_branches("test-proc", demos=demos)
        assert len(branches) == 1
        assert branches[0].step_id == "step_3"

        extractor.merge_branches_into_procedure("test-proc", branches)
        loaded = kb.get_procedure("test-proc")
        assert len(loaded["branches"]) == 1
        assert loaded["branches"][0]["step_id"] == "step_3"

    def test_step_id_fallback_to_positional(
        self, extractor: BranchExtractor
    ) -> None:
        """When steps lack step_id, the extractor generates positional IDs."""
        demos = [
            [{"action": "open"}, {"action": "save"}],
            [{"action": "open"}, {"action": "export"}],
        ]
        branches = extractor.extract_branches("slug", demos=demos)
        assert len(branches) == 1
        # Should get a positional step_id like "step_2"
        assert branches[0].step_id == "step_2"
