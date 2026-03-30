"""Tests for the correction detector module."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest

from agenthandover_worker.correction_detector import (
    Correction,
    CorrectionDetector,
    CorrectionSummary,
    _parse_annotation,
    _parse_timestamp,
    _get_app,
    _get_what_doing,
    _get_location,
)
from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.llm_reasoning import LLMReasoner, ReasoningResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TIME = datetime(2026, 3, 11, 10, 0, 0, tzinfo=timezone.utc)


def _ts(offset_seconds: int = 0) -> str:
    """Return an ISO timestamp offset from the base time."""
    return (_BASE_TIME + timedelta(seconds=offset_seconds)).isoformat()


def make_event(
    app: str = "Chrome",
    what_doing: str = "",
    location: str = "",
    timestamp: str | None = None,
) -> dict:
    """Create a mock event dict with annotation and window JSON."""
    annotation = {
        "task_context": {"what_doing": what_doing},
        "visual_context": {"active_app": app, "location": location},
    }
    return {
        "kind_json": json.dumps({"FocusChange": {}}),
        "window_json": json.dumps({"app": app, "title": f"{app} Window"}),
        "metadata_json": "{}",
        "scene_annotation_json": json.dumps(annotation),
        "timestamp": timestamp or _ts(),
    }


def make_bare_event() -> dict:
    """Create a minimal event without annotation or window JSON."""
    return {
        "kind_json": json.dumps({"FocusChange": {}}),
        "metadata_json": "{}",
    }


def _make_procedure(slug: str = "test-proc", steps: list | None = None):
    """Create a minimal v3 procedure dict."""
    return {
        "schema_version": "3.0.0",
        "id": slug,
        "title": "Test Procedure",
        "steps": steps or [
            {"step_id": "step_1", "index": 0, "action": "open browser"},
            {"step_id": "step_2", "index": 1, "action": "fill form"},
        ],
    }


def _make_correction(
    slug: str = "test-proc",
    step_id: str | None = None,
    original: str = "old action",
    corrected: str = "new action",
    ctype: str = "edit",
    applied: bool = False,
) -> Correction:
    """Create a Correction for testing."""
    return Correction(
        correction_id="corr-" + corrected.replace(" ", "-"),
        procedure_slug=slug,
        execution_id="exec-1",
        step_id=step_id,
        original_output=original,
        corrected_output=corrected,
        correction_type=ctype,
        detected_at=_ts(),
        applied=applied,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kb(tmp_path):
    """Create a KnowledgeBase rooted in a temp directory."""
    kb = KnowledgeBase(root=tmp_path / "knowledge")
    kb.ensure_structure()
    return kb


@pytest.fixture()
def detector(kb):
    """Create a CorrectionDetector backed by a temp KB."""
    return CorrectionDetector(kb)


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestCorrectionDataclass:
    """Verify Correction and CorrectionSummary dataclass structure."""

    def test_correction_fields(self) -> None:
        c = Correction(
            correction_id="abc",
            procedure_slug="my-proc",
            execution_id="exec-1",
            step_id="step_1",
            original_output="old",
            corrected_output="new",
            correction_type="edit",
            detected_at="2026-03-11T10:00:00+00:00",
        )
        assert c.correction_id == "abc"
        assert c.procedure_slug == "my-proc"
        assert c.correction_type == "edit"
        assert c.applied is False

    def test_correction_applied_default(self) -> None:
        c = _make_correction()
        assert c.applied is False

    def test_summary_fields(self) -> None:
        s = CorrectionSummary(
            procedure_slug="proc",
            total_corrections=3,
            correction_types={"edit": 2, "revert": 1},
            most_corrected_steps=[{"step_id": "step_1", "count": 2}],
            last_correction="2026-03-11T10:00:00+00:00",
        )
        assert s.total_corrections == 3
        assert s.correction_types["edit"] == 2


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for private helper functions."""

    def test_parse_annotation_string(self) -> None:
        event = {"scene_annotation_json": json.dumps({"task_context": {}})}
        assert _parse_annotation(event) == {"task_context": {}}

    def test_parse_annotation_dict(self) -> None:
        event = {"scene_annotation_json": {"task_context": {}}}
        assert _parse_annotation(event) == {"task_context": {}}

    def test_parse_annotation_none(self) -> None:
        assert _parse_annotation({}) is None

    def test_parse_annotation_invalid_json(self) -> None:
        event = {"scene_annotation_json": "{{bad"}
        assert _parse_annotation(event) is None

    def test_parse_annotation_non_dict(self) -> None:
        event = {"scene_annotation_json": json.dumps([1, 2])}
        assert _parse_annotation(event) is None

    def test_parse_timestamp_iso(self) -> None:
        event = {"timestamp": "2026-03-11T10:00:00+00:00"}
        ts = _parse_timestamp(event)
        assert ts is not None
        assert ts.year == 2026

    def test_parse_timestamp_missing(self) -> None:
        assert _parse_timestamp({}) is None

    def test_get_app_from_window_json(self) -> None:
        event = {"window_json": json.dumps({"app": "Safari"})}
        assert _get_app(event) == "Safari"

    def test_get_app_missing(self) -> None:
        assert _get_app({}) == ""

    def test_get_what_doing(self) -> None:
        ann = {"task_context": {"what_doing": "writing code"}}
        assert _get_what_doing(ann) == "writing code"

    def test_get_location(self) -> None:
        ann = {"visual_context": {"location": "https://example.com"}}
        assert _get_location(ann) == "https://example.com"


# ---------------------------------------------------------------------------
# Record and retrieve
# ---------------------------------------------------------------------------


class TestRecordAndRetrieve:
    """Tests for recording and retrieving corrections."""

    def test_record_correction(self, detector: CorrectionDetector) -> None:
        c = _make_correction()
        detector.record_correction(c)
        assert len(detector.get_corrections()) == 1
        assert detector.get_corrections()[0].correction_id == c.correction_id

    def test_record_multiple_corrections(
        self, detector: CorrectionDetector
    ) -> None:
        detector.record_correction(_make_correction(corrected="fix A"))
        detector.record_correction(_make_correction(corrected="fix B"))
        assert len(detector.get_corrections()) == 2

    def test_filter_by_procedure(
        self, detector: CorrectionDetector
    ) -> None:
        detector.record_correction(_make_correction(slug="proc-a"))
        detector.record_correction(_make_correction(slug="proc-b"))
        assert len(detector.get_corrections("proc-a")) == 1
        assert len(detector.get_corrections("proc-b")) == 1
        assert len(detector.get_corrections("proc-c")) == 0

    def test_filter_none_returns_all(
        self, detector: CorrectionDetector
    ) -> None:
        detector.record_correction(_make_correction(slug="proc-a"))
        detector.record_correction(_make_correction(slug="proc-b"))
        assert len(detector.get_corrections(None)) == 2


# ---------------------------------------------------------------------------
# Detection: re-edit pattern
# ---------------------------------------------------------------------------


class TestDetectReEdit:
    """Tests for detecting re-edit corrections."""

    def test_same_app_location_different_action(
        self, detector: CorrectionDetector
    ) -> None:
        events = [
            make_event(
                app="VS Code",
                what_doing="write function",
                location="/src/main.py",
                timestamp=_ts(0),
            ),
            make_event(
                app="VS Code",
                what_doing="rewrite function",
                location="/src/main.py",
                timestamp=_ts(30),
            ),
        ]
        corrections = detector.detect_correction(events, procedure_slug="my-proc")
        assert len(corrections) >= 1
        assert corrections[0].correction_type == "edit"
        assert corrections[0].original_output == "write function"
        assert corrections[0].corrected_output == "rewrite function"

    def test_no_correction_if_same_what_doing(
        self, detector: CorrectionDetector
    ) -> None:
        events = [
            make_event(
                app="VS Code",
                what_doing="write function",
                location="/src/main.py",
                timestamp=_ts(0),
            ),
            make_event(
                app="VS Code",
                what_doing="write function",
                location="/src/main.py",
                timestamp=_ts(30),
            ),
        ]
        corrections = detector.detect_correction(events)
        assert len(corrections) == 0

    def test_no_correction_beyond_120s(
        self, detector: CorrectionDetector
    ) -> None:
        events = [
            make_event(
                app="VS Code",
                what_doing="write function",
                location="/src/main.py",
                timestamp=_ts(0),
            ),
            make_event(
                app="VS Code",
                what_doing="rewrite function",
                location="/src/main.py",
                timestamp=_ts(150),
            ),
        ]
        corrections = detector.detect_correction(events)
        assert len(corrections) == 0

    def test_no_correction_different_app(
        self, detector: CorrectionDetector
    ) -> None:
        events = [
            make_event(
                app="VS Code",
                what_doing="write function",
                location="/src/main.py",
                timestamp=_ts(0),
            ),
            make_event(
                app="Chrome",
                what_doing="rewrite function",
                location="/src/main.py",
                timestamp=_ts(30),
            ),
        ]
        corrections = detector.detect_correction(events)
        assert len(corrections) == 0

    def test_no_correction_different_location(
        self, detector: CorrectionDetector
    ) -> None:
        events = [
            make_event(
                app="VS Code",
                what_doing="write function",
                location="/src/main.py",
                timestamp=_ts(0),
            ),
            make_event(
                app="VS Code",
                what_doing="rewrite function",
                location="/src/other.py",
                timestamp=_ts(30),
            ),
        ]
        corrections = detector.detect_correction(events)
        assert len(corrections) == 0


# ---------------------------------------------------------------------------
# Detection: undo/revert pattern
# ---------------------------------------------------------------------------


class TestDetectRevert:
    """Tests for detecting undo/revert corrections."""

    def test_undo_keyword_in_re_edit(
        self, detector: CorrectionDetector
    ) -> None:
        events = [
            make_event(
                app="VS Code",
                what_doing="add feature",
                location="/src/main.py",
                timestamp=_ts(0),
            ),
            make_event(
                app="VS Code",
                what_doing="undo the change",
                location="/src/main.py",
                timestamp=_ts(10),
            ),
        ]
        corrections = detector.detect_correction(events)
        assert len(corrections) >= 1
        assert corrections[0].correction_type == "revert"

    def test_revert_keyword(self, detector: CorrectionDetector) -> None:
        events = [
            make_event(
                app="VS Code",
                what_doing="apply patch",
                location="/src/main.py",
                timestamp=_ts(0),
            ),
            make_event(
                app="VS Code",
                what_doing="revert the patch",
                location="/src/main.py",
                timestamp=_ts(15),
            ),
        ]
        corrections = detector.detect_correction(events)
        assert any(c.correction_type == "revert" for c in corrections)

    def test_fixing_keyword(self, detector: CorrectionDetector) -> None:
        events = [
            make_event(
                app="Chrome",
                what_doing="submit form",
                location="https://app.com/form",
                timestamp=_ts(0),
            ),
            make_event(
                app="Chrome",
                what_doing="fixing the form values",
                location="https://app.com/form",
                timestamp=_ts(20),
            ),
        ]
        corrections = detector.detect_correction(events)
        assert any(c.correction_type == "revert" for c in corrections)

    def test_standalone_undo_without_re_edit_pair(
        self, detector: CorrectionDetector
    ) -> None:
        """An undo keyword in a standalone event (no matching pair) still
        gets detected via heuristic 2."""
        events = [
            make_event(
                app="VS Code",
                what_doing="undo last change",
                location="/unique/location",
                timestamp=_ts(0),
            ),
        ]
        corrections = detector.detect_correction(events)
        assert len(corrections) == 1
        assert corrections[0].correction_type == "revert"
        assert corrections[0].original_output == ""


# ---------------------------------------------------------------------------
# Detection: edge cases
# ---------------------------------------------------------------------------


class TestDetectEdgeCases:
    """Edge cases in correction detection."""

    def test_empty_events(self, detector: CorrectionDetector) -> None:
        assert detector.detect_correction([]) == []

    def test_no_annotations(self, detector: CorrectionDetector) -> None:
        events = [make_bare_event(), make_bare_event()]
        assert detector.detect_correction(events) == []

    def test_missing_annotation_json(
        self, detector: CorrectionDetector
    ) -> None:
        events = [
            {"kind_json": "{}", "timestamp": _ts()},
            {"kind_json": "{}", "timestamp": _ts(10)},
        ]
        assert detector.detect_correction(events) == []

    def test_invalid_annotation_json(
        self, detector: CorrectionDetector
    ) -> None:
        events = [
            {
                "scene_annotation_json": "{{bad json",
                "timestamp": _ts(),
                "window_json": json.dumps({"app": "Chrome"}),
            },
        ]
        assert detector.detect_correction(events) == []

    def test_events_without_what_doing(
        self, detector: CorrectionDetector
    ) -> None:
        events = [
            make_event(app="Chrome", what_doing="", timestamp=_ts(0)),
            make_event(app="Chrome", what_doing="", timestamp=_ts(10)),
        ]
        assert detector.detect_correction(events) == []


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


class TestCorrectionSummary:
    """Tests for get_summary()."""

    def test_empty_summary(self, detector: CorrectionDetector) -> None:
        summary = detector.get_summary("nonexistent")
        assert summary.total_corrections == 0
        assert summary.correction_types == {}
        assert summary.most_corrected_steps == []
        assert summary.last_correction is None

    def test_summary_counts(self, detector: CorrectionDetector) -> None:
        detector.record_correction(
            _make_correction(slug="proc", ctype="edit", corrected="a")
        )
        detector.record_correction(
            _make_correction(slug="proc", ctype="edit", corrected="b")
        )
        detector.record_correction(
            _make_correction(slug="proc", ctype="revert", corrected="c")
        )
        summary = detector.get_summary("proc")
        assert summary.total_corrections == 3
        assert summary.correction_types == {"edit": 2, "revert": 1}
        assert summary.last_correction is not None

    def test_summary_most_corrected_steps(
        self, detector: CorrectionDetector
    ) -> None:
        detector.record_correction(
            _make_correction(step_id="step_1", corrected="a")
        )
        detector.record_correction(
            _make_correction(step_id="step_1", corrected="b")
        )
        detector.record_correction(
            _make_correction(step_id="step_2", corrected="c")
        )
        summary = detector.get_summary("test-proc")
        assert summary.most_corrected_steps[0]["step_id"] == "step_1"
        assert summary.most_corrected_steps[0]["count"] == 2


# ---------------------------------------------------------------------------
# Apply corrections
# ---------------------------------------------------------------------------


class TestApplyCorrections:
    """Tests for apply_corrections()."""

    def test_apply_with_sufficient_occurrences(
        self, detector: CorrectionDetector, kb: KnowledgeBase
    ) -> None:
        kb.save_procedure(_make_procedure("proc", steps=[
            {"step_id": "step_1", "index": 0, "action": "old action"},
        ]))
        detector.record_correction(
            _make_correction(slug="proc", step_id="step_1", corrected="new action")
        )
        detector.record_correction(
            _make_correction(slug="proc", step_id="step_1", corrected="new action")
        )
        result = detector.apply_corrections("proc", min_occurrences=2)
        assert result["applied"] == 2
        assert result["skipped"] == 0
        # Verify procedure was updated
        proc = kb.get_procedure("proc")
        assert proc is not None
        assert proc["steps"][0]["action"] == "new action"

    def test_skip_with_insufficient_occurrences(
        self, detector: CorrectionDetector, kb: KnowledgeBase
    ) -> None:
        kb.save_procedure(_make_procedure("proc"))
        detector.record_correction(
            _make_correction(slug="proc", step_id="step_1", corrected="new action")
        )
        result = detector.apply_corrections("proc", min_occurrences=2)
        assert result["applied"] == 0
        assert result["skipped"] == 1

    def test_apply_by_original_output_match(
        self, detector: CorrectionDetector, kb: KnowledgeBase
    ) -> None:
        """When step_id is None, match by original_output against step action."""
        kb.save_procedure(_make_procedure("proc", steps=[
            {"step_id": "step_1", "index": 0, "action": "open browser"},
        ]))
        detector.record_correction(
            _make_correction(
                slug="proc", step_id=None,
                original="open browser", corrected="open chrome browser",
            )
        )
        detector.record_correction(
            _make_correction(
                slug="proc", step_id=None,
                original="open browser", corrected="open chrome browser",
            )
        )
        result = detector.apply_corrections("proc", min_occurrences=2)
        assert result["applied"] == 2
        proc = kb.get_procedure("proc")
        assert proc["steps"][0]["action"] == "open chrome browser"

    def test_no_procedure_in_kb(
        self, detector: CorrectionDetector
    ) -> None:
        detector.record_correction(_make_correction(slug="missing"))
        detector.record_correction(_make_correction(slug="missing"))
        result = detector.apply_corrections("missing", min_occurrences=1)
        assert result["skipped"] == 2
        assert result["applied"] == 0

    def test_no_corrections_to_apply(
        self, detector: CorrectionDetector
    ) -> None:
        result = detector.apply_corrections("no-corrections")
        assert result["applied"] == 0
        assert result["skipped"] == 0

    def test_already_applied_corrections_skipped(
        self, detector: CorrectionDetector, kb: KnowledgeBase
    ) -> None:
        kb.save_procedure(_make_procedure("proc", steps=[
            {"step_id": "step_1", "index": 0, "action": "old"},
        ]))
        c = _make_correction(slug="proc", step_id="step_1", corrected="new")
        c.applied = True
        detector.record_correction(c)
        result = detector.apply_corrections("proc", min_occurrences=1)
        assert result["applied"] == 0


# ---------------------------------------------------------------------------
# Persistence (save / load)
# ---------------------------------------------------------------------------


class TestPersistence:
    """Tests for save and load round-trip."""

    def test_round_trip(self, kb: KnowledgeBase) -> None:
        det1 = CorrectionDetector(kb)
        det1.record_correction(_make_correction(corrected="fix alpha"))
        det1.record_correction(_make_correction(corrected="fix beta"))

        # Create a new detector that loads from the same KB
        det2 = CorrectionDetector(kb)
        assert len(det2.get_corrections()) == 2
        outputs = {c.corrected_output for c in det2.get_corrections()}
        assert "fix alpha" in outputs
        assert "fix beta" in outputs

    def test_load_empty_kb(self, kb: KnowledgeBase) -> None:
        det = CorrectionDetector(kb)
        assert det.get_corrections() == []

    def test_load_corrupted_file(self, kb: KnowledgeBase) -> None:
        path = kb.root / "observations" / "corrections.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{{not valid json")
        det = CorrectionDetector(kb)
        assert det.get_corrections() == []

    def test_load_non_dict_file(self, kb: KnowledgeBase) -> None:
        path = kb.root / "observations" / "corrections.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([1, 2, 3]))
        det = CorrectionDetector(kb)
        assert det.get_corrections() == []

    def test_persistence_file_structure(self, kb: KnowledgeBase) -> None:
        det = CorrectionDetector(kb)
        det.record_correction(_make_correction())
        path = kb.root / "observations" / "corrections.json"
        assert path.is_file()
        data = json.loads(path.read_text())
        assert "corrections" in data
        assert "updated_at" in data
        assert len(data["corrections"]) == 1


# ---------------------------------------------------------------------------
# LLM-based guardrail learning
# ---------------------------------------------------------------------------


class TestCorrectionGuardrails:
    """Tests for LLM-based correction pattern analysis and guardrails."""

    def test_analyze_3_corrections_calls_llm(
        self, kb: KnowledgeBase
    ) -> None:
        """When 3+ corrections exist for a group, LLM is called."""
        reasoner = LLMReasoner()
        reasoner.reason_json = MagicMock(return_value=ReasoningResult(
            value={
                "guardrail": "Always use full browser name, not abbreviation",
                "improved_condition": "when specifying the browser application",
                "confidence": 0.85,
            },
            success=True,
        ))

        det = CorrectionDetector(kb, llm_reasoner=reasoner)
        # Record 3 corrections for the same step
        for i in range(3):
            det.record_correction(_make_correction(
                slug="proc-a",
                step_id="step_1",
                original=f"open browser {i}",
                corrected=f"open chrome browser {i}",
            ))

        guardrails = det.analyze_correction_patterns("proc-a", min_corrections=3)

        assert len(guardrails) == 1
        assert guardrails[0]["guardrail"] == "Always use full browser name, not abbreviation"
        assert guardrails[0]["confidence"] == 0.85
        reasoner.reason_json.assert_called_once()

    def test_analyze_below_threshold_skips_llm(
        self, kb: KnowledgeBase
    ) -> None:
        """When fewer than min_corrections exist, LLM is not called."""
        reasoner = LLMReasoner()
        reasoner.reason_json = MagicMock()

        det = CorrectionDetector(kb, llm_reasoner=reasoner)
        # Record only 2 corrections (below threshold of 3)
        for i in range(2):
            det.record_correction(_make_correction(
                slug="proc-b",
                step_id="step_1",
                corrected=f"fix {i}",
            ))

        guardrails = det.analyze_correction_patterns("proc-b", min_corrections=3)

        assert guardrails == []
        reasoner.reason_json.assert_not_called()

    def test_apply_guardrails_updates_procedure(
        self, kb: KnowledgeBase
    ) -> None:
        """apply_guardrails_to_procedure appends to constraints.guardrails."""
        kb.save_procedure(_make_procedure("proc-g", steps=[
            {"step_id": "step_1", "index": 0, "action": "open browser"},
        ]))

        det = CorrectionDetector(kb)
        guardrails = [
            {"guardrail": "Always specify full app name", "confidence": 0.9},
            {"guardrail": "Verify URL before navigating", "confidence": 0.8},
        ]

        result = det.apply_guardrails_to_procedure("proc-g", guardrails)

        assert result is True
        proc = kb.get_procedure("proc-g")
        assert proc is not None
        assert "constraints" in proc
        assert "guardrails" in proc["constraints"]
        assert "Always specify full app name" in proc["constraints"]["guardrails"]
        assert "Verify URL before navigating" in proc["constraints"]["guardrails"]
