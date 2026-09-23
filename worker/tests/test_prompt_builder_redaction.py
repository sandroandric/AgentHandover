"""The six prompt builders WP-D-F1 left unrouted now go through
``redaction.redact`` (WP-F5): a password phrase and a token URL in their
input never reach the prompt the model sees.

Each test drives the builder's real public entry point with a reasoner
that records the prompt, then checks the prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agenthandover_worker.llm_reasoning import ReasoningResult

PASSWORD = "Hunter2!xQ"
TOKEN = "tok3nS3cretValue42"
# What a user or the screen could put into any of these inputs.
LEAKY = (
    f"my password is {PASSWORD} then open "
    f"https://app.example.com/reset?token={TOKEN}&page=2"
)


class RecordingReasoner:
    """Stands in for LLMReasoner: records each prompt, answers nothing
    useful (the builders' fallbacks handle an abstention)."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def _record(self, prompt: str) -> ReasoningResult:
        self.prompts.append(prompt)
        return ReasoningResult(success=True, abstained=True)

    def reason_json(self, prompt: str, **_kwargs) -> ReasoningResult:
        return self._record(prompt)

    def reason_yesno(self, prompt: str, **_kwargs) -> ReasoningResult:
        return self._record(prompt)

    def make_provenance(self, *_args, **_kwargs) -> dict:
        return {}


def assert_clean(prompts: list[str]) -> None:
    assert prompts, "the builder sent no prompt"
    for prompt in prompts:
        assert PASSWORD not in prompt
        assert TOKEN not in prompt
        assert "token=" not in prompt
        # The redaction left its marks, so the input did reach the prompt.
        assert "[REDACTED_PASSWORD]" in prompt
        assert "https://app.example.com/reset?page=2" in prompt


@pytest.fixture()
def reasoner() -> RecordingReasoner:
    return RecordingReasoner()


def test_session_linker_semantic_check(tmp_path: Path, reasoner) -> None:
    from agenthandover_worker.knowledge_base import KnowledgeBase
    from agenthandover_worker.session_linker import SessionLinker

    kb = KnowledgeBase(root=tmp_path / "kb")
    kb.ensure_structure()
    linker = SessionLinker(kb, llm_reasoner=reasoner)

    linker._semantic_check(LEAKY, f"reset account, {LEAKY}")

    assert_clean(reasoner.prompts)


def test_correction_detector_guardrail_prompt(tmp_path: Path, reasoner) -> None:
    from agenthandover_worker.correction_detector import Correction, CorrectionDetector
    from agenthandover_worker.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(root=tmp_path / "kb")
    kb.ensure_structure()
    detector = CorrectionDetector(kb, llm_reasoner=reasoner)
    for i in range(3):
        detector.record_correction(Correction(
            correction_id=f"c{i}", procedure_slug="reset-account",
            execution_id=f"e{i}", step_id="s1",
            original_output=LEAKY, corrected_output=f"fixed: {LEAKY}",
            correction_type="edit", detected_at="2026-09-23T10:00:00Z",
        ))

    detector.analyze_correction_patterns("reset-account", min_corrections=3)

    assert_clean(reasoner.prompts)


def test_outcome_tracker_llm_outcomes(reasoner) -> None:
    from agenthandover_worker.outcome_tracker import OutcomeTracker

    events = [
        {
            "kind_json": json.dumps({"FocusChange": {}}),
            "scene_annotation_json": json.dumps({
                "visual_context": {"active_app": "Safari", "location": LEAKY},
                "task_context": {"what_doing": LEAKY},
            }),
        }
        for _ in range(3)
    ]

    OutcomeTracker(llm_reasoner=reasoner)._detect_outcomes_with_llm(events)

    assert_clean(reasoner.prompts)


def test_quality_judge_assessment_prompt(reasoner) -> None:
    from agenthandover_worker.quality_judge import QualityJudge

    procedure = {
        "id": "reset-account",
        "title": f"Reset account ({LEAKY})",
        "steps": [{"action": LEAKY, "app": "Safari"}],
        "evidence": {},
    }

    QualityJudge(reasoner).assess(procedure, "reviewed")

    assert_clean(reasoner.prompts)


def test_sop_dedup_conflict_prompt(reasoner) -> None:
    from agenthandover_worker.sop_dedup import _resolve_conflict_with_llm

    _resolve_conflict_with_llm(
        reasoner, {"step": LEAKY}, {"action": f"type {LEAKY}"},
    )

    assert_clean(reasoner.prompts)


def test_style_analyzer_samples(reasoner) -> None:
    from agenthandover_worker.style_analyzer import analyze_style

    analyze_style([LEAKY, f"Hi team, {LEAKY}"], llm_reasoner=reasoner)

    assert_clean(reasoner.prompts)
