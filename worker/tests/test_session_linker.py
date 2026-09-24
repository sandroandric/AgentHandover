"""Tests for the SessionLinker module."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from agenthandover_worker.knowledge_base import KnowledgeBase
from daemon_db import kb_db
from agenthandover_worker.session_linker import (
    STOP_WORDS,
    LinkedTask,
    SessionLinker,
    _JACCARD_THRESHOLD,
    _STALE_DAYS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    """Create a KnowledgeBase in a temp directory."""
    kb = KnowledgeBase(root=tmp_path / "knowledge", db_path=kb_db(tmp_path / "knowledge"))
    kb.ensure_structure()
    return kb


def _save_daily(
    kb: KnowledgeBase,
    date: str,
    tasks: list[dict],
) -> None:
    """Helper: save a daily summary with given tasks."""
    kb.save_daily_summary(date, {"tasks": tasks})


# ---------------------------------------------------------------------------
# Tests: intent normalization
# ---------------------------------------------------------------------------


class TestNormalizeIntent:
    def test_basic_normalization(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        result = linker._normalize_intent("Deploy the API server")
        assert "deploy" in result
        assert "api" in result
        assert "server" in result
        # Stop words removed
        assert "the" not in result

    def test_punctuation_removed(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        result = linker._normalize_intent("Deploy! the API, server.")
        assert "deploy" in result
        assert "!" not in result
        assert "," not in result
        assert "." not in result

    def test_case_insensitive(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        r1 = linker._normalize_intent("Deploy API")
        r2 = linker._normalize_intent("deploy api")
        assert r1 == r2

    def test_stop_words_removed(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        result = linker._normalize_intent("a test for the project")
        # Only "test" and "project" should remain (single-char tokens removed)
        assert "test" in result
        assert "project" in result
        assert "a" not in result.split()
        assert "the" not in result.split()
        assert "for" not in result.split()

    def test_single_char_tokens_removed(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        result = linker._normalize_intent("I do X on Y")
        # Single char tokens 'I', 'X', 'Y' removed; stop word 'on' removed;
        # 'do' is kept (2 chars, not a stop word)
        assert result == "do"


# ---------------------------------------------------------------------------
# Tests: intent similarity
# ---------------------------------------------------------------------------


class TestIntentSimilarity:
    def test_identical_intents(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        sim, _ = linker._intent_similarity("Deploy API server", "Deploy API server")
        assert sim == 1.0

    def test_completely_different(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        sim, _ = linker._intent_similarity("Deploy API server", "Cook dinner tonight")
        assert sim == 0.0

    def test_partial_overlap(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        sim, _ = linker._intent_similarity("Deploy API server", "Deploy API client")
        # {"deploy", "api", "server"} vs {"deploy", "api", "client"}
        # intersection=2, union=4 -> 0.5
        assert sim == pytest.approx(0.5, abs=0.01)

    def test_above_threshold(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        sim, _ = linker._intent_similarity(
            "Update project documentation", "Update documentation project"
        )
        assert sim >= _JACCARD_THRESHOLD

    def test_both_empty_returns_one(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        sim, _ = linker._intent_similarity("the", "a")  # All stop words
        assert sim == 1.0

    def test_one_empty_returns_zero(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        sim, _ = linker._intent_similarity("the", "deploy server")
        assert sim == 0.0


# ---------------------------------------------------------------------------
# Tests: link creation from daily summaries
# ---------------------------------------------------------------------------


class TestLinkCreation:
    def test_same_intent_across_days_creates_link(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, "2025-03-02", [
            {"intent": "Deploy API server", "duration_minutes": 20, "apps": ["Terminal"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert len(links) == 1
        assert links[0].intent == "Deploy API server"
        assert links[0].total_duration_minutes == 50
        assert links[0].span_days >= 2
        assert len(links[0].sessions) == 2

    def test_similar_intent_creates_link(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", [
            {"intent": "Deploy API server to production", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, "2025-03-02", [
            {"intent": "Deploy API server staging", "duration_minutes": 20, "apps": ["Terminal"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        # "deploy api server production" vs "deploy api server staging"
        # intersection=3, union=5 -> 0.6 >= 0.4
        assert len(links) == 1

    def test_unrelated_tasks_no_link(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, "2025-03-02", [
            {"intent": "Cook dinner tonight", "duration_minutes": 60, "apps": ["Safari"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert len(links) == 0

    def test_same_day_tasks_not_linked(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
            {"intent": "Deploy API server again", "duration_minutes": 15, "apps": ["Terminal"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        # Same day — should not create a cross-day link
        assert len(links) == 0

    def test_matched_procedure_links_tasks(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", [
            {"intent": "something unique alpha", "duration_minutes": 10, "apps": ["Xcode"],
             "matched_procedure": "deploy-api"},
        ])
        _save_daily(kb, "2025-03-02", [
            {"intent": "totally different beta", "duration_minutes": 20, "apps": ["Xcode"],
             "matched_procedure": "deploy-api"},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert len(links) == 1
        assert links[0].matched_procedure == "deploy-api"

    def test_multiple_distinct_links(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
            {"intent": "Write unit tests", "duration_minutes": 45, "apps": ["VS Code"]},
        ])
        _save_daily(kb, "2025-03-02", [
            {"intent": "Deploy API server", "duration_minutes": 20, "apps": ["Terminal"]},
            {"intent": "Write unit tests", "duration_minutes": 30, "apps": ["VS Code"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert len(links) == 2


# ---------------------------------------------------------------------------
# Tests: active vs stale status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_recent_link_is_active(self, kb: KnowledgeBase) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        _save_daily(kb, yesterday, [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, today, [
            {"intent": "Deploy API server", "duration_minutes": 20, "apps": ["Terminal"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert len(links) == 1
        assert links[0].status == "active"

    def test_old_link_becomes_stale(self, kb: KnowledgeBase) -> None:
        old_date1 = (datetime.now(timezone.utc) - timedelta(days=_STALE_DAYS + 5)).strftime("%Y-%m-%d")
        old_date2 = (datetime.now(timezone.utc) - timedelta(days=_STALE_DAYS + 3)).strftime("%Y-%m-%d")
        _save_daily(kb, old_date1, [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, old_date2, [
            {"intent": "Deploy API server", "duration_minutes": 20, "apps": ["Terminal"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert len(links) == 1
        assert links[0].status == "stale"


# ---------------------------------------------------------------------------
# Tests: mark completed
# ---------------------------------------------------------------------------


class TestMarkCompleted:
    def test_mark_existing_completed(self, kb: KnowledgeBase) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        _save_daily(kb, yesterday, [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, today, [
            {"intent": "Deploy API server", "duration_minutes": 20, "apps": ["Terminal"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert len(links) == 1

        result = linker.mark_completed(links[0].link_id)
        assert result is True
        assert links[0].status == "completed"
        assert len(linker.get_active_links()) == 0

    def test_mark_nonexistent_returns_false(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        assert linker.mark_completed("nonexistent-id") is False


# ---------------------------------------------------------------------------
# Tests: get_active_links
# ---------------------------------------------------------------------------


class TestGetActiveLinks:
    def test_only_active_returned(self, kb: KnowledgeBase) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        _save_daily(kb, yesterday, [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
            {"intent": "Write unit tests", "duration_minutes": 45, "apps": ["VS Code"]},
        ])
        _save_daily(kb, today, [
            {"intent": "Deploy API server", "duration_minutes": 20, "apps": ["Terminal"]},
            {"intent": "Write unit tests", "duration_minutes": 30, "apps": ["VS Code"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert len(links) == 2

        linker.mark_completed(links[0].link_id)
        active = linker.get_active_links()
        assert len(active) == 1


# ---------------------------------------------------------------------------
# Tests: persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_links_survive_reload(self, kb: KnowledgeBase) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        _save_daily(kb, yesterday, [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, today, [
            {"intent": "Deploy API server", "duration_minutes": 20, "apps": ["Terminal"]},
        ])

        linker1 = SessionLinker(kb)
        linker1.analyze_daily_summaries()
        assert len(linker1._links) == 1

        linker2 = SessionLinker(kb)
        assert len(linker2._links) == 1
        assert linker2._links[0].intent == "Deploy API server"

    def test_completed_state_persists(self, kb: KnowledgeBase) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        _save_daily(kb, yesterday, [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, today, [
            {"intent": "Deploy API server", "duration_minutes": 20, "apps": ["Terminal"]},
        ])

        linker1 = SessionLinker(kb)
        links = linker1.analyze_daily_summaries()
        linker1.mark_completed(links[0].link_id)

        linker2 = SessionLinker(kb)
        assert linker2._links[0].status == "completed"

    def test_empty_kb_loads_no_links(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        assert linker._links == []
        assert linker.get_active_links() == []


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_intent_skipped(self, kb: KnowledgeBase) -> None:
        _save_daily(kb, "2025-03-01", [
            {"intent": "", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, "2025-03-02", [
            {"intent": "", "duration_minutes": 20, "apps": ["Terminal"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert len(links) == 0

    def test_no_daily_summaries(self, kb: KnowledgeBase) -> None:
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert links == []

    def test_description_fallback_used(self, kb: KnowledgeBase) -> None:
        """When 'intent' is missing, fall back to 'description'."""
        _save_daily(kb, "2025-03-01", [
            {"description": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, "2025-03-02", [
            {"description": "Deploy API server", "duration_minutes": 20, "apps": ["Terminal"]},
        ])
        linker = SessionLinker(kb)
        links = linker.analyze_daily_summaries()
        assert len(links) == 1


# ---------------------------------------------------------------------------
# Tests: LLM-based session linking
# ---------------------------------------------------------------------------


class TestLLMSessionLinking:
    """Tests for LLM semantic check in ambiguous similarity range."""

    def test_ambiguous_similarity_llm_yes_links(self, kb: KnowledgeBase) -> None:
        """Ambiguous similarity (0.2-0.5) + LLM YES = tasks linked."""
        from agenthandover_worker.llm_reasoning import LLMReasoner

        # "Deploy staging server" vs "Push staging build"
        # Jaccard: {"deploy","staging","server"} vs {"push","staging","build"}
        # intersection=1, union=5 -> 0.2 (ambiguous range)
        _save_daily(kb, "2025-03-01", [
            {"intent": "Deploy staging server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, "2025-03-02", [
            {"intent": "Push staging build", "duration_minutes": 20, "apps": ["Terminal"]},
        ])

        reasoner = LLMReasoner()

        def mock_ollama(prompt, system, num_predict=None, **kwargs):
            return ("Yes, these describe the same workflow", 0.3)

        with patch.object(LLMReasoner, "_call_ollama", side_effect=mock_ollama):
            linker = SessionLinker(kb, llm_reasoner=reasoner)
            links = linker.analyze_daily_summaries()

        assert len(links) == 1

    def test_ambiguous_similarity_llm_no_skips(self, kb: KnowledgeBase) -> None:
        """Ambiguous similarity + LLM NO = tasks not linked."""
        from agenthandover_worker.llm_reasoning import LLMReasoner

        _save_daily(kb, "2025-03-01", [
            {"intent": "Deploy staging server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, "2025-03-02", [
            {"intent": "Push staging build", "duration_minutes": 20, "apps": ["Terminal"]},
        ])

        reasoner = LLMReasoner()

        def mock_ollama(prompt, system, num_predict=None, **kwargs):
            return ("No, these are different workflows", 0.3)

        with patch.object(LLMReasoner, "_call_ollama", side_effect=mock_ollama):
            linker = SessionLinker(kb, llm_reasoner=reasoner)
            links = linker.analyze_daily_summaries()

        assert len(links) == 0

    def test_high_similarity_skips_llm(self, kb: KnowledgeBase) -> None:
        """High similarity (>0.5) should not call LLM, just use heuristic."""
        from agenthandover_worker.llm_reasoning import LLMReasoner

        # Identical intents -> similarity 1.0, well above threshold
        _save_daily(kb, "2025-03-01", [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, "2025-03-02", [
            {"intent": "Deploy API server", "duration_minutes": 20, "apps": ["Terminal"]},
        ])

        reasoner = LLMReasoner()
        call_count = 0

        def mock_ollama(prompt, system, num_predict=None, **kwargs):
            nonlocal call_count
            call_count += 1
            return ("Yes", 0.3)

        with patch.object(LLMReasoner, "_call_ollama", side_effect=mock_ollama):
            linker = SessionLinker(kb, llm_reasoner=reasoner)
            links = linker.analyze_daily_summaries()

        assert len(links) == 1
        # LLM should not have been called (similarity > 0.5)
        assert call_count == 0

    def test_low_similarity_skips_llm(self, kb: KnowledgeBase) -> None:
        """Low similarity (<0.2) should not call LLM."""
        from agenthandover_worker.llm_reasoning import LLMReasoner

        # Completely different intents -> similarity 0.0
        _save_daily(kb, "2025-03-01", [
            {"intent": "Deploy API server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, "2025-03-02", [
            {"intent": "Cook dinner tonight", "duration_minutes": 60, "apps": ["Safari"]},
        ])

        reasoner = LLMReasoner()
        call_count = 0

        def mock_ollama(prompt, system, num_predict=None, **kwargs):
            nonlocal call_count
            call_count += 1
            return ("Yes", 0.3)

        with patch.object(LLMReasoner, "_call_ollama", side_effect=mock_ollama):
            linker = SessionLinker(kb, llm_reasoner=reasoner)
            links = linker.analyze_daily_summaries()

        assert len(links) == 0
        # LLM should not have been called (similarity < 0.2)
        assert call_count == 0

    def test_llm_failure_uses_threshold(self, kb: KnowledgeBase) -> None:
        """When LLM fails, fall back to heuristic threshold."""
        from agenthandover_worker.llm_reasoning import LLMReasoner

        # Ambiguous similarity — LLM fails, should fall through to heuristic
        _save_daily(kb, "2025-03-01", [
            {"intent": "Deploy staging server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, "2025-03-02", [
            {"intent": "Push staging build", "duration_minutes": 20, "apps": ["Terminal"]},
        ])

        reasoner = LLMReasoner()

        def mock_ollama(prompt, system, num_predict=None, **kwargs):
            raise ConnectionError("Ollama not reachable")

        with patch.object(LLMReasoner, "_call_ollama", side_effect=mock_ollama):
            linker = SessionLinker(kb, llm_reasoner=reasoner)
            links = linker.analyze_daily_summaries()

        # Similarity ~0.2, below _JACCARD_THRESHOLD (0.4), LLM failed -> no link
        assert len(links) == 0

    def test_llm_called_for_wider_range(self, kb: KnowledgeBase) -> None:
        """LLM fires for similarity 0.18 (was 0.2+ before the wider range)."""
        from agenthandover_worker.llm_reasoning import LLMReasoner

        # "Update staging backend" vs "Patch production frontend"
        # Jaccard: {"update","staging","backend"} vs {"patch","production","frontend"}
        # intersection=0, union=6 -> 0.0
        # That's too low. Let's pick something ~0.18:
        # "Deploy staging server" vs "Rebuild staging cache"
        # {"deploy","staging","server"} vs {"rebuild","staging","cache"}
        # intersection=1 ("staging"), union=5 -> 0.2
        # Need ~0.18. Tricky to get exact. Let's use a slightly different pair:
        # "Deploy staging API server" vs "Rebuild staging cache service"
        # {"deploy","staging","api","server"} vs {"rebuild","staging","cache","service"}
        # intersection=1, union=7 -> ~0.143 -- too low
        # "Deploy staging build server" vs "Rebuild staging cache"
        # {"deploy","staging","build","server"} vs {"rebuild","staging","cache"}
        # intersection=1, union=6 -> ~0.167 -- slightly above 0.15, in new range
        _save_daily(kb, "2025-03-01", [
            {"intent": "Deploy staging build server", "duration_minutes": 30, "apps": ["Terminal"]},
        ])
        _save_daily(kb, "2025-03-02", [
            {"intent": "Rebuild staging cache", "duration_minutes": 20, "apps": ["Terminal"]},
        ])

        reasoner = LLMReasoner()
        call_count = 0

        def mock_ollama(prompt, system, num_predict=None, **kwargs):
            nonlocal call_count
            call_count += 1
            return ("Yes, same workflow", 0.3)

        with patch.object(LLMReasoner, "_call_ollama", side_effect=mock_ollama):
            linker = SessionLinker(kb, llm_reasoner=reasoner)
            links = linker.analyze_daily_summaries()

        # LLM called: similarity ~0.167 is in the band just below 0.4
        assert call_count >= 1
        # LLM said YES, so tasks should be linked
        assert len(links) == 1


# ---------------------------------------------------------------------------
# The LLM band and the per-metric thresholds (#163)
# ---------------------------------------------------------------------------


class _CountingReasoner:
    """An LLM reasoner that counts its YES/NO calls."""

    def __init__(self, answer: bool) -> None:
        self.calls = 0
        self._answer = answer

    def reason_yesno(self, prompt: str, caller: str = ""):
        from agenthandover_worker.llm_reasoning import ReasoningResult

        self.calls += 1
        return ReasoningResult(value=self._answer, success=True)


class _FakeVectorKB:
    """compute_embeddings returning fixed vectors per text."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.embedded: list[str] = []

    def compute_embeddings(self, texts, *, prefix=""):
        self.embedded += list(texts)
        return [self.vectors[t] for t in texts]


class TestLLMBand:
    def test_pair_above_the_threshold_links_without_a_model_call(self, kb: KnowledgeBase) -> None:
        """Similarity 0.55 used to fall in the old [0.15, 0.6] band: a model
        call whose verdict was discarded (the pair linked either way)."""
        a = "Review quarterly sales pipeline report draft alpha"
        b = "Review quarterly sales pipeline report final beta"
        linker = SessionLinker(kb, llm_reasoner=_CountingReasoner(answer=False))
        score, metric = linker._intent_similarity(a, b)
        assert metric == "jaccard" and score == pytest.approx(5 / 9)  # 0.556
        _save_daily(kb, "2025-03-01", [{"intent": a, "duration_minutes": 5, "apps": []}])
        _save_daily(kb, "2025-03-02", [{"intent": b, "duration_minutes": 5, "apps": []}])

        links = linker.analyze_daily_summaries()

        assert linker._llm_reasoner.calls == 0
        assert len(links) == 1

    def test_similarity_055_makes_no_model_call(self, kb: KnowledgeBase) -> None:
        reasoner = _CountingReasoner(answer=False)
        linker = SessionLinker(kb, llm_reasoner=reasoner)
        with patch.object(SessionLinker, "_intent_similarity", return_value=(0.55, "jaccard")):
            assert linker._intents_match("a", "b") is True
        assert reasoner.calls == 0

    def test_band_below_the_threshold_asks_the_model(self, kb: KnowledgeBase) -> None:
        for answer in (True, False):
            reasoner = _CountingReasoner(answer=answer)
            linker = SessionLinker(kb, llm_reasoner=reasoner)
            with patch.object(SessionLinker, "_intent_similarity", return_value=(0.3, "jaccard")):
                assert linker._intents_match("a", "b") is answer
            assert reasoner.calls == 1

    def test_cosine_is_compared_with_the_cosine_threshold(self, kb: KnowledgeBase) -> None:
        """A cosine of 0.6 is far below the cosine threshold, though it
        would pass the Jaccard-calibrated 0.4."""
        from agenthandover_worker.session_linker import _COSINE_THRESHOLD, _LLM_BAND

        vectors = {"x": [1.0, 0.0], "y": [0.6, 0.8]}  # cosine 0.6
        linker = SessionLinker(kb, vector_kb=_FakeVectorKB(vectors))
        score, metric = linker._intent_similarity("x", "y")
        assert metric == "cosine" and score == pytest.approx(0.6)
        assert score < _COSINE_THRESHOLD - _LLM_BAND["cosine"]
        assert linker._intents_match("x", "y") is False

    def test_each_intent_is_embedded_once(self, kb: KnowledgeBase) -> None:
        vkb = _FakeVectorKB({"x": [1.0, 0.0], "y": [0.0, 1.0], "z": [1.0, 1.0]})
        linker = SessionLinker(kb, vector_kb=vkb)
        for a, b in (("x", "y"), ("x", "z"), ("y", "z"), ("x", "y")):
            linker._intent_similarity(a, b)
        assert sorted(vkb.embedded) == ["x", "y", "z"]
