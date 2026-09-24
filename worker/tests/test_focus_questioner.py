"""Tests for focus_questioner.py: merging the focus Q&A answers into the
procedure and the Q&A file IPC (the questions themselves are
``questions.py``'s, tested in test_questions.py)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from agenthandover_worker.focus_questioner import (
    FocusQuestion,
    FocusQAResult,
    ask_focus_questions,
    clear_focus_qa_files,
    merge_answers,
    parse_qa_result_from_file,
    publish_next_focus_questions,
    read_focus_pending,
    read_focus_questions,
    restore_focus_queue,
    write_focus_pending,
    write_focus_questions,
    FOCUS_QUESTIONS_FILE,
    FOCUS_PENDING_DIR,
    FOCUS_PENDING_FILE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_procedure(**overrides) -> dict:
    """Create a sample procedure dict for testing."""
    proc = {
        "title": "Reddit Community Engagement",
        "description": "Browse Reddit and engage with startup community posts",
        "slug": "reddit-community-engagement",
        "strategy": "",
        "steps": [
            {
                "step_id": "step_1",
                "action": "Open browser to Reddit",
                "app": "Chrome",
                "url": "https://reddit.com/r/startups",
            },
            {
                "step_id": "step_2",
                "action": "Browse post list",
                "app": "Chrome",
            },
            {
                "step_id": "step_3",
                "action": "Open interesting post",
                "app": "Chrome",
            },
            {
                "step_id": "step_4",
                "action": "Write comment",
                "app": "Chrome",
            },
        ],
        "apps_involved": ["com.google.Chrome"],
        "variables": [],
        "inputs": [],
    }
    proc.update(overrides)
    return proc


# ---------------------------------------------------------------------------
# Tests: Answer merging
# ---------------------------------------------------------------------------


class TestMergeAnswers:

    def _make_qa(self, questions, answers):
        return FocusQAResult(questions=questions, answers=answers)

    def test_merge_credentials_does_not_corrupt_accounts(self):
        """Credentials Q&A answers stay in agent_clarifications only.

        Regression test for v0.2.x bug: free-text credential answers like
        "Assume credentials are provided via a logged in browser" used to
        be inserted into ``environment.accounts[0].service`` as if the
        narrative answer were a service name, then exposed downstream as
        a Skill account requirement.
        """
        questions = [
            FocusQuestion(
                question="Requires login?",
                category="credentials",
                context="ctx",
                default="No login required",
            )
        ]
        qa = self._make_qa(
            questions,
            {0: "Assume credentials are provided via a logged in browser"},
        )

        result = merge_answers(_make_procedure(), qa)

        # The free-text answer must NOT corrupt the accounts list.
        assert result.get("environment", {}).get("accounts", []) == []
        # And it must NOT appear as a fake credential input variable.
        cred_inputs = [
            i for i in result.get("inputs", [])
            if isinstance(i, dict) and i.get("credential")
        ]
        assert cred_inputs == []
        # The answer is preserved in agent_clarifications for the UI.
        clarifications = result.get("agent_clarifications", [])
        assert len(clarifications) == 1
        assert "logged in browser" in clarifications[0]["answer"]

    def test_merge_credentials_no_login(self):
        """A "No login required" answer still produces no account entry.

        This already worked under the old code (early return on the
        no-login string match) and continues to work because the new
        code routes credential answers exclusively to
        agent_clarifications.
        """
        questions = [
            FocusQuestion(
                question="Requires login?",
                category="credentials",
                context="ctx",
                default="No login required",
            )
        ]
        qa = self._make_qa(questions, {0: "No login required"})

        result = merge_answers(_make_procedure(), qa)

        assert result.get("environment", {}).get("accounts", []) == []
        assert len(result["agent_clarifications"]) == 1

    def test_merge_strategy_answer(self):
        questions = [
            FocusQuestion(
                question="What's the goal?",
                category="strategy",
                context="ctx",
                default="Browse posts",
            )
        ]
        qa = self._make_qa(questions, {0: "Find and engage with SaaS founders"})

        result = merge_answers(_make_procedure(), qa)
        assert "Find and engage with SaaS founders" in result["strategy"]

    def test_merge_strategy_does_not_clobber_existing(self):
        """Synthesised strategy stays intact when Q&A also provides one.

        The behavioral_synthesizer pass is structurally richer than user
        Q&A clarifications, so we only fill ``strategy`` when it's empty.
        Q&A clarifications still land in ``agent_clarifications``.
        """
        proc = _make_procedure(strategy="Existing strategy.")
        questions = [
            FocusQuestion(
                question="More detail?",
                category="strategy",
                context="ctx",
                default="default",
            )
        ]
        qa = self._make_qa(questions, {0: "Focus on B2B posts"})

        result = merge_answers(proc, qa)
        # Existing strategy preserved verbatim.
        assert result["strategy"] == "Existing strategy."
        # New answer recorded in clarifications for visibility.
        assert any(
            "B2B posts" in c.get("answer", "")
            for c in result.get("agent_clarifications", [])
        )

    def test_merge_decision_does_not_corrupt_branches(self):
        """Decision Q&A answers stay in agent_clarifications only.

        Regression test: free-text decision answers like "Skip posts older
        than 24h" used to be auto-inserted into ``branches[0].condition``
        as if they were boolean conditions, then exposed downstream as
        Skill branches that an executing agent could not evaluate.
        """
        questions = [
            FocusQuestion(
                question="How do you decide?",
                category="decision",
                context="Branch detected",
                default="default",
            )
        ]
        qa = self._make_qa(questions, {0: "Skip posts older than 24h"})

        result = merge_answers(_make_procedure(), qa)
        # No bogus branch created from free-text answer.
        assert result.get("branches", []) == []
        # Answer preserved in agent_clarifications.
        assert len(result["agent_clarifications"]) == 1
        assert "Skip posts older than 24h" in result["agent_clarifications"][0]["answer"]

    def test_merge_verification_does_not_create_outcomes(self):
        """Verification Q&A answers stay in agent_clarifications only.

        Q&A produces narrative free text ("At least 3 comments posted")
        that doesn't fit the structured ``expected_outcomes`` schema.
        Behavioral synthesis is the canonical producer of expected outcomes.
        """
        questions = [
            FocusQuestion(
                question="What does done look like?",
                category="verification",
                context="ctx",
                default="default",
            )
        ]
        qa = self._make_qa(questions, {0: "At least 3 comments posted"})

        result = merge_answers(_make_procedure(), qa)
        # Don't auto-create expected_outcomes from Q&A.
        assert result.get("expected_outcomes", []) == []
        # Answer recorded in agent_clarifications.
        assert "3 comments" in result["agent_clarifications"][0]["answer"]

    def test_merge_scope_recurring(self):
        questions = [
            FocusQuestion(
                question="Schedule?",
                category="scope",
                context="ctx",
                default="One-off",
            )
        ]
        qa = self._make_qa(questions, {0: "Daily at 9am"})

        result = merge_answers(_make_procedure(), qa)
        assert result["recurrence"] == "Daily at 9am"

    def test_merge_scope_one_off(self):
        questions = [
            FocusQuestion(
                question="Schedule?",
                category="scope",
                context="ctx",
                default="One-off",
            )
        ]
        qa = self._make_qa(questions, {0: "one-off"})

        result = merge_answers(_make_procedure(), qa)
        assert result["recurrence"] is None

    def test_merge_does_not_mutate_input(self):
        proc = _make_procedure()
        original_strategy = proc.get("strategy", "")
        questions = [
            FocusQuestion(
                question="Goal?",
                category="strategy",
                context="ctx",
                default="default",
            )
        ]
        qa = self._make_qa(questions, {0: "New strategy"})

        merge_answers(proc, qa)
        assert proc.get("strategy", "") == original_strategy

    def test_merge_empty_answer_is_unanswered_never_the_default(self):
        questions = [
            FocusQuestion(
                question="Goal?",
                category="strategy",
                context="ctx",
                default="Browse and engage",
                step_indexes=[0],
            ),
            FocusQuestion(
                question="Which subreddit?",
                category="target",
                context="ctx",
                default="r/assumed",
                step_indexes=[0],
            ),
        ]
        # Question 0 left blank, question 1 absent from the answers.
        qa = self._make_qa(questions, {0: "  "})

        proc = _make_procedure()
        proc["strategy"] = ""
        result = merge_answers(proc, qa)
        assert result["strategy"] == ""
        assert result["steps"] == proc["steps"]
        assert result["agent_clarifications"] == [
            {"category": "strategy", "question": "Goal?", "answer": "",
             "status": "unanswered", "context": "ctx"},
            {"category": "target", "question": "Which subreddit?", "answer": "",
             "status": "unanswered", "context": "ctx"},
        ]
        assert "Browse and engage" not in json.dumps(result)
        assert "r/assumed" not in json.dumps(result)

    def test_merge_out_of_range_index_ignored(self):
        questions = [
            FocusQuestion(
                question="Q?",
                category="strategy",
                context="ctx",
                default="default",
            )
        ]
        qa = self._make_qa(questions, {5: "bogus", -1: "also bogus"})

        result = merge_answers(_make_procedure(), qa)
        # Should not crash and strategy should remain empty
        assert result.get("strategy", "") == ""

    def test_merge_multiple_answers(self):
        """Each answer routes correctly without corrupting structured fields."""
        questions = [
            FocusQuestion("Login?", "credentials", "ctx", "No"),
            FocusQuestion("Goal?", "strategy", "ctx", "Browse"),
            FocusQuestion("Schedule?", "scope", "ctx", "One-off"),
        ]
        qa = self._make_qa(questions, {
            0: "Reddit personal account",
            1: "Find SaaS founders",
            2: "Weekly on Monday",
        })

        result = merge_answers(_make_procedure(), qa)
        # Credentials answer goes to clarifications only — no fake account.
        assert result.get("environment", {}).get("accounts", []) == []
        # Strategy is filled because procedure had no synthesised one.
        assert "SaaS" in result["strategy"]
        # Scope answer maps to recurrence directly.
        assert result["recurrence"] == "Weekly on Monday"
        # All three answers recorded in clarifications.
        assert len(result["agent_clarifications"]) == 3

    def test_merge_step_rewrite_for_target_clarification(self):
        """A target clarification with explicit step_indexes rewrites that step.

        This covers the v0.2.x bug where Q&A asked "Which subreddit?" and
        the user answered "r/ClaudeAI", but the step text kept the wrongly
        inferred "r/midclaw" target. With step_indexes specified, the
        answer now propagates into the step's target field.
        """
        proc = _make_procedure()
        proc["steps"] = [
            {
                "step_id": "step_1",
                "action": "Navigate to subreddit",
                "target": "r/midclaw",
                "parameters": {"target": "r/midclaw", "location": "r/midclaw"},
            }
        ]
        questions = [
            FocusQuestion(
                question="Which subreddit?",
                category="decision",
                context="Step 1 references a subreddit",
                default="r/all",
                step_indexes=[0],
            )
        ]
        qa = self._make_qa(questions, {0: "r/ClaudeAI"})

        result = merge_answers(proc, qa)
        step = result["steps"][0]
        assert step["target"] == "r/ClaudeAI"
        assert step["parameters"]["target"] == "r/ClaudeAI"
        assert step["parameters"]["location"] == "r/ClaudeAI"

    def test_merge_step_rewrite_skips_when_no_step_indexes(self):
        """Free-text answers with no step_indexes do not modify steps.

        The step-rewrite logic is conservative: when the LLM tags a
        question as general (no step_indexes), we never touch step text.
        """
        proc = _make_procedure()
        proc["steps"] = [{"action": "Navigate to subreddit", "target": "r/midclaw"}]
        questions = [
            FocusQuestion(
                question="What's the strategy?",
                category="strategy",
                context="Overall approach",
                default="Browse",
                step_indexes=[],
            )
        ]
        qa = self._make_qa(questions, {0: "r/ClaudeAI"})

        result = merge_answers(proc, qa)
        # Step target unchanged because question had no step_indexes.
        assert result["steps"][0]["target"] == "r/midclaw"

    def test_merge_step_rewrite_skips_long_narrative_answers(self):
        """Answers longer than 80 chars are not used to rewrite step text.

        Long narrative answers (e.g. "I usually decide based on the
        sender's relationship to the company and the urgency of the
        request, but only between 9am and 5pm") shouldn't be substituted
        into step targets — they're context, not values.
        """
        proc = _make_procedure()
        proc["steps"] = [{"action": "Navigate", "target": "r/midclaw"}]
        long_answer = (
            "I usually decide based on the sender relationship and urgency "
            "but only between 9am and 5pm during weekdays only"
        )
        questions = [
            FocusQuestion(
                question="How do you decide?",
                category="decision",
                context="Branch detected",
                default="default",
                step_indexes=[0],
            )
        ]
        qa = self._make_qa(questions, {0: long_answer})

        result = merge_answers(proc, qa)
        # Step target unchanged: long answer doesn't override step text.
        assert result["steps"][0]["target"] == "r/midclaw"
        # Answer still recorded in clarifications for the agent to consult.
        assert long_answer in result["agent_clarifications"][0]["answer"]


# ---------------------------------------------------------------------------
# Tests: File IPC helpers
# ---------------------------------------------------------------------------


class TestFileIPC:

    def test_write_and_read_questions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            questions = [
                FocusQuestion("Q1?", "credentials", "ctx1", "def1"),
                FocusQuestion("Q2?", "strategy", "ctx2", "def2"),
            ]

            path = write_focus_questions(state_dir, "sess-123", "my-slug", questions)
            assert path.exists()

            data = read_focus_questions(state_dir)
            assert data is not None
            assert data["session_id"] == "sess-123"
            assert data["slug"] == "my-slug"
            assert data["status"] == "pending"
            assert len(data["questions"]) == 2
            assert data["questions"][0]["question"] == "Q1?"
            assert data["questions"][1]["category"] == "strategy"

    def test_write_and_read_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            sop = {"title": "Test", "slug": "test"}
            proc = {"title": "Test", "strategy": "strat"}

            path = write_focus_pending(state_dir, "sess-123", "test", sop, proc)
            assert path.exists()

            data = read_focus_pending(state_dir)
            assert data is not None
            assert data["session_id"] == "sess-123"
            assert data["sop_template"]["title"] == "Test"
            assert data["procedure"]["strategy"] == "strat"

    def test_read_missing_questions_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert read_focus_questions(Path(tmpdir)) is None

    def test_read_missing_pending_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert read_focus_pending(Path(tmpdir)) is None

    def test_clear_qa_files(self):
        """Clearing a session removes the questions file and that
        session's queued state; another queued session stays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            (state_dir / FOCUS_QUESTIONS_FILE).write_text("{}")
            write_focus_pending(state_dir, "sess-1", "a", {}, {})
            write_focus_pending(state_dir, "sess-2", "b", {}, {})

            clear_focus_qa_files(state_dir, "sess-1")

            assert not (state_dir / FOCUS_QUESTIONS_FILE).exists()
            assert read_focus_pending(state_dir, "sess-1") is None
            assert read_focus_pending(state_dir, "sess-2")["slug"] == "b"

    def test_clear_nonexistent_files_no_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clear_focus_qa_files(Path(tmpdir), "sess-1")  # Should not raise

    def test_read_invalid_json_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            (state_dir / FOCUS_QUESTIONS_FILE).write_text("not valid json{{{")
            assert read_focus_questions(state_dir) is None


def _qs(*texts: str) -> list[FocusQuestion]:
    return [FocusQuestion(t, "goal", "", "", id=f"q-{t}") for t in texts]


def _ask(state_dir: Path, session_id: str, *texts: str) -> bool:
    return ask_focus_questions(
        state_dir, session_id, f"slug-{session_id}", _qs(*texts),
        title=f"Title {session_id}",
        sop_template={"title": session_id}, procedure={"id": f"slug-{session_id}"},
    )


def _shown(state_dir: Path) -> dict:
    return json.loads((state_dir / FOCUS_QUESTIONS_FILE).read_text(encoding="utf-8"))


class TestFocusQueue:
    """WP-F15: each session's pending Q&A is queued under its id; the app
    sees one session's questions at a time, the oldest first."""

    def test_a_second_session_waits_behind_the_first(self, tmp_path: Path) -> None:
        assert _ask(tmp_path, "A", "a1?", "a2?", "a3?") is True
        before = (tmp_path / FOCUS_QUESTIONS_FILE).read_bytes()
        assert _ask(tmp_path, "B", "b1?", "b2?") is False
        # The app still shows A's questions, byte for byte.
        assert (tmp_path / FOCUS_QUESTIONS_FILE).read_bytes() == before
        assert _shown(tmp_path)["session_id"] == "A"
        # Both sessions keep their own pending state.
        a, b = read_focus_pending(tmp_path, "A"), read_focus_pending(tmp_path, "B")
        assert a["procedure"] == {"id": "slug-A"} and b["procedure"] == {"id": "slug-B"}
        assert [q["question"] for q in b["questions"]] == ["b1?", "b2?"]
        assert b["title"] == "Title B" and a["seq"] < b["seq"]
        assert read_focus_pending(tmp_path)["session_id"] == "A"

    def test_an_answered_file_is_never_overwritten_by_a_later_session(self, tmp_path: Path) -> None:
        _ask(tmp_path, "A", "a1?")
        answered = dict(_shown(tmp_path), status="answered", answers={"0": "yes"})
        (tmp_path / FOCUS_QUESTIONS_FILE).write_text(json.dumps(answered), encoding="utf-8")
        before = (tmp_path / FOCUS_QUESTIONS_FILE).read_bytes()
        assert _ask(tmp_path, "B", "b1?") is False
        assert (tmp_path / FOCUS_QUESTIONS_FILE).read_bytes() == before
        assert publish_next_focus_questions(tmp_path) is None
        assert (tmp_path / FOCUS_QUESTIONS_FILE).read_bytes() == before

    def test_the_next_session_is_shown_once_the_first_is_cleared(self, tmp_path: Path) -> None:
        _ask(tmp_path, "A", "a1?")
        _ask(tmp_path, "B", "b1?", "b2?")
        _ask(tmp_path, "C", "c1?")
        clear_focus_qa_files(tmp_path, "A")
        assert publish_next_focus_questions(tmp_path) == "B"
        shown = _shown(tmp_path)
        # The app's format, unchanged.
        assert set(shown) == {"session_id", "slug", "title", "questions", "status"}
        assert (shown["session_id"], shown["slug"], shown["title"], shown["status"]) == (
            "B", "slug-B", "Title B", "pending",
        )
        assert [(q["index"], q["id"], q["question"]) for q in shown["questions"]] == [
            (0, "q-b1?", "b1?"), (1, "q-b2?", "b2?"),
        ]
        # The same file the worker wrote before the queue.
        direct = write_focus_questions(
            tmp_path / "direct", "B", "slug-B", _qs("b1?", "b2?"), title="Title B",
        )
        assert json.loads(direct.read_text(encoding="utf-8")) == shown
        clear_focus_qa_files(tmp_path, "B")
        assert publish_next_focus_questions(tmp_path) == "C"
        clear_focus_qa_files(tmp_path, "C")
        assert publish_next_focus_questions(tmp_path) is None
        assert not (tmp_path / FOCUS_QUESTIONS_FILE).exists()

    def test_the_same_session_asking_again_replaces_its_pending_questions(
        self, tmp_path: Path,
    ) -> None:
        _ask(tmp_path, "A", "a1?")
        _ask(tmp_path, "B", "b1?")
        seq = read_focus_pending(tmp_path, "A")["seq"]
        assert _ask(tmp_path, "A", "a2?") is True
        assert [q["question"] for q in _shown(tmp_path)["questions"]] == ["a2?"]
        assert read_focus_pending(tmp_path, "A")["seq"] == seq
        assert read_focus_pending(tmp_path)["session_id"] == "A"

    def test_the_queue_is_oldest_first_not_by_name(self, tmp_path: Path) -> None:
        _ask(tmp_path, "zeta", "z?")
        _ask(tmp_path, "alpha", "a?")
        clear_focus_qa_files(tmp_path, "zeta")
        assert publish_next_focus_questions(tmp_path) == "alpha"

    def test_a_corrupt_queue_entry_is_skipped(self, tmp_path: Path, caplog) -> None:
        queue = tmp_path / FOCUS_PENDING_DIR
        queue.mkdir()
        (queue / "0-broken.json").write_text("{not json", encoding="utf-8")
        (queue / "1-list.json").write_text("[1, 2]", encoding="utf-8")
        (queue / "2-no-id.json").write_text('{"slug": "x", "questions": [{}]}', encoding="utf-8")
        _ask(tmp_path, "B", "b1?")
        assert _shown(tmp_path)["session_id"] == "B"
        assert read_focus_pending(tmp_path)["session_id"] == "B"
        assert "Skipping unreadable queued focus Q&A entry" in caplog.text
        # A corrupt entry under a session's own name is not that session's.
        (queue / "C.json").write_text("{oops", encoding="utf-8")
        assert read_focus_pending(tmp_path, "C") is None

    def test_an_entry_without_questions_is_passed_over(self, tmp_path: Path) -> None:
        write_focus_pending(tmp_path, "A", "a", {}, {})
        _ask(tmp_path, "B", "b1?")
        assert _shown(tmp_path)["session_id"] == "B"

    def test_an_unreadable_questions_file_is_replaced_by_the_next_session(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / FOCUS_QUESTIONS_FILE).write_text("{half", encoding="utf-8")
        assert _ask(tmp_path, "B", "b1?") is True
        assert _shown(tmp_path)["session_id"] == "B"

    def test_a_session_id_is_kept_inside_the_queue_directory(self, tmp_path: Path) -> None:
        path = write_focus_pending(tmp_path, "../../etc/x", "a", {}, {})
        assert path.parent == tmp_path / FOCUS_PENDING_DIR
        assert read_focus_pending(tmp_path, "../../etc/x")["slug"] == "a"


class TestLegacyPendingMigration:
    """A 0.4.x/0.5.0 single ``focus-pending.json`` moves into the queue."""

    def _legacy(self, state_dir: Path, *, status: str = "pending") -> bytes:
        (state_dir / FOCUS_PENDING_FILE).write_text(json.dumps({
            "session_id": "old", "slug": "old-slug",
            "sop_template": {"title": "Old"}, "procedure": {"id": "old-slug"},
        }), encoding="utf-8")
        write_focus_questions(state_dir, "old", "old-slug", _qs("o1?", "o2?"), title="Old")
        if status != "pending":
            data = dict(_shown(state_dir), status=status, answers={"0": "x"})
            (state_dir / FOCUS_QUESTIONS_FILE).write_text(json.dumps(data), encoding="utf-8")
        return (state_dir / FOCUS_QUESTIONS_FILE).read_bytes()

    @pytest.mark.parametrize("status", ["pending", "answered"])
    def test_the_legacy_file_is_queued_under_its_session(self, tmp_path: Path, status) -> None:
        before = self._legacy(tmp_path, status=status)
        assert restore_focus_queue(tmp_path) is None  # the app already shows it
        assert not (tmp_path / FOCUS_PENDING_FILE).exists()
        entry = read_focus_pending(tmp_path, "old")
        assert (entry["slug"], entry["title"], entry["procedure"]) == (
            "old-slug", "Old", {"id": "old-slug"},
        )
        assert [q["question"] for q in entry["questions"]] == ["o1?", "o2?"]
        assert (tmp_path / FOCUS_QUESTIONS_FILE).read_bytes() == before
        # Idempotent: a second start changes nothing.
        queued = sorted(p.read_bytes() for p in (tmp_path / FOCUS_PENDING_DIR).iterdir())
        assert restore_focus_queue(tmp_path) is None
        assert sorted(p.read_bytes() for p in (tmp_path / FOCUS_PENDING_DIR).iterdir()) == queued
        assert (tmp_path / FOCUS_QUESTIONS_FILE).read_bytes() == before

    def test_an_already_queued_session_is_kept(self, tmp_path: Path) -> None:
        self._legacy(tmp_path)
        write_focus_pending(tmp_path, "old", "old-slug", {}, {"id": "newer"})
        restore_focus_queue(tmp_path)
        assert not (tmp_path / FOCUS_PENDING_FILE).exists()
        assert read_focus_pending(tmp_path, "old")["procedure"] == {"id": "newer"}

    def test_a_corrupt_legacy_file_is_left_and_the_worker_goes_on(self, tmp_path: Path) -> None:
        (tmp_path / FOCUS_PENDING_FILE).write_text("{broken", encoding="utf-8")
        _ask(tmp_path, "B", "b1?")
        clear_focus_qa_files(tmp_path, "nobody")  # the questions file goes
        assert restore_focus_queue(tmp_path) == "B"
        assert (tmp_path / FOCUS_PENDING_FILE).exists()

    def test_startup_shows_the_oldest_queued_session(self, tmp_path: Path) -> None:
        _ask(tmp_path, "A", "a1?")
        _ask(tmp_path, "B", "b1?")
        (tmp_path / FOCUS_QUESTIONS_FILE).unlink()
        assert restore_focus_queue(tmp_path) == "A"
        assert restore_focus_queue(tmp_path) is None


class TestParseQAResult:

    def test_parse_answered(self):
        data = {
            "session_id": "sess-1",
            "slug": "test",
            "questions": [
                {"question": "Q1?", "category": "strategy", "context": "c", "default": "d"},
                {"question": "Q2?", "category": "credentials", "context": "c", "default": "d"},
            ],
            "status": "answered",
            "answers": {"0": "Answer one", "1": "Answer two"},
        }

        result = parse_qa_result_from_file(data)
        assert result is not None
        assert len(result.questions) == 2
        assert result.answers[0] == "Answer one"
        assert result.answers[1] == "Answer two"

    def test_parse_skipped_records_no_answers(self):
        data = {
            "session_id": "sess-1",
            "slug": "test",
            "questions": [
                {"question": "Q1?", "category": "strategy", "context": "c", "default": "Default 1"},
            ],
            "status": "skipped",
        }

        result = parse_qa_result_from_file(data)
        assert result is not None
        assert result.answers == {}
        assert result.questions[0].default == "Default 1"  # kept as the guess only

        merged = merge_answers(
            _make_procedure(), result,
        )
        assert merged["agent_clarifications"][0]["status"] == "unanswered"
        assert merged["agent_clarifications"][0]["answer"] == ""
        assert "Default 1" not in json.dumps(merged)

    def test_parse_pending_returns_none(self):
        data = {
            "session_id": "sess-1",
            "slug": "test",
            "questions": [],
            "status": "pending",
        }
        assert parse_qa_result_from_file(data) is None

    def test_parse_invalid_status_returns_none(self):
        data = {
            "status": "unknown",
            "questions": [],
        }
        assert parse_qa_result_from_file(data) is None

    def test_parse_invalid_answer_keys_ignored(self):
        data = {
            "session_id": "sess-1",
            "slug": "test",
            "questions": [
                {"question": "Q?", "category": "strategy", "context": "c", "default": "d"},
            ],
            "status": "answered",
            "answers": {"0": "Good", "not_a_number": "Bad"},
        }

        result = parse_qa_result_from_file(data)
        assert result is not None
        assert 0 in result.answers
        assert len(result.answers) == 1
