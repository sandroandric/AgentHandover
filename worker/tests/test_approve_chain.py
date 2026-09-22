"""Unit-level pins for the approve chain (WP-P6).

The app's "Approve for agents" writes one ``lifecycle-promote`` trigger;
the worker approves the pending draft, walks the lifecycle to
``agent_ready`` and refreshes what Claude Code sees.  These pins drive
the real spool, DB, knowledge base, lifecycle manager, procedure writer
and Claude Code writers; only the model output is a stored fixture.
The journey version (J-approve) waits for the journey harness.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agenthandover_worker import main as main_mod
from agenthandover_worker import trigger_spool
from agenthandover_worker.claude_skill_writer import ClaudeSkillWriter
from agenthandover_worker.db import WorkerDB
from agenthandover_worker.evidence_tracker import EvidenceTracker
from agenthandover_worker.exporter import IndexGenerator
from agenthandover_worker.focus_questioner import FocusQuestioner, write_focus_pending
from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.lifecycle_manager import LifecycleManager
from agenthandover_worker.openclaw_writer import OpenClawWriter
from agenthandover_worker.procedure_schema import sop_to_procedure
from agenthandover_worker.procedure_writer import ProcedureWriter
from fixtures.vlm_answers import generated_sop_template

SLUG = "check-domain-availability"


class Env:
    def __init__(self, tmp_path: Path, db: WorkerDB) -> None:
        self.home = Path.home()
        self.state = tmp_path / "state"
        self.state.mkdir()
        self.db = db
        self.kb = KnowledgeBase(root=self.home / ".agenthandover" / "knowledge")
        self.kb.ensure_structure()
        self.lm = LifecycleManager(self.kb)
        self.procedure_writer = ProcedureWriter(
            self.kb, EvidenceTracker(self.kb), lifecycle_manager=self.lm,
        )
        self.export_dir = tmp_path / "workspace"
        self.skills_dir = self.home / ".claude" / "skills"
        self.commands_dir = self.home / ".claude" / "commands"
        self.claude_skill_writer = ClaudeSkillWriter(skills_dir=self.skills_dir)
        self.export_kwargs = dict(
            openclaw_writer=OpenClawWriter(workspace_dir=self.export_dir),
            claude_skill_writer=self.claude_skill_writer,
            index_generator=IndexGenerator(),
            procedure_writer=self.procedure_writer,
        )

    def handlers(self) -> dict:
        """The worker's handler table for the kinds these pins drive."""
        return {
            "approve": lambda t: main_mod._process_approval_triggers(
                t, self.db, **self.export_kwargs,
            ),
            "lifecycle-promote": lambda t: main_mod._process_lifecycle_promote_trigger(
                t,
                lifecycle_manager=self.lm,
                db=self.db,
                approve=lambda slug: main_mod._approve_sop(
                    self.db, slug, **self.export_kwargs,
                ),
                knowledge_base=self.kb,
                claude_skill_writer=self.claude_skill_writer,
            ),
        }

    def send(self, kind: str, payload: dict) -> None:
        """What the app does: one spool file.  Then one worker tick."""
        trigger_spool.write_trigger(self.state, kind, payload)
        assert trigger_spool.drain(self.state, self.handlers()) == 1

    def skills(self) -> list[str]:
        """Slugs of the Claude Code Skills installed."""
        return sorted(p.parent.name for p in self.skills_dir.glob("*/SKILL.md"))

    def save_focus_draft(self, **overrides) -> dict:
        """What the focus v2 path stores for review: a v3 procedure."""
        proc = sop_to_procedure(generated_sop_template("check_domain_availability"))
        proc.update({
            "id": SLUG, "slug": SLUG, "title": "Check Domain Availability",
            "source": "focus", "lifecycle_state": "draft",
        })
        proc.update(overrides)
        self.kb.save_procedure(proc)
        self.db.save_generated_sop(
            slug=SLUG, title=proc["title"], source="focus",
            sop_template=proc, confidence=0.9, source_id="sess-1",
            auto_approve=False,
        )
        return proc


@pytest.fixture()
def env(tmp_path: Path, tmp_db_path: Path, monkeypatch) -> Env:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path / "state")
    with WorkerDB(tmp_db_path) as db:
        yield Env(tmp_path, db)


class TestApproveForAgents:
    def test_one_trigger_approves_promotes_and_hands_off(self, env: Env) -> None:
        env.save_focus_draft()
        assert env.skills() == []

        env.send("lifecycle-promote", {
            "kind": "lifecycle-promote", "slug": SLUG, "to_state": "agent_ready",
        })

        record = env.db.get_generated_sop_by_slug(SLUG)
        assert record["status"] == "approved"
        proc = env.kb.get_procedure(SLUG)
        assert proc["lifecycle_state"] == "agent_ready"
        assert [h["to_state"] for h in proc["lifecycle_history"]] == [
            "reviewed", "verified", "agent_ready",
        ]
        assert env.skills() == [SLUG]
        assert (env.skills_dir / SLUG / "SKILL.md").is_file()
        # The commands dir holds only the pointer to the Skills.
        assert [p.name for p in env.commands_dir.iterdir()] == ["agenthandover-list.md"]
        assert f"`/{SLUG}`" in (env.commands_dir / "agenthandover-list.md").read_text()

    def test_the_advertised_skill_path_is_the_path_written(self, env: Env) -> None:
        """#17: query_api (and SOPDetailView) show ~/.claude/skills/<slug>/SKILL.md."""
        from agenthandover_worker.query_api import procedure_export_paths

        env.save_focus_draft()
        env.send("lifecycle-promote", {"slug": SLUG, "to_state": "agent_ready"})

        advertised = Path(procedure_export_paths(env.kb, SLUG)["claude_skill"])
        assert advertised == env.home / ".claude" / "skills" / SLUG / "SKILL.md"
        assert advertised.resolve() == (env.skills_dir / SLUG / "SKILL.md").resolve()
        assert "report_execution_start" in advertised.read_text(encoding="utf-8")

    def test_demotion_removes_the_command_and_the_skill(self, env: Env) -> None:
        env.save_focus_draft()
        env.send("lifecycle-promote", {"slug": SLUG, "to_state": "agent_ready"})
        assert env.skills() == [SLUG]

        env.send("lifecycle-promote", {"procedure_slug": SLUG, "to_state": "stale"})

        assert env.kb.get_procedure(SLUG)["lifecycle_state"] == "stale"
        assert env.skills() == []
        assert not (env.skills_dir / SLUG).exists()

    def test_a_draft_that_fails_lint_is_not_promoted(self, env: Env) -> None:
        env.save_focus_draft(steps=[])

        env.send("lifecycle-promote", {"slug": SLUG, "to_state": "agent_ready"})

        assert env.db.get_generated_sop_by_slug(SLUG)["status"] == "draft"
        assert env.kb.get_procedure(SLUG)["lifecycle_state"] == "draft"
        assert env.skills() == []
        assert not env.export_dir.exists()

    def test_a_failing_refresh_does_not_undo_the_promotion(
        self, env: Env, monkeypatch,
    ) -> None:
        from agenthandover_worker import agent_connect

        def broken(**_: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(agent_connect, "connect_claude_code", broken)
        env.save_focus_draft()

        env.send("lifecycle-promote", {"slug": SLUG, "to_state": "agent_ready"})

        assert env.kb.get_procedure(SLUG)["lifecycle_state"] == "agent_ready"
        assert env.db.get_generated_sop_by_slug(SLUG)["status"] == "approved"

    def test_approving_keeps_the_stored_procedure(self, env: Env) -> None:
        # The stored record is the v3 procedure with the user's answers;
        # approval must not re-convert it and drop them.
        env.save_focus_draft(
            agent_clarifications=[{"question": "Which registrar?", "answer": "Namecheap"}],
            strategy="Check the registrar first.",
        )

        env.send("approve", {"sop_id": SLUG, "action": "approve"})

        proc = env.kb.get_procedure(SLUG)
        assert proc["agent_clarifications"][0]["answer"] == "Namecheap"
        assert proc["strategy"] == "Check the registrar first."
        assert proc["lifecycle_state"] == "draft"  # approval alone does not promote
        assert env.db.get_generated_sop_by_slug(SLUG)["status"] == "approved"
        assert env.skills() == []


class TestQAMergeUpdatesTheReviewRecord:
    def test_answers_land_in_the_existing_row(self, env: Env) -> None:
        proc = env.save_focus_draft()
        write_focus_pending(env.state, "sess-1", SLUG, sop_template={}, procedure=proc)
        (env.state / "focus-questions.json").write_text(json.dumps({
            "session_id": "sess-1", "slug": SLUG, "status": "answered",
            "questions": [{"index": 0, "question": "Which registrar?",
                           "category": "strategy", "context": "", "default": ""}],
            "answers": {"0": "Namecheap"},
        }))

        exported = main_mod._check_focus_answers(
            env.state, db=env.db,
            focus_questioner=FocusQuestioner(MagicMock()),
            sop_auto_approve=False, knowledge_base=env.kb,
            **env.export_kwargs,
        )

        assert exported == 0
        rows = env.db.get_generated_sops()
        assert len(rows) == 1  # an UPDATE, not a second INSERT
        stored = env.db.get_generated_sop(rows[0]["sop_id"])
        assert rows[0]["status"] == "draft"
        answers = [c.get("answer") for c in stored["sop_json"].get("agent_clarifications", [])]
        assert "Namecheap" in answers

    def test_update_sop_json_without_a_record(self, env: Env) -> None:
        assert env.db.update_sop_json("nope", {"slug": "nope"}) is False
        assert env.db.get_generated_sops() == []


class TestProcedureIdentifiers:
    def test_procedure_carries_slug_and_id(self) -> None:
        proc = sop_to_procedure(generated_sop_template("check_domain_availability"))
        assert proc["slug"] == proc["id"] != "unknown"

    def test_linting_a_procedure_keyed_by_id_logs_its_slug(self, caplog) -> None:
        proc = sop_to_procedure(generated_sop_template("check_domain_availability"))
        del proc["slug"]
        proc["steps"] = [{"step_id": "s1"}]  # no action: one lint error
        with caplog.at_level(logging.ERROR, logger=main_mod.logger.name):
            assert main_mod._lint_and_log(proc, "approval") is False
        messages = [r.getMessage() for r in caplog.records]
        assert any(f"[{proc['id']}]" in m for m in messages)
        assert not any("Missing or empty slug" in m for m in messages)


class TestMinedTemplatesRespectReview:
    """#146: the v1 PrefixSpan path exported every poll with no gates."""

    def _tick(self, env: Env, *, auto_approve: bool, template: dict | None = None) -> dict:
        env.db.get_all_episode_steps = lambda max_age_days=14: [[{"action": "x"}]]
        inducer = MagicMock()
        inducer.induce.return_value = [
            template or generated_sop_template("check_domain_availability"),
        ]
        builder = MagicMock()
        builder.process_events.return_value = []
        linker = MagicMock()
        linker.find_links.return_value = []
        return main_mod.run_pipeline(
            [{"id": "e1"}],
            episode_builder=builder,
            clipboard_linker=linker,
            pruner=MagicMock(),
            translator=MagicMock(),
            scorer=MagicMock(),
            vlm_queue=MagicMock(),
            sop_inducer=inducer,
            db=env.db,
            sop_auto_approve=auto_approve,
            **env.export_kwargs,
        )

    def test_review_on_queues_one_draft_and_exports_nothing(self, env: Env) -> None:
        recorded = env.state / "sop-registry.json"
        recorded.write_text(json.dumps([{"slug": "recorded-skill", "title": "Recorded"}]))
        before = recorded.read_bytes()

        self._tick(env, auto_approve=False)
        self._tick(env, auto_approve=False)  # mining re-runs every poll

        assert not env.export_dir.exists()
        assert not env.skills_dir.exists()
        assert env.kb.list_procedures() == []
        rows = env.db.get_generated_sops()
        assert [(r["source"], r["status"]) for r in rows] == [("v1_mined", "draft")]
        assert recorded.read_bytes() == before
        assert (env.state / "sop-registry-v1-mined.json").is_file()
        assert not (env.state / "sop-templates-cache.json").exists()

    def test_auto_approve_exports(self, env: Env) -> None:
        summary = self._tick(env, auto_approve=True)
        assert summary["sops_exported"] == 1
        assert not (env.state / "sop-registry.json").exists()

    def test_a_template_that_fails_lint_is_dropped(self, env: Env) -> None:
        bad = generated_sop_template("check_domain_availability")
        bad["steps"] = []
        self._tick(env, auto_approve=True, template=bad)
        assert not env.export_dir.exists()
        self._tick(env, auto_approve=False, template=bad)
        assert env.db.get_generated_sops() == []
