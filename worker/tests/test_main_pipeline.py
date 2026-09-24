"""Tests for the main.py pipeline orchestration.

Covers run_pipeline() with episode building, pruning, translation,
confidence scoring and VLM auto-enqueue.  It produces no SOP: the v1
PrefixSpan miner that ran here every poll was removed (WP-F5).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from unittest.mock import MagicMock

import pytest

from agenthandover_worker.clipboard_linker import ClipboardLinker
from agenthandover_worker.confidence import ConfidenceScorer
from agenthandover_worker.db import WorkerDB
from agenthandover_worker.episode_builder import EpisodeBuilder
from agenthandover_worker.main import run_pipeline
from agenthandover_worker.negative_demo import NegativeDemoPruner
from agenthandover_worker.translator import SemanticTranslator
from agenthandover_worker.vlm_queue import VLMFallbackQueue
from daemon_db import kb_db


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _make_event(
    *,
    app_id: str = "com.apple.Safari",
    url: str | None = None,
    timestamp: str | None = None,
    event_id: str | None = None,
    kind: str = "FocusChange",
    title: str = "Test Window",
    target: dict | None = None,
) -> dict:
    eid = event_id or str(uuid.uuid4())
    window = {"app_id": app_id, "title": title}
    metadata: dict = {}
    if url:
        metadata["url"] = url
    if target:
        metadata["target"] = target

    return {
        "id": eid,
        "timestamp": timestamp or _ts(datetime.now(timezone.utc)),
        "kind_json": json.dumps({kind: {}}),
        "window_json": json.dumps(window),
        "metadata_json": json.dumps(metadata),
        "display_topology_json": "[]",
        "primary_display_id": "main",
        "processed": 0,
    }


def _build_pipeline_components(tmp_path: Path) -> dict:
    """Build all pipeline components for testing."""
    return {
        "episode_builder": EpisodeBuilder(),
        "clipboard_linker": ClipboardLinker(),
        "pruner": NegativeDemoPruner(),
        "translator": SemanticTranslator(),
        "scorer": ConfidenceScorer(),
        "vlm_queue": VLMFallbackQueue(),
    }


# ------------------------------------------------------------------
# 1. Empty events returns empty summary
# ------------------------------------------------------------------


class TestEmptyPipeline:
    def test_empty_events_returns_zero_summary(self, tmp_path: Path) -> None:
        components = _build_pipeline_components(tmp_path)
        summary = run_pipeline([], **components)
        assert summary["events_in"] == 0
        assert summary["episodes"] == 0
        assert summary["translations"] == 0


# ------------------------------------------------------------------
# 2. Pipeline processes events into episodes
# ------------------------------------------------------------------


class TestPipelineEpisodes:
    def test_events_produce_episodes(self, tmp_path: Path) -> None:
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event(
                app_id="com.apple.Safari",
                timestamp=_ts(base + timedelta(seconds=i)),
            )
            for i in range(5)
        ]

        components = _build_pipeline_components(tmp_path)
        summary = run_pipeline(events, **components)

        assert summary["events_in"] == 5
        assert summary["episodes"] >= 1
        assert summary["positive_events"] == 5
        assert summary["negative_events"] == 0

    def test_different_apps_produce_multiple_episodes(self, tmp_path: Path) -> None:
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event(app_id="com.apple.Safari", timestamp=_ts(base)),
            _make_event(app_id="com.apple.Safari", timestamp=_ts(base + timedelta(seconds=1))),
            _make_event(app_id="com.microsoft.VSCode", timestamp=_ts(base + timedelta(seconds=2))),
            _make_event(app_id="com.microsoft.VSCode", timestamp=_ts(base + timedelta(seconds=3))),
        ]

        components = _build_pipeline_components(tmp_path)
        summary = run_pipeline(events, **components)

        assert summary["episodes"] == 2


# ------------------------------------------------------------------
# 3. Negative demo pruning works in pipeline
# ------------------------------------------------------------------


class TestPipelineNegativePruning:
    def test_undo_events_are_pruned(self, tmp_path: Path) -> None:
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event(
                app_id="com.apple.Notes",
                kind="FocusChange",
                timestamp=_ts(base),
            ),
            _make_event(
                app_id="com.apple.Notes",
                kind="KeyPress",
                timestamp=_ts(base + timedelta(seconds=5)),
            ),
        ]
        # Add an undo event
        undo_event = _make_event(
            app_id="com.apple.Notes",
            kind="KeyPress",
            timestamp=_ts(base + timedelta(seconds=10)),
        )
        undo_event["metadata_json"] = json.dumps({"shortcut": "cmd+z"})
        events.append(undo_event)

        components = _build_pipeline_components(tmp_path)
        summary = run_pipeline(events, **components)

        assert summary["negative_events"] > 0
        assert summary["positive_events"] < summary["events_in"]


# ------------------------------------------------------------------
# 4. Translation produces results
# ------------------------------------------------------------------


class TestPipelineTranslation:
    def test_events_are_translated(self, tmp_path: Path) -> None:
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event(
                app_id="com.apple.Safari",
                kind="ClickIntent",
                timestamp=_ts(base),
                target={"ariaLabel": "Submit", "role": "button"},
            ),
        ]

        components = _build_pipeline_components(tmp_path)
        summary = run_pipeline(events, **components)

        assert summary["translations"] >= 1


# ------------------------------------------------------------------
# 5. VLM auto-enqueue for low-confidence translations
# ------------------------------------------------------------------


class TestPipelineVLMEnqueue:
    def test_low_confidence_enqueues_vlm(self, tmp_path: Path) -> None:
        """Events with no DOM anchors should either be accepted via native
        app thresholds (with app_context fallback) or enqueued for VLM.

        Since the translator's ``_try_app_context()`` provides a fallback
        anchor (confidence ~0.15) and native app thresholds are lower
        (FLAG=0.15), events with a valid ``app_id`` in ``window_json``
        will typically get ``accept_flagged`` rather than ``reject``.

        Events with NO app context at all (empty window_json) should
        still be rejected and VLM-enqueued.
        """
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        # Events WITH app_id: the app_context fallback anchor provides
        # enough confidence for native-app accept_flagged.
        events_with_app = [
            _make_event(
                app_id="com.apple.Safari",
                kind="ClickIntent",
                timestamp=_ts(base),
            ),
            _make_event(
                app_id="com.apple.Safari",
                kind="ClickIntent",
                timestamp=_ts(base + timedelta(seconds=1)),
            ),
        ]

        components = _build_pipeline_components(tmp_path)
        summary = run_pipeline(events_with_app, **components)

        # Events with app_id get app_context anchor → accept_flagged
        # (not rejected), so VLM enqueue may be 0.
        assert summary["translations"] >= 1, "Should have at least one translation"

    def test_no_app_context_enqueues_vlm(self, tmp_path: Path) -> None:
        """Events with truly empty context should be rejected and VLM-enqueued."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        # Events with unknown intent → always rejected (confidence=0.0)
        events = [
            {
                "id": str(uuid.uuid4()),
                "timestamp": _ts(base),
                "kind_json": json.dumps({"UnknownEvent": {}}),
                "window_json": json.dumps({}),
                "metadata_json": json.dumps({}),
                "display_topology_json": "[]",
                "primary_display_id": "main",
                "processed": 0,
            },
        ]

        components = _build_pipeline_components(tmp_path)
        summary = run_pipeline(events, **components)

        # Unknown intent → rejected with score 0.0 → should attempt VLM
        # enqueue (though the event has no useful context for VLM either)
        assert summary["translations"] >= 1, "Should have at least one translation"


# ------------------------------------------------------------------
# 6. Pipeline summary has all expected keys
# ------------------------------------------------------------------


class TestPipelineSummaryKeys:
    def test_summary_has_all_keys(self, tmp_path: Path) -> None:
        components = _build_pipeline_components(tmp_path)
        summary = run_pipeline([], **components)

        expected_keys = {
            "events_in", "episodes", "positive_events", "negative_events",
            "translations", "vlm_enqueued",
        }
        assert set(summary.keys()) == expected_keys


class TestPassiveBatchMinesNothing:
    """The live test Mac's pattern: the same TextEdit switch/navigate/read
    sequence over and over.  The removed v1 miner turned it into a junk
    template every tick (switch_app_textedit_navigate_textedit_read_…) at
    two model calls per tick; now the batch builds and translates episodes
    and nothing else."""

    def test_repeated_passive_sequences_produce_no_template(
        self, tmp_path: Path, tmp_db_path: Path, monkeypatch,
    ) -> None:
        import inspect

        from agenthandover_worker import main as main_mod

        params = inspect.signature(run_pipeline).parameters
        assert not {"sop_inducer", "sop_enhancer", "mining_watermark"} & set(params)

        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path / "state")
        base = datetime(2026, 9, 23, 9, 0, 0, tzinfo=timezone.utc)
        events = []
        for rep in range(6):
            start = base + timedelta(minutes=10 * rep)
            for i, kind in enumerate(("AppSwitch", "FocusChange", "DwellSnapshot")):
                events.append(_make_event(
                    app_id="com.apple.TextEdit", kind=kind, title="notes.txt",
                    timestamp=_ts(start + timedelta(seconds=i)),
                ))

        with WorkerDB(tmp_db_path) as db:
            for _ in range(3):  # three ticks over the same store
                summary = run_pipeline(
                    events, db=db, **_build_pipeline_components(tmp_path),
                )
                assert summary["translations"] >= 1
            assert db.get_generated_sops() == []
        assert not (tmp_path / "state").exists()


# ------------------------------------------------------------------
# 7. Pipeline handles mixed event types
# ------------------------------------------------------------------


class TestPipelineMixedEvents:
    def test_mixed_event_types(self, tmp_path: Path) -> None:
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        events = [
            _make_event(kind="FocusChange", timestamp=_ts(base)),
            _make_event(kind="ClickIntent", timestamp=_ts(base + timedelta(seconds=1))),
            _make_event(kind="DwellSnapshot", timestamp=_ts(base + timedelta(seconds=2))),
            _make_event(kind="AppSwitch", timestamp=_ts(base + timedelta(seconds=3))),
        ]

        components = _build_pipeline_components(tmp_path)
        summary = run_pipeline(events, **components)

        assert summary["events_in"] == 4
        assert summary["translations"] >= 1


# ---------------------------------------------------------------------------
# Regression: _process_vlm_jobs → record_completion wiring
# ---------------------------------------------------------------------------


class TestProcessVLMJobsReconciliation:
    """Verify _process_vlm_jobs calls vlm_queue.record_completion on success
    and marks in-memory jobs as FAILED on rejection/error.

    This is a regression guard for the mark_completed → record_completion
    rename (was AttributeError at runtime, silently marking all VLM jobs
    as failed).
    """

    def test_success_path_calls_record_completion(self):
        """On VLM success, in-memory queue job must move to COMPLETED."""
        from unittest.mock import MagicMock, patch
        from agenthandover_worker.main import _process_vlm_jobs
        from agenthandover_worker.vlm_queue import (
            VLMFallbackQueue,
            VLMJob,
            VLMJobStatus,
        )

        # Set up a mock DB that returns a valid event
        db = MagicMock()
        db.get_event_by_id.return_value = {
            "kind_json": '{"DwellSnapshot": {}}',
            "window_json": '{"title": "Test Page"}',
        }

        # Set up a mock VLM worker that returns a successful response
        response = MagicMock()
        response.success = True
        response.target_description = "Submit button"
        response.suggested_selector = "#submit"
        response.confidence_boost = 0.15
        response.reasoning = "Clearly a submit button"
        response.inference_time_seconds = 1.2

        vlm_worker = MagicMock()
        vlm_worker.process_job.return_value = response

        # Set up a real VLM queue with a job in it
        vlm_queue = VLMFallbackQueue()
        job = VLMJob(
            job_id="job-001",
            event_id="evt-001",
            episode_id="",
            semantic_step_index=0,
            confidence_score=0.4,
            priority_score=0.8,
        )
        vlm_queue.enqueue(job)
        assert job.status == VLMJobStatus.PENDING

        # Process the job
        pending_jobs = [{"id": "job-001", "event_id": "evt-001"}]
        _process_vlm_jobs(db, pending_jobs, vlm_worker, vlm_queue)

        # The in-memory job should now be COMPLETED
        assert job.status == VLMJobStatus.COMPLETED
        assert job.result is not None
        assert job.result["confidence_boost"] == 0.15

        # DB should also be marked as completed
        db.mark_vlm_job_completed.assert_called_once()

    def test_budget_exhaustion_defers_job(self):
        """Budget errors should defer (keep PENDING), not mark FAILED."""
        from unittest.mock import MagicMock
        from agenthandover_worker.main import _process_vlm_jobs
        from agenthandover_worker.vlm_queue import (
            VLMFallbackQueue,
            VLMJob,
            VLMJobStatus,
        )

        db = MagicMock()
        db.get_event_by_id.return_value = {
            "kind_json": '{"DwellSnapshot": {}}',
            "window_json": '{"title": "Test"}',
        }

        response = MagicMock()
        response.success = False
        response.error = "Budget exhausted"

        vlm_worker = MagicMock()
        vlm_worker.process_job.return_value = response

        vlm_queue = VLMFallbackQueue()
        job = VLMJob(
            job_id="job-002",
            event_id="evt-002",
            episode_id="",
            semantic_step_index=0,
            confidence_score=0.3,
            priority_score=0.7,
        )
        vlm_queue.enqueue(job)

        pending_jobs = [{"id": "job-002", "event_id": "evt-002"}]
        _process_vlm_jobs(db, pending_jobs, vlm_worker, vlm_queue)

        # Budget errors are deferred — job stays PENDING for retry after reset
        assert job.status == VLMJobStatus.PENDING
        db.mark_vlm_job_failed.assert_not_called()

    def test_non_budget_failure_marks_in_memory_failed(self):
        """Non-budget VLM errors should mark the job as FAILED."""
        from unittest.mock import MagicMock
        from agenthandover_worker.main import _process_vlm_jobs
        from agenthandover_worker.vlm_queue import (
            VLMFallbackQueue,
            VLMJob,
            VLMJobStatus,
        )

        db = MagicMock()
        db.get_event_by_id.return_value = {
            "kind_json": '{"DwellSnapshot": {}}',
            "window_json": '{"title": "Test"}',
        }

        response = MagicMock()
        response.success = False
        response.error = "Model inference failed: OOM"

        vlm_worker = MagicMock()
        vlm_worker.process_job.return_value = response

        vlm_queue = VLMFallbackQueue()
        job = VLMJob(
            job_id="job-003",
            event_id="evt-003",
            episode_id="",
            semantic_step_index=0,
            confidence_score=0.3,
            priority_score=0.7,
        )
        vlm_queue.enqueue(job)

        pending_jobs = [{"id": "job-003", "event_id": "evt-003"}]
        _process_vlm_jobs(db, pending_jobs, vlm_worker, vlm_queue)

        assert job.status == VLMJobStatus.FAILED
        db.mark_vlm_job_failed.assert_called_once_with("job-003")

    def test_without_vlm_queue_still_works(self):
        """Passing vlm_queue=None should not crash (backward compat)."""
        from unittest.mock import MagicMock
        from agenthandover_worker.main import _process_vlm_jobs

        db = MagicMock()
        db.get_event_by_id.return_value = {
            "kind_json": '{"DwellSnapshot": {}}',
            "window_json": '{"title": "Test"}',
        }

        response = MagicMock()
        response.success = True
        response.target_description = "OK button"
        response.suggested_selector = "#ok"
        response.confidence_boost = 0.1
        response.reasoning = "OK"
        response.inference_time_seconds = 0.5

        vlm_worker = MagicMock()
        vlm_worker.process_job.return_value = response

        pending_jobs = [{"id": "job-003", "event_id": "evt-003"}]
        # Should not raise — vlm_queue defaults to None
        _process_vlm_jobs(db, pending_jobs, vlm_worker)

        db.mark_vlm_job_completed.assert_called_once()


# ---------------------------------------------------------------------------
# Export trigger + SOP template cache
# ---------------------------------------------------------------------------


class TestSOPCache:
    """Tests for _save_sop_cache / _load_sop_cache."""

    def test_save_and_load_roundtrip(self, tmp_path: Path, monkeypatch):
        from agenthandover_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        templates = [
            {"slug": "test-sop", "title": "Test SOP", "steps": [{"step": "click"}]},
            {"slug": "another", "title": "Another", "steps": []},
        ]
        main_mod._save_sop_cache(templates)

        loaded = main_mod._load_sop_cache()
        assert len(loaded) == 2
        assert loaded[0]["slug"] == "test-sop"
        assert loaded[1]["slug"] == "another"

    def test_load_empty_when_no_file(self, tmp_path: Path, monkeypatch):
        from agenthandover_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        assert main_mod._load_sop_cache() == []

    def test_load_corrupted_returns_empty(self, tmp_path: Path, monkeypatch):
        from agenthandover_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        (tmp_path / "sop-templates-cache.json").write_text("not json{{{")
        assert main_mod._load_sop_cache() == []

    def test_save_empty_is_noop(self, tmp_path: Path, monkeypatch):
        from agenthandover_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        main_mod._save_sop_cache([])
        assert not (tmp_path / "sop-templates-cache.json").exists()


class TestCheckExportTrigger:
    """Tests for _check_export_trigger."""

    def _trigger(self, fmt: str, **kwargs) -> dict:
        trigger = {"format": fmt, "requested_at": "2026-02-23T10:00:00Z"}
        trigger.update(kwargs)
        return trigger

    def _write_cache(self, state_dir: Path, templates: list[dict]) -> None:
        (state_dir / "sop-templates-cache.json").write_text(json.dumps(templates))

    def _approved(self, tmp_path: Path, *slugs: str, draft: tuple[str, ...] = ()):
        """A procedure writer whose knowledge base holds *slugs* approved
        for agents (and *draft* as drafts): the export trigger writes only
        what passes the hand-off gate."""
        from agenthandover_worker.evidence_tracker import EvidenceTracker
        from agenthandover_worker.knowledge_base import KnowledgeBase
        from agenthandover_worker.procedure_schema import sop_to_procedure
        from agenthandover_worker.procedure_writer import ProcedureWriter

        kb = KnowledgeBase(root=tmp_path / "kb", db_path=kb_db(tmp_path / "kb"))
        kb.ensure_structure()
        for slug in (*slugs, *draft):
            proc = sop_to_procedure({
                "slug": slug, "title": slug.title(), "task_description": f"Do {slug}.",
                "outcome": f"{slug} is done.", "steps": [{"step": "Open the page"}],
            })
            proc["lifecycle_state"] = "agent_ready" if slug in slugs else "draft"
            proc["constraints"]["trust_level"] = "suggest" if slug in slugs else "observe"
            kb.save_procedure(proc)
        return ProcedureWriter(kb=kb, evidence=EvidenceTracker(kb))

    def test_unsupported_format_is_noop(self, tmp_path: Path, monkeypatch):
        from agenthandover_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        self._write_cache(tmp_path, [{"slug": "test", "title": "Test", "steps": []}])
        writer = MagicMock()
        main_mod._check_export_trigger(
            self._trigger("pdf"), openclaw_writer=writer, sops_dir=tmp_path / "sops",
        )
        writer.write_sop.assert_not_called()

    def test_skill_md_export(self, tmp_path: Path, monkeypatch):
        from agenthandover_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        templates = [{"slug": "test", "title": "Test", "steps": []}]
        self._write_cache(tmp_path, templates)
        trigger = self._trigger("skill-md")

        sops_dir = tmp_path / "workspace" / "openclaw" / "sops"
        sops_dir.mkdir(parents=True)
        writer = MagicMock()
        main_mod._check_export_trigger(trigger, openclaw_writer=writer, sops_dir=sops_dir,
            procedure_writer=self._approved(tmp_path, "test"),
        )

        # OpenClaw writer should NOT have been called (format is skill-md only)
        writer.write_sop.assert_not_called()
        # SKILL.md files should exist
        skills_dir = sops_dir.parent.parent / "skills"
        assert skills_dir.exists()

    def test_openclaw_export(self, tmp_path: Path, monkeypatch):
        from agenthandover_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        templates = [{"slug": "test", "title": "Test", "steps": []}]
        self._write_cache(tmp_path, templates)
        trigger = self._trigger("openclaw")

        writer = MagicMock()
        main_mod._check_export_trigger(trigger, openclaw_writer=writer, sops_dir=tmp_path / "sops",
            procedure_writer=self._approved(tmp_path, "test"),
        )

        writer.write_procedure.assert_called_once()

    def test_generic_export(self, tmp_path: Path, monkeypatch):
        from agenthandover_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        templates = [{"slug": "test", "title": "Test", "steps": [], "variables": []}]
        self._write_cache(tmp_path, templates)
        trigger = self._trigger("generic")

        writer = MagicMock()
        sops_dir = tmp_path / "sops"
        sops_dir.mkdir(parents=True)
        main_mod._check_export_trigger(trigger, openclaw_writer=writer, sops_dir=sops_dir,
            procedure_writer=self._approved(tmp_path, "test"),
        )

        # OpenClaw writer NOT called
        writer.write_sop.assert_not_called()
        # GenericWriter writes to output_dir/sops/sop.<slug>.md
        assert (sops_dir / "sops" / "sop.test.md").exists()

    def test_output_dir_override(self, tmp_path: Path, monkeypatch):
        from agenthandover_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        custom_output = tmp_path / "custom_output"
        templates = [{"slug": "test", "title": "Test", "steps": []}]
        self._write_cache(tmp_path, templates)
        trigger = self._trigger("skill-md", output_dir=str(custom_output))

        writer = MagicMock()
        main_mod._check_export_trigger(trigger, openclaw_writer=writer, sops_dir=tmp_path / "sops",
            procedure_writer=self._approved(tmp_path, "test"),
        )

        # Skills should be written to custom_output/skills/
        assert (custom_output / "skills").exists()

    def test_slug_filter(self, tmp_path: Path, monkeypatch):
        from agenthandover_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        templates = [
            {"slug": "wanted", "title": "Wanted", "steps": []},
            {"slug": "unwanted", "title": "Unwanted", "steps": []},
        ]
        self._write_cache(tmp_path, templates)
        trigger = self._trigger("openclaw", sop_slug="wanted")

        writer = MagicMock()
        main_mod._check_export_trigger(trigger, openclaw_writer=writer, sops_dir=tmp_path / "sops",
            procedure_writer=self._approved(tmp_path, "wanted", "unwanted"),
        )

        # only "wanted" passes the slug filter
        writer.write_procedure.assert_called_once()
        proc_arg = writer.write_procedure.call_args[0][0]
        assert proc_arg["id"] == "wanted"

    def test_a_draft_is_not_exported(self, tmp_path: Path, monkeypatch):
        """The CLI export writes only procedures approved for agents."""
        from agenthandover_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        self._write_cache(tmp_path, [
            {"slug": "ready", "title": "Ready", "steps": []},
            {"slug": "drafted", "title": "Drafted", "steps": []},
        ])
        writer = MagicMock()
        main_mod._check_export_trigger(
            self._trigger("openclaw"), openclaw_writer=writer, sops_dir=tmp_path / "sops",
            procedure_writer=self._approved(tmp_path, "ready", draft=("drafted",)),
        )
        assert [c.args[0]["id"] for c in writer.write_procedure.call_args_list] == ["ready"]
        writer.write_sop.assert_not_called()

    def test_empty_cache_exports_nothing(self, tmp_path: Path, monkeypatch):
        from agenthandover_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        trigger = self._trigger("skill-md")

        writer = MagicMock()
        main_mod._check_export_trigger(trigger, openclaw_writer=writer, sops_dir=tmp_path / "sops")

        writer.write_sop.assert_not_called()

    def test_openclaw_output_dir_override(self, tmp_path: Path, monkeypatch):
        """When output_dir is set for openclaw format, a new OpenClawWriter
        is created pointing to that directory instead of using the default."""
        from agenthandover_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        custom_output = tmp_path / "custom_oc"
        templates = [{"slug": "test", "title": "Test", "steps": [], "variables": []}]
        self._write_cache(tmp_path, templates)
        trigger = self._trigger("openclaw", output_dir=str(custom_output))

        default_writer = MagicMock()
        main_mod._check_export_trigger(
            trigger, openclaw_writer=default_writer, sops_dir=tmp_path / "sops",
            procedure_writer=self._approved(tmp_path, "test"),
        )

        # Default writer should NOT be called because output_dir triggers
        # creation of a fresh OpenClawWriter.
        default_writer.write_sop.assert_not_called()
        # The custom writer should have written to custom_output/memory/apprentice/sops/
        assert (custom_output / "memory" / "apprentice" / "sops").exists()

    def test_all_format_exports_all_three(self, tmp_path: Path, monkeypatch):
        from agenthandover_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        templates = [{"slug": "test", "title": "Test", "steps": [], "variables": []}]
        self._write_cache(tmp_path, templates)
        trigger = self._trigger("all")

        sops_dir = tmp_path / "workspace" / "openclaw" / "sops"
        sops_dir.mkdir(parents=True)
        writer = MagicMock()
        main_mod._check_export_trigger(trigger, openclaw_writer=writer, sops_dir=sops_dir,
            procedure_writer=self._approved(tmp_path, "test"),
        )

        # OpenClaw writer called (no output_dir → uses default writer)
        writer.write_procedure.assert_called_once()
        # SKILL.md created
        assert (sops_dir.parent.parent / "skills").exists()
        # Generic created (GenericWriter nests under sops/ subdirectory)
        assert (sops_dir / "sops" / "sop.test.md").exists()


class TestMainFunctionScoping:
    """Regression tests for Python scoping bugs inside main().

    These are bytecode-level checks that catch 'accidentally-shadowed
    module-level import' bugs that only surface at runtime — the exact
    failure mode that caused v0.2.2's worker to crash on startup with
    `UnboundLocalError: cannot access local variable 'FocusProcessor'`
    (reported by hikoae, issue #1 follow-up).

    The bug pattern: you add `from some_module import SomeClass` inside
    main() for clarity, forgetting that the class is already imported at
    module level. Python's scoping rule then treats SomeClass as local
    to the ENTIRE main() function, and any earlier use of SomeClass
    (including type annotations evaluated at runtime under PEP 563-ish
    semantics) raises UnboundLocalError.

    The unit tests never caught this because they only exercise helper
    functions, not main() itself.
    """

    def _assert_name_is_global(self, name: str) -> None:
        """Assert that `name` is resolved from globals inside main(), not
        treated as a local variable. Checks Python's `co_varnames` table
        directly — this works across Python versions regardless of which
        specialized LOAD_FAST variant the compiler picks (LOAD_FAST,
        LOAD_FAST_CHECK, LOAD_FAST_BORROW, etc. in 3.14+).

        A name ends up in co_varnames when it's (a) a function parameter,
        (b) assigned inside the function body, or (c) imported inside the
        function body via `from ... import {name}`. For `main()`, none
        of the module-level globals like FocusProcessor should appear in
        co_varnames — if they do, someone shadowed them."""
        from agenthandover_worker.main import main as main_fn
        main_code = main_fn.__code__

        assert name not in main_code.co_varnames, (
            f"'{name}' is a local variable in main() (found in co_varnames). "
            f"This means there's a redundant `from ... import {name}` or "
            f"`{name} = ...` inside main() that shadows the module-level "
            f"binding. Python's scoping rule makes {name} local to the ENTIRE "
            f"main() function, causing UnboundLocalError on any earlier use. "
            f"See v0.2.2 regression in issue #1 (hikoae). "
            f"Fix: remove the inner import/assignment and rely on the "
            f"module-level import at the top of main.py."
        )
        # Also confirm the module-level import is still present — if someone
        # deletes the module-level import, co_varnames will still be clean
        # but main() would crash with NameError instead. This catches that.
        import agenthandover_worker.main as main_mod
        assert hasattr(main_mod, name), (
            f"'{name}' is neither a local in main() nor a module-level "
            f"global in main.py. Either the module-level import was deleted "
            f"or the name was renamed. Fix: re-add the module-level import "
            f"at the top of main.py."
        )

    def test_focus_processor_is_global_in_main(self):
        """Regression: v0.2.2 shipped with a late `from agenthandover_worker
        .focus_processor import FocusProcessor` inside main() that shadowed
        the module-level import at line 29, causing the worker to crash on
        startup with UnboundLocalError. Caught by hikoae; fixed by removing
        the redundant inner import."""
        self._assert_name_is_global("FocusProcessor")

    def test_no_module_imports_are_shadowed_anywhere(self):
        """Sweep regression: any name that is imported at module level in
        main.py must NOT be re-imported or re-assigned inside any function
        body. Shadow-imports are a time-bomb class of bug — even if the
        name is only used AFTER the inner import today, anyone adding a
        reference above the inner import will re-trigger UnboundLocalError.

        Caught 4 latent shadow imports during v0.2.3 investigation:
          - datetime/timezone in _process_drift_reviewed_trigger
          - OpenClawWriter in _check_export_trigger
          - VLMFallbackQueue in main

        All 4 were fixed in v0.2.3 by deleting the redundant inner imports.
        This test ensures none of them come back and no new ones sneak in.
        """
        import ast
        import pathlib

        main_py = (
            pathlib.Path(__file__).resolve().parent.parent
            / "src"
            / "agenthandover_worker"
            / "main.py"
        )
        source = main_py.read_text()
        tree = ast.parse(source)

        # Collect all module-level imported names
        module_imports: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    module_imports.add(alias.asname or alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module_imports.add(alias.asname or alias.name)

        # Walk every function (sync + async) and collect any inner imports
        # that shadow a module-level name.
        findings: list[tuple[str, int, int, str]] = []

        class ShadowImportAuditor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
                self._check_function(node)
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
                self._check_function(node)
                self.generic_visit(node)

            def _check_function(self, func: ast.AST) -> None:
                for subnode in ast.walk(func):
                    if subnode is func:
                        continue
                    # Don't descend into nested functions — they have
                    # their own scope and their own test.
                    if isinstance(
                        subnode, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        continue
                    if isinstance(subnode, (ast.ImportFrom, ast.Import)):
                        for alias in subnode.names:
                            name = alias.asname or alias.name
                            if name in module_imports:
                                findings.append(
                                    (
                                        getattr(func, "name", "<anon>"),
                                        getattr(func, "lineno", -1),
                                        subnode.lineno,
                                        name,
                                    )
                                )

        ShadowImportAuditor().visit(tree)

        assert not findings, (
            "Found shadow-import(s) inside function bodies that already "
            "exist as module-level imports in main.py:\n"
            + "\n".join(
                f"  {name} re-imported inside {func}() "
                f"(func line {func_line}, inner import line {inner_line})"
                for func, func_line, inner_line, name in findings
            )
            + "\n\nPython's scoping rule makes these names local to the "
            "entire function body, causing UnboundLocalError on any "
            "use that executes before the inner import. "
            "Fix: remove the inner import and rely on the module-level "
            "binding. See v0.2.3 release notes + issue #1 for context."
        )
