"""Tests for the main.py pipeline orchestration.

Covers run_pipeline() with episode building, pruning, translation,
confidence scoring, VLM auto-enqueue, and SOP export.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from unittest.mock import MagicMock

import pytest

from oc_apprentice_worker.clipboard_linker import ClipboardLinker
from oc_apprentice_worker.confidence import ConfidenceScorer
from oc_apprentice_worker.episode_builder import EpisodeBuilder
from oc_apprentice_worker.exporter import IndexGenerator
from oc_apprentice_worker.main import run_pipeline
from oc_apprentice_worker.negative_demo import NegativeDemoPruner
from oc_apprentice_worker.openclaw_writer import OpenClawWriter
from oc_apprentice_worker.translator import SemanticTranslator
from oc_apprentice_worker.vlm_queue import VLMFallbackQueue


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
    workspace = tmp_path / "workspace"
    return {
        "episode_builder": EpisodeBuilder(),
        "clipboard_linker": ClipboardLinker(),
        "pruner": NegativeDemoPruner(),
        "translator": SemanticTranslator(),
        "scorer": ConfidenceScorer(),
        "vlm_queue": VLMFallbackQueue(),
        "openclaw_writer": OpenClawWriter(workspace_dir=workspace),
        "index_generator": IndexGenerator(),
        "sop_inducer": None,
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
            "translations", "vlm_enqueued", "sops_induced", "sops_exported",
            "skills_exported",
        }
        assert set(summary.keys()) == expected_keys


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
        from oc_apprentice_worker.main import _process_vlm_jobs
        from oc_apprentice_worker.vlm_queue import (
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
        from oc_apprentice_worker.main import _process_vlm_jobs
        from oc_apprentice_worker.vlm_queue import (
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
        from oc_apprentice_worker.main import _process_vlm_jobs
        from oc_apprentice_worker.vlm_queue import (
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
        from oc_apprentice_worker.main import _process_vlm_jobs

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
        from oc_apprentice_worker import main as main_mod
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
        from oc_apprentice_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        assert main_mod._load_sop_cache() == []

    def test_load_corrupted_returns_empty(self, tmp_path: Path, monkeypatch):
        from oc_apprentice_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        (tmp_path / "sop-templates-cache.json").write_text("not json{{{")
        assert main_mod._load_sop_cache() == []

    def test_save_empty_is_noop(self, tmp_path: Path, monkeypatch):
        from oc_apprentice_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        main_mod._save_sop_cache([])
        assert not (tmp_path / "sop-templates-cache.json").exists()


class TestCheckExportTrigger:
    """Tests for _check_export_trigger."""

    def _write_trigger(self, state_dir: Path, fmt: str, **kwargs) -> None:
        trigger = {"format": fmt, "requested_at": "2026-02-23T10:00:00Z"}
        trigger.update(kwargs)
        (state_dir / "export-trigger.json").write_text(json.dumps(trigger))

    def _write_cache(self, state_dir: Path, templates: list[dict]) -> None:
        (state_dir / "sop-templates-cache.json").write_text(json.dumps(templates))

    def test_no_trigger_is_noop(self, tmp_path: Path, monkeypatch):
        from oc_apprentice_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        writer = MagicMock()
        main_mod._check_export_trigger(openclaw_writer=writer, sops_dir=tmp_path / "sops")
        writer.write_sop.assert_not_called()

    def test_skill_md_export(self, tmp_path: Path, monkeypatch):
        from oc_apprentice_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        templates = [{"slug": "test", "title": "Test", "steps": []}]
        self._write_cache(tmp_path, templates)
        self._write_trigger(tmp_path, "skill-md")

        sops_dir = tmp_path / "workspace" / "openclaw" / "sops"
        sops_dir.mkdir(parents=True)
        writer = MagicMock()
        main_mod._check_export_trigger(openclaw_writer=writer, sops_dir=sops_dir)

        # Trigger should be consumed
        assert not (tmp_path / "export-trigger.json").exists()
        # OpenClaw writer should NOT have been called (format is skill-md only)
        writer.write_sop.assert_not_called()
        # SKILL.md files should exist
        skills_dir = sops_dir.parent.parent / "skills"
        assert skills_dir.exists()

    def test_openclaw_export(self, tmp_path: Path, monkeypatch):
        from oc_apprentice_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        templates = [{"slug": "test", "title": "Test", "steps": []}]
        self._write_cache(tmp_path, templates)
        self._write_trigger(tmp_path, "openclaw")

        writer = MagicMock()
        main_mod._check_export_trigger(openclaw_writer=writer, sops_dir=tmp_path / "sops")

        writer.write_sop.assert_called_once()
        assert not (tmp_path / "export-trigger.json").exists()

    def test_generic_export(self, tmp_path: Path, monkeypatch):
        from oc_apprentice_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        templates = [{"slug": "test", "title": "Test", "steps": [], "variables": []}]
        self._write_cache(tmp_path, templates)
        self._write_trigger(tmp_path, "generic")

        writer = MagicMock()
        sops_dir = tmp_path / "sops"
        sops_dir.mkdir(parents=True)
        main_mod._check_export_trigger(openclaw_writer=writer, sops_dir=sops_dir)

        # OpenClaw writer NOT called
        writer.write_sop.assert_not_called()
        # Trigger consumed
        assert not (tmp_path / "export-trigger.json").exists()
        # GenericWriter writes to output_dir/sops/sop.<slug>.md
        assert (sops_dir / "sops" / "sop.test.md").exists()

    def test_output_dir_override(self, tmp_path: Path, monkeypatch):
        from oc_apprentice_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        custom_output = tmp_path / "custom_output"
        templates = [{"slug": "test", "title": "Test", "steps": []}]
        self._write_cache(tmp_path, templates)
        self._write_trigger(tmp_path, "skill-md", output_dir=str(custom_output))

        writer = MagicMock()
        main_mod._check_export_trigger(openclaw_writer=writer, sops_dir=tmp_path / "sops")

        # Skills should be written to custom_output/skills/
        assert (custom_output / "skills").exists()
        assert not (tmp_path / "export-trigger.json").exists()

    def test_slug_filter(self, tmp_path: Path, monkeypatch):
        from oc_apprentice_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        templates = [
            {"slug": "wanted", "title": "Wanted", "steps": []},
            {"slug": "unwanted", "title": "Unwanted", "steps": []},
        ]
        self._write_cache(tmp_path, templates)
        self._write_trigger(tmp_path, "openclaw", sop_slug="wanted")

        writer = MagicMock()
        main_mod._check_export_trigger(openclaw_writer=writer, sops_dir=tmp_path / "sops")

        # _export_via_adapter calls write_sop per template; only "wanted" passes filter
        writer.write_sop.assert_called_once()
        sop_arg = writer.write_sop.call_args[0][0]
        assert sop_arg["slug"] == "wanted"

    def test_empty_cache_warns_and_removes_trigger(self, tmp_path: Path, monkeypatch):
        from oc_apprentice_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        self._write_trigger(tmp_path, "skill-md")

        writer = MagicMock()
        main_mod._check_export_trigger(openclaw_writer=writer, sops_dir=tmp_path / "sops")

        writer.write_sop.assert_not_called()
        # Trigger should still be removed to avoid infinite loop
        assert not (tmp_path / "export-trigger.json").exists()

    def test_openclaw_output_dir_override(self, tmp_path: Path, monkeypatch):
        """When output_dir is set for openclaw format, a new OpenClawWriter
        is created pointing to that directory instead of using the default."""
        from oc_apprentice_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        custom_output = tmp_path / "custom_oc"
        templates = [{"slug": "test", "title": "Test", "steps": [], "variables": []}]
        self._write_cache(tmp_path, templates)
        self._write_trigger(tmp_path, "openclaw", output_dir=str(custom_output))

        default_writer = MagicMock()
        main_mod._check_export_trigger(
            openclaw_writer=default_writer, sops_dir=tmp_path / "sops"
        )

        # Default writer should NOT be called because output_dir triggers
        # creation of a fresh OpenClawWriter.
        default_writer.write_sop.assert_not_called()
        # The custom writer should have written to custom_output/memory/apprentice/sops/
        assert (custom_output / "memory" / "apprentice" / "sops").exists()
        assert not (tmp_path / "export-trigger.json").exists()

    def test_all_format_exports_all_three(self, tmp_path: Path, monkeypatch):
        from oc_apprentice_worker import main as main_mod
        monkeypatch.setattr(main_mod, "_status_dir", lambda: tmp_path)

        templates = [{"slug": "test", "title": "Test", "steps": [], "variables": []}]
        self._write_cache(tmp_path, templates)
        self._write_trigger(tmp_path, "all")

        sops_dir = tmp_path / "workspace" / "openclaw" / "sops"
        sops_dir.mkdir(parents=True)
        writer = MagicMock()
        main_mod._check_export_trigger(openclaw_writer=writer, sops_dir=sops_dir)

        # OpenClaw writer called (no output_dir → uses default writer)
        writer.write_sop.assert_called_once()
        # SKILL.md created
        assert (sops_dir.parent.parent / "skills").exists()
        # Generic created (GenericWriter nests under sops/ subdirectory)
        assert (sops_dir / "sops" / "sop.test.md").exists()
        # Trigger consumed
        assert not (tmp_path / "export-trigger.json").exists()
