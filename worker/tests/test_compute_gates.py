"""Regression pins for WP-C5's compute gates.

What each pin holds:

* passive annotation stops at the daily budget with rows still pending,
  their screenshots kept, and the reason published in worker-status.json;
* the annotation cache answers an identical screen without the model;
* SOP mining is skipped while the episode store is unchanged, and re-mined
  templates are exported only when their fingerprint changed;
* the procedure list, the sops index and the curation queue do no repeated
  work when nothing changed;
* the passive backlog is capped at a day, and worker state persists.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agenthandover_worker import main as wm
from agenthandover_worker.annotation_cache import (
    MIN_OCR_CHARS,
    AnnotationCache,
    CacheKey,
    cache_key,
    normalise_screen_text,
    simhash64,
)
from agenthandover_worker.db import WorkerDB
from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.ollama import Response
from agenthandover_worker.ops_telemetry import MODEL_USAGE
from agenthandover_worker.scene_annotator import (
    AnnotationConfig,
    AnnotationResult,
    SceneAnnotator,
)
from agenthandover_worker.scheduler import AnnotationBudget
from agenthandover_worker.sop_dedup import deduplicate_templates
from daemon_db import create_daemon_db


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


@pytest.fixture()
def events_db(tmp_path: Path) -> Path:
    return create_daemon_db(tmp_path / "events.db")


def _insert(db_path: Path, eid: str, when: datetime, metadata: dict | None = None) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO events (id, timestamp, kind_json, window_json, "
        "display_topology_json, primary_display_id, artifact_ids_json, metadata_json) "
        "VALUES (?, ?, '{\"type\":\"DwellSnapshot\"}', '{}', '[]', 'main', '[]', ?)",
        (eid, _iso(when), json.dumps(metadata or {})),
    )
    conn.commit()
    conn.close()


def _statuses(db_path: Path) -> dict[str, str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return dict(conn.execute("SELECT id, annotation_status FROM events").fetchall())
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Daily annotation budget
# ---------------------------------------------------------------------------


class TestAnnotationBudget:

    def test_compute_cap_stops_the_run_with_rows_pending(self, events_db, tmp_path):
        """Cap 5 minutes, each frame costs 2: three frames run, the rest stay
        pending with their JPEGs, and the status names the reason."""
        model = "budget-test-model:1b"
        now = datetime.now(timezone.utc)
        shots = tmp_path / "shots"
        shots.mkdir()
        for i in range(6):
            img = shots / f"p{i}.jpg"
            img.write_bytes(b"\xff\xd8\xff")
            _insert(events_db, f"p{i}", now - timedelta(minutes=10 - i),
                    {"screenshot_path": str(img)})

        annotator = SceneAnnotator(AnnotationConfig(model=model))

        def annotate(event, **kwargs):
            # Two minutes of Ollama compute per frame (1 prompt + 1 generation).
            MODEL_USAGE.record(model, "generate", Response(
                text="{}", prompt_eval_duration=60 * 10**9, eval_duration=60 * 10**9,
            ))
            return AnnotationResult(
                event_id=event["id"], status="completed",
                annotation={"app": "Chrome", "task_context": {"what_doing": "x"}},
            )

        budget = AnnotationBudget(max_frames=1500, max_compute_seconds=5 * 60)
        budget.roll(datetime(2026, 9, 23, 12, 0))
        with WorkerDB(events_db) as db, \
                patch.object(annotator, "ready", return_value=True), \
                patch.object(annotator, "annotate_event", side_effect=annotate):
            first = wm._process_annotations(db, annotator, shots, budget=budget)
            second = wm._process_annotations(db, annotator, shots, budget=budget)

        assert first["annotated"] == 3 and first["budget_exhausted"] is True
        assert second["annotated"] == 0 and second["budget_exhausted"] is True
        assert budget.frames == 3 and budget.compute_seconds == pytest.approx(360.0)
        statuses = _statuses(events_db)
        assert [statuses[f"p{i}"] for i in range(6)] == ["completed"] * 3 + ["pending"] * 3
        assert all((shots / f"p{i}.jpg").exists() for i in range(3, 6))

        wm._write_worker_status(
            started_at="x", events_processed_today=0, sops_generated=0,
            last_pipeline_duration_ms=None, consecutive_errors=0,
            vlm_available=True, sop_inducer_available=True,
            v2_annotation_enabled=True,
            annotation_paused_reason="annotation_budget_exhausted",
            annotation_budget=budget.status(),
        )
        status = json.loads((wm._status_dir() / "worker-status.json").read_text())
        assert status["annotation_paused_reason"] == "annotation_budget_exhausted"
        assert status["annotation_budget"]["compute_minutes"] == 6.0

    def test_frame_cap_and_cache_hits_are_free(self, events_db, tmp_path):
        now = datetime.now(timezone.utc)
        for i in range(4):
            _insert(events_db, f"q{i}", now - timedelta(minutes=5 - i))
        annotator = SceneAnnotator(AnnotationConfig(model="frame-cap-model:1b"))
        results = iter([True, False, False, False])  # the first is a cache hit

        def annotate(event, **kwargs):
            return AnnotationResult(
                event_id=event["id"], status="completed",
                annotation={"app": "Chrome"}, cached=next(results),
            )

        budget = AnnotationBudget(max_frames=2, max_compute_seconds=3600)
        with WorkerDB(events_db) as db, \
                patch.object(annotator, "ready", return_value=True), \
                patch.object(annotator, "annotate_event", side_effect=annotate):
            stats = wm._process_annotations(db, annotator, tmp_path, budget=budget)
        assert stats["annotated"] == 3 and stats["budget_exhausted"] is True
        assert _statuses(events_db)["q3"] == "pending"

    def test_counters_start_over_on_a_new_local_day_and_persist(self, events_db):
        budget = AnnotationBudget(max_frames=10, max_compute_seconds=60)
        budget.roll(datetime(2026, 9, 22, 12, 0))
        budget.charge_frame(model_call=True)
        budget.charge_compute(30.0)
        with WorkerDB(events_db) as db:
            db.set_state("annotation_budget", budget.to_json())
            restored = AnnotationBudget(max_frames=10, max_compute_seconds=60)
            restored.load(db.get_state("annotation_budget"))
        assert (restored.date, restored.frames, restored.compute_seconds) == (
            "2026-09-22", 1, 30.0,
        )
        restored.roll(datetime(2026, 9, 22, 12, 30))
        assert restored.frames == 1
        restored.roll(datetime(2026, 9, 23, 0, 5))
        assert (restored.frames, restored.compute_seconds) == (0, 0.0)


# ---------------------------------------------------------------------------
# Annotation cache
# ---------------------------------------------------------------------------


_SCREEN = (
    "Inbox (3) Compose Reply to Alice about the Q3 budget review meeting on "
    "Thursday with the finance team and the updated forecast spreadsheet"
)


class TestAnnotationCache:

    def test_identical_and_near_identical_screens_hit(self, events_db):
        cache = AnnotationCache(events_db)
        key = cache_key("Chrome", "Inbox", _SCREEN)
        cache.store(key, {"app": "Chrome", "location": "Inbox"})
        assert cache.lookup(cache_key("Chrome", "Inbox", _SCREEN)) == {
            "app": "Chrome", "location": "Inbox",
        }
        # A simhash a few bits away (OCR noise, a clock) is the same screen;
        # one further away is not.
        near = CacheKey(key.app, key.title_hash, key.ocr_simhash ^ 0b101)
        assert cache.lookup(near) is not None
        far = CacheKey(key.app, key.title_hash, key.ocr_simhash ^ 0b1111)
        assert cache.lookup(far) is None
        assert simhash64(normalise_screen_text(_SCREEN.upper())) == key.ocr_simhash  # case-insensitive
        cache.close()

    def test_different_screen_title_or_app_misses(self, events_db):
        cache = AnnotationCache(events_db)
        cache.store(cache_key("Chrome", "Inbox", _SCREEN), {"app": "Chrome"})
        other = "Quarterly report draft section two revenue by region and churn analysis for the board"
        assert cache.lookup(cache_key("Chrome", "Inbox", other)) is None
        assert cache.lookup(cache_key("Chrome", "Sent", _SCREEN)) is None
        assert cache.lookup(cache_key("Mail", "Inbox", _SCREEN)) is None
        cache.close()

    def test_too_little_screen_text_is_never_cached(self):
        assert cache_key("Chrome", "Inbox", "x" * (MIN_OCR_CHARS - 1)) is None

    def test_prune_drops_unused_rows(self, events_db):
        cache = AnnotationCache(events_db)
        cache.store(cache_key("Chrome", "Inbox", _SCREEN), {"app": "Chrome"})
        cache.store(cache_key("Chrome", "Old", _SCREEN), {"app": "Chrome"})
        cache._conn.execute(
            "UPDATE annotation_cache SET used_at = '2020-01-01T00:00:00.000Z' "
            "WHERE title_hash = ?", (cache_key("Chrome", "Old", _SCREEN).title_hash,),
        )
        cache._conn.commit()
        assert cache.prune(max_age_days=14) == 1
        assert cache.lookup(cache_key("Chrome", "Inbox", _SCREEN)) is not None
        cache.close()


# ---------------------------------------------------------------------------
# SOP mining gated on new input; changed-only export
# ---------------------------------------------------------------------------


class TestMiningGate:

    def test_unchanged_episode_store_skips_induction(self, events_db, tmp_path):
        inducer = MagicMock()
        inducer.induce.return_value = []
        episode_builder = MagicMock()
        episode_builder.process_events.return_value = []
        watermark: dict = {}
        event = {
            "id": "e1", "timestamp": _iso(datetime.now(timezone.utc)),
            "kind_json": "{}", "window_json": "{}", "metadata_json": "{}",
        }

        with WorkerDB(events_db) as db:
            db.save_episode_steps("ep1", "t1", [{"step": "open"}, {"step": "send"}])

            def tick():
                return wm.run_pipeline(
                    [event],
                    episode_builder=episode_builder,
                    clipboard_linker=wm.ClipboardLinker(),
                    pruner=wm.NegativeDemoPruner(),
                    translator=wm.SemanticTranslator(),
                    scorer=wm.ConfidenceScorer(),
                    vlm_queue=wm.VLMFallbackQueue(),
                    openclaw_writer=MagicMock(),
                    index_generator=MagicMock(),
                    sop_inducer=inducer,
                    db=db,
                    mining_watermark=watermark,
                )

            tick()
            tick()
            assert inducer.induce.call_count == 1
            db.save_episode_steps("ep2", "t2", [{"step": "open"}, {"step": "send"}])
            tick()
            assert inducer.induce.call_count == 2
            # Re-saving an episode (INSERT OR REPLACE) is new input too.
            db.save_episode_steps("ep2", "t2", [{"step": "open"}, {"step": "archive"}])
            tick()
            assert inducer.induce.call_count == 3

    def test_changed_only_dedup_returns_new_and_changed_templates(self, tmp_path):
        def template(title: str) -> dict:
            return {
                "slug": "reply-to-email", "title": title,
                "apps_involved": ["Mail"],
                "steps": [{"step": "open", "target": "Mail"}, {"step": "send"}],
                "episode_count": 1,
            }

        first = deduplicate_templates([template("Reply to email")], tmp_path, changed_only=True)
        assert len(first) == 1
        again = deduplicate_templates([template("Reply to email")], tmp_path, changed_only=True)
        assert again == []
        renamed = deduplicate_templates(
            [template("Reply to customer email")], tmp_path, changed_only=True,
        )
        assert len(renamed) == 1
        # The default (recorded Skills) still hands every template on.
        assert len(deduplicate_templates([template("Reply to customer email")], tmp_path)) == 1


# ---------------------------------------------------------------------------
# Procedure list cache, sops index and curation queue do no repeated work
# ---------------------------------------------------------------------------


def _write_procedures(kb: KnowledgeBase, count: int) -> None:
    for i in range(count):
        kb.save_procedure({
            "id": f"proc-{i:02d}", "title": f"Procedure {i}",
            "source": "focus", "lifecycle_state": "draft", "steps": [],
        })


class TestNoRepeatedWork:

    def test_second_list_with_no_change_parses_nothing(self, tmp_path):
        kb = KnowledgeBase(root=tmp_path / "kb")
        kb.ensure_structure()
        _write_procedures(kb, 60)
        loads = MagicMock(side_effect=json.load)
        with patch("agenthandover_worker.knowledge_base.json.load", loads):
            assert len(kb.list_procedures()) == 60
            assert loads.call_count == 60
            assert len(kb.list_procedures()) == 60
            assert loads.call_count == 60
            kb.save_procedure({"id": "proc-07", "title": "Changed", "steps": []})
            procs = kb.list_procedures()
            assert loads.call_count == 61
            assert procs[7]["title"] == "Changed"
            (tmp_path / "kb" / "procedures" / "proc-08.json").unlink()
            assert len(kb.list_procedures()) == 59
            assert loads.call_count == 61

    def test_second_sops_index_refresh_opens_nothing_and_parses_nothing(
        self, events_db, tmp_path,
    ):
        kb = KnowledgeBase(root=tmp_path / "kb")
        kb.ensure_structure()
        _write_procedures(kb, 60)
        with WorkerDB(events_db) as db:
            for i in range(0, 60, 3):
                db.save_generated_sop(
                    slug=f"proc-{i:02d}", title=f"Procedure {i}", source="focus",
                    sop_template={"slug": f"proc-{i:02d}"}, confidence=0.5,
                    auto_approve=False,
                )
            wm._write_sops_index(db, kb, force=True)
            connects = MagicMock(side_effect=sqlite3.connect)
            loads = MagicMock(side_effect=json.load)
            with patch("agenthandover_worker.db.sqlite3.connect", connects), \
                    patch("agenthandover_worker.knowledge_base.json.load", loads):
                wm._write_sops_index(db, kb, force=True)
            assert connects.call_count == 0
            assert loads.call_count == 0
        index = json.loads((wm._status_dir() / "sops-index.json").read_text())
        assert len(index["sops"]) == 60

    def test_curation_queue_is_written_only_when_it_changes(self):
        merge = MagicMock(
            slug_a="a", slug_b="b", similarity=0.91, explanation="same steps",
            shared_apps=["Mail"],
        )
        curator = MagicMock()
        curator.detect_merge_candidates.return_value = [merge]
        target = wm._status_dir() / "curation-queue.json"
        target.unlink(missing_ok=True)

        assert wm._write_curation_queue(curator) is True
        first = target.stat().st_mtime_ns
        assert wm._write_curation_queue(curator) is False
        assert target.stat().st_mtime_ns == first
        curator.detect_merge_candidates.return_value = []
        assert wm._write_curation_queue(curator) is True
        assert json.loads(target.read_text())["merge_candidates"] == []


# ---------------------------------------------------------------------------
# Backlog cap and worker state
# ---------------------------------------------------------------------------


class TestBacklogAndState:

    def test_passive_frames_older_than_a_day_are_given_up(self, events_db):
        now = datetime.now(timezone.utc)
        _insert(events_db, "old", now - timedelta(hours=25))
        _insert(events_db, "fresh", now - timedelta(hours=23))
        _insert(events_db, "old-focus", now - timedelta(hours=30), {"focus_session_id": "s"})
        with WorkerDB(events_db) as db:
            expired = db.expire_annotation_backlog(max_age_hours=24)
        assert [r["id"] for r in expired] == ["old"]
        assert _statuses(events_db) == {
            "old": "skipped", "fresh": "pending", "old-focus": "pending",
        }

    def test_worker_state_round_trip(self, events_db):
        with WorkerDB(events_db) as db:
            assert db.get_state("daily_jobs_date") is None
            assert db.set_state("daily_jobs_date", "2026-09-23")
        with WorkerDB(events_db) as db:
            assert db.get_state("daily_jobs_date") == "2026-09-23"

    def test_daily_maintenance_prunes_every_store(self):
        vector_kb, searcher, cache = MagicMock(), MagicMock(), MagicMock()
        searcher.prune_index.side_effect = RuntimeError("locked")
        wm._run_daily_maintenance(
            vector_kb=vector_kb, activity_searcher=searcher, annotation_cache=cache,
        )
        vector_kb.purge_stale.assert_called_once_with()
        cache.prune.assert_called_once_with()
        wm._run_daily_maintenance()  # nothing configured: nothing to do
