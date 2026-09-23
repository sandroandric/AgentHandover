"""Regression pins for WP-C5's compute gates.

What each pin holds:

* passive annotation stops at the daily budget with rows still pending,
  their screenshots kept, and the reason published in worker-status.json;
* the annotation cache answers an identical screen without the model;
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
from agenthandover_worker.scheduler import AnnotationBudget, charged_to
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
            vlm_available=True,
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
# Hourly gate and the charging helper (WP-F2)
# ---------------------------------------------------------------------------


def _usage(model: str, seconds: float) -> None:
    """One Ollama reply of *seconds* compute (half prompt, half generation)."""
    half = int(seconds / 2 * 10**9)
    MODEL_USAGE.record(model, "generate", Response(
        text="{}", prompt_eval_duration=half, eval_duration=half,
    ))


class TestHourlyGate:

    T0 = datetime(2026, 9, 23, 10, 0)

    def test_model_calls_close_the_hour_and_reopen_as_it_rolls(self):
        budget = AnnotationBudget(
            max_frames=1500, max_compute_seconds=3600, max_frames_per_hour=3,
        )
        for minute in (0, 10, 20):
            budget.roll(self.T0 + timedelta(minutes=minute))
            assert budget.pause_reason() is None
            budget.charge_frame(model_call=True)
        assert budget.pause_reason() == "annotation_hourly_budget_exhausted"
        assert (budget.frames, budget.hour()) == (3, (3, 0.0))
        # An hour after the first frame, one slot is free again.
        budget.roll(self.T0 + timedelta(minutes=60, seconds=1))
        assert budget.hour() == (2, 0.0)
        assert budget.pause_reason() is None

    def test_cache_hits_are_free_in_the_hour_and_shown_in_the_status(self):
        """WP-F6: the hourly frame cap counts model calls only."""
        budget = AnnotationBudget(
            max_frames=1500, max_compute_seconds=3600, max_frames_per_hour=2,
        )
        budget.roll(self.T0)
        budget.charge_frame(model_call=True)
        for _ in range(50):
            budget.charge_frame(model_call=False)
        assert budget.pause_reason() is None
        assert (budget.frames, budget.hour(), budget.hour_cache_hits()) == (
            1, (1, 0.0), 50,
        )
        status = budget.status()
        assert (status["hour_frames"], status["hour_cache_hits"]) == (1, 50)
        budget.charge_frame(model_call=True)
        assert budget.pause_reason() == "annotation_hourly_budget_exhausted"
        # Hits leave the hour like any other charge.
        budget.roll(self.T0 + timedelta(minutes=60, seconds=1))
        assert (budget.hour(), budget.hour_cache_hits()) == ((0, 0.0), 0)

    def test_state_saved_before_f6_still_loads(self):
        """Three-element entries (no hit count) load as model calls."""
        budget = AnnotationBudget(max_frames=1500, max_compute_seconds=3600)
        budget.load(json.dumps({
            "date": "2026-09-23", "frames": 1, "compute_seconds": 2.0,
            "recent": [[self.T0.timestamp(), 1, 0.0], [self.T0.timestamp(), 0, 2.0]],
        }))
        budget.roll(self.T0 + timedelta(minutes=1))
        assert (budget.hour(), budget.hour_cache_hits()) == ((1, 2.0), 0)
        restored = AnnotationBudget(max_frames=1500, max_compute_seconds=3600)
        restored.load(budget.to_json())
        restored.roll(self.T0 + timedelta(minutes=1))
        assert restored.recent == budget.recent

    def test_compute_closes_the_hour_and_the_day_wins_over_the_hour(self):
        budget = AnnotationBudget(
            max_frames=1500, max_compute_seconds=900, max_compute_seconds_per_hour=600,
        )
        budget.roll(self.T0)
        budget.charge_compute(600.0)
        assert budget.pause_reason() == "annotation_hourly_budget_exhausted"
        budget.roll(self.T0 + timedelta(hours=1, seconds=1))
        assert budget.pause_reason() is None
        budget.charge_compute(300.0)
        assert budget.pause_reason() == "annotation_budget_exhausted"
        status = budget.status()
        assert (status["hour_compute_minutes"], status["compute_minutes"]) == (5.0, 15.0)

    def test_the_hour_survives_a_restart(self, events_db):
        budget = AnnotationBudget(max_frames=1500, max_compute_seconds=3600)
        budget.roll(self.T0)
        budget.charge_compute(599.0)
        budget.charge_frame(model_call=True)
        with WorkerDB(events_db) as db:
            db.set_state("annotation_budget", budget.to_json())
            restored = AnnotationBudget(max_frames=1500, max_compute_seconds=3600)
            restored.load(db.get_state("annotation_budget"))
        restored.roll(self.T0 + timedelta(minutes=30))
        assert restored.hour() == (1, 599.0)
        restored.charge_compute(1.0)
        assert restored.pause_reason() == "annotation_hourly_budget_exhausted"

    def test_passive_annotation_stops_at_the_hourly_gate_with_rows_pending(
        self, events_db, tmp_path,
    ):
        """The hour counts frames sent to the model; cache hits pass free
        until the third model call closes it, and the rest stays pending
        with its JPEG."""
        now = datetime.now(timezone.utc)
        shots = tmp_path / "shots"
        shots.mkdir()
        for i in range(7):
            img = shots / f"h{i}.jpg"
            img.write_bytes(b"\xff\xd8\xff")
            _insert(events_db, f"h{i}", now - timedelta(minutes=10 - i),
                    {"screenshot_path": str(img)})
        annotator = SceneAnnotator(AnnotationConfig(model="hour-gate-model:1b"))
        cached = iter([False, True, True, False, False])

        def annotate(event, **kwargs):
            return AnnotationResult(
                event_id=event["id"], status="completed",
                annotation={"app": "Chrome"}, cached=next(cached),
            )

        budget = AnnotationBudget(
            max_frames=1500, max_compute_seconds=3600, max_frames_per_hour=3,
        )
        budget.roll(datetime.now())
        with WorkerDB(events_db) as db, \
                patch.object(annotator, "ready", return_value=True), \
                patch.object(annotator, "annotate_event", side_effect=annotate):
            stats = wm._process_annotations(
                db, annotator, shots, batch_size=10, budget=budget,
            )
        assert stats["annotated"] == 5 and stats["budget_exhausted"] is True
        assert budget.pause_reason() == "annotation_hourly_budget_exhausted"
        assert (budget.frames, budget.hour()[0], budget.hour_cache_hits()) == (3, 3, 2)
        statuses = _statuses(events_db)
        assert [statuses[f"h{i}"] for i in (5, 6)] == ["pending", "pending"]
        assert (shots / "h5.jpg").exists() and (shots / "h6.jpg").exists()

    def test_charged_to_charges_every_model_in_the_block(self):
        budget = AnnotationBudget(max_frames=10, max_compute_seconds=3600)
        budget.roll(self.T0)
        _usage("charge-a:1b", 4.0)  # outside the block: not charged
        with charged_to(budget):
            _usage("charge-a:1b", 2.0)
            _usage("charge-b:1b", 1.0)
        assert budget.compute_seconds == pytest.approx(3.0)
        assert budget.hour()[1] == pytest.approx(3.0)
        with charged_to(None):  # focus: nothing to charge
            _usage("charge-a:1b", 5.0)
        assert budget.compute_seconds == pytest.approx(3.0)

    def test_a_frame_diff_call_reduces_the_remaining_budget(self, events_db):
        """A passive frame's LLM diff is charged; a focus frame's is not."""
        now = datetime.now(timezone.utc)
        annotation = json.dumps({"app": "Chrome", "task_context": {"what_doing": "x"}})
        for eid, minute, meta in (
            ("d0", 0, {}), ("d1", 1, {}), ("d2", 2, {"focus_session_id": "s1"}),
        ):
            _insert(events_db, eid, now - timedelta(minutes=10 - minute), meta)
        conn = sqlite3.connect(str(events_db))
        conn.execute(
            "UPDATE events SET annotation_status = 'completed', scene_annotation_json = ?",
            (annotation,),
        )
        conn.execute("UPDATE events SET frame_diff_json = '{}' WHERE id = 'd0'")
        conn.commit()
        conn.close()

        differ = wm.FrameDiffer()
        calls = []

        def diff_pair(prev, curr):
            calls.append(curr["id"])
            _usage("diff-model:1b", 3.0)
            return MagicMock(diff={"diff_type": "action", "step_description": "x"},
                             inference_time_seconds=3.0)

        budget = AnnotationBudget(
            max_frames=1500, max_compute_seconds=3600, max_compute_seconds_per_hour=600,
        )
        budget.roll(datetime.now())
        before = budget.status()
        with WorkerDB(events_db) as db, patch.object(differ, "diff_pair", side_effect=diff_pair):
            stats = wm._process_diffs(db, differ, budget=budget)
        assert calls == ["d1", "d2"] and stats["diffs"] == 2
        assert budget.compute_seconds == pytest.approx(3.0)  # d1 only
        assert budget.hour() == (0, pytest.approx(3.0))
        assert budget.status()["hour_compute_minutes"] > before["hour_compute_minutes"]


# ---------------------------------------------------------------------------
# Cache key normalisation (WP-F2)
# ---------------------------------------------------------------------------


class TestNormaliseScreenText:

    @pytest.mark.parametrize("volatile, steady", [
        ("updated 5s ago", "updated 42s ago"),
        ("updated 3 min ago", "updated 1 hour ago"),
        ("Edited just now", "Edited 2 days ago"),
        ("sent a minute ago", "sent 15 mins ago"),
        ("Last sync 09:41", "Last sync 23:05:17"),
        ("Meeting at 9:30 AM", "Meeting at 11:00 pm"),
        ("Ln 12, Col 4", "Ln 318, Col 27"),
        ("Line 7, Column 1", "Line 90, Column 12"),
        ("Inbox (3)", "Inbox (1,204)"),
        ("backlog at 80 frames", "backlog at 12345 frames"),
    ])
    def test_volatile_parts_fold_to_the_same_text(self, volatile, steady):
        assert normalise_screen_text(volatile) == normalise_screen_text(steady)

    def test_each_kind_becomes_one_token(self):
        assert normalise_screen_text(
            "updated 5s ago at 10:42 Ln 3, Col 9 item 12"
        ).split() == ["updated", "_reltime_", "at", "_clock_", "_textpos_", "item", "0"]

    def test_words_stay_apart(self):
        assert normalise_screen_text("Quarterly report") != normalise_screen_text("Annual report")
        assert normalise_screen_text("Sonnet ago") == "Sonnet ago"  # no number, no time

    def test_the_key_is_normalised_and_the_title_is_not(self, events_db):
        screen = ("Grafana worker latency p95 annotation latency queue depth "
                  "updated {n}s ago, {m} events, last scrape 10:{s}")
        a = cache_key("Chrome", "Grafana", screen.format(n=3, m=120, s="04"))
        b = cache_key("Chrome", "Grafana", screen.format(n=58, m=9001, s="59"))
        assert a == b
        assert cache_key("Chrome", "Invoice 1234", screen.format(n=3, m=1, s="00")).title_hash != \
            cache_key("Chrome", "Invoice 5678", screen.format(n=3, m=1, s="00")).title_hash
        cache = AnnotationCache(events_db)
        cache.store(a, {"app": "Chrome"})
        assert cache.lookup(b) == {"app": "Chrome"}
        cache.close()


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
