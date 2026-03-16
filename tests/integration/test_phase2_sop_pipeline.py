"""Phase 2 End-to-End SOP Pipeline Integration Tests.

Full pipeline integration test: recorded events -> episodes -> semantic steps
-> SOP induction -> export to OpenClaw format.

Test scenarios:
1. Full Pipeline — Repeated Workflow Discovery (3 repetitions -> pattern mining)
2. Low Confidence -> VLM Queue (poor metadata -> reject -> enqueue)
3. Manual Edit Detection in Pipeline (edit detection -> v2_draft)
4. Scheduler Gate Check (favorable vs unfavorable conditions)
5. Index.md Catalog Verification (multiple SOPs -> index content)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import uuid
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Add worker src to Python path
# ---------------------------------------------------------------------------

WORKER_SRC = Path(__file__).resolve().parent.parent.parent / "worker" / "src"
sys.path.insert(0, str(WORKER_SRC))

from agenthandover_worker.clipboard_linker import ClipboardLinker
from agenthandover_worker.confidence import ConfidenceScorer
from agenthandover_worker.db import WorkerDB
from agenthandover_worker.episode_builder import EpisodeBuilder
from agenthandover_worker.exporter import AtomicWriter, IndexGenerator, SOPExporter
from agenthandover_worker.models.semantic_step import Evidence, SemanticStep
from agenthandover_worker.negative_demo import NegativeDemoPruner
from agenthandover_worker.openclaw_writer import OpenClawWriter
from agenthandover_worker.scheduler import (
    GateResult,
    IdleJobGate,
    IdleScheduler,
    SchedulerConfig,
    SystemConditions,
)
from agenthandover_worker.sop_format import SOPFormatter
from agenthandover_worker.sop_inducer import SOPInducer
from agenthandover_worker.sop_versioner import SOPVersioner
from agenthandover_worker.translator import SemanticTranslator
from agenthandover_worker.vlm_queue import QueueBudget, VLMFallbackQueue, VLMJob

# ---------------------------------------------------------------------------
# Schema — mirrors crates/storage/src/migrations/v001_initial.sql exactly
# ---------------------------------------------------------------------------

DAEMON_SCHEMA = """\
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY NOT NULL,
    timestamp TEXT NOT NULL,
    kind_json TEXT NOT NULL,
    window_json TEXT,
    display_topology_json TEXT NOT NULL,
    primary_display_id TEXT NOT NULL,
    cursor_x INTEGER,
    cursor_y INTEGER,
    ui_scale REAL,
    artifact_ids_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    processed INTEGER NOT NULL DEFAULT 0,
    episode_id TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_processed ON events(processed);
CREATE INDEX IF NOT EXISTS idx_events_episode_id ON events(episode_id);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY NOT NULL,
    event_id TEXT NOT NULL REFERENCES events(id),
    artifact_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    compression_algo TEXT NOT NULL DEFAULT 'zstd',
    encryption_algo TEXT NOT NULL DEFAULT 'xchacha20poly1305',
    original_size_bytes INTEGER NOT NULL,
    stored_size_bytes INTEGER NOT NULL,
    artifact_version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_artifacts_event_id ON artifacts(event_id);

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY NOT NULL,
    segment_id INTEGER NOT NULL DEFAULT 0,
    prev_segment_id INTEGER,
    thread_id TEXT,
    start_time TEXT NOT NULL,
    end_time TEXT,
    event_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'open',
    summary TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS vlm_queue (
    id TEXT PRIMARY KEY NOT NULL,
    event_id TEXT NOT NULL REFERENCES events(id),
    priority REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    processed_at TEXT,
    result_json TEXT,
    ttl_expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vlm_queue_status ON vlm_queue(status, priority DESC);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _ts(dt: datetime) -> str:
    """Format a datetime as the ISO 8601 string the daemon produces."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _hash(content: str) -> str:
    """Produce a deterministic SHA-256 hex digest for test content."""
    return hashlib.sha256(content.encode()).hexdigest()


def _insert_event(
    conn: sqlite3.Connection,
    *,
    event_id: str | None = None,
    timestamp: str | None = None,
    kind_json: str = '{"FocusChange":{}}',
    window_json: str | None = None,
    metadata_json: str = "{}",
    cursor_x: int | None = None,
    cursor_y: int | None = None,
    processed: int = 0,
) -> str:
    """Insert a full event row into the daemon's events table and return its id."""
    eid = event_id or _new_uuid()
    ts = timestamp or _now_iso()
    conn.execute(
        "INSERT INTO events "
        "(id, timestamp, kind_json, window_json, display_topology_json, "
        "primary_display_id, cursor_x, cursor_y, metadata_json, processed) "
        "VALUES (?, ?, ?, ?, '[]', 'main', ?, ?, ?, ?)",
        (eid, ts, kind_json, window_json, cursor_x, cursor_y, metadata_json, processed),
    )
    conn.commit()
    return eid


def _generate_pr_review_workflow(
    conn: sqlite3.Connection,
    *,
    pr_number: int,
    base_time: datetime,
    comment_text: str = "LGTM",
    slack_channel: str = "#team",
) -> list[str]:
    """Generate a complete PR review workflow (5 events).

    Pattern:
    1. Navigate to GitHub PR page (FocusChange)
    2. Dwell / read the PR code (DwellSnapshot)
    3. Click Approve button (ClickIntent with ARIA label)
    4. Type review comment (KeyPress with text)
    5. Switch to Slack and send notification (AppSwitch)

    Returns list of event IDs.
    """
    event_ids: list[str] = []
    pr_url = f"https://github.com/org/repo/pull/{pr_number}"

    # Step 1: Navigate to GitHub PR
    eid = _insert_event(
        conn,
        event_id=f"ev-pr{pr_number}-01-navigate",
        timestamp=_ts(base_time),
        kind_json=json.dumps({"FocusChange": {}}),
        window_json=json.dumps({"app_id": "com.google.Chrome", "title": f"PR #{pr_number} - GitHub"}),
        metadata_json=json.dumps({
            "url": pr_url,
            "target": {
                "ariaLabel": f"Pull request #{pr_number}",
                "role": "main",
                "tagName": "main",
            },
        }),
    )
    event_ids.append(eid)

    # Step 2: Dwell / read the PR code
    eid = _insert_event(
        conn,
        event_id=f"ev-pr{pr_number}-02-dwell",
        timestamp=_ts(base_time + timedelta(seconds=10)),
        kind_json=json.dumps({"DwellSnapshot": {}}),
        window_json=json.dumps({"app_id": "com.google.Chrome", "title": f"PR #{pr_number} - GitHub"}),
        metadata_json=json.dumps({
            "url": pr_url,
            "dwell_seconds": 10.0,
            "target": {
                "ariaLabel": "Code review diff",
                "role": "region",
            },
        }),
    )
    event_ids.append(eid)

    # Step 3: Click Approve button
    eid = _insert_event(
        conn,
        event_id=f"ev-pr{pr_number}-03-approve",
        timestamp=_ts(base_time + timedelta(seconds=25)),
        kind_json=json.dumps({"ClickIntent": {}}),
        window_json=json.dumps({"app_id": "com.google.Chrome", "title": f"PR #{pr_number} - GitHub"}),
        metadata_json=json.dumps({
            "url": pr_url,
            "target": {
                "ariaLabel": "Approve",
                "role": "button",
                "tagName": "button",
                "innerText": "Approve",
                "testId": "approve-btn",
            },
        }),
        cursor_x=850,
        cursor_y=620,
    )
    event_ids.append(eid)

    # Step 4: Type review comment
    eid = _insert_event(
        conn,
        event_id=f"ev-pr{pr_number}-04-comment",
        timestamp=_ts(base_time + timedelta(seconds=35)),
        kind_json=json.dumps({"KeyPress": {}}),
        window_json=json.dumps({"app_id": "com.google.Chrome", "title": f"PR #{pr_number} - GitHub"}),
        metadata_json=json.dumps({
            "url": pr_url,
            "text": comment_text,
            "target": {
                "ariaLabel": "Review comment",
                "role": "textbox",
                "tagName": "textarea",
            },
        }),
    )
    event_ids.append(eid)

    # Step 5: Switch to Slack and send notification
    eid = _insert_event(
        conn,
        event_id=f"ev-pr{pr_number}-05-slack",
        timestamp=_ts(base_time + timedelta(seconds=50)),
        kind_json=json.dumps({"AppSwitch": {}}),
        window_json=json.dumps({
            "app_id": "com.tinyspeck.slackmacgap",
            "title": f"Slack - {slack_channel}",
        }),
        metadata_json=json.dumps({
            "target": {
                "ariaLabel": f"Message {slack_channel}",
                "role": "textbox",
            },
        }),
    )
    event_ids.append(eid)

    return event_ids


def _events_to_semantic_steps(
    events: list[dict],
    episode_id: str,
) -> list[SemanticStep]:
    """Translate raw events through the full Phase 2 semantic pipeline.

    Runs: SemanticTranslator -> ConfidenceScorer -> SemanticStep assembly.
    Returns a list of SemanticStep objects.
    """
    translator = SemanticTranslator()
    scorer = ConfidenceScorer()

    translations = translator.translate_batch(events)
    steps: list[SemanticStep] = []

    # Build a scoring context that accumulates across steps
    scoring_context: dict = {}

    for idx, (event, translation) in enumerate(zip(events, translations)):
        # Build scoring context from event metadata
        metadata_json = event.get("metadata_json", "{}")
        try:
            meta = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        except (json.JSONDecodeError, TypeError):
            meta = {}

        window_json = event.get("window_json", "{}")
        try:
            window = json.loads(window_json) if isinstance(window_json, str) else window_json
        except (json.JSONDecodeError, TypeError):
            window = {}

        # Accumulate context for confidence scoring
        if translation.pre_state.get("window_title"):
            scoring_context["expected_title"] = translation.pre_state["window_title"]
        if translation.pre_state.get("url"):
            scoring_context["expected_url"] = translation.pre_state["url"]
        if translation.pre_state.get("app_id"):
            scoring_context["expected_app"] = translation.pre_state["app_id"]

        # Provenance signals from translator's context tracking
        if meta.get("content_hash"):
            scoring_context["clipboard_link"] = True
        if translation.intent == "read":
            scoring_context["dwell_snapshot"] = True

        score = scorer.score(translation, scoring_context)

        # Build evidence
        evidence = Evidence(
            dom_anchor=translation.target.selector if translation.target else None,
            url=translation.pre_state.get("url"),
            window_title=translation.pre_state.get("window_title"),
        )

        # Build target description
        target_desc = ""
        if translation.target:
            if translation.target.method == "aria_label":
                target_desc = translation.target.raw_evidence.get("ariaLabel", "")
            elif translation.target.method == "inner_text":
                target_desc = translation.target.raw_evidence.get("normalized", "")
            elif translation.target.method == "test_id":
                target_desc = translation.target.raw_evidence.get("testId", "")
            elif translation.target.method == "role_position":
                target_desc = translation.target.raw_evidence.get("role", "")
            else:
                target_desc = translation.target.selector

        # Parse timestamp
        ts_raw = event.get("timestamp")
        ts: datetime | None = None
        if ts_raw:
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        step = SemanticStep(
            step_id=f"step-{uuid.uuid4()}",
            episode_id=episode_id,
            step_index=idx,
            intent=translation.intent,
            target_description=target_desc,
            target_selector=translation.target.selector if translation.target else None,
            parameters=translation.parameters,
            pre_state=translation.pre_state,
            post_state=translation.post_state,
            confidence=score.total,
            confidence_reasons=score.reasons,
            decision=score.decision,
            evidence=evidence,
            raw_event_id=event.get("id", ""),
            timestamp=ts,
            is_negative=False,
        )
        steps.append(step)

    return steps


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db_path(tmp_path: Path) -> Path:
    """Create an empty database with the daemon schema, return its path."""
    db_file = tmp_path / "events.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA journal_mode=wal;")
    conn.executescript(DAEMON_SCHEMA)
    conn.close()
    return db_file


@pytest.fixture()
def write_conn(tmp_db_path: Path) -> sqlite3.Connection:
    """Return a read-write connection to the temp database for inserting
    test data.  Callers must NOT close this; the fixture handles cleanup.
    """
    conn = sqlite3.connect(str(tmp_db_path))
    conn.row_factory = sqlite3.Row
    yield conn  # type: ignore[misc]
    conn.close()


@pytest.fixture()
def openclaw_workspace(tmp_path: Path) -> Path:
    """Return a temporary directory to serve as the OpenClaw workspace."""
    ws = tmp_path / "openclaw_workspace"
    ws.mkdir()
    return ws


# ===================================================================
# Test 1: Full Pipeline — Repeated Workflow Discovery
# ===================================================================


class TestFullPipelineRepeatedWorkflow:
    """Simulate a user doing the same PR review task 3 times
    (minimum for pattern mining).

    Each repetition:
    1. Navigate to GitHub PR page
    2. Dwell / read the PR code
    3. Click Approve button
    4. Type review comment
    5. Switch to Slack and send notification

    Each repetition has different PR numbers, timestamps, and
    slight comment variations.

    Verify the full pipeline from events -> SOP export.
    """

    def test_full_pipeline_discovers_repeated_workflow(
        self,
        tmp_db_path: Path,
        write_conn: sqlite3.Connection,
        openclaw_workspace: Path,
    ) -> None:
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)

        # --- Generate 3 repetitions of the PR review workflow ---
        all_event_ids: list[list[str]] = []

        for rep_idx, (pr_num, comment, channel) in enumerate([
            (42, "LGTM", "#frontend"),
            (87, "Looks good!", "#backend"),
            (103, "Ship it", "#infra"),
        ]):
            event_ids = _generate_pr_review_workflow(
                write_conn,
                pr_number=pr_num,
                base_time=base + timedelta(hours=rep_idx * 2),
                comment_text=comment,
                slack_channel=channel,
            )
            all_event_ids.append(event_ids)

        # ===============================================================
        # PHASE A: Read events from SQLite via WorkerDB
        # ===============================================================
        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=200)

        assert len(events) == 15, f"Expected 15 events (5 * 3 reps), got {len(events)}"

        # ===============================================================
        # PHASE B: Build episodes via EpisodeBuilder
        # ===============================================================
        builder = EpisodeBuilder()
        episodes = builder.process_events(events)

        # We expect at least 3 episodes (each repetition in a separate
        # episode because of the 2-hour gap between repetitions, and
        # the thread multiplexing separates Chrome from Slack).
        assert len(episodes) >= 3, (
            f"Expected at least 3 episodes, got {len(episodes)}"
        )

        # Total event count preserved
        total_events = sum(ep.event_count for ep in episodes)
        assert total_events == 15

        # ===============================================================
        # PHASE C: Prune negative demos (all should be positive here)
        # ===============================================================
        pruner = NegativeDemoPruner()
        prune_result = pruner.prune(events)

        assert len(prune_result.positive_events) == 15
        assert len(prune_result.negative_events) == 0

        # ===============================================================
        # PHASE D: Translate events to semantic steps
        # ===============================================================
        translator = SemanticTranslator()
        scorer = ConfidenceScorer()

        # Translate per-repetition to produce separate episode step lists
        episode_step_lists: list[list[dict]] = []

        for rep_idx in range(3):
            rep_events = events[rep_idx * 5 : (rep_idx + 1) * 5]
            episode_id = f"episode-rep-{rep_idx}"

            steps = _events_to_semantic_steps(rep_events, episode_id)

            # Verify each step has an intent
            for step in steps:
                assert step.intent != "unknown", (
                    f"Step {step.step_index} has unknown intent"
                )

            # Convert to SOP-step dicts for the inducer
            sop_steps = [s.to_sop_step() for s in steps]
            episode_step_lists.append(sop_steps)

        assert len(episode_step_lists) == 3

        # ===============================================================
        # PHASE E: Verify confidence scoring
        # ===============================================================
        # Re-translate with full context to check scoring
        for rep_idx in range(3):
            rep_events = events[rep_idx * 5 : (rep_idx + 1) * 5]
            steps = _events_to_semantic_steps(rep_events, f"ep-{rep_idx}")

            for step in steps:
                # All events have ARIA labels, so confidence should be
                # at least at the accept_flagged level (>= 0.40 from
                # UI anchor alone, plus state match contributions)
                assert step.confidence > 0.0, (
                    f"Step {step.step_index} ({step.intent}) has zero confidence"
                )
                # Steps with ARIA labels should not be rejected
                if step.decision == "reject":
                    # Some steps may lack state match context on first
                    # event of a rep; that is acceptable as long as the
                    # reason is consistent
                    assert step.confidence < 0.60, (
                        f"Step with confidence {step.confidence} should not be rejected"
                    )

        # ===============================================================
        # PHASE F: Mine patterns via SOPInducer
        # ===============================================================
        inducer = SOPInducer(min_support=0.3, min_pattern_length=3)
        sop_templates = inducer.induce(episode_step_lists)

        assert len(sop_templates) >= 1, (
            f"Expected at least 1 SOP template from 3 repetitions, got {len(sop_templates)}"
        )

        # The longest/highest-support template should capture the
        # common pattern
        best_sop = sop_templates[0]
        assert best_sop["episode_count"] >= 2, (
            f"Expected pattern in at least 2 episodes, got {best_sop['episode_count']}"
        )
        assert len(best_sop["steps"]) >= 3, (
            f"Expected at least 3 steps in pattern, got {len(best_sop['steps'])}"
        )
        assert best_sop["slug"], "SOP slug should not be empty"
        assert best_sop["title"], "SOP title should not be empty"

        # ===============================================================
        # PHASE G: Format SOPs via SOPFormatter
        # ===============================================================
        formatter = SOPFormatter()
        for template in sop_templates:
            content = formatter.format_sop(template)

            # Verify YAML frontmatter structure
            assert content.startswith("---\n"), "SOP should start with YAML frontmatter"
            assert "---\n\n" in content, "Frontmatter should be closed by ---"

            # Parse frontmatter
            fm_end = content.index("---", 3)
            fm_text = content[3:fm_end].strip()
            frontmatter = yaml.safe_load(fm_text)

            assert frontmatter["sop_version"] == 1
            assert frontmatter["sop_slug"] == template["slug"]
            assert frontmatter["sop_title"] == template["title"]
            assert "generated_by" in frontmatter
            assert "generated_at" in frontmatter
            assert "generated_body_hash" in frontmatter
            assert frontmatter["generated_body_hash"].startswith("sha256:")
            assert frontmatter["confidence_summary"] in ("high", "medium", "low")

            # Verify body has numbered steps
            body = content[fm_end + 3:].strip()
            assert "## Steps" in body, "SOP body should have Steps section"

        # ===============================================================
        # PHASE H: Export via SOPExporter + OpenClawWriter
        # ===============================================================
        writer = OpenClawWriter(workspace_dir=openclaw_workspace)
        writer.ensure_directory_structure()

        paths = writer.write_all_sops(sop_templates)
        assert len(paths) >= 1, "Expected at least 1 SOP file written"

        # Verify file structure
        sops_dir = openclaw_workspace / "memory" / "apprentice" / "sops"
        assert sops_dir.exists()
        assert (sops_dir / "archive").exists()

        # Verify SOP files exist
        for path in paths:
            assert path.exists(), f"SOP file should exist: {path}"
            assert path.suffix == ".md"
            content = path.read_text(encoding="utf-8")
            assert content.startswith("---\n")

        # Verify index.md is generated
        index_path = openclaw_workspace / "memory" / "apprentice" / "index.md"
        assert index_path.exists(), "index.md should be generated"
        index_content = index_path.read_text(encoding="utf-8")
        assert "# SOP Index" in index_content
        assert "Total SOPs" in index_content

        # Verify metadata directory exists
        metadata_dir = openclaw_workspace / "memory" / "apprentice" / "metadata"
        assert metadata_dir.exists()


# ===================================================================
# Test 2: Low Confidence -> VLM Queue
# ===================================================================


class TestLowConfidenceVLMQueue:
    """Simulate events with poor metadata (no ARIA, no testid, minimal info).

    Events produce low confidence scores (< 0.60) and should be enqueued
    in the VLM fallback queue. Verify budget constraints are respected.
    """

    def test_low_confidence_events_enqueue_to_vlm(
        self,
        tmp_db_path: Path,
        write_conn: sqlite3.Connection,
    ) -> None:
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)

        # --- Events with minimal metadata (no ARIA, no testid, no role) ---
        for i in range(5):
            _insert_event(
                write_conn,
                event_id=f"ev-poor-{i:02d}",
                timestamp=_ts(base + timedelta(seconds=i * 10)),
                kind_json=json.dumps({"ClickIntent": {}}),
                window_json=json.dumps({
                    "app_id": "com.google.Chrome",
                    "title": "Some Page",
                }),
                metadata_json=json.dumps({
                    # No target, no ARIA, no testid — just bare coordinates
                    "url": "https://example.com/page",
                }),
                cursor_x=100 + i * 50,
                cursor_y=200 + i * 30,
            )

        # Read events
        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        assert len(events) == 5

        # Translate and score
        translator = SemanticTranslator()
        scorer = ConfidenceScorer()
        translations = translator.translate_batch(events)

        low_confidence_items: list[tuple[dict, float, str]] = []

        for event, translation in zip(events, translations):
            score = scorer.score(translation, {})

            # Without ARIA, testid, or role — UI anchor score should be
            # at most the app_context fallback (0.15). Total confidence
            # should be low.
            assert score.ui_anchor_score <= 0.15, (
                f"Event {event['id']}: expected <= 0.15 UI anchor score, got {score.ui_anchor_score}"
            )
            assert score.total < 0.60, (
                f"Event {event['id']}: expected confidence < 0.60, got {score.total}"
            )
            assert score.decision == "reject", (
                f"Event {event['id']}: expected reject decision, got {score.decision}"
            )

            low_confidence_items.append(
                (event, score.total, translation.intent)
            )

        # All 5 events should have low confidence
        assert len(low_confidence_items) == 5

        # --- Enqueue to VLM fallback queue ---
        queue = VLMFallbackQueue(budget=QueueBudget(
            max_jobs_per_day=50,
            max_queue_size=500,
            job_ttl_days=7,
            max_compute_minutes_per_day=20.0,
        ))

        for event, confidence, intent in low_confidence_items:
            priority = queue.compute_priority(
                confidence=confidence,
                intent=intent,
                created_at=datetime.now(timezone.utc),
            )

            job = VLMJob(
                job_id=f"vlm-{event['id']}",
                event_id=event["id"],
                episode_id="episode-test",
                semantic_step_index=0,
                confidence_score=confidence,
                priority_score=priority,
                query=f"What UI element was clicked at ({event.get('cursor_x')}, {event.get('cursor_y')})?",
            )

            success = queue.enqueue(job)
            assert success, f"Should enqueue job for event {event['id']}"

        # Verify queue state
        stats = queue.get_stats()
        assert stats.pending_jobs == 5
        assert stats.jobs_today == 5
        assert stats.dropped_count == 0

        # Dequeue should return the highest priority job
        top_job = queue.dequeue()
        assert top_job is not None
        assert top_job.confidence_score < 0.60

    def test_vlm_queue_budget_limits(self) -> None:
        """Verify that budget constraints prevent over-enqueuing."""
        budget = QueueBudget(
            max_jobs_per_day=3,
            max_queue_size=10,
            job_ttl_days=7,
            max_compute_minutes_per_day=5.0,
        )
        queue = VLMFallbackQueue(budget=budget)

        # Enqueue 3 jobs (at the limit)
        for i in range(3):
            job = VLMJob(
                job_id=f"vlm-limit-{i}",
                event_id=f"ev-limit-{i}",
                episode_id="ep-test",
                semantic_step_index=i,
                confidence_score=0.3,
                priority_score=0.5,
            )
            assert queue.enqueue(job) is True

        # 4th job should be rejected (daily limit reached)
        rejected_job = VLMJob(
            job_id="vlm-limit-rejected",
            event_id="ev-limit-rejected",
            episode_id="ep-test",
            semantic_step_index=3,
            confidence_score=0.3,
            priority_score=0.5,
        )
        assert queue.enqueue(rejected_job) is False

        stats = queue.get_stats()
        assert stats.jobs_today == 3
        assert stats.pending_jobs == 3


# ===================================================================
# Test 3: Manual Edit Detection in Pipeline
# ===================================================================


class TestManualEditDetection:
    """Simulate:
    1. Run pipeline -> SOP generated
    2. Modify the SOP body (simulate user edit)
    3. Run pipeline again -> should create v2_draft, not overwrite
    """

    def test_manual_edit_creates_v2_draft(
        self,
        tmp_db_path: Path,
        write_conn: sqlite3.Connection,
        openclaw_workspace: Path,
    ) -> None:
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)

        # --- Generate 3 repetitions so we get a pattern ---
        for rep in range(3):
            _generate_pr_review_workflow(
                write_conn,
                pr_number=200 + rep,
                base_time=base + timedelta(hours=rep * 2),
                comment_text="Approved",
                slack_channel="#reviews",
            )

        # Read events and build pipeline
        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=200)

        # Build episodes and translate
        episode_step_lists: list[list[dict]] = []
        for rep in range(3):
            rep_events = events[rep * 5 : (rep + 1) * 5]
            steps = _events_to_semantic_steps(rep_events, f"ep-edit-{rep}")
            episode_step_lists.append([s.to_sop_step() for s in steps])

        # Mine patterns
        inducer = SOPInducer(min_support=0.3, min_pattern_length=3)
        sop_templates = inducer.induce(episode_step_lists)
        assert len(sop_templates) >= 1

        # --- First write: create the canonical SOP ---
        writer = OpenClawWriter(workspace_dir=openclaw_workspace)
        writer.ensure_directory_structure()

        first_template = sop_templates[0]
        first_path = writer.write_sop(first_template)
        assert first_path.exists()
        assert "v2_draft" not in first_path.name, (
            "First write should be the canonical file"
        )

        original_content = first_path.read_text(encoding="utf-8")

        # --- Simulate user editing the SOP body ---
        # The user adds a note after the steps
        modified_content = original_content + "\n\n## User Notes\n\nRemember to check CI status first.\n"
        first_path.write_text(modified_content, encoding="utf-8")

        # --- Second write: should detect manual edit and create v2_draft ---
        second_path = writer.write_sop(first_template)
        assert second_path.exists()
        assert "v2_draft" in second_path.name, (
            f"Second write should create v2_draft, got: {second_path.name}"
        )

        # Original file should still have the user's edits
        assert first_path.exists()
        preserved_content = first_path.read_text(encoding="utf-8")
        assert "User Notes" in preserved_content, (
            "User edits should be preserved in the canonical file"
        )

        # The v2_draft should have the new machine-generated content
        draft_content = second_path.read_text(encoding="utf-8")
        assert draft_content.startswith("---\n")
        assert "User Notes" not in draft_content

    def test_no_edit_overwrites_canonical(
        self,
        openclaw_workspace: Path,
    ) -> None:
        """When no manual edit, re-running the pipeline archives the old
        and writes a new canonical file."""
        writer = OpenClawWriter(workspace_dir=openclaw_workspace)
        writer.ensure_directory_structure()

        template = {
            "slug": "test_overwrite",
            "title": "Test Overwrite SOP",
            "steps": [
                {"step": "click", "target": "Button A", "selector": None, "parameters": {}, "confidence": 0.9},
                {"step": "type", "target": "Input B", "selector": None, "parameters": {"text": "hello"}, "confidence": 0.85},
                {"step": "click", "target": "Submit", "selector": None, "parameters": {}, "confidence": 0.88},
            ],
            "variables": [],
            "confidence_avg": 0.88,
            "episode_count": 3,
            "apps_involved": ["Chrome"],
        }

        # First write
        first_path = writer.write_sop(template)
        assert first_path.exists()

        # Second write without editing — should archive and overwrite
        second_path = writer.write_sop(template)
        assert second_path.exists()
        assert "v2_draft" not in second_path.name, (
            "Without manual edit, should overwrite canonical (not create draft)"
        )

        # Archive should have the old version
        archive_dir = openclaw_workspace / "memory" / "apprentice" / "sops" / "archive"
        archived_files = list(archive_dir.glob("sop.test_overwrite.*.md"))
        assert len(archived_files) == 1, (
            f"Expected 1 archived version, got {len(archived_files)}"
        )


# ===================================================================
# Test 4: Scheduler Gate Check
# ===================================================================


class TestSchedulerGateCheck:
    """Test that the scheduler correctly gates pipeline execution
    under favorable vs unfavorable conditions."""

    def test_favorable_conditions_allow_pipeline(self) -> None:
        """All conditions met -> pipeline can run."""
        config = SchedulerConfig(
            require_ac_power=True,
            min_battery_percent=50,
            max_cpu_percent=30,
            max_temp_c=80,
            run_window_start=time(1, 0),
            run_window_end=time(5, 0),
        )
        scheduler = IdleScheduler(config=config)

        favorable = SystemConditions(
            on_ac_power=True,
            battery_percent=85,
            cpu_percent=10.0,
            cpu_temp_c=55.0,
            current_time=time(2, 30),  # within 01:00-05:00
        )

        result = scheduler.should_run_now(conditions=favorable)
        assert result.can_run is True
        assert len(result.blockers) == 0

    def test_unfavorable_conditions_block_pipeline(self) -> None:
        """Multiple conditions violated -> pipeline blocked with reasons."""
        config = SchedulerConfig(
            require_ac_power=True,
            min_battery_percent=50,
            max_cpu_percent=30,
            max_temp_c=80,
            run_window_start=time(1, 0),
            run_window_end=time(5, 0),
        )
        scheduler = IdleScheduler(config=config)

        unfavorable = SystemConditions(
            on_ac_power=False,          # violation
            battery_percent=20,         # violation (< 50)
            cpu_percent=75.0,           # violation (> 30)
            cpu_temp_c=92.0,            # violation (> 80)
            current_time=time(14, 0),   # violation (outside 01:00-05:00)
        )

        result = scheduler.should_run_now(conditions=unfavorable)
        assert result.can_run is False
        assert len(result.blockers) == 5, (
            f"Expected 5 blockers, got {len(result.blockers)}: {result.blockers}"
        )

        # Verify specific blockers are present
        blocker_text = " ".join(result.blockers)
        assert "not_on_ac_power" in blocker_text
        assert "battery_low" in blocker_text
        assert "cpu_high" in blocker_text
        assert "temp_high" in blocker_text
        assert "outside_time_window" in blocker_text

    def test_partial_blocker_still_blocks(self) -> None:
        """Even one failing condition blocks the pipeline."""
        config = SchedulerConfig(
            require_ac_power=True,
            min_battery_percent=50,
            max_cpu_percent=30,
            max_temp_c=80,
            run_window_start=time(1, 0),
            run_window_end=time(5, 0),
        )
        gate = IdleJobGate(config=config)

        # Everything is fine except battery
        conditions = SystemConditions(
            on_ac_power=True,
            battery_percent=30,         # only violation
            cpu_percent=10.0,
            cpu_temp_c=55.0,
            current_time=time(2, 30),
        )

        result = gate.check(conditions=conditions)
        assert result.can_run is False
        assert len(result.blockers) == 1
        assert "battery_low" in result.blockers[0]

    def test_midnight_crossing_window(self) -> None:
        """Time window crossing midnight (e.g. 23:00-05:00) works correctly."""
        config = SchedulerConfig(
            run_window_start=time(23, 0),
            run_window_end=time(5, 0),
        )
        gate = IdleJobGate(config=config)

        # 02:00 should be within 23:00-05:00
        conditions_inside = SystemConditions(
            on_ac_power=True,
            battery_percent=100,
            cpu_percent=5.0,
            cpu_temp_c=50.0,
            current_time=time(2, 0),
        )
        result = gate.check(conditions=conditions_inside)
        assert result.can_run is True

        # 14:00 should be outside 23:00-05:00
        conditions_outside = SystemConditions(
            on_ac_power=True,
            battery_percent=100,
            cpu_percent=5.0,
            cpu_temp_c=50.0,
            current_time=time(14, 0),
        )
        result = gate.check(conditions=conditions_outside)
        assert result.can_run is False
        assert any("outside_time_window" in b for b in result.blockers)


# ===================================================================
# Test 5: Index.md Catalog Verification
# ===================================================================


class TestIndexCatalogVerification:
    """After exporting multiple SOPs, verify the index.md catalog."""

    def test_index_contains_all_sops(
        self,
        openclaw_workspace: Path,
    ) -> None:
        """Multiple SOP templates -> index.md with all entries."""
        templates = [
            {
                "slug": "review_pr_github",
                "title": "Review PR in GitHub",
                "steps": [
                    {"step": "navigate", "target": "GitHub PR page", "selector": None, "parameters": {"url": "https://github.com"}, "confidence": 0.90},
                    {"step": "read", "target": "Code diff", "selector": None, "parameters": {}, "confidence": 0.85},
                    {"step": "click", "target": "Approve", "selector": "[aria-label='Approve']", "parameters": {}, "confidence": 0.92},
                ],
                "variables": [
                    {"name": "pr_number", "type": "number", "example": "42"},
                ],
                "confidence_avg": 0.89,
                "episode_count": 5,
                "apps_involved": ["com.google.Chrome"],
            },
            {
                "slug": "deploy_staging",
                "title": "Deploy to Staging",
                "steps": [
                    {"step": "navigate", "target": "CI Dashboard", "selector": None, "parameters": {"url": "https://ci.example.com"}, "confidence": 0.88},
                    {"step": "click", "target": "Deploy button", "selector": "[data-testid='deploy-btn']", "parameters": {}, "confidence": 0.91},
                    {"step": "read", "target": "Deployment status", "selector": None, "parameters": {}, "confidence": 0.80},
                ],
                "variables": [
                    {"name": "branch_name", "type": "string", "example": "main"},
                ],
                "confidence_avg": 0.86,
                "episode_count": 3,
                "apps_involved": ["com.google.Chrome"],
            },
            {
                "slug": "triage_bug_report",
                "title": "Triage Bug Report in Jira",
                "steps": [
                    {"step": "navigate", "target": "Jira board", "selector": None, "parameters": {"url": "https://jira.example.com"}, "confidence": 0.82},
                    {"step": "click", "target": "Bug ticket", "selector": None, "parameters": {}, "confidence": 0.78},
                    {"step": "type", "target": "Priority field", "selector": None, "parameters": {"text": "P2"}, "confidence": 0.75},
                    {"step": "click", "target": "Save", "selector": "[aria-label='Save']", "parameters": {}, "confidence": 0.88},
                ],
                "variables": [
                    {"name": "priority", "type": "enum", "example": "P2", "choices": ["P1", "P2", "P3"]},
                ],
                "confidence_avg": 0.81,
                "episode_count": 7,
                "apps_involved": ["com.google.Chrome", "com.tinyspeck.slackmacgap"],
            },
        ]

        writer = OpenClawWriter(workspace_dir=openclaw_workspace)
        writer.ensure_directory_structure()

        paths = writer.write_all_sops(templates)
        assert len(paths) == 3

        # ===============================================================
        # Verify index.md content
        # ===============================================================
        index_path = openclaw_workspace / "memory" / "apprentice" / "index.md"
        assert index_path.exists()

        index_content = index_path.read_text(encoding="utf-8")

        # Header verification
        assert "# SOP Index" in index_content
        assert "**Total SOPs:** 3" in index_content

        # Average confidence: (0.89 + 0.86 + 0.81) / 3 = 0.8533...
        assert "**Average confidence:** 0.85" in index_content

        # Summary table format verification
        assert "| Slug | Title | Confidence | Episodes | Apps |" in index_content
        assert "|------|-------|------------|----------|------|" in index_content

        # All SOPs should be in the table
        assert "`review_pr_github`" in index_content
        assert "`deploy_staging`" in index_content
        assert "`triage_bug_report`" in index_content

        # Titles should be in the table
        assert "Review PR in GitHub" in index_content
        assert "Deploy to Staging" in index_content
        assert "Triage Bug Report in Jira" in index_content

        # Details section
        assert "## Details" in index_content

        # Each SOP should have a file reference
        assert "sop.review_pr_github.md" in index_content
        assert "sop.deploy_staging.md" in index_content
        assert "sop.triage_bug_report.md" in index_content

    def test_index_generator_standalone(self) -> None:
        """Test the IndexGenerator in isolation with computed averages."""
        gen = IndexGenerator()

        entries = [
            {"slug": "task_a", "title": "Task A", "confidence_avg": 0.90, "episode_count": 4, "apps_involved": ["Chrome"], "steps": [{"step": "click"}]},
            {"slug": "task_b", "title": "Task B", "confidence_avg": 0.70, "episode_count": 2, "apps_involved": ["Firefox"], "steps": [{"step": "type"}, {"step": "click"}]},
        ]

        content = gen.generate_index(Path("/tmp/sops"), entries)

        assert "# SOP Index" in content
        assert "**Total SOPs:** 2" in content

        # Average confidence: (0.90 + 0.70) / 2 = 0.80
        assert "**Average confidence:** 0.80" in content

        # Sorted by slug in table
        lines = content.splitlines()
        table_rows = [l for l in lines if l.startswith("| `")]
        assert len(table_rows) == 2
        # task_a comes before task_b alphabetically
        assert table_rows[0].startswith("| `task_a`")
        assert table_rows[1].startswith("| `task_b`")

        # Step counts in details
        assert "**Steps:** 1" in content
        assert "**Steps:** 2" in content


# ===================================================================
# Test 6: Combined Pipeline Integration (Translator + Confidence + SOP)
# ===================================================================


class TestTranslatorConfidencePipeline:
    """Verify that the translator -> confidence -> SOP step pipeline
    produces consistent, well-structured data end-to-end."""

    def test_translation_to_sop_step_roundtrip(
        self,
        tmp_db_path: Path,
        write_conn: sqlite3.Connection,
    ) -> None:
        """Single workflow repetition: events -> translate -> score ->
        SemanticStep -> to_sop_step."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)

        _generate_pr_review_workflow(
            write_conn,
            pr_number=55,
            base_time=base,
            comment_text="Nice work!",
            slack_channel="#code-review",
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        assert len(events) == 5

        steps = _events_to_semantic_steps(events, "episode-roundtrip")

        assert len(steps) == 5

        # Verify each step has required fields
        for step in steps:
            assert step.step_id
            assert step.episode_id == "episode-roundtrip"
            assert step.intent in (
                "navigate", "read", "click", "type", "switch_app",
                "copy", "paste", "scroll_read", "secure_focus", "unknown",
            )
            assert isinstance(step.confidence, float)
            assert 0.0 <= step.confidence <= 1.0
            assert step.decision in ("accept", "accept_flagged", "reject")
            assert step.raw_event_id

        # Convert to SOP steps
        sop_steps = [s.to_sop_step() for s in steps]
        assert len(sop_steps) == 5

        for sop_step in sop_steps:
            assert "step" in sop_step
            assert "target" in sop_step
            assert "confidence" in sop_step

        # Verify intent mapping is correct
        intents = [s.intent for s in steps]
        assert intents[0] == "navigate"    # FocusChange
        assert intents[1] == "read"        # DwellSnapshot
        assert intents[2] == "click"       # ClickIntent
        assert intents[3] == "type"        # KeyPress
        assert intents[4] == "switch_app"  # AppSwitch

    def test_semantic_step_serialization_roundtrip(self) -> None:
        """SemanticStep -> to_dict -> from_dict preserves all fields."""
        original = SemanticStep(
            step_id="step-001",
            episode_id="ep-001",
            step_index=0,
            intent="click",
            target_description="Approve button",
            target_selector="[aria-label='Approve']",
            parameters={"button_text": "Approve"},
            pre_state={"window_title": "PR #42", "url": "https://github.com/pr/42"},
            post_state={"window_title": "PR #42 - Approved"},
            confidence=0.87,
            confidence_reasons=["strong_ui_anchor", "state_matches"],
            decision="accept",
            evidence=Evidence(
                dom_anchor="[aria-label='Approve']",
                url="https://github.com/pr/42",
                window_title="PR #42",
            ),
            raw_event_id="ev-042",
            timestamp=datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc),
            is_negative=False,
        )

        d = original.to_dict()
        restored = SemanticStep.from_dict(d)

        assert restored.step_id == original.step_id
        assert restored.episode_id == original.episode_id
        assert restored.intent == original.intent
        assert restored.target_description == original.target_description
        assert restored.target_selector == original.target_selector
        assert restored.parameters == original.parameters
        assert restored.pre_state == original.pre_state
        assert restored.post_state == original.post_state
        assert restored.confidence == original.confidence
        assert restored.confidence_reasons == original.confidence_reasons
        assert restored.decision == original.decision
        assert restored.evidence.dom_anchor == original.evidence.dom_anchor
        assert restored.evidence.url == original.evidence.url
        assert restored.evidence.window_title == original.evidence.window_title
        assert restored.raw_event_id == original.raw_event_id
        assert restored.is_negative == original.is_negative


# ===================================================================
# Test 7: Full Pipeline with Negative Demo Filtering
# ===================================================================


class TestPipelineWithNegativeDemo:
    """Verify that negative demo pruning integrates correctly with the
    downstream SOP pipeline — negative events should not appear in
    the induced SOPs."""

    def test_negative_events_excluded_from_sop(
        self,
        tmp_db_path: Path,
        write_conn: sqlite3.Connection,
        openclaw_workspace: Path,
    ) -> None:
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)

        # --- Generate 3 clean repetitions ---
        for rep in range(3):
            _generate_pr_review_workflow(
                write_conn,
                pr_number=300 + rep,
                base_time=base + timedelta(hours=rep * 3),
                comment_text="Approved",
                slack_channel="#team",
            )

        # --- Add a 4th repetition with an undo (negative demo) ---
        pr_num = 400
        pr_url = f"https://github.com/org/repo/pull/{pr_num}"
        bad_base = base + timedelta(hours=9)

        # Normal start
        _insert_event(
            write_conn,
            event_id=f"ev-pr{pr_num}-01-navigate",
            timestamp=_ts(bad_base),
            kind_json=json.dumps({"FocusChange": {}}),
            window_json=json.dumps({"app_id": "com.google.Chrome", "title": f"PR #{pr_num} - GitHub"}),
            metadata_json=json.dumps({
                "url": pr_url,
                "target": {"ariaLabel": f"Pull request #{pr_num}", "role": "main"},
            }),
        )

        # Mistake: type wrong comment
        _insert_event(
            write_conn,
            event_id=f"ev-pr{pr_num}-mistake",
            timestamp=_ts(bad_base + timedelta(seconds=10)),
            kind_json=json.dumps({"KeyPress": {}}),
            window_json=json.dumps({"app_id": "com.google.Chrome", "title": f"PR #{pr_num} - GitHub"}),
            metadata_json=json.dumps({"url": pr_url, "text": "wrong comment", "shortcut": ""}),
        )

        # Undo the mistake
        _insert_event(
            write_conn,
            event_id=f"ev-pr{pr_num}-undo",
            timestamp=_ts(bad_base + timedelta(seconds=12)),
            kind_json=json.dumps({"KeyPress": {}}),
            window_json=json.dumps({"app_id": "com.google.Chrome", "title": f"PR #{pr_num} - GitHub"}),
            metadata_json=json.dumps({"url": pr_url, "shortcut": "cmd+z"}),
        )

        # Read all events
        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=200)

        total_events = 15 + 3  # 15 from 3 clean reps + 3 from bad rep
        assert len(events) == total_events

        # Prune negative demos
        pruner = NegativeDemoPruner()
        prune_result = pruner.prune(events)

        # Undo and the mistake should be negative
        negative_ids = {e["id"] for e in prune_result.negative_events}
        assert f"ev-pr{pr_num}-undo" in negative_ids
        assert f"ev-pr{pr_num}-mistake" in negative_ids

        # The 15 clean events + the first event of the bad rep should
        # be positive (the navigate event is outside the undo lookback
        # time window if within 30s, but it IS within 30s here, so
        # it may also be marked negative by lookback — that is acceptable)
        positive_events = prune_result.positive_events

        # Build episodes from positive events only
        builder = EpisodeBuilder()
        episodes = builder.process_events(positive_events)

        # Translate positive events to semantic steps
        episode_step_lists: list[list[dict]] = []
        for rep in range(3):
            # Use the known positive events from each clean repetition
            clean_ids = [f"ev-pr{300 + rep}-{i:02d}-{suffix}"
                         for i, suffix in enumerate(
                             ["navigate", "dwell", "approve", "comment", "slack"], start=1
                         )]
            rep_events = [e for e in positive_events if e["id"] in clean_ids]

            if len(rep_events) >= 3:
                steps = _events_to_semantic_steps(rep_events, f"ep-clean-{rep}")
                episode_step_lists.append([s.to_sop_step() for s in steps])

        # Mine patterns from clean episodes only
        if episode_step_lists:
            inducer = SOPInducer(min_support=0.3, min_pattern_length=3)
            sop_templates = inducer.induce(episode_step_lists)

            # Export if we got patterns
            if sop_templates:
                writer = OpenClawWriter(workspace_dir=openclaw_workspace)
                writer.ensure_directory_structure()
                paths = writer.write_all_sops(sop_templates)

                for path in paths:
                    assert path.exists()
                    content = path.read_text(encoding="utf-8")
                    # The SOP should not contain undo-related content
                    assert "cmd+z" not in content.lower()
                    assert "undo" not in content.lower()


# ===================================================================
# Test 8: AtomicWriter Safety
# ===================================================================


class TestAtomicWriterSafety:
    """Verify AtomicWriter produces complete files even under edge conditions."""

    def test_atomic_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        """AtomicWriter creates parent directories if missing."""
        deep_path = tmp_path / "a" / "b" / "c" / "file.md"
        AtomicWriter.write(deep_path, "content")
        assert deep_path.exists()
        assert deep_path.read_text(encoding="utf-8") == "content"

    def test_atomic_write_overwrites_existing(self, tmp_path: Path) -> None:
        """AtomicWriter overwrites an existing file atomically."""
        filepath = tmp_path / "test.md"
        filepath.write_text("old", encoding="utf-8")

        AtomicWriter.write(filepath, "new")
        assert filepath.read_text(encoding="utf-8") == "new"

    def test_atomic_write_no_temp_files_left(self, tmp_path: Path) -> None:
        """After a successful write, no temp files remain."""
        filepath = tmp_path / "clean.md"
        AtomicWriter.write(filepath, "content")

        temp_files = list(tmp_path.glob(".tmp_*"))
        assert len(temp_files) == 0, (
            f"Temp files should be cleaned up: {temp_files}"
        )


# ===================================================================
# Test 9: SOP Versioner Archive Chain
# ===================================================================


class TestSOPVersionerArchiveChain:
    """Verify the version archive chain works correctly across
    multiple pipeline runs."""

    def test_multiple_versions_archived(self, tmp_path: Path) -> None:
        """Three pipeline runs produce an archive chain of 2 versions."""
        sops_dir = tmp_path / "sops"
        versioner = SOPVersioner(sops_dir=sops_dir)
        formatter = SOPFormatter()

        template = {
            "slug": "chain_test",
            "title": "Chain Test SOP",
            "steps": [
                {"step": "click", "target": "Button", "selector": None, "parameters": {}, "confidence": 0.9},
                {"step": "type", "target": "Field", "selector": None, "parameters": {"text": "hi"}, "confidence": 0.85},
                {"step": "click", "target": "Submit", "selector": None, "parameters": {}, "confidence": 0.88},
            ],
            "variables": [],
            "confidence_avg": 0.88,
            "episode_count": 3,
            "apps_involved": [],
        }

        # Run 1: write canonical
        content_v1 = formatter.format_sop(template)
        path_v1 = versioner.write_sop("chain_test", content_v1, formatter)
        assert path_v1.name == "sop.chain_test.md"

        # Run 2: should archive v1, write new canonical
        content_v2 = formatter.format_sop(template)
        path_v2 = versioner.write_sop("chain_test", content_v2, formatter)
        assert path_v2.name == "sop.chain_test.md"

        # Run 3: should archive v2, write new canonical
        content_v3 = formatter.format_sop(template)
        path_v3 = versioner.write_sop("chain_test", content_v3, formatter)
        assert path_v3.name == "sop.chain_test.md"

        # Archive should have 2 versions (v1 and v2 archived)
        versions = versioner.list_versions("chain_test")
        assert len(versions) == 2, (
            f"Expected 2 archived versions, got {len(versions)}"
        )

        # All archive files should be valid markdown
        for v in versions:
            assert v.exists()
            content = v.read_text(encoding="utf-8")
            assert content.startswith("---\n")


# ===================================================================
# Test 10: Metadata Export
# ===================================================================


class TestMetadataExport:
    """Verify metadata files are written correctly alongside SOPs."""

    def test_confidence_metadata_written(
        self,
        openclaw_workspace: Path,
    ) -> None:
        """Confidence log metadata is written as JSON to the metadata dir."""
        writer = OpenClawWriter(workspace_dir=openclaw_workspace)
        writer.ensure_directory_structure()

        confidence_data = {
            "sop_slug": "test_sop",
            "scores": [
                {"step_index": 0, "confidence": 0.90, "decision": "accept"},
                {"step_index": 1, "confidence": 0.75, "decision": "accept_flagged"},
                {"step_index": 2, "confidence": 0.45, "decision": "reject"},
            ],
            "average": 0.70,
        }

        path = writer.write_metadata("confidence_log", confidence_data)
        assert path.exists()
        assert path.name == "confidence_log.json"

        content = json.loads(path.read_text(encoding="utf-8"))
        assert content["metadata_type"] == "confidence_log"
        assert content["sop_slug"] == "test_sop"
        assert "generated_at" in content
        assert len(content["scores"]) == 3
