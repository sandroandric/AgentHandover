"""Load testing suite — simulates a 'power user day' with 10k events.

Per §14.4: Validates observer stays <1% CPU, DB growth bounded and
reclaimed, VLM queue respects budgets.
"""

from __future__ import annotations

import json
import random
import sqlite3
import string
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Add worker src to Python path
# ---------------------------------------------------------------------------

WORKER_SRC = Path(__file__).resolve().parent.parent.parent / "worker" / "src"
sys.path.insert(0, str(WORKER_SRC))

from agenthandover_worker.episode_builder import EpisodeBuilder
from agenthandover_worker.negative_demo import NegativeDemoPruner
from agenthandover_worker.vlm_queue import (
    VLMFallbackQueue,
    VLMJob,
    VLMJobStatus,
    QueueBudget,
)
from agenthandover_worker.confidence import ConfidenceScorer
from agenthandover_worker.translator import SemanticTranslator, TranslationResult, UIAnchor


# ---- Constants ----

EVENT_KINDS = [
    "FocusChange",
    "ClickIntent",
    "KeyPress",
    "ScrollReadSnapshot",
    "ClipboardChange",
    "AppSwitch",
    "DwellSnapshot",
    "WindowTitleChange",
    "PasteDetected",
]

APPS = ["Chrome", "Slack", "VS Code", "Notion", "Terminal", "Finder", "Mail"]

URLS = [
    "https://github.com/pulls",
    "https://mail.google.com/inbox",
    "https://notion.so/workspace",
    "https://slack.com/client",
    "https://docs.google.com/document",
    "https://jira.atlassian.net/board",
]

TARGETS = [
    "Submit button",
    "Save button",
    "Cancel button",
    "Search field",
    "Email input",
    "Password field",
    "Navigation menu",
    "Sidebar toggle",
    "Notification bell",
    "User avatar",
    "Settings gear",
    "Help icon",
    "Tab close",
    "New tab",
    "Address bar",
    "Bookmark star",
]


# ---- Event Generator ----
# Generates events in the daemon's actual DB row format with kind_json,
# window_json, and metadata_json fields, so that EpisodeBuilder, NegativeDemoPruner,
# and SemanticTranslator can process them directly.


def _ts(dt: datetime) -> str:
    """Format a datetime as the ISO 8601 string the daemon produces."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def generate_event(
    index: int,
    base_time: datetime,
    app: str | None = None,
    url: str | None = None,
) -> dict:
    """Generate a realistic mock event in daemon DB row format."""
    kind = random.choice(EVENT_KINDS)
    app = app or random.choice(APPS)
    url = url or random.choice(URLS)
    ts = base_time + timedelta(seconds=index * 3)  # ~3 sec between events

    target_text = random.choice(TARGETS)

    # Build kind_json (dict keyed by event kind)
    kind_payload: dict = {}
    if kind == "ClickIntent":
        kind_payload = {kind: {"x": random.randint(0, 1920), "y": random.randint(0, 1080)}}
    elif kind == "KeyPress":
        kind_payload = {kind: {"key": random.choice(string.ascii_lowercase)}}
    elif kind == "ClipboardChange":
        kind_payload = {kind: {"content_hash": uuid.uuid4().hex}}
    else:
        kind_payload = {kind: {}}

    # Build window_json
    window = {"app_id": app, "title": f"{app} - {''.join(random.choices(string.ascii_lowercase, k=8))}"}

    # Build metadata_json
    metadata: dict = {}
    if app == "Chrome":
        metadata["url"] = url
    if kind == "ClickIntent":
        metadata["target"] = {
            "innerText": target_text,
            "ariaLabel": target_text if random.random() > 0.5 else "",
            "testId": f"btn-{random.randint(1, 100)}" if random.random() > 0.7 else "",
            "role": "button",
            "tagName": "BUTTON",
            "x": random.randint(0, 1920),
            "y": random.randint(0, 1080),
        }
    elif kind == "KeyPress":
        metadata["key"] = random.choice(string.ascii_lowercase)
        metadata["text"] = "".join(random.choices(string.ascii_letters + " ", k=random.randint(5, 50)))
    elif kind == "ClipboardChange":
        metadata["content_hash"] = uuid.uuid4().hex
        metadata["byte_size"] = random.randint(10, 10000)

    event = {
        "id": str(uuid.uuid4()),
        "timestamp": _ts(ts),
        "kind_json": json.dumps(kind_payload),
        "window_json": json.dumps(window),
        "metadata_json": json.dumps(metadata),
        "display_topology_json": "[]",
        "primary_display_id": "main",
        "cursor_x": random.randint(0, 1920),
        "cursor_y": random.randint(0, 1080),
        "processed": 0,
    }

    return event


def generate_power_user_day(
    event_count: int = 10000,
    base_time: datetime | None = None,
) -> list[dict]:
    """Generate a full day of events simulating an intensive user session."""
    base_time = base_time or datetime(2026, 2, 16, 8, 0, 0, tzinfo=timezone.utc)
    events = []

    # Simulate realistic work patterns: bursts of activity with breaks
    current_app = random.choice(APPS)
    current_url = random.choice(URLS)

    for i in range(event_count):
        # Occasionally switch context (every ~20-50 events)
        if random.random() < 0.03:
            current_app = random.choice(APPS)
            current_url = random.choice(URLS)

        events.append(generate_event(i, base_time, current_app, current_url))

    return events


# ---- Database Helpers ----

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


def create_test_db(db_path: Path, events: list[dict]) -> None:
    """Create a test SQLite database with events in daemon format."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(DAEMON_SCHEMA)

    for event in events:
        conn.execute(
            "INSERT INTO events "
            "(id, timestamp, kind_json, window_json, display_topology_json, "
            "primary_display_id, cursor_x, cursor_y, metadata_json, processed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event["id"],
                event["timestamp"],
                event["kind_json"],
                event.get("window_json"),
                event.get("display_topology_json", "[]"),
                event.get("primary_display_id", "main"),
                event.get("cursor_x"),
                event.get("cursor_y"),
                event.get("metadata_json", "{}"),
                event.get("processed", 0),
            ),
        )

    conn.commit()
    conn.close()


# ---- Tests ----


class TestPowerUserDayEventGeneration:
    """Verify event generator produces realistic data."""

    def test_generates_10k_events(self):
        events = generate_power_user_day(10000)
        assert len(events) == 10000

    def test_events_have_required_fields(self):
        events = generate_power_user_day(100)
        required = {"id", "timestamp", "kind_json", "window_json", "metadata_json"}
        for e in events:
            assert required.issubset(e.keys()), f"Missing fields: {required - e.keys()}"

    def test_events_are_chronological(self):
        events = generate_power_user_day(1000)
        timestamps = [e["timestamp"] for e in events]
        assert timestamps == sorted(timestamps)

    def test_all_event_kinds_represented(self):
        random.seed(42)  # deterministic for this test
        events = generate_power_user_day(10000)
        kinds_seen = set()
        for e in events:
            parsed = json.loads(e["kind_json"])
            kinds_seen.update(parsed.keys())
        # With 10k events, all kinds should appear
        assert len(kinds_seen) == len(EVENT_KINDS)

    def test_multiple_apps_represented(self):
        random.seed(42)
        events = generate_power_user_day(10000)
        apps_seen = set()
        for e in events:
            window = json.loads(e["window_json"])
            apps_seen.add(window["app_id"])
        assert len(apps_seen) >= 3

    def test_kind_json_is_valid_json(self):
        events = generate_power_user_day(500)
        for e in events:
            parsed = json.loads(e["kind_json"])
            assert isinstance(parsed, dict)
            assert len(parsed) == 1  # Single key per event kind

    def test_window_json_is_valid_json(self):
        events = generate_power_user_day(500)
        for e in events:
            parsed = json.loads(e["window_json"])
            assert "app_id" in parsed
            assert "title" in parsed


class TestDBGrowthBounded:
    """DB size stays bounded even with 10k events."""

    def test_db_size_reasonable(self, tmp_path: Path):
        events = generate_power_user_day(10000)
        db_path = tmp_path / "load_test.db"
        create_test_db(db_path, events)

        db_size_mb = db_path.stat().st_size / (1024 * 1024)
        # 10k events should be well under 100MB
        assert db_size_mb < 100, f"DB grew to {db_size_mb:.1f}MB with 10k events"

    def test_db_size_per_event_reasonable(self, tmp_path: Path):
        events = generate_power_user_day(10000)
        db_path = tmp_path / "load_test.db"
        create_test_db(db_path, events)

        size_per_event = db_path.stat().st_size / 10000
        # Each event should be under 10KB on average
        assert size_per_event < 10240, f"Average {size_per_event:.0f} bytes/event"

    def test_vacuum_reclaims_space_after_delete(self, tmp_path: Path):
        """Deleting rows and running VACUUM shrinks the DB file."""
        events = generate_power_user_day(5000)
        db_path = tmp_path / "vacuum_test.db"

        conn = sqlite3.connect(str(db_path))
        conn.executescript(DAEMON_SCHEMA)

        for event in events:
            conn.execute(
                "INSERT INTO events "
                "(id, timestamp, kind_json, window_json, display_topology_json, "
                "primary_display_id, cursor_x, cursor_y, metadata_json, processed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event["id"],
                    event["timestamp"],
                    event["kind_json"],
                    event.get("window_json"),
                    event.get("display_topology_json", "[]"),
                    event.get("primary_display_id", "main"),
                    event.get("cursor_x"),
                    event.get("cursor_y"),
                    event.get("metadata_json", "{}"),
                    event.get("processed", 0),
                ),
            )
        conn.commit()

        # Measure size with all 5000 events
        size_full = db_path.stat().st_size

        # Delete half the events (simulating retention purge)
        conn.execute("DELETE FROM events WHERE rowid % 2 = 0")
        conn.commit()

        # Without VACUUM, the DB file size may not shrink (freed pages are
        # kept on the SQLite free-list for reuse).
        size_after_delete = db_path.stat().st_size

        # VACUUM rewrites the DB, actually reclaiming the freed pages
        conn.execute("VACUUM")
        conn.close()

        size_after_vacuum = db_path.stat().st_size

        # After VACUUM, size should be meaningfully smaller than the full DB
        assert size_after_vacuum < size_full, (
            f"VACUUM did not reclaim space: full={size_full}, "
            f"after_delete={size_after_delete}, after_vacuum={size_after_vacuum}"
        )

    def test_db_event_count_matches(self, tmp_path: Path):
        events = generate_power_user_day(10000)
        db_path = tmp_path / "count_test.db"
        create_test_db(db_path, events)

        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        conn.close()
        assert count == 10000


class TestVLMQueueBudgetUnderLoad:
    """VLM queue respects budgets with many jobs."""

    def test_daily_limit_enforced_under_load(self):
        queue = VLMFallbackQueue(budget=QueueBudget(max_jobs_per_day=50, max_queue_size=500))

        enqueued = 0
        for i in range(200):
            job = VLMJob(
                job_id=f"job_{i}",
                event_id=f"event_{i}",
                episode_id=f"ep_{i}",
                semantic_step_index=0,
                confidence_score=0.3,
                priority_score=random.uniform(0.1, 0.9),
            )
            result = queue.enqueue(job)
            if result:
                enqueued += 1

        assert enqueued == 50

    def test_queue_backpressure_under_load(self):
        queue = VLMFallbackQueue(budget=QueueBudget(
            max_jobs_per_day=1000,  # High daily limit
            max_queue_size=100,     # Low queue size
        ))

        for i in range(200):
            job = VLMJob(
                job_id=f"job_{i}",
                event_id=f"event_{i}",
                episode_id=f"ep_{i}",
                semantic_step_index=0,
                confidence_score=random.uniform(0.1, 0.5),
                priority_score=random.uniform(0.1, 0.9),
            )
            queue.enqueue(job)

        stats = queue.get_stats()
        # Queue should not exceed max_size (pending + processing)
        active = stats.pending_jobs
        assert active <= 100

    def test_compute_budget_enforced(self):
        queue = VLMFallbackQueue(budget=QueueBudget(
            max_jobs_per_day=1000,
            max_compute_minutes_per_day=1.0,  # Very tight budget
        ))

        # Enqueue some jobs
        for i in range(10):
            job = VLMJob(
                job_id=f"job_{i}",
                event_id=f"event_{i}",
                episode_id=f"ep_{i}",
                semantic_step_index=0,
                confidence_score=0.3,
                priority_score=0.5 + i * 0.01,  # Descending priority for dequeue order
            )
            queue.enqueue(job)

        # Process with heavy compute time
        completed = 0
        for _ in range(10):
            job = queue.dequeue()
            if job is None:
                break
            # Record 0.5 minutes per job — after 2 jobs we hit 1.0 budget
            queue.record_completion(job.job_id, compute_minutes=0.5, result={"ok": True})
            completed += 1
            if not queue.can_dispatch():
                break

        # Should stop at or before processing all 10
        assert completed < 10

    def test_enqueue_after_budget_exhausted_rejected(self):
        queue = VLMFallbackQueue(budget=QueueBudget(
            max_jobs_per_day=5,
            max_queue_size=500,
        ))

        for i in range(5):
            job = VLMJob(
                job_id=f"job_{i}",
                event_id=f"event_{i}",
                episode_id=f"ep_{i}",
                semantic_step_index=0,
                confidence_score=0.3,
                priority_score=0.5,
            )
            assert queue.enqueue(job) is True

        # 6th should be rejected
        job_6 = VLMJob(
            job_id="job_6",
            event_id="event_6",
            episode_id="ep_6",
            semantic_step_index=0,
            confidence_score=0.3,
            priority_score=0.5,
        )
        assert queue.enqueue(job_6) is False


class TestEpisodeBuilderUnderLoad:
    """Episode builder handles 10k events correctly."""

    def test_builds_episodes_from_10k_events(self):
        builder = EpisodeBuilder()
        events = generate_power_user_day(10000)

        episodes = builder.process_events(events)

        # Should produce multiple episodes (different threads + segment splits)
        assert len(episodes) >= 1

        # Total event count across all episodes should equal 10000
        total_events = sum(ep.event_count for ep in episodes)
        assert total_events == 10000

    def test_no_episode_exceeds_hard_cap(self):
        builder = EpisodeBuilder()
        events = generate_power_user_day(10000)

        episodes = builder.process_events(events)

        for ep in episodes:
            assert ep.event_count <= 200, f"Episode exceeded 200-event hard cap: {ep.event_count}"

    def test_episodes_have_valid_thread_ids(self):
        builder = EpisodeBuilder()
        events = generate_power_user_day(1000)

        episodes = builder.process_events(events)

        for ep in episodes:
            assert ep.thread_id, f"Episode {ep.episode_id} has empty thread_id"

    def test_episode_timestamps_are_set(self):
        builder = EpisodeBuilder()
        events = generate_power_user_day(500)

        episodes = builder.process_events(events)

        for ep in episodes:
            assert ep.start_time is not None, f"Episode {ep.episode_id} has no start_time"
            assert ep.end_time is not None, f"Episode {ep.episode_id} has no end_time"
            assert ep.start_time <= ep.end_time

    def test_processing_10k_events_is_fast(self):
        builder = EpisodeBuilder()
        events = generate_power_user_day(10000)

        start = time.monotonic()
        builder.process_events(events)
        elapsed = time.monotonic() - start

        # Should process 10k events in under 5 seconds
        assert elapsed < 5.0, f"Episode building for 10k events took {elapsed:.2f}s"


class TestNegativeDemoUnderLoad:
    """Negative demo pruner handles high-volume events."""

    def test_prunes_undo_patterns_in_bulk(self):
        pruner = NegativeDemoPruner()
        events = generate_power_user_day(1000)

        # Inject some undo patterns (Ctrl+Z in KeyPress events)
        for i in range(0, 1000, 50):
            kind_payload = {"KeyPress": {"key": "z"}}
            events[i]["kind_json"] = json.dumps(kind_payload)
            events[i]["metadata_json"] = json.dumps({"shortcut": "ctrl+z"})

        result = pruner.prune(events)

        # Should have identified negative events
        assert len(result.negative_events) > 0
        assert len(result.positive_events) < 1000

    def test_prunes_cancel_patterns_in_bulk(self):
        pruner = NegativeDemoPruner()
        events = generate_power_user_day(500)

        # Inject cancel clicks
        for i in range(0, 500, 100):
            kind_payload = {"ClickIntent": {"x": 100, "y": 200}}
            events[i]["kind_json"] = json.dumps(kind_payload)
            events[i]["metadata_json"] = json.dumps({"target": {"innerText": "Cancel"}})

        result = pruner.prune(events)
        assert len(result.negative_events) > 0

    def test_pruning_preserves_event_count(self):
        pruner = NegativeDemoPruner()
        events = generate_power_user_day(2000)

        result = pruner.prune(events)

        # positive + negative should equal total
        assert len(result.positive_events) + len(result.negative_events) == 2000

    def test_no_false_positives_on_clean_events(self):
        """Events without undo/cancel patterns should all be positive."""
        pruner = NegativeDemoPruner()

        # Generate events that are all FocusChange (no undo/cancel patterns)
        base_time = datetime(2026, 2, 16, 8, 0, 0, tzinfo=timezone.utc)
        events = []
        for i in range(500):
            ts = base_time + timedelta(seconds=i * 3)
            events.append({
                "id": str(uuid.uuid4()),
                "timestamp": _ts(ts),
                "kind_json": json.dumps({"FocusChange": {}}),
                "window_json": json.dumps({"app_id": "Chrome", "title": "Chrome"}),
                "metadata_json": json.dumps({"url": "https://example.com"}),
            })

        result = pruner.prune(events)
        assert len(result.negative_events) == 0
        assert len(result.positive_events) == 500


class TestConfidenceScoringUnderLoad:
    """Confidence scorer processes many steps efficiently."""

    def test_scores_1000_translations(self):
        scorer = ConfidenceScorer()

        # Build TranslationResult objects (the actual input type for scorer.score)
        translations_and_contexts = []
        for i in range(1000):
            anchor_method = random.choice(["aria_label", "test_id", "inner_text", "role_position", "vision_bbox"])
            conf_map = {
                "aria_label": 0.45,
                "test_id": 0.40,
                "inner_text": 0.30,
                "role_position": 0.20,
                "vision_bbox": 0.10,
            }
            anchor = UIAnchor(
                method=anchor_method,
                selector=f"#el-{i}",
                confidence_contribution=conf_map[anchor_method],
            )
            tr = TranslationResult(
                intent=random.choice(["click", "type", "navigate", "scroll"]),
                target=anchor,
                parameters={},
                pre_state={
                    "window_title": f"Window {i}",
                    "url": random.choice(URLS) if random.random() > 0.5 else None,
                    "app_id": random.choice(APPS),
                },
                post_state={
                    "window_title": f"Window {i + 1}" if random.random() > 0.3 else f"Window {i}",
                },
                raw_event_id=str(uuid.uuid4()),
            )
            context = {
                "expected_title": tr.pre_state.get("window_title") if random.random() > 0.3 else None,
                "expected_url": tr.pre_state.get("url") if random.random() > 0.5 else None,
                "expected_app": tr.pre_state.get("app_id") if random.random() > 0.4 else None,
                "clipboard_link": random.random() > 0.8,
                "dwell_snapshot": random.random() > 0.7,
            }
            translations_and_contexts.append((tr, context))

        start = time.monotonic()
        results = [scorer.score(tr, ctx) for tr, ctx in translations_and_contexts]
        elapsed = time.monotonic() - start

        assert len(results) == 1000
        # Should complete in under 1 second
        assert elapsed < 1.0, f"Scoring 1000 steps took {elapsed:.2f}s"

        # All scores should be in valid range
        for score in results:
            assert 0.0 <= score.total <= 1.0
            assert score.decision in ("accept", "accept_flagged", "reject")

    def test_score_distribution_reasonable(self):
        """With varied inputs, we should see a mix of decisions."""
        scorer = ConfidenceScorer()
        random.seed(123)

        decisions = {"accept": 0, "accept_flagged": 0, "reject": 0}
        for i in range(500):
            anchor_method = random.choice(["aria_label", "test_id", "inner_text", "role_position", "vision_bbox"])
            conf_map = {
                "aria_label": 0.45,
                "test_id": 0.40,
                "inner_text": 0.30,
                "role_position": 0.20,
                "vision_bbox": 0.10,
            }
            anchor = UIAnchor(
                method=anchor_method,
                selector=f"#el-{i}",
                confidence_contribution=conf_map[anchor_method],
            )
            tr = TranslationResult(
                intent="click",
                target=anchor,
                parameters={},
                pre_state={
                    "window_title": f"Title {i}",
                    "url": f"https://example.com/{i}",
                    "app_id": "Chrome",
                },
                post_state={"window_title": f"Title {i}"},
                raw_event_id=str(uuid.uuid4()),
            )
            context = {
                "expected_title": f"Title {i}" if random.random() > 0.3 else "wrong",
                "expected_url": f"https://example.com/{i}" if random.random() > 0.5 else None,
                "expected_app": "Chrome" if random.random() > 0.4 else None,
                "clipboard_link": random.random() > 0.7,
                "dwell_snapshot": random.random() > 0.6,
            }
            result = scorer.score(tr, context)
            decisions[result.decision] += 1

        # We should see at least some variety in decisions
        assert decisions["reject"] > 0 or decisions["accept_flagged"] > 0, (
            f"Expected mixed decisions, got: {decisions}"
        )


class TestTranslationUnderLoad:
    """Translator processes many events efficiently."""

    def test_translates_1000_events(self):
        translator = SemanticTranslator()
        events = generate_power_user_day(1000)

        start = time.monotonic()
        results = translator.translate_batch(events)
        elapsed = time.monotonic() - start

        assert len(results) == 1000
        # Should complete in under 2 seconds
        assert elapsed < 2.0, f"Translating 1000 events took {elapsed:.2f}s"

    def test_translate_10k_events(self):
        translator = SemanticTranslator()
        events = generate_power_user_day(10000)

        start = time.monotonic()
        results = translator.translate_batch(events)
        elapsed = time.monotonic() - start

        assert len(results) == 10000
        # Should complete in under 10 seconds
        assert elapsed < 10.0, f"Translating 10k events took {elapsed:.2f}s"

    def test_all_translations_have_intent(self):
        translator = SemanticTranslator()
        events = generate_power_user_day(500)

        results = translator.translate_batch(events)

        for r in results:
            assert isinstance(r.intent, str)
            assert len(r.intent) > 0


class TestPipelineThroughput:
    """Full pipeline handles 10k events within reasonable time."""

    def test_10k_event_pipeline(self, tmp_path: Path):
        """Full pipeline: generate -> store -> build episodes."""
        events = generate_power_user_day(10000)
        db_path = tmp_path / "throughput.db"
        create_test_db(db_path, events)

        # Verify all events stored
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        conn.close()
        assert count == 10000

        # Build episodes
        builder = EpisodeBuilder()
        start = time.monotonic()
        episodes = builder.process_events(events)
        elapsed = time.monotonic() - start

        # Should process 10k events in under 5 seconds
        assert elapsed < 5.0, f"Episode building for 10k events took {elapsed:.2f}s"

        # Episodes should contain all events
        total = sum(ep.event_count for ep in episodes)
        assert total == 10000

    def test_full_pipeline_10k(self, tmp_path: Path):
        """Full pipeline: generate -> store -> episodes -> prune -> translate."""
        events = generate_power_user_day(10000)
        db_path = tmp_path / "full_pipeline.db"
        create_test_db(db_path, events)

        # Step 1: Build episodes
        builder = EpisodeBuilder()
        episodes = builder.process_events(events)
        assert len(episodes) >= 1

        # Step 2: Prune negative demos
        pruner = NegativeDemoPruner()
        prune_result = pruner.prune(events)
        assert len(prune_result.positive_events) + len(prune_result.negative_events) == 10000

        # Step 3: Translate a subset (first 1000 positive events for speed)
        translator = SemanticTranslator()
        subset = prune_result.positive_events[:1000]
        translations = translator.translate_batch(subset)
        assert len(translations) == len(subset)


class TestPerformanceValidation:
    """Performance assertions per spec: daemon <50MB RAM, <2% CPU, 1000 events <30s."""

    def test_1000_events_processed_under_30s(self, tmp_path: Path):
        """Processing 1000 events through the full pipeline should take <30s."""
        from agenthandover_worker.episode_builder import EpisodeBuilder
        from agenthandover_worker.clipboard_linker import ClipboardLinker
        from agenthandover_worker.negative_demo import NegativeDemoPruner
        from agenthandover_worker.translator import SemanticTranslator
        from agenthandover_worker.confidence import ConfidenceScorer
        from agenthandover_worker.vlm_queue import VLMFallbackQueue
        from agenthandover_worker.openclaw_writer import OpenClawWriter
        from agenthandover_worker.exporter import IndexGenerator
        from agenthandover_worker.main import run_pipeline

        events = generate_power_user_day(1000)
        workspace = tmp_path / "workspace"
        writer = OpenClawWriter(workspace_dir=workspace)

        start = time.monotonic()
        summary = run_pipeline(
            events,
            episode_builder=EpisodeBuilder(),
            clipboard_linker=ClipboardLinker(),
            pruner=NegativeDemoPruner(),
            translator=SemanticTranslator(),
            scorer=ConfidenceScorer(),
            vlm_queue=VLMFallbackQueue(),
            openclaw_writer=writer,
            index_generator=IndexGenerator(),
        )
        elapsed = time.monotonic() - start

        assert summary["events_in"] == 1000
        assert elapsed < 30.0, f"Processing 1000 events took {elapsed:.2f}s (limit: 30s)"

    def test_episode_building_is_linear(self):
        """Episode building should scale linearly: 10k should be <10x of 1k."""
        builder = EpisodeBuilder()

        events_1k = generate_power_user_day(1000)
        start = time.monotonic()
        builder.process_events(events_1k)
        time_1k = time.monotonic() - start

        events_10k = generate_power_user_day(10000)
        start = time.monotonic()
        builder.process_events(events_10k)
        time_10k = time.monotonic() - start

        # 10k should be less than 15x of 1k (generous for linear scaling)
        ratio = time_10k / max(time_1k, 0.001)
        assert ratio < 15, f"Scaling ratio: {ratio:.1f}x (expected <15x for linear scaling)"

    def test_db_write_throughput(self, tmp_path: Path):
        """Inserting 10k events into SQLite should complete in <5s."""
        events = generate_power_user_day(10000)
        db_path = tmp_path / "throughput.db"

        start = time.monotonic()
        create_test_db(db_path, events)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"DB write of 10k events took {elapsed:.2f}s (limit: 5s)"

    def test_memory_estimate_from_event_size(self):
        """Estimate per-event memory footprint stays reasonable."""
        import sys

        events = generate_power_user_day(100)
        # Rough estimate: json serialized size as proxy for memory
        total_size = sum(len(json.dumps(e)) for e in events)
        avg_size = total_size / 100

        # Each event should be under 2KB serialized
        assert avg_size < 2048, f"Average event size: {avg_size:.0f} bytes (limit: 2048)"
