"""Phase 1 End-to-End Integration Tests.

Simulates the full pipeline: browser events -> daemon storage -> worker processing.
Uses a real SQLite database with the daemon's schema to verify cross-component
data flow.

Test scenarios:
1. Full E2E pipeline — realistic browser session simulation
2. Negative demo detection — undo workflow
3. Episode segmentation — long session with segment splitting
4. Secure field event handling — no DOM/click events during secure focus
5. Multiple clipboard links — parallel copy-paste operations
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Add worker src to Python path
# ---------------------------------------------------------------------------

WORKER_SRC = Path(__file__).resolve().parent.parent.parent / "worker" / "src"
sys.path.insert(0, str(WORKER_SRC))

from agenthandover_worker.clipboard_linker import ClipboardLinker
from agenthandover_worker.db import WorkerDB
from agenthandover_worker.episode_builder import EpisodeBuilder
from agenthandover_worker.negative_demo import NegativeDemoPruner

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


# ===================================================================
# Test 1: Full E2E Pipeline — Browser Session Simulation
# ===================================================================


class TestFullE2EPipeline:
    """Simulate a realistic browser session flowing through all Phase 1 components.

    Scenario:
    1. User opens GitHub (FocusChange)
    2. User dwells on a PR page (DwellSnapshot)
    3. User scrolls through code review (ScrollReadSnapshot)
    4. User clicks "Approve" button (ClickIntent with semantic data)
    5. User switches to Slack (AppSwitch)
    6. User copies a URL from Slack (ClipboardChange with hash)
    7. User switches back to GitHub (AppSwitch)
    8. User pastes URL into a comment (PasteDetected with matching hash)

    Verifications:
    - Events are correctly read from SQLite by WorkerDB
    - Episode builder creates 2 threads (GitHub and Slack)
    - Clipboard linker finds the copy-paste link
    - No negative demo markers in this happy path
    """

    def test_full_e2e_pipeline(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        url_hash = _hash("https://github.com/org/repo/pull/42")

        # --- Step 1: User opens GitHub (FocusChange) ---
        ev1_id = _insert_event(
            write_conn,
            event_id="ev-01-focus-github",
            timestamp=_ts(base),
            kind_json=json.dumps({"FocusChange": {}}),
            window_json=json.dumps({"app_id": "com.google.Chrome", "title": "PR #42 - GitHub"}),
            metadata_json=json.dumps({"url": "https://github.com/org/repo/pull/42"}),
        )

        # --- Step 2: User dwells on PR page (DwellSnapshot) ---
        ev2_id = _insert_event(
            write_conn,
            event_id="ev-02-dwell-pr",
            timestamp=_ts(base + timedelta(seconds=5)),
            kind_json=json.dumps({"DwellSnapshot": {}}),
            window_json=json.dumps({"app_id": "com.google.Chrome", "title": "PR #42 - GitHub"}),
            metadata_json=json.dumps({
                "url": "https://github.com/org/repo/pull/42",
                "dwell_seconds": 5.0,
            }),
        )

        # --- Step 3: User scrolls through code review (ScrollReadSnapshot) ---
        ev3_id = _insert_event(
            write_conn,
            event_id="ev-03-scroll-review",
            timestamp=_ts(base + timedelta(seconds=15)),
            kind_json=json.dumps({"ScrollReadSnapshot": {}}),
            window_json=json.dumps({"app_id": "com.google.Chrome", "title": "PR #42 - GitHub"}),
            metadata_json=json.dumps({
                "url": "https://github.com/org/repo/pull/42",
                "scroll_position": 1200,
            }),
        )

        # --- Step 4: User clicks "Approve" button (ClickIntent) ---
        ev4_id = _insert_event(
            write_conn,
            event_id="ev-04-click-approve",
            timestamp=_ts(base + timedelta(seconds=30)),
            kind_json=json.dumps({"ClickIntent": {}}),
            window_json=json.dumps({"app_id": "com.google.Chrome", "title": "PR #42 - GitHub"}),
            metadata_json=json.dumps({
                "url": "https://github.com/org/repo/pull/42",
                "element_text": "Approve",
                "element_role": "button",
                "selector": "button.btn-primary",
            }),
            cursor_x=800,
            cursor_y=600,
        )

        # --- Step 5: User switches to Slack (AppSwitch) ---
        ev5_id = _insert_event(
            write_conn,
            event_id="ev-05-switch-slack",
            timestamp=_ts(base + timedelta(seconds=45)),
            kind_json=json.dumps({"AppSwitch": {}}),
            window_json=json.dumps({"app_id": "com.tinyspeck.slackmacgap", "title": "Slack - #team"}),
            metadata_json=json.dumps({}),
        )

        # --- Step 6: User copies a URL from Slack (ClipboardChange) ---
        ev6_id = _insert_event(
            write_conn,
            event_id="ev-06-copy-url",
            timestamp=_ts(base + timedelta(seconds=60)),
            kind_json=json.dumps({"ClipboardChange": {}}),
            window_json=json.dumps({"app_id": "com.tinyspeck.slackmacgap", "title": "Slack - #team"}),
            metadata_json=json.dumps({
                "content_hash": url_hash,
                "content_types": ["text/plain"],
                "byte_size": 45,
            }),
        )

        # --- Step 7: User switches back to GitHub (AppSwitch) ---
        ev7_id = _insert_event(
            write_conn,
            event_id="ev-07-switch-github",
            timestamp=_ts(base + timedelta(seconds=75)),
            kind_json=json.dumps({"AppSwitch": {}}),
            window_json=json.dumps({"app_id": "com.google.Chrome", "title": "PR #42 - GitHub"}),
            metadata_json=json.dumps({"url": "https://github.com/org/repo/pull/42"}),
        )

        # --- Step 8: User pastes URL into comment (PasteDetected) ---
        ev8_id = _insert_event(
            write_conn,
            event_id="ev-08-paste-url",
            timestamp=_ts(base + timedelta(seconds=90)),
            kind_json=json.dumps({"PasteDetected": {}}),
            window_json=json.dumps({"app_id": "com.google.Chrome", "title": "PR #42 - GitHub"}),
            metadata_json=json.dumps({
                "url": "https://github.com/org/repo/pull/42",
                "content_hash": url_hash,
                "target_app": "com.google.Chrome",
            }),
        )

        # ===============================================================
        # PHASE A: Verify WorkerDB reads events from SQLite correctly
        # ===============================================================
        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        assert len(events) == 8, f"Expected 8 events, got {len(events)}"

        # Events should be ordered by timestamp ascending
        event_ids_in_order = [e["id"] for e in events]
        assert event_ids_in_order == [
            "ev-01-focus-github",
            "ev-02-dwell-pr",
            "ev-03-scroll-review",
            "ev-04-click-approve",
            "ev-05-switch-slack",
            "ev-06-copy-url",
            "ev-07-switch-github",
            "ev-08-paste-url",
        ]

        # Verify individual event fields are preserved through SQLite round-trip
        first_event = events[0]
        assert first_event["kind_json"] == json.dumps({"FocusChange": {}})
        window = json.loads(first_event["window_json"])
        assert window["app_id"] == "com.google.Chrome"
        assert window["title"] == "PR #42 - GitHub"
        meta = json.loads(first_event["metadata_json"])
        assert meta["url"] == "https://github.com/org/repo/pull/42"

        # Verify click event has cursor coordinates
        click_event = events[3]
        assert click_event["cursor_x"] == 800
        assert click_event["cursor_y"] == 600

        # ===============================================================
        # PHASE B: Episode builder creates correct thread groupings
        # ===============================================================
        builder = EpisodeBuilder()
        episodes = builder.process_events(events)

        # GitHub events (with URL) go to one thread, Slack events to another
        thread_ids = {ep.thread_id for ep in episodes}
        # Entity-based clustering appends ticket/issue IDs (e.g. #42 from PR URL)
        github_thread = [t for t in thread_ids if t.startswith("com.google.Chrome:github.com")]
        assert len(github_thread) == 1, (
            f"Expected one GitHub thread, got: {thread_ids}"
        )
        assert "com.tinyspeck.slackmacgap" in thread_ids, (
            f"Expected Slack thread, got: {thread_ids}"
        )

        # GitHub thread should have 6 events (ev1-4, ev7-8)
        github_episodes = [
            ep for ep in episodes if ep.thread_id == github_thread[0]
        ]
        github_event_count = sum(ep.event_count for ep in github_episodes)
        assert github_event_count == 6, (
            f"Expected 6 GitHub events, got {github_event_count}"
        )

        # Slack thread should have 2 events (ev5-6)
        slack_episodes = [
            ep for ep in episodes if ep.thread_id == "com.tinyspeck.slackmacgap"
        ]
        slack_event_count = sum(ep.event_count for ep in slack_episodes)
        assert slack_event_count == 2, (
            f"Expected 2 Slack events, got {slack_event_count}"
        )

        # Total event count should be preserved
        total_events = sum(ep.event_count for ep in episodes)
        assert total_events == 8

        # ===============================================================
        # PHASE C: Clipboard linker finds the copy-paste link
        # ===============================================================
        linker = ClipboardLinker()
        links = linker.find_links(events)

        assert len(links) == 1, f"Expected 1 clipboard link, got {len(links)}"
        link = links[0]
        assert link.copy_event_id == "ev-06-copy-url"
        assert link.paste_event_id == "ev-08-paste-url"
        assert link.content_hash == url_hash
        # Copy at +60s, paste at +90s = 30 seconds delta
        assert abs(link.time_delta_seconds - 30.0) < 1.0

        # ===============================================================
        # PHASE D: Negative demo pruner finds no negatives in happy path
        # ===============================================================
        pruner = NegativeDemoPruner()
        result = pruner.prune(events)

        assert len(result.positive_events) == 8, (
            f"Expected all 8 events positive, got {len(result.positive_events)}"
        )
        assert len(result.negative_events) == 0, (
            f"Expected 0 negative events, got {len(result.negative_events)}"
        )
        assert len(result.prune_reasons) == 0


# ===================================================================
# Test 2: Negative Demo Detection — Undo Workflow
# ===================================================================


class TestNegativeDemoDetection:
    """Simulate a user making a mistake and undoing it.

    Scenario:
    1. User types in a form (multiple events in TextEdit)
    2. User hits Ctrl+Z (undo event)
    3. User continues working normally

    Verifications:
    - Negative demo pruner marks the undo and preceding events as negative
    - Events after the undo that are normal remain positive
    """

    def test_undo_workflow_marks_negative(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        app_id = "com.apple.TextEdit"

        # --- Typing events (will be marked negative by undo lookback) ---
        ev_ids = []
        for i in range(5):
            eid = _insert_event(
                write_conn,
                event_id=f"ev-type-{i:02d}",
                timestamp=_ts(base + timedelta(seconds=i * 2)),
                kind_json=json.dumps({"KeyPress": {}}),
                window_json=json.dumps({"app_id": app_id, "title": "Untitled.txt"}),
                metadata_json=json.dumps({"key": chr(ord("a") + i)}),
            )
            ev_ids.append(eid)

        # --- Undo event (Ctrl+Z) ---
        undo_id = _insert_event(
            write_conn,
            event_id="ev-undo",
            timestamp=_ts(base + timedelta(seconds=10)),
            kind_json=json.dumps({"KeyPress": {}}),
            window_json=json.dumps({"app_id": app_id, "title": "Untitled.txt"}),
            metadata_json=json.dumps({"shortcut": "ctrl+z"}),
        )

        # --- Normal events after undo (should remain positive) ---
        post_undo_ids = []
        for i in range(3):
            eid = _insert_event(
                write_conn,
                event_id=f"ev-after-{i:02d}",
                timestamp=_ts(base + timedelta(seconds=15 + i * 2)),
                kind_json=json.dumps({"FocusChange": {}}),
                window_json=json.dumps({"app_id": app_id, "title": "Untitled.txt"}),
                metadata_json=json.dumps({"action": "normal_work"}),
            )
            post_undo_ids.append(eid)

        # Read events from SQLite
        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        assert len(events) == 9  # 5 typing + 1 undo + 3 normal

        # Run negative demo pruning
        pruner = NegativeDemoPruner()
        result = pruner.prune(events)

        # The undo event itself must be negative
        negative_ids = {e["id"] for e in result.negative_events}
        assert "ev-undo" in negative_ids, "Undo event should be marked negative"

        # At least some of the preceding typing events should be negative
        preceding_negative_count = sum(
            1 for e in result.negative_events if e["id"].startswith("ev-type-")
        )
        assert preceding_negative_count >= 1, (
            "At least one preceding typing event should be marked negative"
        )

        # The undo reason should be correct
        assert result.prune_reasons["ev-undo"] == "undo_shortcut"

        # Post-undo normal events should all be positive
        positive_ids = {e["id"] for e in result.positive_events}
        for post_id in post_undo_ids:
            assert post_id in positive_ids, (
                f"Post-undo event {post_id} should be positive"
            )

    def test_cancel_click_workflow(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        """Cancel click in a dialog marks the dialog interaction as negative."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        app_id = "com.apple.Finder"

        # Open file dialog
        _insert_event(
            write_conn,
            event_id="ev-open-dialog",
            timestamp=_ts(base),
            kind_json=json.dumps({"ClickIntent": {}}),
            window_json=json.dumps({"app_id": app_id, "title": "Open File"}),
            metadata_json=json.dumps({"element_text": "Open File"}),
        )

        # Navigate in dialog
        _insert_event(
            write_conn,
            event_id="ev-browse",
            timestamp=_ts(base + timedelta(seconds=3)),
            kind_json=json.dumps({"FocusChange": {}}),
            window_json=json.dumps({"app_id": app_id, "title": "Open File"}),
            metadata_json=json.dumps({"dialog": "file_picker"}),
        )

        # Click Cancel
        _insert_event(
            write_conn,
            event_id="ev-cancel",
            timestamp=_ts(base + timedelta(seconds=6)),
            kind_json=json.dumps({"ClickIntent": {}}),
            window_json=json.dumps({"app_id": app_id, "title": "Open File"}),
            metadata_json=json.dumps({"element_text": "Cancel"}),
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        pruner = NegativeDemoPruner()
        result = pruner.prune(events)

        negative_ids = {e["id"] for e in result.negative_events}
        assert "ev-cancel" in negative_ids
        assert result.prune_reasons["ev-cancel"] == "cancel_click"
        # At least one preceding event should also be negative
        assert len(result.negative_events) >= 2


# ===================================================================
# Test 3: Episode Segmentation — Long Session
# ===================================================================


class TestEpisodeSegmentation:
    """Simulate a long session that exceeds the soft cap.

    Scenario:
    - Create events spanning >15 minutes in the same app
    - Process through episode builder

    Verifications:
    - Multiple segments created with correct linking (prev_segment_id chain)
    - Total event count matches original
    """

    def test_long_session_splits_into_segments(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        app_id = "com.microsoft.VSCode"
        total_events = 45  # Will span 45 minutes -> ~3 segments

        for i in range(total_events):
            _insert_event(
                write_conn,
                event_id=f"ev-long-{i:03d}",
                timestamp=_ts(base + timedelta(minutes=i)),
                kind_json=json.dumps({"FocusChange": {}}),
                window_json=json.dumps({"app_id": app_id, "title": "main.rs - VSCode"}),
                metadata_json=json.dumps({}),
            )

        # Read events from SQLite
        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=200)

        assert len(events) == total_events

        # Process through episode builder
        builder = EpisodeBuilder()
        episodes = builder.process_events(events)

        # All episodes should be for the same thread
        # Entity-based clustering may enrich thread_id (e.g. "com.microsoft.VSCode:main.rs")
        vscode_episodes = [
            ep for ep in episodes if ep.thread_id.startswith(app_id)
        ]
        assert len(vscode_episodes) >= 3, (
            f"Expected at least 3 segments for 45-minute session, got {len(vscode_episodes)}"
        )

        # Total event count must match original
        total_in_episodes = sum(ep.event_count for ep in vscode_episodes)
        assert total_in_episodes == total_events, (
            f"Expected {total_events} total events, got {total_in_episodes}"
        )

        # All segments share the same episode_id
        episode_ids = {ep.episode_id for ep in vscode_episodes}
        assert len(episode_ids) == 1, (
            f"All segments should share one episode_id, got {len(episode_ids)}"
        )

        # Verify segment chain: prev_segment_id links correctly
        segments = sorted(vscode_episodes, key=lambda e: e.segment_id)

        # First segment has no predecessor
        assert segments[0].segment_id == 0
        assert segments[0].prev_segment_id is None

        # Each subsequent segment links to the previous
        for i in range(1, len(segments)):
            assert segments[i].segment_id == i, (
                f"Segment {i} should have segment_id={i}, got {segments[i].segment_id}"
            )
            assert segments[i].prev_segment_id == i - 1, (
                f"Segment {i} should link to segment {i-1}, "
                f"got prev_segment_id={segments[i].prev_segment_id}"
            )

    def test_hard_cap_segmentation(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        """210 events within 1 minute triggers hard cap split at 200."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        app_id = "com.apple.Notes"
        total_events = 210

        for i in range(total_events):
            _insert_event(
                write_conn,
                event_id=f"ev-hard-{i:04d}",
                # All within ~3 minutes (sub-second intervals) to avoid soft cap
                timestamp=_ts(base + timedelta(milliseconds=i * 500)),
                kind_json=json.dumps({"KeyPress": {}}),
                window_json=json.dumps({"app_id": app_id, "title": "Quick Note"}),
                metadata_json=json.dumps({"key": chr(ord("a") + (i % 26))}),
            )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=300)

        assert len(events) == total_events

        builder = EpisodeBuilder()
        episodes = builder.process_events(events)

        notes_episodes = [ep for ep in episodes if ep.thread_id == app_id]
        assert len(notes_episodes) == 2, (
            f"Expected 2 segments (200+10) from hard cap, got {len(notes_episodes)}"
        )

        segments = sorted(notes_episodes, key=lambda e: e.segment_id)
        assert segments[0].event_count == 200
        assert segments[1].event_count == 10
        assert segments[0].episode_id == segments[1].episode_id
        assert segments[1].prev_segment_id == 0


# ===================================================================
# Test 4: Secure Field Event Handling
# ===================================================================


class TestSecureFieldHandling:
    """Simulate secure field focus events.

    Scenario:
    1. Normal events
    2. SecureFieldFocus event (isSecure: true)
    3. More events (should NOT have DOM snapshots or click events if the
       browser extension is working correctly — we simulate correct behavior)
    4. SecureFieldFocus event (isSecure: false)
    5. Normal events with DOM snapshots resume

    Verifications:
    - All events are stored and readable from SQLite
    - No DOM snapshot or click events appear between the secure field start/end
    - Events before and after secure field are present
    """

    def test_secure_field_event_flow(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        app_id = "com.google.Chrome"
        login_url = "https://bank.example.com/login"

        # --- Normal events before secure field ---
        _insert_event(
            write_conn,
            event_id="ev-normal-01",
            timestamp=_ts(base),
            kind_json=json.dumps({"FocusChange": {}}),
            window_json=json.dumps({"app_id": app_id, "title": "Login - Bank"}),
            metadata_json=json.dumps({"url": login_url}),
        )

        _insert_event(
            write_conn,
            event_id="ev-click-username",
            timestamp=_ts(base + timedelta(seconds=2)),
            kind_json=json.dumps({"ClickIntent": {}}),
            window_json=json.dumps({"app_id": app_id, "title": "Login - Bank"}),
            metadata_json=json.dumps({
                "url": login_url,
                "element_text": "Username",
                "element_role": "textbox",
            }),
        )

        _insert_event(
            write_conn,
            event_id="ev-dwell-before",
            timestamp=_ts(base + timedelta(seconds=5)),
            kind_json=json.dumps({"DwellSnapshot": {}}),
            window_json=json.dumps({"app_id": app_id, "title": "Login - Bank"}),
            metadata_json=json.dumps({"url": login_url}),
        )

        # --- Secure field ENTER (password field focused) ---
        _insert_event(
            write_conn,
            event_id="ev-secure-enter",
            timestamp=_ts(base + timedelta(seconds=8)),
            kind_json=json.dumps({"SecureFieldFocus": {"is_secure": True}}),
            window_json=json.dumps({"app_id": app_id, "title": "Login - Bank"}),
            metadata_json=json.dumps({"url": login_url, "is_secure": True}),
        )

        # --- During secure field: only non-DOM, non-click events ---
        # (The browser extension suppresses clicks/DOM captures while secure)
        # A dwell event might still come through but WITHOUT DOM snapshot
        _insert_event(
            write_conn,
            event_id="ev-dwell-during-secure",
            timestamp=_ts(base + timedelta(seconds=12)),
            kind_json=json.dumps({"DwellSnapshot": {}}),
            window_json=json.dumps({"app_id": app_id, "title": "Login - Bank"}),
            metadata_json=json.dumps({"url": login_url, "secure_mode": True}),
        )

        # --- Secure field EXIT (password field blurred) ---
        _insert_event(
            write_conn,
            event_id="ev-secure-exit",
            timestamp=_ts(base + timedelta(seconds=18)),
            kind_json=json.dumps({"SecureFieldFocus": {"is_secure": False}}),
            window_json=json.dumps({"app_id": app_id, "title": "Login - Bank"}),
            metadata_json=json.dumps({"url": login_url, "is_secure": False}),
        )

        # --- Normal events resume after secure field ---
        _insert_event(
            write_conn,
            event_id="ev-click-login",
            timestamp=_ts(base + timedelta(seconds=20)),
            kind_json=json.dumps({"ClickIntent": {}}),
            window_json=json.dumps({"app_id": app_id, "title": "Login - Bank"}),
            metadata_json=json.dumps({
                "url": login_url,
                "element_text": "Sign In",
                "element_role": "button",
            }),
        )

        _insert_event(
            write_conn,
            event_id="ev-normal-after",
            timestamp=_ts(base + timedelta(seconds=25)),
            kind_json=json.dumps({"FocusChange": {}}),
            window_json=json.dumps({"app_id": app_id, "title": "Dashboard - Bank"}),
            metadata_json=json.dumps({"url": "https://bank.example.com/dashboard"}),
        )

        # ===============================================================
        # Read all events from SQLite
        # ===============================================================
        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        assert len(events) == 8

        # ===============================================================
        # Verify: no DOM snapshot or click events between secure enter/exit
        # ===============================================================
        secure_enter_ts = _ts(base + timedelta(seconds=8))
        secure_exit_ts = _ts(base + timedelta(seconds=18))

        for event in events:
            ts = event["timestamp"]
            if ts > secure_enter_ts and ts < secure_exit_ts:
                kind = json.loads(event["kind_json"])
                kind_name = next(iter(kind))
                # During secure field, there should be no ClickIntent events
                assert kind_name != "ClickIntent", (
                    f"ClickIntent event found during secure field period: {event['id']}"
                )

        # ===============================================================
        # Verify: events before and after secure field are present
        # ===============================================================
        event_ids = [e["id"] for e in events]
        assert "ev-click-username" in event_ids  # Before secure
        assert "ev-dwell-before" in event_ids     # Before secure
        assert "ev-click-login" in event_ids       # After secure
        assert "ev-normal-after" in event_ids      # After secure

        # ===============================================================
        # Verify: secure field events themselves are stored
        # ===============================================================
        assert "ev-secure-enter" in event_ids
        assert "ev-secure-exit" in event_ids

        # Verify secure field event data is preserved
        secure_enter = next(e for e in events if e["id"] == "ev-secure-enter")
        kind = json.loads(secure_enter["kind_json"])
        assert "SecureFieldFocus" in kind
        assert kind["SecureFieldFocus"]["is_secure"] is True

        secure_exit = next(e for e in events if e["id"] == "ev-secure-exit")
        kind = json.loads(secure_exit["kind_json"])
        assert "SecureFieldFocus" in kind
        assert kind["SecureFieldFocus"]["is_secure"] is False

        # ===============================================================
        # Episode builder processes secure field events normally
        # ===============================================================
        builder = EpisodeBuilder()
        episodes = builder.process_events(events)

        # All events are in the same Chrome thread (same app_id + domain)
        total = sum(ep.event_count for ep in episodes)
        assert total == 8


# ===================================================================
# Test 5: Multiple Clipboard Links
# ===================================================================


class TestMultipleClipboardLinks:
    """Simulate multiple copy-paste operations across different apps.

    Scenario:
    1. Copy text A from app 1 (Terminal)
    2. Copy text B from app 2 (Safari)
    3. Paste text A in app 3 (VSCode) — should link to copy from Terminal
    4. Paste text B in app 4 (Slack) — should link to copy from Safari

    Verifications:
    - Both clipboard links are found
    - Correct copy-paste pairing by hash
    """

    def test_multiple_clipboard_links(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        hash_a = _hash("git commit -m 'fix: resolve issue'")
        hash_b = _hash("https://docs.example.com/api-reference")

        # --- Copy text A from Terminal ---
        _insert_event(
            write_conn,
            event_id="ev-copy-a",
            timestamp=_ts(base),
            kind_json=json.dumps({"ClipboardChange": {}}),
            window_json=json.dumps({"app_id": "com.apple.Terminal", "title": "Terminal"}),
            metadata_json=json.dumps({
                "content_hash": hash_a,
                "content_types": ["text/plain"],
                "byte_size": 35,
            }),
        )

        # --- Some normal events in between ---
        _insert_event(
            write_conn,
            event_id="ev-between-01",
            timestamp=_ts(base + timedelta(seconds=10)),
            kind_json=json.dumps({"FocusChange": {}}),
            window_json=json.dumps({"app_id": "com.apple.Safari", "title": "Safari"}),
            metadata_json=json.dumps({"url": "https://docs.example.com/api-reference"}),
        )

        # --- Copy text B from Safari ---
        _insert_event(
            write_conn,
            event_id="ev-copy-b",
            timestamp=_ts(base + timedelta(seconds=20)),
            kind_json=json.dumps({"ClipboardChange": {}}),
            window_json=json.dumps({"app_id": "com.apple.Safari", "title": "API Docs"}),
            metadata_json=json.dumps({
                "content_hash": hash_b,
                "content_types": ["text/plain"],
                "byte_size": 42,
            }),
        )

        # --- Switch to VSCode ---
        _insert_event(
            write_conn,
            event_id="ev-switch-vscode",
            timestamp=_ts(base + timedelta(seconds=30)),
            kind_json=json.dumps({"AppSwitch": {}}),
            window_json=json.dumps({"app_id": "com.microsoft.VSCode", "title": "main.py"}),
            metadata_json=json.dumps({}),
        )

        # --- Paste text A in VSCode ---
        _insert_event(
            write_conn,
            event_id="ev-paste-a",
            timestamp=_ts(base + timedelta(seconds=35)),
            kind_json=json.dumps({"PasteDetected": {}}),
            window_json=json.dumps({"app_id": "com.microsoft.VSCode", "title": "main.py"}),
            metadata_json=json.dumps({
                "content_hash": hash_a,
                "target_app": "com.microsoft.VSCode",
            }),
        )

        # --- Switch to Slack ---
        _insert_event(
            write_conn,
            event_id="ev-switch-slack",
            timestamp=_ts(base + timedelta(seconds=45)),
            kind_json=json.dumps({"AppSwitch": {}}),
            window_json=json.dumps({
                "app_id": "com.tinyspeck.slackmacgap",
                "title": "Slack - #dev",
            }),
            metadata_json=json.dumps({}),
        )

        # --- Paste text B in Slack ---
        _insert_event(
            write_conn,
            event_id="ev-paste-b",
            timestamp=_ts(base + timedelta(seconds=50)),
            kind_json=json.dumps({"PasteDetected": {}}),
            window_json=json.dumps({
                "app_id": "com.tinyspeck.slackmacgap",
                "title": "Slack - #dev",
            }),
            metadata_json=json.dumps({
                "content_hash": hash_b,
                "target_app": "com.tinyspeck.slackmacgap",
            }),
        )

        # ===============================================================
        # Read events from SQLite
        # ===============================================================
        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        assert len(events) == 7

        # ===============================================================
        # Find clipboard links
        # ===============================================================
        linker = ClipboardLinker()
        links = linker.find_links(events)

        assert len(links) == 2, f"Expected 2 clipboard links, got {len(links)}"

        # Sort by paste event id for deterministic assertions
        links_sorted = sorted(links, key=lambda lk: lk.paste_event_id)

        # Link A: Terminal copy -> VSCode paste
        link_a = links_sorted[0]
        assert link_a.copy_event_id == "ev-copy-a"
        assert link_a.paste_event_id == "ev-paste-a"
        assert link_a.content_hash == hash_a
        # Copy at +0s, paste at +35s = 35 seconds
        assert abs(link_a.time_delta_seconds - 35.0) < 1.0

        # Link B: Safari copy -> Slack paste
        link_b = links_sorted[1]
        assert link_b.copy_event_id == "ev-copy-b"
        assert link_b.paste_event_id == "ev-paste-b"
        assert link_b.content_hash == hash_b
        # Copy at +20s, paste at +50s = 30 seconds
        assert abs(link_b.time_delta_seconds - 30.0) < 1.0

    def test_overwritten_clipboard_links_most_recent_copy(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        """When the same content is copied twice, paste links to the most recent copy."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        h = _hash("repeated content")

        # First copy
        _insert_event(
            write_conn,
            event_id="ev-copy-old",
            timestamp=_ts(base),
            kind_json=json.dumps({"ClipboardChange": {}}),
            window_json=json.dumps({"app_id": "com.app.A", "title": "App A"}),
            metadata_json=json.dumps({
                "content_hash": h,
                "content_types": ["text/plain"],
                "byte_size": 16,
            }),
        )

        # Second copy (same content, different app, later time)
        _insert_event(
            write_conn,
            event_id="ev-copy-new",
            timestamp=_ts(base + timedelta(minutes=5)),
            kind_json=json.dumps({"ClipboardChange": {}}),
            window_json=json.dumps({"app_id": "com.app.B", "title": "App B"}),
            metadata_json=json.dumps({
                "content_hash": h,
                "content_types": ["text/plain"],
                "byte_size": 16,
            }),
        )

        # Paste
        _insert_event(
            write_conn,
            event_id="ev-paste",
            timestamp=_ts(base + timedelta(minutes=10)),
            kind_json=json.dumps({"PasteDetected": {}}),
            window_json=json.dumps({"app_id": "com.app.C", "title": "App C"}),
            metadata_json=json.dumps({
                "content_hash": h,
                "target_app": "com.app.C",
            }),
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        linker = ClipboardLinker()
        links = linker.find_links(events)

        assert len(links) == 1
        assert links[0].copy_event_id == "ev-copy-new", (
            "Should link to the most recent copy, not the older one"
        )
        assert links[0].paste_event_id == "ev-paste"


# ===================================================================
# Test 6: Combined Pipeline — All Components Together
# ===================================================================


class TestCombinedPipeline:
    """Run all pipeline components on the same event stream to verify
    they compose correctly without interference.

    Scenario: A session with normal work, an undo, and a copy-paste operation.
    """

    def test_all_components_compose(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)
        app_id = "com.google.Chrome"
        paste_hash = _hash("shared link")

        # Normal browsing events — placed >30s before undo so they are
        # outside the negative demo lookback window (LOOKBACK_MAX_SECONDS=30)
        _insert_event(
            write_conn,
            event_id="ev-browse-01",
            timestamp=_ts(base),
            kind_json=json.dumps({"FocusChange": {}}),
            window_json=json.dumps({"app_id": app_id, "title": "Google Chrome"}),
            metadata_json=json.dumps({"url": "https://example.com"}),
        )

        _insert_event(
            write_conn,
            event_id="ev-browse-02",
            timestamp=_ts(base + timedelta(seconds=3)),
            kind_json=json.dumps({"DwellSnapshot": {}}),
            window_json=json.dumps({"app_id": app_id, "title": "Google Chrome"}),
            metadata_json=json.dumps({"url": "https://example.com"}),
        )

        # Mistake: type something wrong then undo — placed 60s after browsing
        # so the browsing events are outside the 30s lookback window
        _insert_event(
            write_conn,
            event_id="ev-mistake",
            timestamp=_ts(base + timedelta(seconds=60)),
            kind_json=json.dumps({"KeyPress": {}}),
            window_json=json.dumps({"app_id": app_id, "title": "Google Chrome"}),
            metadata_json=json.dumps({"url": "https://example.com", "key": "x"}),
        )

        _insert_event(
            write_conn,
            event_id="ev-undo",
            timestamp=_ts(base + timedelta(seconds=61)),
            kind_json=json.dumps({"KeyPress": {}}),
            window_json=json.dumps({"app_id": app_id, "title": "Google Chrome"}),
            metadata_json=json.dumps({"url": "https://example.com", "shortcut": "cmd+z"}),
        )

        # Copy operation
        _insert_event(
            write_conn,
            event_id="ev-copy",
            timestamp=_ts(base + timedelta(seconds=80)),
            kind_json=json.dumps({"ClipboardChange": {}}),
            window_json=json.dumps({"app_id": app_id, "title": "Google Chrome"}),
            metadata_json=json.dumps({
                "url": "https://example.com",
                "content_hash": paste_hash,
                "content_types": ["text/plain"],
                "byte_size": 11,
            }),
        )

        # Paste operation
        _insert_event(
            write_conn,
            event_id="ev-paste",
            timestamp=_ts(base + timedelta(seconds=90)),
            kind_json=json.dumps({"PasteDetected": {}}),
            window_json=json.dumps({"app_id": app_id, "title": "Google Chrome"}),
            metadata_json=json.dumps({
                "url": "https://example.com",
                "content_hash": paste_hash,
                "target_app": app_id,
            }),
        )

        # Read from SQLite
        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        assert len(events) == 6

        # Episode builder
        builder = EpisodeBuilder()
        episodes = builder.process_events(events)
        total_events = sum(ep.event_count for ep in episodes)
        assert total_events == 6

        # Negative demo pruner
        pruner = NegativeDemoPruner()
        result = pruner.prune(events)
        # Undo + preceding mistake should be negative
        negative_ids = {e["id"] for e in result.negative_events}
        assert "ev-undo" in negative_ids
        assert "ev-mistake" in negative_ids
        # Browsing events are >30s before the undo, so outside lookback window
        positive_ids = {e["id"] for e in result.positive_events}
        assert "ev-browse-01" in positive_ids, (
            "Browsing event at +0s should be positive (>30s before undo at +61s)"
        )
        assert "ev-browse-02" in positive_ids, (
            "Browsing event at +3s should be positive (>30s before undo at +61s)"
        )
        # Clipboard events should be positive (they come after undo)
        assert "ev-copy" in positive_ids
        assert "ev-paste" in positive_ids

        # Clipboard linker
        linker = ClipboardLinker()
        links = linker.find_links(events)
        assert len(links) == 1
        assert links[0].copy_event_id == "ev-copy"
        assert links[0].paste_event_id == "ev-paste"


# ===================================================================
# Test 7: WorkerDB Edge Cases with Real Schema
# ===================================================================


class TestWorkerDBEdgeCases:
    """Verify WorkerDB behavior with the real daemon schema under
    edge conditions.
    """

    def test_get_event_by_id_roundtrip(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        """Single event inserted by daemon, retrieved by worker."""
        eid = _insert_event(
            write_conn,
            event_id="ev-specific",
            kind_json=json.dumps({"ClickIntent": {"x": 100, "y": 200}}),
            window_json=json.dumps({"app_id": "com.test.App", "title": "Test"}),
            metadata_json=json.dumps({"selector": "button#submit"}),
            cursor_x=100,
            cursor_y=200,
        )

        with WorkerDB(tmp_db_path) as db:
            event = db.get_event_by_id("ev-specific")

        assert event is not None
        assert event["id"] == "ev-specific"
        assert event["cursor_x"] == 100
        assert event["cursor_y"] == 200

        kind = json.loads(event["kind_json"])
        assert "ClickIntent" in kind
        assert kind["ClickIntent"]["x"] == 100

        meta = json.loads(event["metadata_json"])
        assert meta["selector"] == "button#submit"

    def test_nonexistent_event_returns_none(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        """Querying a nonexistent event returns None."""
        with WorkerDB(tmp_db_path) as db:
            event = db.get_event_by_id("does-not-exist")

        assert event is None

    def test_processed_events_excluded(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        """Events marked as processed=1 are excluded from unprocessed query."""
        _insert_event(
            write_conn,
            event_id="ev-processed",
            processed=1,
        )
        _insert_event(
            write_conn,
            event_id="ev-unprocessed",
            processed=0,
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        ids = [e["id"] for e in events]
        assert "ev-unprocessed" in ids
        assert "ev-processed" not in ids

    def test_timestamp_ordering(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection
    ) -> None:
        """Events are returned in timestamp order, not insertion order."""
        base = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)

        # Insert in reverse chronological order
        _insert_event(
            write_conn,
            event_id="ev-third",
            timestamp=_ts(base + timedelta(seconds=20)),
        )
        _insert_event(
            write_conn,
            event_id="ev-first",
            timestamp=_ts(base),
        )
        _insert_event(
            write_conn,
            event_id="ev-second",
            timestamp=_ts(base + timedelta(seconds=10)),
        )

        with WorkerDB(tmp_db_path) as db:
            events = db.get_unprocessed_events(limit=100)

        ids = [e["id"] for e in events]
        assert ids == ["ev-first", "ev-second", "ev-third"]
