"""Tests for oc_apprentice_worker.activity_search — Activity Search + Session Recall.

Each test gets a temporary SQLite database (created by conftest fixtures)
pre-initialised with the daemon's schema.  We insert events with
scene_annotation_json via the write connection, then verify that
ActivitySearcher indexes and queries them correctly.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from oc_apprentice_worker.activity_search import (
    ActivitySearcher,
    ActivityTimeline,
    SearchResult,
    _calculate_active_minutes,
    _escape_fts5_query,
    _iso_to_epoch,
    _parse_annotation,
)
from conftest import insert_event


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_annotation(
    what_doing: str = "Browsing the web",
    app: str = "Google Chrome",
    location: str = "https://example.com",
    is_workflow: bool = True,
) -> str:
    """Build a scene_annotation_json string."""
    return json.dumps({
        "task_context": {
            "what_doing": what_doing,
            "likely_next": "continue",
            "is_workflow": is_workflow,
        },
        "visual_context": {
            "active_app": app,
            "location": location,
        },
    })


def _set_annotation(
    conn: sqlite3.Connection,
    event_id: str,
    annotation_json: str,
    status: str = "completed",
) -> None:
    """Set the scene_annotation_json and annotation_status on an event."""
    conn.execute(
        "UPDATE events SET scene_annotation_json = ?, annotation_status = ? "
        "WHERE id = ?",
        (annotation_json, status, event_id),
    )
    conn.commit()


def _insert_annotated_event(
    conn: sqlite3.Connection,
    *,
    timestamp: str | None = None,
    what_doing: str = "Browsing the web",
    app: str = "Google Chrome",
    location: str = "https://example.com",
    is_workflow: bool = True,
    event_id: str | None = None,
) -> str:
    """Insert an event and set its annotation in one call."""
    eid = insert_event(conn, timestamp=timestamp, event_id=event_id)
    ann = _make_annotation(
        what_doing=what_doing, app=app, location=location, is_workflow=is_workflow,
    )
    _set_annotation(conn, eid, ann)
    return eid


# ---------------------------------------------------------------------------
# 1. FTS5 query escaping
# ---------------------------------------------------------------------------


class TestFTS5QueryEscaping:
    """Tests for the _escape_fts5_query helper."""

    def test_simple_term(self) -> None:
        assert _escape_fts5_query("domains") == '"domains"'

    def test_multiple_terms(self) -> None:
        result = _escape_fts5_query("expired domains")
        assert result == '"expired" "domains"'

    def test_empty_query(self) -> None:
        assert _escape_fts5_query("") == ""

    def test_whitespace_only(self) -> None:
        assert _escape_fts5_query("   ") == ""

    def test_special_characters_stripped(self) -> None:
        # Double quotes are stripped to prevent FTS5 injection.
        result = _escape_fts5_query('test"injection')
        assert '"' not in result.replace('"test', "").replace('injection"', "")
        assert "testinjection" in result

    def test_operators_quoted(self) -> None:
        # FTS5 operators like AND, OR, NOT should be wrapped in quotes.
        result = _escape_fts5_query("NOT domains AND expired")
        assert result == '"NOT" "domains" "AND" "expired"'

    def test_asterisk_in_term(self) -> None:
        result = _escape_fts5_query("domain*")
        # The asterisk is kept inside quotes where it is literal.
        assert '"domain*"' == result

    def test_parentheses_in_term(self) -> None:
        result = _escape_fts5_query("func(x)")
        assert '"func(x)"' == result


# ---------------------------------------------------------------------------
# 2. Annotation parsing
# ---------------------------------------------------------------------------


class TestAnnotationParsing:
    """Tests for the _parse_annotation helper."""

    def test_nested_format(self) -> None:
        ann = json.dumps({
            "task_context": {"what_doing": "Reading email"},
            "visual_context": {"active_app": "Mail", "location": "inbox"},
        })
        result = _parse_annotation(ann)
        assert result["what_doing"] == "Reading email"
        assert result["app"] == "Mail"
        assert result["location"] == "inbox"

    def test_flat_format(self) -> None:
        ann = json.dumps({
            "task_context": {"what_doing": "Writing code"},
            "app": "VS Code",
            "location": "/src/main.py",
        })
        result = _parse_annotation(ann)
        assert result["what_doing"] == "Writing code"
        assert result["app"] == "VS Code"
        assert result["location"] == "/src/main.py"

    def test_none_input(self) -> None:
        result = _parse_annotation(None)
        assert result == {"what_doing": "", "app": "", "location": ""}

    def test_invalid_json(self) -> None:
        result = _parse_annotation("not-json")
        assert result == {"what_doing": "", "app": "", "location": ""}

    def test_empty_string(self) -> None:
        result = _parse_annotation("")
        assert result == {"what_doing": "", "app": "", "location": ""}

    def test_missing_fields(self) -> None:
        ann = json.dumps({"task_context": {}})
        result = _parse_annotation(ann)
        assert result["what_doing"] == ""
        assert result["app"] == ""

    def test_nested_preferred_over_flat(self) -> None:
        """visual_context.active_app takes precedence over top-level app."""
        ann = json.dumps({
            "task_context": {"what_doing": "test"},
            "visual_context": {"active_app": "Nested App"},
            "app": "Flat App",
        })
        result = _parse_annotation(ann)
        assert result["app"] == "Nested App"


# ---------------------------------------------------------------------------
# 3. ISO to epoch conversion
# ---------------------------------------------------------------------------


class TestISOToEpoch:
    """Tests for _iso_to_epoch."""

    def test_z_suffix(self) -> None:
        epoch = _iso_to_epoch("2026-03-10T12:00:00Z")
        assert epoch > 0

    def test_offset_suffix(self) -> None:
        epoch = _iso_to_epoch("2026-03-10T12:00:00+00:00")
        assert epoch > 0

    def test_empty_string(self) -> None:
        assert _iso_to_epoch("") == 0.0

    def test_invalid_string(self) -> None:
        assert _iso_to_epoch("not-a-date") == 0.0

    def test_none_input(self) -> None:
        assert _iso_to_epoch(None) == 0.0  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. Active minutes calculation
# ---------------------------------------------------------------------------


class TestActiveMinutes:
    """Tests for _calculate_active_minutes."""

    def test_no_timestamps(self) -> None:
        assert _calculate_active_minutes([]) == 0

    def test_single_timestamp(self) -> None:
        assert _calculate_active_minutes(["2026-03-10T12:00:00Z"]) == 0

    def test_two_close_events(self) -> None:
        # 3 minutes apart → 3 active minutes.
        ts = [
            "2026-03-10T12:00:00Z",
            "2026-03-10T12:03:00Z",
        ]
        assert _calculate_active_minutes(ts) == 3

    def test_gap_exceeding_threshold(self) -> None:
        # 10 minutes apart → exceeds 5-minute threshold → 0 active.
        ts = [
            "2026-03-10T12:00:00Z",
            "2026-03-10T12:10:00Z",
        ]
        assert _calculate_active_minutes(ts) == 0

    def test_mixed_active_and_idle(self) -> None:
        # 3 active minutes, then 10-min gap, then 2 active minutes.
        ts = [
            "2026-03-10T12:00:00Z",
            "2026-03-10T12:03:00Z",  # +3 min (active)
            "2026-03-10T12:13:00Z",  # +10 min (idle)
            "2026-03-10T12:15:00Z",  # +2 min (active)
        ]
        assert _calculate_active_minutes(ts) == 5

    def test_exactly_at_threshold(self) -> None:
        # Exactly 5 minutes = threshold → should be counted.
        ts = [
            "2026-03-10T12:00:00Z",
            "2026-03-10T12:05:00Z",
        ]
        assert _calculate_active_minutes(ts) == 5


# ---------------------------------------------------------------------------
# 5. ActivitySearcher — constructor / lifecycle
# ---------------------------------------------------------------------------


class TestSearcherLifecycle:
    """Tests for ActivitySearcher open/close behaviour."""

    def test_opens_existing_database(self, tmp_db_path: Path) -> None:
        searcher = ActivitySearcher(tmp_db_path)
        searcher.close()

    def test_rejects_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ActivitySearcher(tmp_path / "nonexistent.db")

    def test_context_manager(self, tmp_db_path: Path) -> None:
        with ActivitySearcher(tmp_db_path) as s:
            assert s is not None

    def test_double_close_is_safe(self, tmp_db_path: Path) -> None:
        s = ActivitySearcher(tmp_db_path)
        s.close()
        s.close()  # Should not raise.


# ---------------------------------------------------------------------------
# 6. FTS5 index creation (lazy)
# ---------------------------------------------------------------------------


class TestFTSIndexCreation:
    """The FTS5 index is created lazily on first use."""

    def test_fts_table_created_on_search(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(write_conn, what_doing="Test activity")

        with ActivitySearcher(tmp_db_path) as s:
            # Before search, _fts_ready is False.
            assert not s._fts_ready
            s.search("Test")
            assert s._fts_ready

        # Verify the table exists.
        cur = write_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='events_fts'"
        )
        assert cur.fetchone() is not None

    def test_fts_table_created_on_session_recall(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T12:00:00.000000Z",
            what_doing="Some work",
        )

        with ActivitySearcher(tmp_db_path) as s:
            s.session_recall(date="2026-03-10")
            assert s._fts_ready

    def test_fts_table_not_created_before_first_use(
        self, tmp_db_path: Path,
    ) -> None:
        with ActivitySearcher(tmp_db_path) as s:
            assert not s._fts_ready

        # Table should not exist.
        conn = sqlite3.connect(str(tmp_db_path))
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='events_fts'"
        )
        assert cur.fetchone() is None
        conn.close()


# ---------------------------------------------------------------------------
# 7. FTS5 index population
# ---------------------------------------------------------------------------


class TestFTSIndexPopulation:
    """The FTS5 index is populated from existing annotations."""

    def test_populates_from_existing_annotations(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(write_conn, what_doing="Checking email")
        _insert_annotated_event(write_conn, what_doing="Writing code")

        with ActivitySearcher(tmp_db_path) as s:
            s._ensure_fts_index()

            cur = s._conn.execute("SELECT COUNT(*) AS cnt FROM events_fts")
            assert cur.fetchone()["cnt"] == 2

    def test_skips_unannotated_events(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        # One annotated, one not.
        _insert_annotated_event(write_conn, what_doing="Annotated event")
        insert_event(write_conn)  # No annotation.

        with ActivitySearcher(tmp_db_path) as s:
            s._ensure_fts_index()
            cur = s._conn.execute("SELECT COUNT(*) AS cnt FROM events_fts")
            assert cur.fetchone()["cnt"] == 1

    def test_idempotent_population(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(write_conn, what_doing="Test event")

        with ActivitySearcher(tmp_db_path) as s:
            s._ensure_fts_index()
            # Call again — should not duplicate.
            s._fts_ready = False
            s._ensure_fts_index()

            cur = s._conn.execute("SELECT COUNT(*) AS cnt FROM events_fts")
            assert cur.fetchone()["cnt"] == 1

    def test_refresh_adds_new_events(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(write_conn, what_doing="First event")

        with ActivitySearcher(tmp_db_path) as s:
            s._ensure_fts_index()

            # Add a new event after initial population.
            _insert_annotated_event(write_conn, what_doing="Second event")
            added = s.refresh_index()

            assert added == 1
            cur = s._conn.execute("SELECT COUNT(*) AS cnt FROM events_fts")
            assert cur.fetchone()["cnt"] == 2


# ---------------------------------------------------------------------------
# 8. search() — basic matching
# ---------------------------------------------------------------------------


class TestSearchBasic:
    """Basic full-text search over annotations."""

    def test_search_by_keyword(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(
            write_conn,
            what_doing="Searching for expired domains on GoDaddy",
            app="Google Chrome",
            location="https://auctions.godaddy.com",
        )
        _insert_annotated_event(
            write_conn,
            what_doing="Writing Python code in VS Code",
            app="VS Code",
            location="/src/main.py",
        )

        with ActivitySearcher(tmp_db_path) as s:
            results = s.search("expired domains")

        assert len(results) == 1
        assert results[0].what_doing == "Searching for expired domains on GoDaddy"
        assert results[0].app == "Google Chrome"

    def test_search_matches_app_field(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(write_conn, app="Slack", what_doing="Chatting")
        _insert_annotated_event(write_conn, app="Mail", what_doing="Reading")

        with ActivitySearcher(tmp_db_path) as s:
            results = s.search("Slack")

        assert len(results) == 1
        assert results[0].app == "Slack"

    def test_search_matches_location(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(
            write_conn,
            location="https://github.com/openmimic",
            what_doing="Reviewing PRs",
        )

        with ActivitySearcher(tmp_db_path) as s:
            results = s.search("github")

        assert len(results) == 1
        assert "github" in results[0].location

    def test_search_empty_query_returns_empty(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(write_conn, what_doing="Something")

        with ActivitySearcher(tmp_db_path) as s:
            results = s.search("")

        assert results == []

    def test_search_no_matches(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(write_conn, what_doing="Writing code")

        with ActivitySearcher(tmp_db_path) as s:
            results = s.search("elephant")

        assert results == []

    def test_search_empty_database(self, tmp_db_path: Path) -> None:
        with ActivitySearcher(tmp_db_path) as s:
            results = s.search("anything")

        assert results == []


# ---------------------------------------------------------------------------
# 9. search() — filters
# ---------------------------------------------------------------------------


class TestSearchFilters:
    """Search with date, app, time_range filters."""

    def test_filter_by_date(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:00:00.000000Z",
            what_doing="Morning work on domains",
        )
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-11T10:00:00.000000Z",
            what_doing="Next day work on domains",
        )

        with ActivitySearcher(tmp_db_path) as s:
            results = s.search("domains", date="2026-03-10")

        assert len(results) == 1
        assert "2026-03-10" in results[0].timestamp

    def test_filter_by_app(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(
            write_conn, what_doing="Coding in editor", app="VS Code",
        )
        _insert_annotated_event(
            write_conn, what_doing="Coding in terminal", app="Terminal",
        )

        with ActivitySearcher(tmp_db_path) as s:
            results = s.search("Coding", app="VS Code")

        assert len(results) == 1
        assert results[0].app == "VS Code"

    def test_filter_by_time_range(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T08:00:00.000000Z",
            what_doing="Early morning coding",
        )
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T14:00:00.000000Z",
            what_doing="Afternoon coding",
        )

        with ActivitySearcher(tmp_db_path) as s:
            results = s.search(
                "coding",
                time_range=("2026-03-10T07:00:00Z", "2026-03-10T10:00:00Z"),
            )

        assert len(results) == 1
        assert "08:00" in results[0].timestamp

    def test_combined_filters(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:00:00.000000Z",
            what_doing="Reviewing pull requests",
            app="Google Chrome",
        )
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:00:00.000000Z",
            what_doing="Reviewing code locally",
            app="VS Code",
        )

        with ActivitySearcher(tmp_db_path) as s:
            results = s.search(
                "Reviewing", date="2026-03-10", app="Chrome",
            )

        assert len(results) == 1
        assert results[0].app == "Google Chrome"


# ---------------------------------------------------------------------------
# 10. search() — limit and relevance
# ---------------------------------------------------------------------------


class TestSearchLimitAndRelevance:
    """Result limiting and relevance scoring."""

    def test_respects_limit(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        for i in range(10):
            _insert_annotated_event(
                write_conn, what_doing=f"Coding task number {i}",
            )

        with ActivitySearcher(tmp_db_path) as s:
            results = s.search("Coding", limit=3)

        assert len(results) == 3

    def test_relevance_score_is_between_0_and_1(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(write_conn, what_doing="expired domains search")
        _insert_annotated_event(write_conn, what_doing="domain registration")
        _insert_annotated_event(write_conn, what_doing="unrelated cooking recipe")

        with ActivitySearcher(tmp_db_path) as s:
            results = s.search("expired domains")

        for r in results:
            assert 0.0 <= r.relevance_score <= 1.0

    def test_best_match_has_highest_score(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(
            write_conn, what_doing="Searching for expired domains on GoDaddy",
        )
        _insert_annotated_event(
            write_conn, what_doing="Domains are interesting",
        )

        with ActivitySearcher(tmp_db_path) as s:
            results = s.search("expired domains")

        # The first result should have the highest relevance.
        if len(results) > 1:
            assert results[0].relevance_score >= results[1].relevance_score


# ---------------------------------------------------------------------------
# 11. search() — special characters and edge cases
# ---------------------------------------------------------------------------


class TestSearchSpecialCharacters:
    """FTS5 safety with special characters and edge cases."""

    def test_query_with_double_quotes(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(write_conn, what_doing="Test activity")

        with ActivitySearcher(tmp_db_path) as s:
            # Should not raise.
            results = s.search('"test"')

        # The quotes are stripped, so it should still match.
        assert len(results) >= 0  # Just ensure no crash.

    def test_query_with_parentheses(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(write_conn, what_doing="Calling function()")

        with ActivitySearcher(tmp_db_path) as s:
            results = s.search("function()")

        assert len(results) >= 0

    def test_query_with_asterisk(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(write_conn, what_doing="file search with glob pattern")

        with ActivitySearcher(tmp_db_path) as s:
            results = s.search("glob*")

        assert len(results) >= 0

    def test_query_with_colon(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(write_conn, what_doing="localhost:3000 server")

        with ActivitySearcher(tmp_db_path) as s:
            results = s.search("localhost:3000")

        assert len(results) >= 0

    def test_query_with_fts5_operators(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(write_conn, what_doing="NOT a bug AND not OR an issue")

        with ActivitySearcher(tmp_db_path) as s:
            # These should be escaped and treated as literal words.
            results = s.search("NOT AND OR NEAR")

        assert len(results) >= 0


# ---------------------------------------------------------------------------
# 12. session_recall() — basic
# ---------------------------------------------------------------------------


class TestSessionRecallBasic:
    """Basic session recall functionality."""

    def test_recall_returns_ordered_timeline(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:00:00.000000Z",
            what_doing="First task",
        )
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:05:00.000000Z",
            what_doing="Second task",
        )
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:10:00.000000Z",
            what_doing="Third task",
        )

        with ActivitySearcher(tmp_db_path) as s:
            timeline = s.session_recall(date="2026-03-10")

        assert len(timeline.entries) == 3
        assert timeline.entries[0].what_doing == "First task"
        assert timeline.entries[1].what_doing == "Second task"
        assert timeline.entries[2].what_doing == "Third task"

    def test_recall_returns_activity_timeline(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:00:00.000000Z",
            what_doing="Working",
        )

        with ActivitySearcher(tmp_db_path) as s:
            timeline = s.session_recall(date="2026-03-10")

        assert isinstance(timeline, ActivityTimeline)
        assert timeline.date == "2026-03-10"

    def test_recall_empty_database(self, tmp_db_path: Path) -> None:
        with ActivitySearcher(tmp_db_path) as s:
            timeline = s.session_recall(date="2026-03-10")

        assert timeline.entries == []
        assert timeline.total_active_minutes == 0
        assert timeline.apps_used == []


# ---------------------------------------------------------------------------
# 13. session_recall() — filters
# ---------------------------------------------------------------------------


class TestSessionRecallFilters:
    """Session recall with date, app, time range filters."""

    def test_recall_filters_by_date(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:00:00.000000Z",
            what_doing="March 10 work",
        )
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-11T10:00:00.000000Z",
            what_doing="March 11 work",
        )

        with ActivitySearcher(tmp_db_path) as s:
            timeline = s.session_recall(date="2026-03-10")

        assert len(timeline.entries) == 1
        assert timeline.entries[0].what_doing == "March 10 work"

    def test_recall_filters_by_app(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:00:00.000000Z",
            what_doing="Browsing",
            app="Google Chrome",
        )
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:05:00.000000Z",
            what_doing="Coding",
            app="VS Code",
        )

        with ActivitySearcher(tmp_db_path) as s:
            timeline = s.session_recall(date="2026-03-10", app="VS Code")

        assert len(timeline.entries) == 1
        assert timeline.entries[0].app == "VS Code"

    def test_recall_filters_by_start_and_end_time(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T08:00:00.000000Z",
            what_doing="Early morning",
        )
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T12:00:00.000000Z",
            what_doing="Midday work",
        )
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T18:00:00.000000Z",
            what_doing="Evening work",
        )

        with ActivitySearcher(tmp_db_path) as s:
            timeline = s.session_recall(
                start_time="2026-03-10T10:00:00Z",
                end_time="2026-03-10T15:00:00Z",
            )

        assert len(timeline.entries) == 1
        assert timeline.entries[0].what_doing == "Midday work"


# ---------------------------------------------------------------------------
# 14. session_recall() — active minutes
# ---------------------------------------------------------------------------


class TestSessionRecallActiveMinutes:
    """Active minutes calculation in session recall."""

    def test_calculates_active_minutes(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        # 4 events, 2 minutes apart each → total 6 minutes active.
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:00:00.000000Z",
            what_doing="Task A",
        )
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:02:00.000000Z",
            what_doing="Task B",
        )
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:04:00.000000Z",
            what_doing="Task C",
        )
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:06:00.000000Z",
            what_doing="Task D",
        )

        with ActivitySearcher(tmp_db_path) as s:
            timeline = s.session_recall(date="2026-03-10")

        assert timeline.total_active_minutes == 6

    def test_idle_gaps_not_counted(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        # 2 events, 10 minutes apart → exceeds threshold → 0 minutes.
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:00:00.000000Z",
            what_doing="Task A",
        )
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:10:00.000000Z",
            what_doing="Task B",
        )

        with ActivitySearcher(tmp_db_path) as s:
            timeline = s.session_recall(date="2026-03-10")

        assert timeline.total_active_minutes == 0

    def test_single_event_zero_minutes(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:00:00.000000Z",
            what_doing="Solo task",
        )

        with ActivitySearcher(tmp_db_path) as s:
            timeline = s.session_recall(date="2026-03-10")

        assert timeline.total_active_minutes == 0


# ---------------------------------------------------------------------------
# 15. session_recall() — apps used
# ---------------------------------------------------------------------------


class TestSessionRecallAppsUsed:
    """Apps used list in session recall."""

    def test_lists_distinct_apps(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:00:00.000000Z",
            app="Google Chrome",
            what_doing="Browsing",
        )
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:05:00.000000Z",
            app="VS Code",
            what_doing="Coding",
        )
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:10:00.000000Z",
            app="Google Chrome",
            what_doing="More browsing",
        )

        with ActivitySearcher(tmp_db_path) as s:
            timeline = s.session_recall(date="2026-03-10")

        assert "Google Chrome" in timeline.apps_used
        assert "VS Code" in timeline.apps_used
        assert len(timeline.apps_used) == 2

    def test_apps_ordered_by_first_appearance(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:00:00.000000Z",
            app="Terminal",
            what_doing="Setup",
        )
        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:05:00.000000Z",
            app="VS Code",
            what_doing="Coding",
        )

        with ActivitySearcher(tmp_db_path) as s:
            timeline = s.session_recall(date="2026-03-10")

        assert timeline.apps_used[0] == "Terminal"
        assert timeline.apps_used[1] == "VS Code"


# ---------------------------------------------------------------------------
# 16. session_recall() — defaults to today
# ---------------------------------------------------------------------------


class TestSessionRecallDefaults:
    """session_recall with no arguments defaults to today."""

    def test_defaults_to_today(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        _insert_annotated_event(
            write_conn,
            timestamp=now_ts,
            what_doing="Current work",
        )

        with ActivitySearcher(tmp_db_path) as s:
            timeline = s.session_recall()

        assert timeline.date == today


# ---------------------------------------------------------------------------
# 17. Multiple annotations for same query
# ---------------------------------------------------------------------------


class TestMultipleAnnotations:
    """Multiple events matching the same search query."""

    def test_returns_all_matching_events(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(
            write_conn,
            what_doing="Deploying to production server",
            timestamp="2026-03-10T10:00:00.000000Z",
        )
        _insert_annotated_event(
            write_conn,
            what_doing="Deploying to staging server",
            timestamp="2026-03-10T11:00:00.000000Z",
        )
        _insert_annotated_event(
            write_conn,
            what_doing="Deploying hotfix to production",
            timestamp="2026-03-10T12:00:00.000000Z",
        )

        with ActivitySearcher(tmp_db_path) as s:
            results = s.search("Deploying")

        assert len(results) == 3

    def test_each_result_has_correct_event_id(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        eid1 = _insert_annotated_event(
            write_conn,
            what_doing="Deploy alpha",
            event_id="evt-alpha",
        )
        eid2 = _insert_annotated_event(
            write_conn,
            what_doing="Deploy beta",
            event_id="evt-beta",
        )

        with ActivitySearcher(tmp_db_path) as s:
            results = s.search("Deploy")

        event_ids = {r.event_id for r in results}
        assert "evt-alpha" in event_ids
        assert "evt-beta" in event_ids


# ---------------------------------------------------------------------------
# 18. SearchResult dataclass
# ---------------------------------------------------------------------------


class TestSearchResultDataclass:
    """SearchResult fields and defaults."""

    def test_all_fields_present(self) -> None:
        r = SearchResult(
            timestamp="2026-03-10T10:00:00Z",
            app="Chrome",
            location="https://example.com",
            what_doing="Browsing",
            relevance_score=0.95,
            event_id="e1",
        )
        assert r.timestamp == "2026-03-10T10:00:00Z"
        assert r.app == "Chrome"
        assert r.location == "https://example.com"
        assert r.what_doing == "Browsing"
        assert r.relevance_score == 0.95
        assert r.event_id == "e1"
        assert r.screenshot_id is None

    def test_optional_screenshot_id(self) -> None:
        r = SearchResult(
            timestamp="ts", app="a", location="l", what_doing="w",
            relevance_score=0.5, event_id="e", screenshot_id="scr-1",
        )
        assert r.screenshot_id == "scr-1"


# ---------------------------------------------------------------------------
# 19. ActivityTimeline dataclass
# ---------------------------------------------------------------------------


class TestActivityTimelineDataclass:
    """ActivityTimeline fields."""

    def test_all_fields_present(self) -> None:
        t = ActivityTimeline(
            entries=[],
            date="2026-03-10",
            total_active_minutes=42,
            apps_used=["Chrome", "VS Code"],
        )
        assert t.date == "2026-03-10"
        assert t.total_active_minutes == 42
        assert len(t.apps_used) == 2


# ---------------------------------------------------------------------------
# 20. Refresh index
# ---------------------------------------------------------------------------


class TestRefreshIndex:
    """refresh_index() picks up new annotations."""

    def test_refresh_returns_count_of_new_entries(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(write_conn, what_doing="Initial")

        with ActivitySearcher(tmp_db_path) as s:
            s._ensure_fts_index()

            _insert_annotated_event(write_conn, what_doing="Added later")
            _insert_annotated_event(write_conn, what_doing="Also added later")

            count = s.refresh_index()
            assert count == 2

    def test_refresh_with_nothing_new_returns_zero(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        _insert_annotated_event(write_conn, what_doing="Only event")

        with ActivitySearcher(tmp_db_path) as s:
            s._ensure_fts_index()
            count = s.refresh_index()
            assert count == 0


# ---------------------------------------------------------------------------
# 21. Recall with only pending annotations (should be excluded)
# ---------------------------------------------------------------------------


class TestPendingAnnotationsExcluded:
    """Events with annotation_status != 'completed' are excluded."""

    def test_pending_events_not_in_recall(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        eid = insert_event(
            write_conn, timestamp="2026-03-10T10:00:00.000000Z",
        )
        # Set annotation but leave status as something other than completed.
        _set_annotation(
            write_conn, eid,
            _make_annotation(what_doing="Pending work"),
            status="pending",
        )

        _insert_annotated_event(
            write_conn,
            timestamp="2026-03-10T10:05:00.000000Z",
            what_doing="Completed work",
        )

        with ActivitySearcher(tmp_db_path) as s:
            timeline = s.session_recall(date="2026-03-10")

        assert len(timeline.entries) == 1
        assert timeline.entries[0].what_doing == "Completed work"

    def test_failed_events_not_in_recall(
        self, tmp_db_path: Path, write_conn: sqlite3.Connection,
    ) -> None:
        eid = insert_event(
            write_conn, timestamp="2026-03-10T10:00:00.000000Z",
        )
        _set_annotation(
            write_conn, eid,
            _make_annotation(what_doing="Failed annotation"),
            status="failed",
        )

        with ActivitySearcher(tmp_db_path) as s:
            timeline = s.session_recall(date="2026-03-10")

        assert len(timeline.entries) == 0
