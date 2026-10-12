"""Forget (``memory_forget``): every store emptied for what is forgotten,
everything else intact."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from agenthandover_worker.memory_forget import Selector, forget, moment_rows, parse_selector
from memory_world import NOW, World

H = timedelta(hours=1)
M = timedelta(minutes=1)


@pytest.fixture()
def world(tmp_path):
    w = World(tmp_path)
    yield w
    w.close()


def run(world: World, selector: Selector) -> dict:
    return forget(
        selector, db_path=world.db, data_dir=world.data_dir,
        knowledge_dir=world.kb.root, local_tz=timezone.utc,
    )


def everything_left(world: World) -> dict:
    return {
        "events": world.ids("SELECT id FROM events"),
        "memory": world.ids("SELECT id FROM memory_frames"),
        "fts": world.fts_events(),
        "preserved": world.ids("SELECT event_id FROM preserved_text"),
        "annotation_vectors": world.vector_sources("annotation"),
        "preserved_vectors": world.vector_sources("preserved_text"),
        "daily_vectors": world.vector_sources("daily_summary"),
        "artifacts": world.ids("SELECT event_id FROM artifacts"),
        "jobs": world.ids("SELECT event_id FROM vlm_queue"),
        "screenshots": world.files(world.shots),
        "thumbnails": world.files(world.thumbs),
        "artifact_files": world.files(world.artifacts),
        "segments": world.ids("SELECT segment_id FROM task_segments"),
        "observations": world.ids("SELECT obs_id FROM workflow_observations"),
        "cache_apps": world.ids("SELECT app FROM annotation_cache"),
        "daily": world.daily_days(),
        "digests": world.digest_days(),
        "recent": world.recent_days(),
    }


def three_frames(world: World) -> None:
    """A at 09:00 (Chrome), B at 11:30 (Slack), C at 11:45 (Chrome Docs);
    a frame of yesterday; Skills and the profile vector."""
    world.frame("ev-a", NOW - 3 * H, "Google Chrome", "Inbox", "Reading the Acme invoice")
    world.frame("ev-b", NOW - 30 * M, "Slack", "#design", "Discussing the hero banner")
    world.frame("ev-c", NOW - 15 * M, "Google Chrome", "Q3 plan - Docs", "Editing the Q3 plan")
    world.frame("ev-y", NOW - 24 * H, "Slack", "#general", "Reading announcements")
    world.segment("seg-a", ["ev-a"], NOW - 3 * H)
    world.segment("seg-bc", ["ev-b", "ev-c"], NOW - 30 * M)
    world.observation("obs-a", ["ev-a"], NOW - 3 * H)
    world.observation("obs-b", ["ev-b"], NOW - 30 * M)
    world.daily("2026-10-11", "Slack announcements")
    world.daily("2026-10-12", "Acme invoice and the Q3 plan")
    world.kb.save_procedure({"id": "pay-invoice", "title": "Pay an invoice",
                             "lifecycle_state": "agent_ready"})
    world.vkb.upsert("procedure", "pay-invoice", "Pay an invoice", embedding=world.next_axis())
    world.vkb.upsert("profile", "user", "profile", embedding=world.next_axis())
    world.index_activity()
    for name in ("search-query-result.json", "recall-query-result.json"):
        (world.data_dir / name).write_text("{}")


def test_a_range_empties_every_store_for_it_and_keeps_the_rest(world):
    three_frames(world)
    # OCR text kept past its event's purge (the event is gone, its text is
    # not), and its vector.
    world.sql(
        "INSERT INTO preserved_text (event_id, source, timestamp, text_content, app) "
        "VALUES ('gone-1', 'visual_proxy', ?, 'payroll numbers', 'pid:1:Numbers')",
        ((NOW - 50 * M).strftime("%Y-%m-%dT%H:%M:%SZ"),),
    )
    world.vkb.upsert("preserved_text", "gone-1:visual_proxy", "payroll", embedding=world.next_axis())

    counts = run(world, Selector(since=NOW - H, until=NOW))

    left = everything_left(world)
    assert left["events"] == {"ev-a", "ev-y"}
    assert left["memory"] == {"ev-a", "ev-y"}
    assert left["fts"] == {"ev-a", "ev-y"}
    assert left["preserved"] == {"ev-a", "ev-y"}
    assert left["annotation_vectors"] == {"ev-a", "ev-y"}
    assert left["preserved_vectors"] == {"ev-a:ocr", "ev-y:ocr"}
    assert left["artifacts"] == left["jobs"] == {"ev-a", "ev-y"}
    assert left["screenshots"] == left["thumbnails"] == left["artifact_files"] == {"ev-a", "ev-y"}
    assert left["segments"] == {"seg-a"}
    assert left["observations"] == {"obs-a"}
    assert left["cache_apps"] == {"Google Chrome", "Slack"}  # Inbox and #general stay
    assert world.sql("SELECT count(*) FROM annotation_cache")[0][0] == 2
    # The day the range touches: its summary, digest, vector and recent entry.
    assert left["daily"] == left["digests"] == left["recent"] == {"2026-10-11"}
    assert left["daily_vectors"] == {"2026-10-11"}
    # Skills and the profile are never touched.
    assert world.vector_sources("procedure") == {"pay-invoice"}
    assert world.vector_sources("profile") == {"user"}
    assert world.kb.get_procedure("pay-invoice") is not None
    assert not (world.data_dir / "search-query-result.json").exists()
    assert not (world.data_dir / "recall-query-result.json").exists()
    assert counts == {
        "events": 2, "memory_moments": 2, "thumbnails": 2, "screenshots": 2,
        # annotation x2, preserved_text x3 (two frames and gone-1), the day.
        "vectors": 6, "daily_summaries": 1,
    }


def test_a_moment_overlapping_the_range_is_forgotten_whole(world):
    # Docs open 10:50-11:05 (one moment, three frames); a Terminal frame
    # at 10:58 inside its span is another window.
    for eid, minutes in (("d1", -70), ("d2", -65), ("d3", -55)):
        world.frame(eid, NOW + minutes * M, "Google Chrome", "Q3 plan - Docs",
                    "Editing the Q3 plan", memory=False)
    world.frame("t1", NOW - 62 * M, "Terminal", "zsh", "Running tests", memory=False)
    world.moment("d1", NOW - 70 * M, NOW - 55 * M, "Google Chrome", "Q3 plan - Docs",
                 "Editing the Q3 plan", frames=3, thumb_id="d1")
    world.moment("t1", NOW - 62 * M, NOW - 62 * M, "Terminal", "zsh", "Running tests",
                 thumb_id="t1")
    world.frame("old", NOW - 3 * H, "Google Chrome", "Q3 plan - Docs", "Editing the Q3 plan")

    counts = run(world, Selector(since=NOW - H, until=NOW))

    assert world.ids("SELECT id FROM memory_frames") == {"t1", "old"}
    # The moment's frames before the range went with it; the Terminal
    # frame inside its span did not.
    assert world.ids("SELECT id FROM events") == {"t1", "old"}
    assert world.files(world.thumbs) == {"t1", "old"}
    assert counts["memory_moments"] == 1
    assert counts["events"] == 3


def test_an_app_is_forgotten_on_every_day_and_nothing_else(world):
    three_frames(world)
    counts = run(world, Selector(app="slack"))

    left = everything_left(world)
    assert left["events"] == left["memory"] == left["fts"] == {"ev-a", "ev-c"}
    assert left["annotation_vectors"] == {"ev-a", "ev-c"}
    assert left["screenshots"] == left["thumbnails"] == {"ev-a", "ev-c"}
    assert left["segments"] == {"seg-a"}  # seg-bc held a Slack frame
    assert left["observations"] == {"obs-a"}
    # Both days had Slack frames.
    assert left["daily"] == set()
    assert counts["events"] == 2 and counts["memory_moments"] == 2


def test_a_summary_naming_the_app_goes_even_when_its_frames_are_purged(world):
    world.frame("ev-a", NOW - 3 * H, "Google Chrome", "Inbox", "Reading")
    world.daily("2026-09-01", "Mostly Figma and Slack")
    world.daily("2026-09-02", "Mostly Xcode")
    run(world, Selector(app="Figma"))
    assert world.daily_days() == {"2026-09-02"}
    assert world.ids("SELECT id FROM events") == {"ev-a"}


def test_a_range_with_an_app_keeps_the_other_apps(world):
    three_frames(world)
    run(world, Selector(since=NOW - H, until=NOW, app="Google Chrome"))
    assert world.ids("SELECT id FROM events") == {"ev-a", "ev-b", "ev-y"}
    assert world.ids("SELECT id FROM memory_frames") == {"ev-a", "ev-b", "ev-y"}
    # The Slack frame in the range keeps its thumbnail (no mtime sweep
    # with an app).
    assert world.files(world.thumbs) == {"ev-a", "ev-b", "ev-y"}


def test_a_moment_id_forgets_its_rows_and_the_time_they_span(world):
    title = "Q3 plan - Docs"
    world.frame("m1", NOW - 40 * M, "Google Chrome", title, "Editing", memory=False)
    world.frame("m2", NOW - 33 * M, "Google Chrome", title, "Editing", memory=False)
    world.frame("x1", NOW - 20 * M, "Slack", "#design", "Chatting", memory=False)
    world.frame("m3", NOW - 10 * M, "Google Chrome", title, "Editing", memory=False)
    world.moment("m1", NOW - 40 * M, NOW - 36 * M, "Google Chrome", title, "Editing")
    world.moment("m2", NOW - 33 * M, NOW - 30 * M, "Google Chrome", title, "Editing")
    world.moment("x1", NOW - 20 * M, NOW - 20 * M, "Slack", "#design", "Chatting")
    world.moment("m3", NOW - 10 * M, NOW - 10 * M, "Google Chrome", title, "Editing")

    import sqlite3
    conn = sqlite3.connect(world.db)
    conn.row_factory = sqlite3.Row
    assert [r["id"] for r in moment_rows(conn, "m2")] == ["m1", "m2"]
    assert [r["id"] for r in moment_rows(conn, "m3")] == ["m3"]
    assert moment_rows(conn, "nope") == []
    conn.close()

    run(world, Selector(moment_id="m2"))
    assert world.ids("SELECT id FROM memory_frames") == {"x1", "m3"}
    assert world.ids("SELECT id FROM events") == {"x1", "m3"}
    with pytest.raises(LookupError):
        run(world, Selector(moment_id="nope"))


def test_all_forgets_everything_but_skills_and_the_profile(world):
    three_frames(world)
    counts = run(world, Selector(all=True))
    left = everything_left(world)
    assert all(not v for v in left.values()), left
    assert world.vector_sources("procedure") == {"pay-invoice"}
    assert world.vector_sources("profile") == {"user"}
    assert counts["events"] == 4 and counts["daily_summaries"] == 2


def test_files_written_in_the_range_before_their_event_go_too(world):
    # The daemon writes the thumbnail and screenshot before the event row.
    early = world.thumbs / "not-yet.ahthumb"
    early.write_bytes(b"x")
    shot = world.shots / "not-yet.jpg"
    shot.write_bytes(b"x")
    old = world.thumbs / "older.ahthumb"
    old.write_bytes(b"x")
    import os
    stamp = (NOW - 2 * H).timestamp()
    os.utime(old, (stamp, stamp))
    for path in (early, shot):
        os.utime(path, (NOW.timestamp() - 60, NOW.timestamp() - 60))
    counts = run(world, Selector(since=NOW - H, until=NOW))
    assert not early.exists() and not shot.exists()
    assert old.exists()
    assert counts["thumbnails"] == 1 and counts["screenshots"] == 1


@pytest.mark.parametrize("body, selector", [
    ({"moment_id": "m1"}, Selector(moment_id="m1")),
    ({"all": True}, Selector(all=True)),
    ({"app": "Slack"}, Selector(app="Slack")),
    ({"since": "2026-10-12T11:00:00Z", "until": "2026-10-12T12:00:00+00:00"},
     Selector(since=NOW - H, until=NOW)),
    ({"since": "2026-10-12T11:00:00", "until": "2026-10-12T12:00:00Z", "app": "Slack"},
     Selector(since=NOW - H, until=NOW, app="Slack")),
])
def test_parse_selector_accepts_exactly_one_selector(body, selector):
    assert parse_selector(body) == selector


@pytest.mark.parametrize("body, message", [
    ({}, "exactly one"),
    ({"all": True, "app": "Slack"}, "exactly one"),
    ({"moment_id": "m", "since": "2026-10-12T11:00:00Z", "until": "2026-10-12T12:00:00Z"},
     "exactly one"),
    ({"since": "2026-10-12T11:00:00Z"}, "exactly one"),
    ({"all": False}, "'all' must be true"),
    ({"all": "yes"}, "'all' must be true"),
    ({"app": ""}, "'app'"),
    ({"moment_id": 3}, "'moment_id'"),
    ({"since": "yesterday", "until": "2026-10-12T12:00:00Z"}, "'since' must be an ISO-8601"),
    ({"since": "2026-10-12T12:00:00Z", "until": "2026-10-12T11:00:00Z"}, "after"),
    ({"days": 3}, "unknown field"),
    ([], "JSON object"),
])
def test_parse_selector_refuses_ambiguous_or_empty_selectors(body, message):
    with pytest.raises(ValueError, match=message):
        parse_selector(body)


def test_a_frame_forgotten_while_it_is_embedded_does_not_come_back(tmp_path):
    """The writer re-checks its event inside the transaction that writes
    the row: a forget that ran during the embedding call wins."""
    from dataclasses import replace

    from agenthandover_worker.config import MemoryConfig
    from agenthandover_worker.memory import MemoryWriter

    world = World(tmp_path)
    world.frame("ev-1", NOW - M, "Slack", "#design", "Chatting", memory=False)
    world.sql("UPDATE events SET processed = 0")

    def embed(items):
        # The forget lands while the model is answering.
        run(world, Selector(all=True))
        return [world.next_axis()]

    shot = world.shots / "ev-1.jpg"
    writer = MemoryWriter(
        world.db, replace(MemoryConfig(), enabled=True), embed=embed,
        screenshots_dir=world.shots, data_dir=world.data_dir, now=lambda: NOW,
        annotation_enabled=False,
    )
    shot.write_bytes(b"jpeg")
    writer.run()
    writer.close()
    assert world.ids("SELECT id FROM memory_frames") == set()
    world.close()


def test_a_forgotten_latest_row_is_not_extended(tmp_path):
    """After a forget the writer starts a new row instead of extending
    the deleted one (``MemoryWriter.forgotten`` / the UPDATE's rowcount)."""
    from dataclasses import replace

    from agenthandover_worker.config import MemoryConfig
    from agenthandover_worker.memory import MemoryWriter

    world = World(tmp_path)
    vector = world.next_axis()
    writer = MemoryWriter(
        world.db, replace(MemoryConfig(), enabled=True), embed=lambda items: [vector],
        screenshots_dir=world.shots, data_dir=world.data_dir, now=lambda: NOW,
        annotation_enabled=False,
    )
    world.frame("ev-1", NOW - 2 * M, "Slack", "#design", "Chatting", memory=False)
    writer.run()
    assert world.ids("SELECT id FROM memory_frames") == {"ev-1"}
    run(world, Selector(moment_id="ev-1"))
    # Same window, same picture, a minute later: without the forget it
    # would extend ev-1's row.
    world.frame("ev-2", NOW - M, "Slack", "#design", "Chatting", memory=False)
    writer.run()
    rows = world.sql("SELECT id, frames FROM memory_frames")
    assert rows == [("ev-2", 1)]
    writer.close()
    world.close()


def test_the_selector_describes_itself():
    assert Selector(all=True).describe() == "everything"
    assert Selector(app="Slack").describe() == "app Slack"
    assert "moment m1" == Selector(moment_id="m1").describe()
    assert json.dumps(Selector(since=NOW - H, until=NOW).describe())
