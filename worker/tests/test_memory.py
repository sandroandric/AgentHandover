"""Memory's store, writer and search (WP-E4), with a fake embedder.

The writer runs over a real daemon database (the daemon's migrations,
v008 included) and real JPEG paths; only the model is replaced by
deterministic vectors, so these pin the writer's decisions: what is
embedded, what merges into a moment, which screenshots are kept for it
and when they are deleted, what is never kept, retention, and the
search path's scoring, filters and moment grouping.  The journey
(``journeys/test_j_memory.py``) drives the same code through the worker's
tick with recorded EmbeddingGemma answers.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agenthandover_worker import main as wm
from agenthandover_worker.config import Config, MemoryConfig, load_config
from agenthandover_worker.db import WorkerDB
from agenthandover_worker.memory import (
    CURSOR_STATE_KEY,
    HOLD_HOURS,
    KEYWORD_BOOST,
    MemoryIndex,
    MemoryWriter,
    annotation_description,
    embed_text,
    paused,
    utc_text,
)
from agenthandover_worker.model_scheduler import GB, ModelScheduler
from agenthandover_worker.ollama import OllamaUnavailable
from agenthandover_worker.privacy_zones import PrivacyZoneChecker
from agenthandover_worker.scene_annotator import AnnotationConfig, SceneAnnotator
from daemon_db import create_daemon_db

DIM = 512
NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)


def _basis(i: int, wobble: float = 0.0, j: int | None = None) -> list[float]:
    """A unit vector along axis *i*, tilted by *wobble* toward axis *j*."""
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    if wobble:
        v[(i + 1) if j is None else j] = wobble
    return list(v / np.linalg.norm(v))


class FakeEmbedder:
    """Vectors by image content: ``vectors[image bytes]``.  Records every
    request; ``fail`` raises the next N calls."""

    def __init__(self, vectors: dict[bytes, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[dict]] = []
        self.fail = 0

    def __call__(self, items: list) -> list[list[float]]:
        self.calls.append(items)
        if self.fail:
            self.fail -= 1
            raise OllamaUnavailable("Ollama not reachable (fake)")
        return [self.vectors[item["image"]] for item in items]


class Clock:
    def __init__(self) -> None:
        self.now = NOW
        self.mono = 1000.0


@pytest.fixture()
def env(tmp_path: Path):
    db_path = create_daemon_db(tmp_path / "events.db")
    data_dir = tmp_path / "data"
    shots = data_dir / "screenshots"
    shots.mkdir(parents=True)
    (data_dir / "status").mkdir()
    (data_dir / "thumbs").mkdir()
    return SimpleNamespace(db_path=db_path, data_dir=data_dir, shots=shots, tmp=tmp_path)


def _writer(env, embedder, clock, **config) -> MemoryWriter:
    cfg = replace(MemoryConfig(), **{"enabled": True, **config})
    return MemoryWriter(
        env.db_path, cfg,
        embed=embedder,
        screenshots_dir=env.shots,
        data_dir=env.data_dir,
        privacy_checker=PrivacyZoneChecker(),
        now=lambda: clock.now,
        monotonic=lambda: clock.mono,
    )


_counter = iter(range(10_000))


def add_frame(
    env,
    *,
    at: datetime,
    app: str = "Google Chrome",
    title: str = "Inbox",
    image: bytes = b"red",
    status: str = "pending",
    annotation: dict | None = None,
    focus: str | None = None,
    url: str | None = None,
    with_file: bool = True,
    thumb: bool = False,
) -> dict:
    """Insert a captured frame as the daemon does; returns the event dict."""
    eid = f"evt-{next(_counter):04d}"
    shot = env.shots / f"{eid}.jpg"
    if with_file:
        shot.write_bytes(image)
    meta: dict = {"screenshot_path": str(shot)}
    if focus:
        meta["focus_session_id"] = focus
    if url:
        meta["url"] = url
    if thumb:
        (env.data_dir / "thumbs" / f"{eid}.ahthumb").write_bytes(b"sealed")
    conn = sqlite3.connect(env.db_path)
    conn.execute(
        "INSERT INTO events (id, timestamp, kind_json, window_json, display_topology_json, "
        "primary_display_id, metadata_json, annotation_status, scene_annotation_json) "
        "VALUES (?, ?, '{\"FocusChange\":{}}', ?, '[]', 'main', ?, ?, ?)",
        (
            eid, at.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            json.dumps({"app_id": f"pid:1:{app}", "title": title}),
            json.dumps(meta), status, json.dumps(annotation) if annotation else None,
        ),
    )
    conn.commit()
    conn.close()
    return {"id": eid, "path": shot}


def set_status(env, eid: str, status: str, annotation: dict | None = None) -> None:
    conn = sqlite3.connect(env.db_path)
    conn.execute(
        "UPDATE events SET annotation_status = ?, scene_annotation_json = ? WHERE id = ?",
        (status, json.dumps(annotation) if annotation else None, eid),
    )
    conn.commit()
    conn.close()


def rows(env) -> list[dict]:
    conn = sqlite3.connect(env.db_path)
    conn.row_factory = sqlite3.Row
    out = [dict(r) for r in conn.execute("SELECT * FROM memory_frames ORDER BY first_ts")]
    conn.close()
    return out


ANNOTATION = {
    "app": "Chrome",
    "location": "Gmail inbox",
    "task_context": {"what_doing": "Reading the invoice email from Acme"},
}


# ---------------------------------------------------------------------------
# Writer: embed, describe, delete
# ---------------------------------------------------------------------------


def test_an_annotated_frame_is_embedded_described_and_its_screenshot_deleted(env):
    clock = Clock()
    fake = FakeEmbedder({b"red": _basis(0)})
    writer = _writer(env, fake, clock)
    frame = add_frame(
        env, at=NOW - timedelta(minutes=5), status="completed", annotation=ANNOTATION,
        url="https://mail.google.com/mail/u/0/#inbox",
    )

    stats = writer.run()

    assert stats["embedded"] == 1
    (row,) = rows(env)
    assert row["id"] == frame["id"]
    assert row["app"] == "Google Chrome" and row["window_title"] == "Inbox"
    assert row["url_host"] == "mail.google.com"
    assert row["description"] == "Reading the invoice email from Acme (Chrome, Gmail inbox)"
    assert row["described_by"] == frame["id"]
    assert row["model"] == "embeddinggemma-2:440m" and row["dim"] == DIM
    vector = np.frombuffer(row["embedding"], dtype="<f4")
    assert vector.size == DIM and abs(float(np.linalg.norm(vector)) - 1.0) < 1e-6
    # What was sent: the screenshot and "title: app — title | text: ".
    (item,) = fake.calls[0]
    assert item == {"text": "title: Google Chrome — Inbox | text: ", "image": b"red"}
    assert not frame["path"].exists(), "embedded and annotated: the JPEG is deleted"
    assert writer.cursor() > 0


def test_a_frame_awaiting_annotation_is_held_until_embedded_then_released(env):
    """Embed before delete: the annotator keeps a JPEG Memory still needs,
    and once Memory has it, annotation's delete goes through."""
    clock = Clock()
    writer = _writer(env, FakeEmbedder({b"red": _basis(0)}), clock)
    annotator = SceneAnnotator(AnnotationConfig())
    annotator.keep_screenshot = writer.holds
    frame = add_frame(env, at=NOW - timedelta(minutes=1))
    event = {"id": frame["id"], "metadata_json": json.dumps({"screenshot_path": str(frame["path"])})}

    annotator.discard_screenshot(event, env.shots)
    assert frame["path"].exists(), "not embedded yet: kept for Memory"

    writer.run()
    assert frame["path"].exists(), "annotation still pending: Memory leaves it"
    assert not writer.holds(event)
    annotator.discard_screenshot(event, env.shots)
    assert not frame["path"].exists()


def test_without_an_annotation_the_description_is_the_screen_text_until_one_lands(env):
    clock = Clock()
    writer = _writer(env, FakeEmbedder({b"red": _basis(0)}), clock)
    frame = add_frame(env, at=NOW - timedelta(minutes=1))
    conn = sqlite3.connect(env.db_path)
    conn.execute(
        "UPDATE events SET metadata_json = json_set(metadata_json, '$.ocr', json(?)) WHERE id = ?",
        (json.dumps({"full_text": "Invoice 4411 from Acme, mail me at a.person@example.com"}),
         frame["id"]),
    )
    conn.commit()
    conn.close()

    writer.run()
    (row,) = rows(env)
    assert row["described_by"] is None
    assert "Invoice 4411 from Acme" in row["description"]
    assert "a.person@example.com" not in row["description"], "redacted"

    event = {"id": frame["id"], "timestamp": utc_text(NOW - timedelta(minutes=1))}
    assert writer.annotated(event, ANNOTATION)
    (row,) = rows(env)
    assert row["description"].startswith("Reading the invoice email")
    assert row["described_by"] == frame["id"]
    # A later annotation of the moment does not replace the first.
    other = dict(ANNOTATION, task_context={"what_doing": "Something else"})
    assert not writer.annotated(event, other)


def test_annotation_description_is_redacted_and_empty_without_content():
    assert annotation_description(None) == ""
    assert annotation_description({"task_context": {}}) == ""
    text = annotation_description({
        "task_context": {"what_doing": "Emailing jane.doe@example.com"}, "app": "Mail",
    })
    assert "jane.doe@example.com" not in text and text.endswith("(Mail)")


def test_embed_text_matches_the_profile_format():
    assert embed_text("Slack", "general") == "title: Slack — general | text: "


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------


def test_near_identical_frames_of_one_window_extend_one_moment(env):
    clock = Clock()
    fake = FakeEmbedder({
        b"a1": _basis(0), b"a2": _basis(0, 0.05), b"a3": _basis(0, 0.1),
    })
    writer = _writer(env, fake, clock)
    first = add_frame(env, at=NOW - timedelta(minutes=30), image=b"a1", thumb=True)
    add_frame(env, at=NOW - timedelta(minutes=28), image=b"a2", thumb=True)
    add_frame(env, at=NOW - timedelta(minutes=21), image=b"a3")

    stats = writer.run()

    assert stats == {"embedded": 3, "merged": 2, "passed": 0, "waiting": None}
    (row,) = rows(env)
    assert row["id"] == first["id"] and row["frames"] == 3
    assert row["first_ts"] == utc_text(NOW - timedelta(minutes=30))
    assert row["last_ts"] == utc_text(NOW - timedelta(minutes=21))
    assert row["thumb_id"] == first["id"]
    thumbs = sorted(p.name for p in (env.data_dir / "thumbs").iterdir())
    assert thumbs == [f"{first['id']}.ahthumb"], "a merged frame's thumbnail is a copy"


@pytest.mark.parametrize("second", [
    # another window
    dict(title="Sent", image=b"a2", minutes=29),
    # too far apart
    dict(title="Inbox", image=b"a2", minutes=19),
    # the screen changed (cosine below 0.97)
    dict(title="Inbox", image=b"b", minutes=29),
])
def test_a_frame_starts_a_new_moment(env, second):
    clock = Clock()
    fake = FakeEmbedder({b"a1": _basis(0), b"a2": _basis(0, 0.05), b"b": _basis(0, 0.5)})
    writer = _writer(env, fake, clock)
    add_frame(env, at=NOW - timedelta(minutes=30), image=b"a1")
    add_frame(env, at=NOW - timedelta(minutes=second["minutes"]),
              title=second["title"], image=second["image"])

    writer.run()

    assert [r["frames"] for r in rows(env)] == [1, 1]


def test_returning_to_a_window_after_another_is_a_new_moment(env):
    clock = Clock()
    fake = FakeEmbedder({b"a": _basis(0), b"b": _basis(5)})
    writer = _writer(env, fake, clock)
    add_frame(env, at=NOW - timedelta(minutes=10), image=b"a")
    add_frame(env, at=NOW - timedelta(minutes=9), title="Docs", image=b"b")
    add_frame(env, at=NOW - timedelta(minutes=8), image=b"a")

    writer.run()

    assert [(r["window_title"], r["frames"]) for r in rows(env)] == [
        ("Inbox", 1), ("Docs", 1), ("Inbox", 1),
    ]


def test_a_restarted_writer_merges_into_the_stored_latest_row(env):
    clock = Clock()
    vectors = {b"a1": _basis(0), b"a2": _basis(0, 0.05)}
    _writer(env, FakeEmbedder(vectors), clock).run()  # nothing yet
    add_frame(env, at=NOW - timedelta(minutes=5), image=b"a1")
    _writer(env, FakeEmbedder(vectors), clock).run()
    add_frame(env, at=NOW - timedelta(minutes=4), image=b"a2")
    _writer(env, FakeEmbedder(vectors), clock).run()

    (row,) = rows(env)
    assert row["frames"] == 2


# ---------------------------------------------------------------------------
# What Memory never keeps, and what it waits for
# ---------------------------------------------------------------------------


def test_blocked_metadata_only_and_system_ui_frames_never_enter_memory(env):
    clock = Clock()
    fake = FakeEmbedder({b"ok": _basis(0)})
    writer = _writer(env, fake, clock)
    blocked = add_frame(env, at=NOW - timedelta(minutes=5), app="1Password", image=b"x")
    zoned = add_frame(env, at=NOW - timedelta(minutes=4), image=b"x", status="privacy_blocked")
    meta = add_frame(env, at=NOW - timedelta(minutes=3), image=b"x", status="metadata_only")
    dock = add_frame(env, at=NOW - timedelta(minutes=2), app="Dock", image=b"x")
    bank = add_frame(env, at=NOW - timedelta(minutes=2), image=b"x",
                     url="https://www.mybank.com/accounts")
    ok = add_frame(env, at=NOW - timedelta(minutes=1), image=b"ok")

    for frame in (blocked, zoned, meta, dock, bank):
        assert not writer.holds({"id": frame["id"]})
    stats = writer.run()

    assert [r["id"] for r in rows(env)] == [ok["id"]]
    assert stats["passed"] == 5 and len(fake.calls) == 1


def test_a_focus_recordings_frames_wait_for_the_keyframe_decision(env):
    clock = Clock()
    fake = FakeEmbedder({b"k": _basis(0), b"n": _basis(9), b"after": _basis(20)})
    writer = _writer(env, fake, clock)
    key = add_frame(env, at=NOW - timedelta(minutes=5), image=b"k", focus="s1")
    drop = add_frame(env, at=NOW - timedelta(minutes=4), image=b"n", focus="s1")
    after = add_frame(env, at=NOW - timedelta(minutes=3), title="Other", image=b"after")

    stats = writer.run()
    assert stats["embedded"] == 0
    assert stats["waiting"] == "a focus recording's frames are not processed yet"
    assert writer.holds({"id": key["id"]}) and writer.holds({"id": after["id"]})

    # The focus processor: a keyframe annotated, a non-keyframe dropped.
    set_status(env, key["id"], "completed", ANNOTATION)
    set_status(env, drop["id"], "skipped")
    assert not writer.holds({"id": drop["id"]})
    stats = writer.run()

    assert [r["id"] for r in rows(env)] == [key["id"], after["id"]]
    assert rows(env)[0]["focus_session_id"] == "s1"
    assert stats["passed"] == 1
    assert not key["path"].exists()
    assert not drop["path"].exists(), "a dropped frame is never kept: its JPEG is freed"


def test_frames_older_than_the_hold_are_given_up_on(env):
    clock = Clock()
    fake = FakeEmbedder({b"red": _basis(0)})
    writer = _writer(env, fake, clock)
    old = add_frame(env, at=NOW - timedelta(hours=HOLD_HOURS + 1), status="skipped")

    assert not writer.holds({"id": old["id"]})
    stats = writer.run()

    assert stats["passed"] == 1 and fake.calls == [] and rows(env) == []
    assert not old["path"].exists(), "nothing needs it any more"


def test_a_frame_the_backlog_expiry_gives_up_on_is_embedded_before_its_screenshot_goes(env):
    clock = Clock()
    writer = _writer(env, FakeEmbedder({b"red": _basis(0)}), clock)
    annotator = SceneAnnotator(AnnotationConfig())
    annotator.keep_screenshot = writer.holds
    frame = add_frame(env, at=NOW - timedelta(hours=25))
    db = WorkerDB(env.db_path)
    try:
        expired = db.expire_annotation_backlog(max_age_hours=24, now=NOW)
    finally:
        db.close()
    assert [e["id"] for e in expired] == [frame["id"]]
    annotator.discard_screenshot(expired[0], env.shots)
    assert frame["path"].exists()

    writer.run()

    assert [r["id"] for r in rows(env)] == [frame["id"]]
    assert not frame["path"].exists()


def test_paused_memory_embeds_nothing_and_skips_what_was_captured_meanwhile(env):
    clock = Clock()
    fake = FakeEmbedder({b"red": _basis(0)})
    writer = _writer(env, fake, clock)
    frame = add_frame(env, at=NOW - timedelta(minutes=1))
    pause = env.data_dir / "status" / "memory-pause.json"
    pause.write_text(json.dumps({"paused_until": None}))

    assert paused(env.data_dir, NOW)
    assert not writer.holds({"id": frame["id"]})
    assert writer.run()["embedded"] == 0
    assert fake.calls == []

    pause.write_text(json.dumps({"paused_until": utc_text(NOW - timedelta(seconds=1))}))
    assert not paused(env.data_dir, NOW)
    assert writer.run()["embedded"] == 0, "captured while paused: never embedded"
    later = add_frame(env, at=NOW)
    assert writer.run()["embedded"] == 1
    assert [r["id"] for r in rows(env)] == [later["id"]]


def test_a_missing_pause_file_is_not_paused_and_a_broken_one_is(env):
    """As the daemon (status::memory_paused): a broken record never resumes."""
    assert not paused(env.data_dir, NOW)
    path = env.data_dir / "status" / "memory-pause.json"
    for text in ("not json", "[]", "{}", json.dumps({"paused_until": "soon"}),
                 json.dumps({"paused_until": 5})):
        path.write_text(text)
        assert paused(env.data_dir, NOW), text
    path.write_text(json.dumps({"paused_until": utc_text(NOW + timedelta(hours=1))}))
    assert paused(env.data_dir, NOW)


def test_memory_off_holds_nothing_and_embeds_nothing(env):
    clock = Clock()
    fake = FakeEmbedder({b"red": _basis(0)})
    writer = _writer(env, fake, clock, enabled=False)
    frame = add_frame(env, at=NOW - timedelta(minutes=1))
    assert not writer.holds({"id": frame["id"]})
    assert writer.held_screenshot_paths() == set()
    assert writer.run()["embedded"] == 0 and fake.calls == []


def test_the_hourly_budget_makes_frames_wait(env):
    clock = Clock()
    fake = FakeEmbedder({b"a": _basis(0), b"b": _basis(5), b"c": _basis(9)})
    writer = _writer(env, fake, clock, max_frames_per_hour=2)
    third = None
    for image, minutes in ((b"a", 3), (b"b", 2), (b"c", 1)):
        third = add_frame(env, at=NOW - timedelta(minutes=minutes), image=image)

    stats = writer.run()
    assert stats["embedded"] == 2
    assert stats["waiting"] == "2 frames embedded in the last hour"
    assert writer.holds({"id": third["id"]})

    clock.mono += 3601
    assert writer.run()["embedded"] == 1


def test_an_ollama_failure_backs_off_and_keeps_the_frames(env):
    clock = Clock()
    fake = FakeEmbedder({b"red": _basis(0)})
    fake.fail = 1
    writer = _writer(env, fake, clock)
    frame = add_frame(env, at=NOW - timedelta(minutes=1), status="completed")

    stats = writer.run()
    assert stats["embedded"] == 0 and "Ollama failed" in stats["waiting"]
    assert frame["path"].exists() and writer.holds({"id": frame["id"]})
    assert writer.run()["embedded"] == 0, "inside the backoff: no call"
    assert len(fake.calls) == 1

    clock.mono += 61
    assert writer.run()["embedded"] == 1
    assert not frame["path"].exists()


def test_repeated_failures_stop_the_hold_so_annotation_deletes_as_before(env):
    """A model that is not pulled (or an Ollama too old for images) must not
    keep every screenshot on disk for two days."""
    clock = Clock()
    fake = FakeEmbedder({b"red": _basis(0)})
    fake.fail = 3
    writer = _writer(env, fake, clock)
    frame = add_frame(env, at=NOW - timedelta(minutes=1))
    for attempt in range(3):
        assert writer.holds({"id": frame["id"]}), attempt
        writer.run()
        clock.mono += 3600
    assert not writer.holds({"id": frame["id"]})
    assert writer.run()["embedded"] == 1, "the walk keeps retrying"
    assert writer.holds({"id": add_frame(env, at=NOW)["id"]})


def test_a_vector_of_the_wrong_size_is_a_failure_not_a_row(env):
    clock = Clock()
    writer = _writer(env, FakeEmbedder({b"red": [1.0, 0.0]}), clock)
    add_frame(env, at=NOW - timedelta(minutes=1))
    stats = writer.run()
    assert stats["embedded"] == 0 and "not 512" in stats["waiting"]
    assert rows(env) == []


def test_the_model_schedulers_guard_makes_frames_wait(env):
    clock = Clock()
    cfg = replace(MemoryConfig(), enabled=True)
    writer = MemoryWriter(
        env.db_path, cfg, embed=FakeEmbedder({b"red": _basis(0)}),
        screenshots_dir=env.shots, data_dir=env.data_dir,
        blocker=lambda: "20 GB of swap in use", now=lambda: clock.now,
        monotonic=lambda: clock.mono,
    )
    add_frame(env, at=NOW - timedelta(minutes=1))
    assert writer.run()["waiting"] == "20 GB of swap in use"
    assert rows(env) == []


def test_embedding_blocker_reports_pressure_and_a_generation_phase():
    client = SimpleNamespace()
    swapping = ModelScheduler(
        client, annotation_model="a", sop_model="b",
        free_memory=lambda: 8 * GB, swap_used=lambda: 20 * GB, free_disk=lambda: 100 * GB,
    )
    assert "swap" in swapping.embedding_blocker()
    calm = ModelScheduler(
        client, annotation_model="a", sop_model="b",
        free_memory=lambda: 8 * GB, swap_used=lambda: 0, free_disk=lambda: 100 * GB,
    )
    assert calm.embedding_blocker() is None
    calm._generating = True
    assert calm.embedding_blocker() == "b is writing Skills"


def test_held_paths_keep_the_orphan_sweep_off_memorys_frames(env):
    clock = Clock()
    writer = _writer(env, FakeEmbedder({}), clock)
    frame = add_frame(env, at=NOW - timedelta(minutes=1), status="completed")
    old = time.time() - 3600
    os.utime(frame["path"], (old, old))
    held = writer.held_screenshot_paths()
    assert held == {str(frame["path"].resolve())}

    db = WorkerDB(env.db_path)
    try:
        assert wm._sweep_orphan_screenshots(db, env.shots, keep=held) == 0
        assert frame["path"].exists()
        assert wm._sweep_orphan_screenshots(db, env.shots) == 1
    finally:
        db.close()


def test_the_cursor_follows_the_events_table_after_its_newest_rows_are_deleted(env):
    clock = Clock()
    writer = _writer(env, FakeEmbedder({b"a": _basis(0), b"b": _basis(5)}), clock)
    first = add_frame(env, at=NOW - timedelta(minutes=3), image=b"a")
    writer.run()
    conn = sqlite3.connect(env.db_path)
    conn.execute("DELETE FROM events WHERE id = ?", (first["id"],))
    conn.execute("DELETE FROM memory_frames")
    conn.commit()
    conn.close()
    again = add_frame(env, at=NOW - timedelta(minutes=1), image=b"b")
    writer = _writer(env, FakeEmbedder({b"a": _basis(0), b"b": _basis(5)}), clock)
    writer.run()
    assert [r["id"] for r in rows(env)] == [again["id"]]


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


def test_purge_deletes_rows_past_retention_and_forgets_expired_thumbnails(env):
    clock = Clock()
    fake = FakeEmbedder({b"a": _basis(0), b"b": _basis(5), b"c": _basis(9)})
    writer = _writer(env, fake, clock, retention_days=30, thumbnail_days=7)
    clock.now = NOW - timedelta(days=40)
    old = add_frame(env, at=clock.now, image=b"a", thumb=True)
    writer.run()
    clock.now = NOW - timedelta(days=10)
    week = add_frame(env, at=clock.now, image=b"b", thumb=True)
    writer.run()
    clock.now = NOW
    fresh = add_frame(env, at=NOW - timedelta(hours=1), image=b"c", thumb=True)
    writer.run()
    thumbs = env.data_dir / "thumbs"
    # thumbnails.expire_thumbnails (run first by the daily maintenance)
    # deleted the 10-day-old moment's thumbnail.
    (thumbs / f"{week['id']}.ahthumb").unlink()

    assert writer.purge() == {"rows": 1, "thumbnails": 1}

    assert [(r["id"], r["thumb_id"]) for r in rows(env)] == [
        (week["id"], None), (fresh["id"], fresh["id"]),
    ]
    assert sorted(p.name for p in thumbs.iterdir()) == [f"{fresh['id']}.ahthumb"]
    assert writer.purge() == {"rows": 0, "thumbnails": 0}, "idempotent"


def test_daily_maintenance_expires_thumbnails_before_the_memory_purge(monkeypatch):
    from agenthandover_worker import thumbnails

    calls = []
    monkeypatch.setattr(
        thumbnails, "expire_thumbnails",
        lambda db, *, days: calls.append(("expire", days)),
    )
    wm._run_daily_maintenance(
        db=object(), thumbnail_days=7,
        memory_writer=SimpleNamespace(purge=lambda: calls.append(("purge",))),
    )
    assert calls == [("expire", 7), ("purge",)]


def test_a_passed_over_frame_loses_its_thumbnail(env):
    clock = Clock()
    writer = _writer(env, FakeEmbedder({b"a": _basis(0)}), clock)
    clock.now = NOW
    stale = add_frame(env, at=NOW - timedelta(hours=49), image=b"a", thumb=True)
    assert writer.run()["passed"] == 1
    assert not (env.data_dir / "thumbs" / f"{stale['id']}.ahthumb").exists()


def test_daily_maintenance_runs_the_memory_purge():
    calls = []
    wm._run_daily_maintenance(memory_writer=SimpleNamespace(purge=lambda: calls.append(1)))
    assert calls == [1]


# ---------------------------------------------------------------------------
# Wiring: config, the job, the annotation hook
# ---------------------------------------------------------------------------


def test_memory_config_defaults_and_validation(tmp_path):
    assert Config().memory == MemoryConfig(
        enabled=False, model="embeddinggemma-2:440m", dimensions=512, retention_days=365,
        thumbnail_days=7, agents_access=False, max_frames_per_hour=600,
    )
    path = tmp_path / "config.toml"
    path.write_text(
        "[memory]\nenabled = true\nretention_days = 30\nthumbnail_days = 0\n"
        "agents_access = true\nmax_frames_per_hour = \"many\"\n"
    )
    memory = load_config(path).memory
    assert memory.enabled and memory.agents_access and memory.retention_days == 30
    assert memory.thumbnail_days == 0, "0 keeps no thumbnails, as in the daemon"
    assert memory.max_frames_per_hour == 600


@pytest.mark.parametrize("enabled", [True, False])
def test_the_memory_job_runs_before_passive_annotation_only_when_on(enabled):
    from agenthandover_worker.stages import Tick, control

    ctx = SimpleNamespace(
        cfg=SimpleNamespace(memory=SimpleNamespace(enabled=enabled)),
        memory_writer=object(), focus_processor=None, v2_schema_ok=True, runner=None,
    )
    tick = Tick()
    jobs = control.submit_model_jobs(ctx, tick)
    keys = [key for key, _priority, _job in jobs]
    if enabled:
        assert keys[keys.index("memory") + 1] == "passive-annotation"
    else:
        assert "memory" not in keys


def test_a_completed_annotation_describes_the_moment(env, monkeypatch):
    """``_process_annotations`` hands each completed annotation to Memory."""
    from agenthandover_worker.scene_annotator import AnnotationResult

    clock = Clock()
    writer = _writer(env, FakeEmbedder({b"red": _basis(0)}), clock)
    frame = add_frame(env, at=NOW - timedelta(minutes=1))
    writer.run()

    class Annotator:
        config = SimpleNamespace(sliding_window_size=3, sliding_window_max_age_sec=600)

        def ready(self) -> bool:
            return True

        def annotate_event(self, event, **_kw):
            return AnnotationResult(event_id=event["id"], status="completed",
                                    annotation=dict(ANNOTATION))

    db = WorkerDB(env.db_path)
    try:
        stats = wm._process_annotations(db, Annotator(), env.shots, memory=writer)
    finally:
        db.close()
    assert stats["annotated"] == 1
    (row,) = rows(env)
    assert row["described_by"] == frame["id"]
    assert row["description"].startswith("Reading the invoice email")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _seed(env, specs: list[tuple]) -> None:
    """Rows straight into the table: (id, minutes ago, span, app, title,
    description, vector)."""
    conn = sqlite3.connect(env.db_path)
    for rid, ago, span, app, title, description, vector in specs:
        first = NOW - timedelta(minutes=ago)
        conn.execute(
            "INSERT INTO memory_frames (id, first_ts, last_ts, frames, app, window_title, "
            "description, embedding, model, dim) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?)",
            (rid, utc_text(first), utc_text(first + timedelta(minutes=span)), app, title,
             description, np.asarray(vector, dtype="<f4").tobytes(),
             "embeddinggemma-2:440m", DIM),
        )
    conn.commit()
    conn.close()


def _index(env, queries: dict[str, list[float]]) -> MemoryIndex:
    return MemoryIndex(
        env.db_path, model="embeddinggemma-2:440m", dim=DIM,
        embed_query=lambda q: queries[q],
    )


def test_search_ranks_by_vector_and_filters_by_time_and_app(env):
    _seed(env, [
        ("r1", 300, 2, "Google Chrome", "Invoices", "Paying the Acme invoice", _basis(0)),
        ("r2", 200, 2, "Slack", "general", "Chatting about lunch", _basis(5)),
        ("r3", 100, 2, "Xcode", "Build", "Fixing the build", _basis(9)),
    ])
    index = _index(env, {"invoice": _basis(0, 0.3), "lunch": _basis(5)})

    hits = index.search("invoice", limit=3)
    assert hits[0]["id"] == "r1" and {h["id"] for h in hits} == {"r1", "r2", "r3"}
    assert hits[0]["app"] == "Google Chrome" and hits[0]["frames"] == 1
    assert hits[0]["score"] > hits[1]["score"]

    assert [h["id"] for h in index.search("invoice", app="slack", limit=5)] == ["r2"]
    since = NOW - timedelta(minutes=150)
    assert [h["id"] for h in index.search("invoice", since=since, limit=5)] == ["r3"]
    until = NOW - timedelta(minutes=250)
    assert [h["id"] for h in index.search("invoice", until=until, limit=5)] == ["r1"]
    assert index.search("lunch", limit=1)[0]["id"] == "r2"
    assert index.search("   ") == []


def test_search_boosts_rows_whose_description_holds_every_word(env):
    # Equal vector scores; only r2's description holds both words.
    _seed(env, [
        ("r1", 300, 1, "Notes", "a", "Quarterly numbers", _basis(1)),
        ("r2", 200, 1, "Notes", "b", "The quarterly report draft", _basis(2)),
    ])
    index = _index(env, {"quarterly report": list(
        (np.asarray(_basis(1)) + np.asarray(_basis(2))) / np.sqrt(2)
    )})
    hits = index.search("quarterly report", limit=2)
    assert [h["id"] for h in hits] == ["r2", "r1"]
    assert hits[0]["score"] - hits[1]["score"] == pytest.approx(KEYWORD_BOOST, abs=1e-4)


def test_search_groups_adjacent_rows_of_one_window_into_a_moment(env):
    _seed(env, [
        ("r1", 60, 2, "Figma", "Landing page", "Editing the hero", _basis(0)),
        ("r2", 55, 2, "Figma", "Landing page", "Editing the footer", _basis(0, 0.5)),
        ("r3", 40, 1, "Figma", "Landing page", "Later, after a break", _basis(0, 0.5)),
        ("r4", 30, 1, "Mail", "Inbox", "Mail", _basis(7)),
    ])
    index = _index(env, {"hero": _basis(0)})
    hits = index.search("hero", limit=10)
    first = hits[0]
    assert first["rows"] == ["r1", "r2"] and first["id"] == "r1"
    assert first["first_ts"] == utc_text(NOW - timedelta(minutes=60))
    assert first["last_ts"] == utc_text(NOW - timedelta(minutes=53))
    assert first["frames"] == 2
    assert [h["rows"] for h in hits[1:]] == [["r3"], ["r4"]]


def test_the_index_refreshes_with_new_extended_and_deleted_rows(env):
    clock = Clock()
    fake = FakeEmbedder({b"a1": _basis(0), b"a2": _basis(0, 0.05), b"b": _basis(5)})
    writer = _writer(env, fake, clock)
    index = _index(env, {"a": _basis(0), "b": _basis(5)})
    add_frame(env, at=NOW - timedelta(minutes=20), image=b"a1")
    writer.run()
    assert index.search("a")[0]["frames"] == 1
    assert len(index) == 1

    add_frame(env, at=NOW - timedelta(minutes=15), image=b"a2")
    add_frame(env, at=NOW - timedelta(minutes=10), title="Other", image=b"b")
    writer.run()
    hit = index.search("a")[0]
    assert hit["frames"] == 2 and hit["last_ts"] == utc_text(NOW - timedelta(minutes=15))
    assert len(index) == 2
    assert index.search("b")[0]["window_title"] == "Other"

    conn = sqlite3.connect(env.db_path)
    conn.execute("DELETE FROM memory_frames WHERE window_title = 'Other'")
    conn.commit()
    conn.close()
    assert [h["window_title"] for h in index.search("b", limit=5)] == ["Inbox"]
    assert len(index) == 1


def test_search_ignores_rows_of_another_model_or_size(env):
    _seed(env, [("r1", 10, 1, "Notes", "a", "x", _basis(0))])
    conn = sqlite3.connect(env.db_path)
    conn.execute(
        "INSERT INTO memory_frames (id, first_ts, last_ts, embedding, model, dim) "
        "VALUES ('other', ?, ?, ?, 'nomic-embed-text', 768)",
        (utc_text(NOW), utc_text(NOW), np.zeros(768, dtype="<f4").tobytes()),
    )
    conn.commit()
    conn.close()
    index = _index(env, {"x": _basis(0)})
    assert [h["id"] for h in index.search("x", limit=5)] == ["r1"]


def test_the_cursor_is_kept_in_worker_state(env):
    clock = Clock()
    writer = _writer(env, FakeEmbedder({b"red": _basis(0)}), clock)
    frame = add_frame(env, at=NOW - timedelta(minutes=1))
    writer.run()
    db = WorkerDB(env.db_path)
    try:
        state = json.loads(db.get_state(CURSOR_STATE_KEY))
    finally:
        db.close()
    assert state == {
        "rowid": writer.cursor(), "id": frame["id"], "ts": utc_text(NOW - timedelta(minutes=1)),
    }


def test_purged_older_events_do_not_move_the_cursor_back(env):
    clock = Clock()
    fake = FakeEmbedder({b"a": _basis(0), b"b": _basis(5), b"c": _basis(9)})
    writer = _writer(env, fake, clock)
    old = add_frame(env, at=NOW - timedelta(minutes=5), image=b"a")
    add_frame(env, at=NOW - timedelta(minutes=4), image=b"b")
    writer.run()
    conn = sqlite3.connect(env.db_path)
    conn.execute("DELETE FROM events WHERE id = ?", (old["id"],))
    conn.commit()
    conn.close()
    add_frame(env, at=NOW - timedelta(minutes=1), title="Other", image=b"c")
    assert writer.run()["embedded"] == 1
    assert len(fake.calls) == 3, "nothing embedded twice"


def test_with_annotation_off_memory_deletes_every_jpeg_it_passes(env):
    """No annotator will read the frames: Memory frees them, blocked ones
    included, and does not wait for a keyframe decision that never comes."""
    clock = Clock()
    cfg = replace(MemoryConfig(), enabled=True)
    writer = MemoryWriter(
        env.db_path, cfg, embed=FakeEmbedder({b"a": _basis(0), b"f": _basis(5)}),
        screenshots_dir=env.shots, data_dir=env.data_dir,
        privacy_checker=PrivacyZoneChecker(), annotation_enabled=False,
        now=lambda: clock.now, monotonic=lambda: clock.mono,
    )
    kept = add_frame(env, at=NOW - timedelta(minutes=3), image=b"a")
    blocked = add_frame(env, at=NOW - timedelta(minutes=2), app="1Password", image=b"x")
    focus = add_frame(env, at=NOW - timedelta(minutes=1), title="Rec", image=b"f", focus="s1")

    stats = writer.run()

    assert stats["embedded"] == 2 and stats["passed"] == 1
    assert [r["id"] for r in rows(env)] == [kept["id"], focus["id"]]
    for frame in (kept, blocked, focus):
        assert not frame["path"].exists()


@pytest.mark.parametrize(("value", "days"), [("3", 3), ("0", 0), ("-1", 7), ("true", 7)])
def test_thumbnail_days_is_a_whole_number_of_days_at_least_zero(tmp_path, value, days):
    path = tmp_path / "config.toml"
    path.write_text(f"[memory]\nthumbnail_days = {value}\n")
    assert load_config(path).memory.thumbnail_days == days
