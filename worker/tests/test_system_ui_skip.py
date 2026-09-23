"""Regression pins for WP-F3: system UI frames are not annotated.

Real-life finding (test Mac): notification banners, Control Center,
Spotlight and the Dock were captured, annotated and cost a model call
each while carrying no workflow.  Their frames are now marked ``skipped``
with reason ``system_ui`` before any model call; the rows stay, so their
window titles are still there for segmentation.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agenthandover_worker import main as wm
from agenthandover_worker.db import WorkerDB
from agenthandover_worker.privacy_zones import (
    SYSTEM_UI_APPS,
    ObservationTier,
    PrivacyZoneChecker,
    is_system_ui,
)
from agenthandover_worker.scene_annotator import AnnotationConfig, SceneAnnotator
from daemon_db import create_daemon_db


def _event(app_id: str, *, title: str = "", shot: Path | None = None, eid: str = "e1") -> dict:
    meta = {"screenshot_path": str(shot)} if shot else {}
    return {
        "id": eid,
        "timestamp": "2026-09-23T10:00:00.000Z",
        "window_json": json.dumps({"app_id": app_id, "title": title}),
        "metadata_json": json.dumps(meta),
    }


@pytest.mark.parametrize("app_id", [
    "pid:412:UserNotificationCenter",
    "pid:388:ControlCenter",
    "pid:390:NotificationCenter",
    "pid:501:Spotlight",
    "pid:377:Dock",
    "pid:380:SystemUIServer",
    "com.apple.controlcenter",
    "COM.APPLE.DOCK",
])
def test_system_processes_are_system_ui(app_id):
    assert is_system_ui(_event(app_id))


@pytest.mark.parametrize("app_id", [
    "pid:812:Google Chrome", "com.google.Chrome", "pid:1377:Notes", "pid:9:Finder",
    "pid:10:Docker",  # a prefix of a system name is not a match
    "",
])
def test_apps_are_not_system_ui(app_id):
    assert not is_system_ui(_event(app_id, title="Dock"))


def test_system_ui_is_skipped_not_blocked():
    """Blocking is for privacy (the frame is dropped); system UI is only
    not annotated.  The two lists do not overlap."""
    checker = PrivacyZoneChecker()
    for app in ("UserNotificationCenter", "ControlCenter", "Spotlight", "Dock"):
        assert checker.check_app("", app) == ObservationTier.FULL
    assert not SYSTEM_UI_APPS & {a.casefold() for a in PrivacyZoneChecker._DEFAULT_BLOCKED}


def test_the_annotator_makes_no_model_call(tmp_path):
    shot = tmp_path / "cc.jpg"
    shot.write_bytes(b"\xff\xd8\xff")
    annotator = SceneAnnotator(AnnotationConfig())
    urlopen = MagicMock(side_effect=AssertionError("model called"))
    with patch("urllib.request.urlopen", urlopen):
        result = annotator.annotate_event(_event("pid:388:ControlCenter", shot=shot))
    assert result.status == "skipped"
    assert result.annotation == {"skip_reason": "system_ui"}
    assert urlopen.call_count == 0
    assert not shot.exists()  # never annotated, so the JPEG goes now
    assert annotator.stats["system_ui"] == 1


def test_the_passive_loop_marks_the_row_skipped_with_the_reason(tmp_path):
    db_path = create_daemon_db(tmp_path / "events.db")
    now = datetime.now(timezone.utc)
    conn = sqlite3.connect(str(db_path))
    for eid, app, age in (("banner", "pid:412:UserNotificationCenter", 2), ("dock", "pid:377:Dock", 1)):
        shot = tmp_path / f"{eid}.jpg"
        shot.write_bytes(b"\xff\xd8\xff")
        conn.execute(
            "INSERT INTO events (id, timestamp, kind_json, window_json, "
            "display_topology_json, primary_display_id, artifact_ids_json, metadata_json) "
            "VALUES (?, ?, '{\"type\":\"DwellSnapshot\"}', ?, '[]', 'main', '[]', ?)",
            (
                eid,
                (now - timedelta(minutes=age)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                json.dumps({"app_id": app, "title": "Notification"}),
                json.dumps({"screenshot_path": str(shot)}),
            ),
        )
    conn.commit()
    conn.close()

    annotator = SceneAnnotator(AnnotationConfig())
    urlopen = MagicMock(side_effect=AssertionError("model called"))
    with WorkerDB(db_path) as db, patch.object(annotator, "ready", return_value=True), \
            patch("urllib.request.urlopen", urlopen):
        stats = wm._process_annotations(db, annotator, tmp_path)

    assert stats["skipped"] == 2 and stats["annotated"] == 0
    assert urlopen.call_count == 0
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT annotation_status, scene_annotation_json, window_json FROM events"
    ).fetchall()
    conn.close()
    assert len(rows) == 2  # the rows stay, window titles and all
    for status, annotation, window in rows:
        assert status == "skipped"
        assert json.loads(annotation) == {"skip_reason": "system_ui"}
        assert json.loads(window)["title"] == "Notification"
