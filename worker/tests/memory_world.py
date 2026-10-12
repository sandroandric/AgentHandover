"""A data directory where every store holds the frames a test adds.

``World.frame`` writes one captured, annotated frame everywhere
AgentHandover keeps what was on screen, the way the daemon and the
worker's passes leave it: the ``events`` row (window, OCR, URL,
annotation), its screenshot and encrypted thumbnail, an ``artifacts`` row
and file and a ``vlm_queue`` job, ``preserved_text`` and the vectors made
from it and from the annotation, the annotation cache row and, unless
told not to, a Memory row.  The other helpers add a Memory moment, task
segments, workflow observations, daily summaries and digests.  Shared by
the forget unit tests and J-forget.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from agenthandover_worker.activity_search import ActivitySearcher
from agenthandover_worker.annotation_cache import title_hash
from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.memory import utc_text
from agenthandover_worker.model_profiles import MEMORY_MODEL
from agenthandover_worker.vector_kb import VectorKB
from daemon_db import create_daemon_db

DIM = 256


def ts(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def axis(i: int) -> list[float]:
    """A unit vector of ``DIM`` values along axis *i*."""
    v = np.zeros(DIM, dtype=np.float32)
    v[i % DIM] = 1.0
    return [float(x) for x in v]


class World:
    """*data_dir* and *knowledge_dir* default to directories in *tmp_path*;
    the database is ``<data_dir>/events.db``."""

    def __init__(
        self, tmp_path: Path, *, data_dir: Path | None = None,
        knowledge_dir: Path | None = None,
    ) -> None:
        self.tmp = tmp_path
        self.data_dir = data_dir or tmp_path / "data"
        self.shots = self.data_dir / "screenshots"
        self.thumbs = self.data_dir / "thumbs"
        self.artifacts = self.data_dir / "artifacts"
        for d in (self.shots, self.thumbs, self.artifacts, self.data_dir / "status"):
            d.mkdir(parents=True, exist_ok=True)
        self.db = create_daemon_db(self.data_dir / "events.db")
        self.kb = KnowledgeBase(root=knowledge_dir or tmp_path / "knowledge", db_path=self.db)
        self.kb.ensure_structure()
        self.vkb = VectorKB(self.db)
        self._axis = 0

    def close(self) -> None:
        self.vkb.close()

    def sql(self, statement: str, params: tuple = ()) -> list[tuple]:
        conn = sqlite3.connect(self.db)
        try:
            rows = conn.execute(statement, params).fetchall()
            conn.commit()
            return rows
        finally:
            conn.close()

    def ids(self, statement: str, params: tuple = ()) -> set:
        return {row[0] for row in self.sql(statement, params)}

    def frame(
        self,
        eid: str,
        at: datetime,
        app: str,
        title: str,
        doing: str,
        *,
        memory: bool = True,
        url: str | None = None,
        vector: list[float] | None = None,
    ) -> None:
        """*vector* is the annotation's and the screen text's embedding
        (default: the next axis for each)."""
        shot = self.shots / f"{eid}.jpg"
        shot.write_bytes(b"jpeg " + eid.encode())
        (self.thumbs / f"{eid}.ahthumb").write_bytes(b"sealed " + eid.encode())
        artifact = self.artifacts / f"{eid}.bin"
        artifact.write_bytes(b"artifact")
        screen_text = f"{title}: {doing} — screen text of {eid}"
        meta = {"screenshot_path": str(shot), "ocr": {"full_text": screen_text}}
        if url:
            meta["url"] = url
        annotation = {"app": app, "location": title, "task_context": {"what_doing": doing}}
        self.sql(
            "INSERT INTO events (id, timestamp, kind_json, window_json, display_topology_json, "
            "primary_display_id, metadata_json, processed, annotation_status, "
            "scene_annotation_json) VALUES (?, ?, '{\"FocusChange\":{}}', ?, '[]', 'main', ?, 1, "
            "'completed', ?)",
            (
                eid, ts(at), json.dumps({"app_id": f"pid:7:{app}", "title": title}),
                json.dumps(meta), json.dumps(annotation),
            ),
        )
        self.sql(
            "INSERT INTO artifacts (id, event_id, artifact_type, file_path, original_size_bytes, "
            "stored_size_bytes) VALUES (?, ?, 'screenshot', ?, 8, 8)",
            (f"art-{eid}", eid, str(artifact)),
        )
        self.sql(
            "INSERT INTO vlm_queue (id, event_id, ttl_expires_at) VALUES (?, ?, ?)",
            (f"job-{eid}", eid, ts(at + timedelta(days=1))),
        )
        self.sql(
            "INSERT INTO preserved_text (event_id, source, timestamp, text_content, app) "
            "VALUES (?, 'ocr', ?, ?, ?)",
            (eid, ts(at), screen_text, f"pid:7:{app}"),
        )
        self.vkb.upsert("annotation", eid, doing, embedding=vector or self.next_axis())
        self.vkb.upsert("preserved_text", f"{eid}:ocr", screen_text,
                        embedding=vector or self.next_axis())
        self.sql(
            "INSERT OR REPLACE INTO annotation_cache (app, title_hash, ocr_simhash, "
            "annotation_json) VALUES (?, ?, 1, ?)",
            (app, title_hash(title), json.dumps(annotation)),
        )
        if memory:
            self.moment(eid, at, at, app, title, doing, thumb_id=eid)

    def next_axis(self) -> list[float]:
        self._axis += 1
        return axis(self._axis)

    def moment(
        self,
        mid: str,
        first: datetime,
        last: datetime,
        app: str,
        title: str,
        description: str,
        *,
        frames: int = 1,
        thumb_id: str | None = None,
        vector: list[float] | None = None,
    ) -> list[float]:
        vector = vector or self.next_axis()
        self.sql(
            "INSERT INTO memory_frames (id, first_ts, last_ts, frames, app, window_title, "
            "url_host, description, embedding, model, dim, thumb_id) "
            "VALUES (?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?)",
            (
                mid, utc_text(first), utc_text(last), frames, app, title, description,
                np.asarray(vector, dtype="<f4").tobytes(), MEMORY_MODEL, DIM, thumb_id,
            ),
        )
        return vector

    def segment(self, sid: str, event_ids: list[str], start: datetime) -> None:
        self.sql(
            "INSERT INTO task_segments (segment_id, cluster_id, task_label, event_ids_json, "
            "frame_count, start_time, end_time) VALUES (?, 0, ?, ?, ?, ?, ?)",
            (sid, f"label {sid}", json.dumps(event_ids), len(event_ids), ts(start), ts(start)),
        )

    def observation(self, oid: str, event_ids: list[str], started: datetime) -> None:
        self.sql(
            "INSERT INTO workflow_observations (obs_id, workflow_key, started_utc, "
            "started_local_date, started_local_weekday, started_local_minute, duration_s, "
            "event_ids_json) VALUES (?, 'k', ?, ?, 0, 0, 60, ?)",
            (oid, ts(started), started.date().isoformat(), json.dumps(event_ids)),
        )

    def daily(self, day: str, text: str) -> None:
        self.kb.save_daily_summary(day, {"tasks": [text], "top_apps": [text]})
        digests = self.kb.root / "observations" / "digests"
        digests.mkdir(parents=True, exist_ok=True)
        (digests / f"{day}.json").write_text(json.dumps({"date": day, "text": text}))
        self.vkb.upsert("daily_summary", day, f"Daily summary for {day}: {text}",
                        embedding=self.next_axis())
        recent = self.kb.get_context("recent").get("last_7_days", [])
        self.kb.update_context("recent", {"last_7_days": [*recent, {"date": day}]})

    def index_activity(self) -> int:
        with ActivitySearcher(self.db) as searcher:
            return searcher.refresh_index()

    # -- what is left ---------------------------------------------------

    def files(self, directory: Path) -> set[str]:
        return {p.stem for p in directory.iterdir()} if directory.is_dir() else set()

    def daily_days(self) -> set[str]:
        return self.files(self.kb.root / "observations" / "daily")

    def digest_days(self) -> set[str]:
        return self.files(self.kb.root / "observations" / "digests")

    def recent_days(self) -> set[str]:
        return {d["date"] for d in self.kb.get_context("recent").get("last_7_days", [])}

    def vector_sources(self, source_type: str) -> set[str]:
        return self.ids("SELECT source_id FROM vector_store WHERE source_type = ?", (source_type,))

    def fts_events(self) -> set[str]:
        return self.ids("SELECT event_id FROM events_fts") | self.ids(
            "SELECT event_id FROM events_fts_ids"
        )


NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
