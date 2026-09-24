"""Session fixtures: daemon event rows loaded into a database built by the
daemon's own migrations.

A fixture is a directory under ``fixtures/``:

``manifest.json``
    What the rows are (``source``), the focus session they belong to,
    the recorded time range, and the row counts a loaded database must
    show (``row_counts``, checked by ``test_fixtures_load.py``).
``events.jsonl.gz``
    One JSON object per ``events`` row, keyed by the columns the daemon's
    ``EventStore::insert_event`` writes.  The JSON columns (``kind_json``,
    ``window_json``, ``metadata_json``, ...) hold the exact strings the
    daemon stores.  A ``screenshot_path`` in ``metadata_json`` starts with
    ``{screenshots_dir}``; loading points it at the test's screenshots
    directory and writes a 1x1 placeholder JPEG there (the cassette carries
    the model answers, so frame pixels are not needed).

``annotations.jsonl.gz`` (optional)
    What the worker's scene annotator wrote for some rows: ``id``,
    ``annotation_status`` and ``scene_annotation_json``.  A passive fixture
    carries the annotated stream this way (its frames have no pixels to
    annotate); loading applies it to the daemon rows.
``labels.json`` (optional)
    Ground truth for scoring a replay of the fixture.

``load_fixture`` adds to a database already in the data dir (several runs
of one routine share it, as they do in the daemon's), and optionally moves
the whole recording in time (whole seconds, so each timestamp keeps the
daemon's exact text format) so that code reading the wall clock sees a
recent session.
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from daemon_db import create_daemon_db

FIXTURES_DIR = Path(__file__).parent / "fixtures"
PLACEHOLDER_JPEG = FIXTURES_DIR / "placeholder_1x1.jpg"
SCREENSHOTS_TOKEN = "{screenshots_dir}"
EVENTS_FILE = "events.jsonl.gz"
ANNOTATIONS_FILE = "annotations.jsonl.gz"
LABELS_FILE = "labels.json"

# The columns EventStore::insert_event writes (crates/storage/src/lib.rs).
# Everything else (processed, annotation_status, created_at, ...) takes the
# schema default, as it does for a row the daemon inserts.
DAEMON_COLUMNS = (
    "id",
    "timestamp",
    "kind_json",
    "window_json",
    "display_topology_json",
    "primary_display_id",
    "cursor_x",
    "cursor_y",
    "ui_scale",
    "artifact_ids_json",
    "metadata_json",
    "display_ids_spanned_json",
    "focused_element_json",
)

_ISO_SECONDS = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(.*)$")


def shift_timestamp(text: str, shift: timedelta) -> str:
    """Move an ISO-8601 timestamp by whole seconds, keeping its fraction and
    offset text exactly as written (``.123456+00:00``, ``.123Z``, ...)."""
    if not shift:
        return text
    match = _ISO_SECONDS.match(text)
    if match is None:
        raise ValueError(f"not an ISO-8601 timestamp: {text!r}")
    moved = datetime.fromisoformat(match.group(1)) + shift
    return moved.strftime("%Y-%m-%dT%H:%M:%S") + match.group(2)


def parse_timestamp(text: str) -> datetime:
    """Parse a daemon or extension timestamp into an aware UTC datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def read_manifest(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name / "manifest.json").read_text(encoding="utf-8"))


def read_rows(name: str) -> list[dict]:
    with gzip.open(FIXTURES_DIR / name / EVENTS_FILE, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_annotations(name: str) -> list[dict]:
    path = FIXTURES_DIR / name / ANNOTATIONS_FILE
    if not path.is_file():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_labels(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name / LABELS_FILE).read_text(encoding="utf-8"))


def write_rows(fixture_dir: Path, rows: list[dict]) -> None:
    """Write rows as ``events.jsonl.gz`` byte-reproducibly (gzip mtime 0)."""
    payload = "".join(
        json.dumps({col: row.get(col) for col in DAEMON_COLUMNS}, ensure_ascii=False) + "\n"
        for row in rows
    ).encode("utf-8")
    with open(fixture_dir / EVENTS_FILE, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(payload)


def row_counts(rows: list[dict]) -> dict:
    """The ``row_counts`` a fixture's manifest records for its rows."""
    kinds: dict[str, int] = {}
    counts = dict.fromkeys((
        "chrome_extension", "clipboard_with_preview", "dom_snapshots",
        "focus_tagged", "with_ocr_text", "with_screenshot",
    ), 0)
    for row in rows:
        kind = json.loads(row["kind_json"])
        kinds[kind["type"]] = kinds.get(kind["type"], 0) + 1
        metadata = json.loads(row["metadata_json"])
        window = json.loads(row["window_json"]) if row["window_json"] else {}
        ocr = metadata.get("ocr")
        counts["chrome_extension"] += window.get("app_id") == "com.google.Chrome"
        counts["clipboard_with_preview"] += bool(kind.get("content_preview"))
        counts["dom_snapshots"] += any(isinstance(metadata.get(k), list) for k in ("anchors", "nodes"))
        counts["focus_tagged"] += "focus_session_id" in metadata
        counts["with_ocr_text"] += isinstance(ocr, dict) and bool(ocr.get("full_text"))
        counts["with_screenshot"] += "screenshot_path" in metadata
    return {"events": len(rows), "by_kind": dict(sorted(kinds.items())), **counts}


def write_manifest(fixture_dir: Path, manifest: dict) -> None:
    (fixture_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )


def _dump_metadata(metadata: dict) -> str:
    # serde_json without preserve_order: sorted keys, compact separators.
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass
class LoadedFixture:
    name: str
    manifest: dict
    db_path: Path
    screenshots_dir: Path
    shift: timedelta

    @property
    def focus_session(self) -> dict | None:
        """The fixture's focus session with its times moved by ``shift``."""
        session = self.manifest.get("focus_session")
        if not session:
            return None
        moved = dict(session)
        for key in ("started_at", "stopped_at"):
            if moved.get(key):
                moved[key] = shift_timestamp(moved[key], self.shift)
        return moved

    def write_focus_signal(self, state_dir: Path, status: str = "stopped") -> Path:
        """Write ``focus-session.json`` as the app does (crates/common/src/focus_session.rs)."""
        session = self.focus_session
        if session is None:
            raise ValueError(f"fixture {self.name} has no focus session")
        path = state_dir / "focus-session.json"
        state_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "session_id": session["session_id"],
            "title": session["title"],
            "started_at": session["started_at"],
            "status": status,
        }), encoding="utf-8")
        return path


def load_fixture(
    name: str,
    data_dir: Path,
    *,
    rebase_to: datetime | None = None,
) -> LoadedFixture:
    """Load the fixture's rows into ``data_dir/events.db``, creating it
    through the daemon's migrations when it does not exist yet.  Loading
    several fixtures into one data dir keeps all their rows, as the
    daemon's database keeps every day it recorded.

    ``rebase_to`` moves every timestamp so the recording ends at that time
    (rounded to whole seconds).
    """
    manifest = read_manifest(name)
    shift = timedelta(0)
    if rebase_to is not None:
        recorded_end = parse_timestamp(manifest["recorded_end"])
        shift = timedelta(seconds=round((rebase_to - recorded_end).total_seconds()))

    data_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir = data_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)
    db_path = data_dir / "events.db"
    if not db_path.exists():
        create_daemon_db(db_path)

    rows = []
    for row in read_rows(name):
        row = dict(row)
        row["timestamp"] = shift_timestamp(row["timestamp"], shift)
        metadata = json.loads(row["metadata_json"])
        changed = False
        shot = metadata.get("screenshot_path")
        if isinstance(shot, str) and shot.startswith(SCREENSHOTS_TOKEN):
            target = screenshots_dir / shot[len(SCREENSHOTS_TOKEN):].lstrip("/")
            shutil.copyfile(PLACEHOLDER_JPEG, target)
            metadata["screenshot_path"] = str(target)
            changed = True
        # Extension payloads carry the page's own ISO timestamp.
        if shift and isinstance(metadata.get("timestamp"), str):
            metadata["timestamp"] = shift_timestamp(metadata["timestamp"], shift)
            changed = True
        if changed:
            row["metadata_json"] = _dump_metadata(metadata)
        rows.append(tuple(row.get(col) for col in DAEMON_COLUMNS))

    conn = sqlite3.connect(str(db_path))
    try:
        conn.executemany(
            f"INSERT INTO events ({', '.join(DAEMON_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in DAEMON_COLUMNS)})",
            rows,
        )
        conn.executemany(
            "UPDATE events SET annotation_status = ?, scene_annotation_json = ? WHERE id = ?",
            [
                (ann["annotation_status"], ann["scene_annotation_json"], ann["id"])
                for ann in read_annotations(name)
            ],
        )
        conn.commit()
    finally:
        conn.close()

    return LoadedFixture(
        name=name,
        manifest=manifest,
        db_path=db_path,
        screenshots_dir=screenshots_dir,
        shift=shift,
    )
