"""Capture a real focus recording from a daemon database as an anonymised
journey fixture (same format as ``generate_focus_dailynews.py`` writes).

Procedure (on the test Mac, AgentHandover installed and running, Chrome
with the extension connected):

1. Start a focus recording from the menu bar app titled "Daily news digest"
   and do the routine for about ten minutes: read Hacker News in Chrome,
   open three stories, copy each title and link, paste them into a Notes
   note with a one-line summary, copy the finished digest, and post it in a
   Slack channel open in another Chrome tab.  Stop the recording and note
   the time (``date -u +%Y-%m-%dT%H:%M:%SZ``): Chrome and clipboard rows are
   not tagged with the session, so the stop time bounds them.
2. Find the session id (``focus-session.json`` while it is still there, or
   the newest ``focus_session_id`` in the database)::

       sqlite3 -readonly ~/Library/Application\\ Support/agenthandover/events.db \\
         "SELECT DISTINCT json_extract(metadata_json,'$.focus_session_id') \\
          FROM events WHERE json_extract(metadata_json,'$.focus_session_id') \\
          IS NOT NULL ORDER BY timestamp DESC LIMIT 1"

3. From a checkout of this repository::

       python3 worker/tests/journeys/record/record_focus_session.py \\
         --db ~/Library/Application\\ Support/agenthandover/events.db \\
         --session-id <id> --title "Daily news digest" --until <stop time> \\
         --replace "<Your Name>=Alex Example" --replace "<your-company>=Example Co"

   ``--replace OLD=NEW`` (repeatable) rewrites every occurrence in every
   string the rows carry (window titles, OCR text, DOM text, URLs,
   clipboard previews).  Email addresses become ``personN@example.com``;
   home-directory paths become ``/Users/user``.  Screenshots are not
   copied: the fixture points each frame at a 1x1 placeholder.
4. Read the review the script prints (window titles, URLs, apps, the
   strings it rewrote) and grep the output for anything personal it left:
   ``gzip -dc .../events.jsonl.gz | grep -i <word>``.  Re-run with more
   ``--replace`` pairs until clean.
5. Check the rows are still exactly what the daemon writes, and that they
   load::

       cargo test -p agenthandover-storage --test fixture_rows_test
       python -m pytest worker/tests/journeys/test_fixtures_load.py

The database is read through SQLite's backup API from a read-only
connection, so the running daemon is not disturbed.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

JOURNEYS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JOURNEYS_DIR.parent))  # worker/tests, for daemon_db

from daemon_db import schema_version  # noqa: E402
from journeys.fixture_loader import (  # noqa: E402
    DAEMON_COLUMNS,
    FIXTURES_DIR,
    SCREENSHOTS_TOKEN,
    row_counts,
    write_manifest,
    write_rows,
)

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
HOME_PATH = re.compile(r"/Users/[^/\s\"']+")
JSON_COLUMNS = ("kind_json", "window_json", "metadata_json")


def _utc_text(value: str) -> str:
    """An ISO 8601 time as the daemon writes timestamps, so SQL text
    comparison against ``events.timestamp`` is exact."""
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise argparse.ArgumentTypeError(f"{value!r} needs a time zone (e.g. Z)")
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def snapshot_db(db_path: Path, into: Path) -> Path:
    """Copy the live database (WAL included) without writing to it."""
    src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    dst = sqlite3.connect(str(into))
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()
    return into


def session_rows(db_path: Path, session_id: str, until: str | None = None) -> list[dict]:
    """Rows tagged with the session plus every untagged row the daemon wrote
    while it ran (Chrome extension and clipboard events are not tagged):
    from the first tagged row to ``until`` (default: the last tagged row)."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        first, last = conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM events "
            "WHERE json_extract(metadata_json, '$.focus_session_id') = ?",
            (session_id,),
        ).fetchone()
        if first is None:
            raise SystemExit(f"no events tagged with focus_session_id {session_id}")
        last = max(last, until) if until else last
        rows = conn.execute(
            f"SELECT {', '.join(DAEMON_COLUMNS)} FROM events "
            "WHERE json_extract(metadata_json, '$.focus_session_id') = ? "
            "   OR (json_extract(metadata_json, '$.focus_session_id') IS NULL "
            "       AND timestamp >= ? AND timestamp <= ?) "
            "ORDER BY timestamp ASC, rowid ASC",
            (session_id, first, last),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


class Anonymiser:
    def __init__(self, replacements: list[tuple[str, str]]) -> None:
        # Longest first, so "Jane Doe" is replaced before "Jane".
        self.replacements = sorted(replacements, key=lambda p: -len(p[0]))
        self.emails: dict[str, str] = {}
        self.hits: Counter[str] = Counter()

    def text(self, value: str) -> str:
        out = value
        for old, new in self.replacements:
            if old in out:
                self.hits[old] += out.count(old)
                out = out.replace(old, new)
        out = EMAIL.sub(self._email, out)
        out = HOME_PATH.sub(self._home, out)
        return out

    def _email(self, match: re.Match) -> str:
        address = match.group(0)
        if address.endswith("@example.com"):
            return address
        if address not in self.emails:
            self.emails[address] = f"person{len(self.emails) + 1}@example.com"
        self.hits["<email>"] += 1
        return self.emails[address]

    def _home(self, match: re.Match) -> str:
        if match.group(0) != "/Users/user":
            self.hits["<home path>"] += 1
        return "/Users/user"

    def value(self, value):
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.value(v) for v in value]
        if isinstance(value, dict):
            return {k: self.value(v) for k, v in value.items()}
        return value


def anonymise_row(row: dict, anon: Anonymiser) -> dict:
    out = dict(row)
    for col in JSON_COLUMNS:
        if out[col] is None:
            continue
        parsed = json.loads(out[col])
        if col == "metadata_json":
            shot = parsed.get("screenshot_path")
            if isinstance(shot, str) and shot:
                parsed["screenshot_path"] = f"{SCREENSHOTS_TOKEN}/{Path(shot).name}"
        cleaned = anon.value(parsed)
        if cleaned != json.loads(out[col]):
            # serde_json text: compact; metadata keys sorted (no preserve_order),
            # kind/window keep their struct field order.
            out[col] = json.dumps(
                cleaned, separators=(",", ":"), ensure_ascii=False,
                sort_keys=(col == "metadata_json"),
            )
    return out


def review(rows: list[dict], anon: Anonymiser) -> None:
    apps: Counter[str] = Counter()
    titles: Counter[str] = Counter()
    urls: Counter[str] = Counter()
    for row in rows:
        if row["window_json"]:
            window = json.loads(row["window_json"])
            apps[window["app_id"]] += 1
            titles[window["title"]] += 1
        url = json.loads(row["metadata_json"]).get("url")
        if url:
            urls[url] += 1
    print("\napps:", *(f"  {n:4d}  {a}" for a, n in apps.most_common()), sep="\n")
    print("\nwindow titles:", *(f"  {n:4d}  {t}" for t, n in titles.most_common()), sep="\n")
    print("\nurls:", *(f"  {n:4d}  {u}" for u, n in urls.most_common()), sep="\n")
    print("\nrewritten:", *(f"  {n:4d}  {k}" for k, n in anon.hits.most_common()), sep="\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, required=True, help="the daemon's events.db")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--title", required=True, help="the focus session title")
    parser.add_argument(
        "--until", type=_utc_text,
        help="when the recording was stopped (ISO 8601); default: the last tagged row",
    )
    parser.add_argument("--name", default="focus_dailynews", help="fixture name")
    parser.add_argument("--description", default="", help="what the recording shows")
    parser.add_argument(
        "--fixtures-dir", type=Path, default=FIXTURES_DIR,
        help="where fixture directories live (default: the journeys' fixtures)",
    )
    parser.add_argument(
        "--replace", action="append", default=[], metavar="OLD=NEW",
        help="rewrite OLD to NEW in every string (repeatable)",
    )
    args = parser.parse_args(argv)

    pairs = []
    for item in args.replace:
        old, sep, new = item.partition("=")
        if not sep or not old:
            parser.error(f"--replace expects OLD=NEW, got {item!r}")
        pairs.append((old, new))

    with tempfile.TemporaryDirectory() as tmp:
        copy = snapshot_db(args.db.expanduser(), Path(tmp) / "events.db")
        rows = session_rows(copy, args.session_id, args.until)

    anon = Anonymiser(pairs)
    rows = [anonymise_row(row, anon) for row in rows]
    tagged = [
        row["timestamp"] for row in rows
        if json.loads(row["metadata_json"]).get("focus_session_id") == args.session_id
    ]

    out = args.fixtures_dir / args.name
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out, rows)
    write_manifest(out, {
        "name": args.name,
        "description": args.description or f"Focus recording: {args.title}",
        "source": {
            "kind": "recorded",
            "by": "worker/tests/journeys/record/record_focus_session.py",
            "note": "Real daemon rows, anonymised; screenshots replaced by a 1x1 placeholder.",
        },
        "schema_version": schema_version(),
        "recorded_start": rows[0]["timestamp"],
        "recorded_end": rows[-1]["timestamp"],
        "focus_session": {
            "session_id": args.session_id,
            "title": anon.text(args.title),
            "started_at": tagged[0],
            "stopped_at": args.until or tagged[-1],
        },
        "row_counts": row_counts(rows),
    })
    print(f"wrote {len(rows)} rows to {out}")
    review(rows, anon)


if __name__ == "__main__":
    main()
