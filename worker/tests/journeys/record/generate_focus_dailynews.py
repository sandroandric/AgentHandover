"""Generate the ``focus_dailynews`` fixture: a 10-minute focus recording of a
daily-news routine, in the exact row format the daemon writes.

This is the stand-in until a real session is recorded on the test Mac with
``record_focus_session.py`` (same output format, ``source.kind`` becomes
``recorded``).  The routine: read Hacker News in Chrome, open three
stories, copy each title and link, paste it into a Notes note with a
one-line summary, copy the finished digest, and post it in a Slack
channel open in another Chrome tab.

Every row follows the code that writes it:

* native observer events (``AppSwitch``, ``WindowTitleChange`` and the
  focus-capture ``DwellSnapshot`` frames) carry ``window_json`` from the
  app's ``ObservationService`` (``app_id`` is ``pid:<pid>:<owner name>``),
  the display topology, cursor and ``ui_scale``, and ``focus_session_id``
  in ``metadata_json`` (``crates/daemon/src/observer/event_loop.rs``);
* focus frames are taken every 2 s but at most 20 per minute
  (``screenshot_max_per_minute``), each with ``screenshot_path``,
  ``focus_capture: true`` and the Vision OCR result
  (``elements`` / ``full_text`` joined by newlines / ``processing_time_ms``);
* Chrome extension events (``content_ready`` -> ``FocusChange``,
  ``click_intent`` -> ``ClickIntent``, ``dwell_snapshot`` / ``dom_snapshot``
  -> ``DwellSnapshot``, ``scroll_snapshot`` -> ``ScrollReadSnapshot``) carry
  ``app_id`` ``com.google.Chrome``, ``window_id`` ``chrome-tab-<id>``, an
  empty display topology, the payload as ``metadata_json``, and no
  ``focus_session_id`` (``crates/daemon/src/ipc/native_messaging.rs``);
  during focus the daemon asks the last active tab for a DOM snapshot
  every 2 s;
* ``ClipboardChange`` rows have no window, ``primary_display_id``
  ``unknown``, a sha256 ``content_hash`` of the copied bytes and a
  ``content_preview`` of up to 500 characters (``crates/daemon/src/main.rs``);
* timestamps use chrono's ``to_rfc3339()`` text; JSON objects in
  ``metadata_json`` have sorted keys (serde_json without ``preserve_order``).

Usage (from the repository root)::

    .venv/bin/python worker/tests/journeys/record/generate_focus_dailynews.py

The output is deterministic: running it twice gives identical files.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

JOURNEYS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JOURNEYS_DIR.parent))  # worker/tests, for daemon_db

from journeys.fixture_loader import (  # noqa: E402
    FIXTURES_DIR,
    SCREENSHOTS_TOKEN,
    write_rows,
)

NAME = "focus_dailynews"
SEED = 20260915
START = datetime(2026, 9, 15, 8, 30, 0, tzinfo=timezone.utc)
SESSION_TITLE = "Daily news digest"

DISPLAY = {
    "display_id": "1",
    "bounds_global_px": [0, 0, 1512, 982],
    "scale_factor": 2.0,
    "orientation": 0,
}
CHROME_APP = "pid:812:Google Chrome"
NOTES_APP = "pid:1377:Notes"
CHROME_WINDOW = {"window_id": "4127", "bounds": [0, 33, 1512, 949]}
NOTES_WINDOW = {"window_id": "5310", "bounds": [160, 90, 1020, 670]}
NEWS_TAB = 1843
SLACK_TAB = 1851

FRONT_PAGE = {"url": "https://news.ycombinator.com/", "title": "Hacker News"}
STORIES = [
    {
        "rank": 3,
        "title": "SQLite's WAL mode, explained with diagrams",
        "url": "https://blog.example.org/sqlite-wal-explained",
        "points": 412,
        "summary": "Readers keep going while one writer appends; checkpoints fold the log back.",
        "body": [
            "Write-ahead logging keeps a separate log of changed pages.",
            "Readers see the database as of the last commit they started with.",
            "A checkpoint copies committed pages back into the main file.",
        ],
    },
    {
        "rank": 7,
        "title": "A field guide to on-device vision models",
        "url": "https://research.example.com/on-device-vision-guide",
        "points": 288,
        "summary": "Small quantised models now read screenshots well enough for UI tasks.",
        "body": [
            "Quantisation-aware training closes most of the accuracy gap.",
            "Context length, not parameter count, limits screenshot reading.",
            "Latency on a laptop GPU is now under two seconds per frame.",
        ],
    },
    {
        "rank": 12,
        "title": "Show HN: A tiny HTTP cassette recorder for tests",
        "url": "https://github.com/example/cassette",
        "points": 156,
        "summary": "Records real API answers once and replays them byte for byte.",
        "body": [
            "Record mode proxies to the real service and appends answers.",
            "Replay mode fails the test on any request it has not seen.",
            "Keys hash the request body, so option order does not matter.",
        ],
    },
]
OTHER_HEADLINES = [
    (1, "The quiet death of the RSS reader, and its return", 530),
    (2, "Designing a scheduler that respects laptop batteries", 498),
    (4, "Ask HN: How do you keep a daily reading habit?", 377),
    (5, "Postmortem: a leap second in our job queue", 341),
    (6, "Typst 1.0 released", 310),
    (8, "Why most dashboards are ignored", 250),
    (9, "An illustrated guide to CRDTs", 233),
    (10, "The economics of small open-source libraries", 201),
    (11, "Reverse-engineering a 1990s pager protocol", 177),
]
SLACK_PAGE = {
    "url": "https://app.slack.com/client/T0EXAMPLE/C0DAILYNEWS",
    "title": "daily-news (Channel) - Example Co - Slack",
}

# Timeline (seconds from START).  Story i is opened at CYCLE_START[i].
CYCLE_START = [40.6, 190.6, 340.6]
READ_S = 40.0          # reading the story before copying title + link
NOTES_S = 62.0         # pasting and writing the summary in Notes
DIGEST_COPY = 452.0    # select-all + copy in Notes
TO_CHROME_FINAL = 456.2
SLACK_TAB_SWITCH = 458.4
SLACK_CLICK_BOX = 464.1
SLACK_CLICK_SEND = 482.7
STOPPED_AT = 600.0
MINUTE_RESET_PHASE = 17.0  # the daemon's per-minute screenshot counter reset


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def rfc3339(ts: datetime) -> str:
    """chrono ``DateTime<Utc>::to_rfc3339()``: fraction digits 0/3/6/9 (AutoSi)."""
    base = ts.strftime("%Y-%m-%dT%H:%M:%S")
    micros = ts.microsecond
    if micros == 0:
        frac = ""
    elif micros % 1000 == 0:
        frac = f".{micros // 1000:03d}"
    else:
        frac = f".{micros:06d}"
    return f"{base}{frac}+00:00"


def js_iso(ts: datetime) -> str:
    """JavaScript ``Date.prototype.toISOString()``."""
    return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"


def compact(value) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def metadata_text(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class Generator:
    def __init__(self) -> None:
        self.rng = random.Random(SEED)
        self.session_id = str(self._uuid())
        self.rows: list[tuple[float, dict]] = []

    def _uuid(self) -> uuid.UUID:
        return uuid.UUID(int=self.rng.getrandbits(128), version=4)

    def _time(self, offset: float) -> datetime:
        return START + timedelta(seconds=offset, microseconds=self.rng.randrange(1, 999))

    def _cursor(self, window: dict) -> tuple[int, int]:
        x, y, w, h = window["bounds"]
        return (x + self.rng.randrange(40, w - 40), y + self.rng.randrange(60, h - 40))

    def _add(self, offset: float, row: dict) -> None:
        self.rows.append((offset, row))

    # -- native observer (focus-tagged) -------------------------------------

    def native(self, offset: float, kind: dict, app_id: str, title: str,
               metadata: dict | None = None) -> None:
        window = CHROME_WINDOW if app_id == CHROME_APP else NOTES_WINDOW
        cx, cy = self._cursor(window)
        meta = dict(metadata or {})
        meta["focus_session_id"] = self.session_id
        self._add(offset, {
            "id": str(self._uuid()),
            "timestamp": rfc3339(self._time(offset)),
            "kind_json": compact(kind),
            "window_json": compact({
                "window_id": window["window_id"],
                "app_id": app_id,
                "title": title,
                "bounds_global_px": window["bounds"],
                "z_order": 0,
                "is_fullscreen": False,
            }),
            "display_topology_json": compact([DISPLAY]),
            "primary_display_id": DISPLAY["display_id"],
            "cursor_x": cx,
            "cursor_y": cy,
            "ui_scale": DISPLAY["scale_factor"],
            "artifact_ids_json": "[]",
            "metadata_json": metadata_text(meta),
            "display_ids_spanned_json": None,
        })

    def focus_frame(self, offset: float, app_id: str, title: str, lines: list[str]) -> None:
        elements = []
        top = 0.06
        for text in lines:
            height = 0.018
            elements.append({
                "text": text,
                "confidence": 1.0 if self.rng.random() < 0.85 else 0.5,
                "bbox_normalized": [
                    round(0.04 + self.rng.random() * 0.02, 4),
                    round(top, 4),
                    round(min(0.9, 0.012 * len(text) + 0.05), 4),
                    height,
                ],
            })
            top += 0.028
        self.native(offset, {"type": "DwellSnapshot"}, app_id, title, {
            "focus_capture": True,
            "ocr": {
                "elements": elements,
                "full_text": "\n".join(lines),
                "processing_time_ms": self.rng.randrange(180, 420),
            },
            "screenshot_path": f"{SCREENSHOTS_TOKEN}/{self._uuid()}.jpg",
        })

    # -- Chrome extension (not focus-tagged) --------------------------------

    def extension(self, offset: float, kind: dict, tab_id: int, url: str,
                  title: str | None, payload: dict) -> None:
        metadata = {"tabId": tab_id, "url": url, **payload}
        self._add(offset, {
            "id": str(self._uuid()),
            "timestamp": rfc3339(self._time(offset)),
            "kind_json": compact(kind),
            "window_json": compact({
                "window_id": f"chrome-tab-{tab_id}",
                "app_id": "com.google.Chrome",
                "title": title if title is not None else url,
                "bounds_global_px": [0, 0, 0, 0],
                "z_order": 0,
                "is_fullscreen": False,
            }),
            "display_topology_json": "[]",
            "primary_display_id": "",
            "cursor_x": None,
            "cursor_y": None,
            "ui_scale": None,
            "artifact_ids_json": "[]",
            "metadata_json": metadata_text(metadata),
            "display_ids_spanned_json": None,
        })

    def click(self, offset: float, tab_id: int, url: str, target: dict) -> None:
        label = target.get("ariaLabel") or target.get("innerText") or ""
        when = self._time(offset)
        self.extension(
            offset,
            {"type": "ClickIntent", "target_description": f"{label}<{target['tagName']}>"},
            tab_id, url, None,
            {
                "x": self.rng.randrange(80, 900),
                "y": self.rng.randrange(80, 700),
                "target": target,
                "timestamp": js_iso(when),
            },
        )

    # -- clipboard ------------------------------------------------------------

    def clipboard(self, offset: float, text: str) -> None:
        data = text.encode("utf-8")
        self._add(offset, {
            "id": str(self._uuid()),
            "timestamp": rfc3339(self._time(offset)),
            "kind_json": compact({
                "type": "ClipboardChange",
                "content_types": ["public.utf8-plain-text", "public.html"],
                "byte_size": len(data),
                "high_entropy": False,
                "content_hash": hashlib.sha256(data).hexdigest(),
                "content_preview": text.strip()[:500],  # chars().take(500)
            }),
            "window_json": None,
            "display_topology_json": "[]",
            "primary_display_id": "unknown",
            "cursor_x": None,
            "cursor_y": None,
            "ui_scale": None,
            "artifact_ids_json": "[]",
            "metadata_json": "{}",
            "display_ids_spanned_json": None,
        })


# ---------------------------------------------------------------------------
# Screen content
# ---------------------------------------------------------------------------


def front_page_headlines() -> list[tuple[int, str, int, str]]:
    rows = [(r, t, p, "") for r, t, p in OTHER_HEADLINES]
    rows += [(s["rank"], s["title"], s["points"], s["url"]) for s in STORIES]
    return sorted(rows)


def front_page_ocr() -> list[str]:
    lines = ["Hacker News  new | past | comments | ask | show | jobs | submit", "login"]
    for rank, title, points, _ in front_page_headlines():
        lines.append(f"{rank}. {title}")
        lines.append(f"{points} points | {40 + rank * 7} comments")
    return lines


def story_ocr(story: dict) -> list[str]:
    host = story["url"].split("/")[2]
    return [story["url"], story["title"], host, *story["body"]]


def note_lines(stories_done: int, typing_current: bool) -> list[str]:
    lines = ["Notes", "Daily news - 15 Sep"]
    for i in range(stories_done):
        story = STORIES[i]
        lines.append(f"{i + 1}. {story['title']}")
        lines.append(story["url"])
        if i < stories_done - 1 or not typing_current:
            lines.append(story["summary"])
    return lines


def digest_text() -> str:
    parts = ["Daily news - 15 Sep"]
    for i, story in enumerate(STORIES):
        parts.append(f"{i + 1}. {story['title']}\n{story['url']}\n{story['summary']}")
    return "\n".join(parts)


def slack_ocr(posted: bool, composing: bool) -> list[str]:
    lines = ["Example Co", "# daily-news", "Yesterday", "Daily news - 14 Sep (3 links)"]
    if posted:
        lines += ["Today", *digest_text().splitlines()]
    lines.append("Message #daily-news" if not composing else digest_text().splitlines()[0])
    return lines


def dom_node(tag: str, **fields) -> dict:
    node = {"tag": tag}
    node.update({k: v for k, v in fields.items() if v is not None})
    return node


def front_page_dom() -> list[dict]:
    rows = []
    y = 60
    for rank, title, points, url in front_page_headlines():
        rows.append(dom_node(
            "tr",
            bbox={"x": 8, "y": y, "width": 1100, "height": 18},
            children=[
                dom_node("span", innerText=f"{rank}."),
                dom_node(
                    "a", innerText=title,
                    bbox={"x": 40, "y": y, "width": min(900, 7 * len(title)), "height": 16},
                ),
            ],
        ))
        y += 36
    return [
        dom_node("table", bbox={"x": 8, "y": 8, "width": 1180, "height": y}, children=[
            dom_node("tr", children=[
                dom_node("a", innerText="Hacker News"),
                dom_node("a", innerText="new"), dom_node("a", innerText="past"),
                dom_node("a", innerText="submit"),
            ]),
            *rows,
        ]),
    ]


def story_dom(story: dict) -> list[dict]:
    return [
        dom_node("header", role="banner", children=[
            dom_node("h1", innerText=story["title"],
                     bbox={"x": 120, "y": 96, "width": 880, "height": 44}),
        ]),
        dom_node("article", role="article", children=[
            dom_node("p", innerText=line) for line in story["body"]
        ]),
    ]


def slack_dom(posted: bool) -> list[dict]:
    messages = [dom_node("div", role="listitem", innerText="Daily news - 14 Sep (3 links)")]
    if posted:
        messages.append(dom_node("div", role="listitem", innerText=digest_text()[:500]))
    return [
        dom_node("nav", role="navigation", ariaLabel="Channels", children=[
            dom_node("a", role="treeitem", innerText="daily-news"),
        ]),
        dom_node("div", role="list", ariaLabel="Messages in daily-news", children=messages),
        dom_node("div", role="textbox", ariaLabel="Message #daily-news",
                 bbox={"x": 280, "y": 860, "width": 1180, "height": 44},
                 state={"enabled": True}),
        dom_node("button", ariaLabel="Send now",
                 bbox={"x": 1420, "y": 868, "width": 28, "height": 28},
                 state={"enabled": True}),
    ]


def news_tab_page(t: float) -> dict | None:
    """The story open in the news tab at time ``t`` (None: front page)."""
    for i, start in enumerate(CYCLE_START):
        back = start + READ_S + 3 + NOTES_S + 1.4  # Cmd-[ back to the front page
        if i == len(STORIES) - 1:
            back = float("inf")  # stays on the last story
        if start < t < back:
            return STORIES[i]
    return None


def screen_at(t: float) -> tuple[str, str, list[str]]:
    """(app_id, window title, OCR lines) of the focused window at ``t``."""
    for i, start in enumerate(CYCLE_START):
        in_notes_from = start + READ_S + 3
        in_notes_to = in_notes_from + NOTES_S if i < len(STORIES) - 1 else TO_CHROME_FINAL
        if in_notes_from <= t < in_notes_to:
            typing = t < in_notes_from + NOTES_S * 0.6
            return NOTES_APP, "Notes", note_lines(i + 1, typing)
    if t >= SLACK_TAB_SWITCH:
        posted = t > SLACK_CLICK_SEND
        composing = SLACK_CLICK_BOX < t <= SLACK_CLICK_SEND
        return CHROME_APP, SLACK_PAGE["title"], slack_ocr(posted, composing)
    story = news_tab_page(t)
    if story is None:
        return CHROME_APP, FRONT_PAGE["title"], front_page_ocr()
    return CHROME_APP, story["title"], story_ocr(story)


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


def build() -> Generator:
    gen = Generator()

    # Chrome extension: scrolling and reading on the front page.
    gen.extension(14.3, {"type": "ScrollReadSnapshot"}, NEWS_TAB, FRONT_PAGE["url"],
                  FRONT_PAGE["title"], {"title": FRONT_PAGE["title"],
                                        "timestamp": js_iso(START + timedelta(seconds=14.3))})

    for i, (start, story) in enumerate(zip(CYCLE_START, STORIES)):
        gen.click(start, NEWS_TAB, FRONT_PAGE["url"], {
            "tagName": "a",
            "innerText": story["title"],
            "composedPath": ["a", "span.titleline", "td.title", "tr.athing", "tbody",
                             "table", "td", "tr", "tbody", "table#hnmain", "center",
                             "body", "html"],
        })
        gen.extension(start + 0.8, {"type": "FocusChange"}, NEWS_TAB, story["url"], None, {})
        gen.native(start + 1.3, {"type": "WindowTitleChange"}, CHROME_APP, story["title"])
        gen.extension(start + 8.6, {"type": "DwellSnapshot"}, NEWS_TAB, story["url"],
                      story["title"], {"title": story["title"],
                                       "timestamp": js_iso(START + timedelta(seconds=start + 8.6))})
        gen.clipboard(start + READ_S, f"{story['title']}\n{story['url']}")
        gen.native(start + READ_S + 3, {"type": "AppSwitch", "from_app": CHROME_APP,
                                        "to_app": NOTES_APP}, NOTES_APP, "Notes")
        if i < len(STORIES) - 1:
            back_in_chrome = start + READ_S + 3 + NOTES_S
            gen.native(back_in_chrome, {"type": "AppSwitch", "from_app": NOTES_APP,
                                        "to_app": CHROME_APP}, CHROME_APP, story["title"])
            gen.extension(back_in_chrome + 1.4, {"type": "FocusChange"}, NEWS_TAB,
                          FRONT_PAGE["url"], None, {})
            gen.native(back_in_chrome + 1.8, {"type": "WindowTitleChange"}, CHROME_APP,
                       FRONT_PAGE["title"])

    # Copy the finished digest, switch to the Slack tab and post it.
    gen.clipboard(DIGEST_COPY, digest_text())
    gen.native(TO_CHROME_FINAL, {"type": "AppSwitch", "from_app": NOTES_APP,
                                 "to_app": CHROME_APP}, CHROME_APP, STORIES[-1]["title"])
    gen.native(SLACK_TAB_SWITCH, {"type": "WindowTitleChange"}, CHROME_APP, SLACK_PAGE["title"])
    gen.click(SLACK_CLICK_BOX, SLACK_TAB, SLACK_PAGE["url"], {
        "tagName": "div", "role": "textbox", "ariaLabel": "Message #daily-news",
        "composedPath": ["div.ql-editor", "div.ql-container", "div.c-texty_input",
                         "div.p-message_input", "body", "html"],
    })
    gen.click(SLACK_CLICK_SEND, SLACK_TAB, SLACK_PAGE["url"], {
        "tagName": "button", "ariaLabel": "Send now",
        "composedPath": ["button.c-wysiwyg_container__button--send",
                         "div.c-wysiwyg_container__footer", "div.p-message_input",
                         "body", "html"],
    })

    # Focus frames: every 2 s, at most 20 per daemon minute window.
    per_window: Counter[int] = Counter()
    t = 0.9
    while t < STOPPED_AT:
        window_index = int((t + MINUTE_RESET_PHASE) // 60)
        if per_window[window_index] < 20:
            per_window[window_index] += 1
            app_id, title, lines = screen_at(t)
            gen.focus_frame(t, app_id, title, lines)
        t += 2.0

    # DOM snapshots: the daemon asks the last active tab every 2 s.
    t = 1.3
    while t < STOPPED_AT:
        if t > SLACK_CLICK_BOX:
            tab, url, nodes = SLACK_TAB, SLACK_PAGE["url"], slack_dom(t > SLACK_CLICK_SEND)
        else:
            story = news_tab_page(t)
            tab = NEWS_TAB
            url = story["url"] if story else FRONT_PAGE["url"]
            nodes = story_dom(story) if story else front_page_dom()
        gen.extension(t, {"type": "DwellSnapshot"}, tab, url, None,
                      {"nodes": nodes, "captureReason": "request"})
        t += 2.0

    gen.rows.sort(key=lambda item: item[0])
    return gen


def manifest_for(gen: Generator) -> dict:
    rows = [row for _, row in gen.rows]
    kinds: Counter[str] = Counter()
    counts = Counter()
    for row in rows:
        kinds[json.loads(row["kind_json"])["type"]] += 1
        metadata = json.loads(row["metadata_json"])
        window = json.loads(row["window_json"]) if row["window_json"] else {}
        counts["focus_tagged"] += "focus_session_id" in metadata
        counts["with_screenshot"] += "screenshot_path" in metadata
        counts["with_ocr_text"] += bool(metadata.get("ocr", {}).get("full_text"))
        counts["dom_snapshots"] += isinstance(metadata.get("nodes"), list)
        counts["chrome_extension"] += window.get("app_id") == "com.google.Chrome"
        kind = json.loads(row["kind_json"])
        counts["clipboard_with_preview"] += bool(kind.get("content_preview"))
    return {
        "name": NAME,
        "description": (
            "10-minute focus recording: read Hacker News in Chrome, copy three "
            "stories' titles and links into a Notes note with a summary each, "
            "copy the digest and post it in a Slack channel in Chrome."
        ),
        "source": {
            "kind": "generated",
            "by": "worker/tests/journeys/record/generate_focus_dailynews.py",
            "note": "Replace with a real session via record/record_focus_session.py.",
        },
        "schema_version": 3,
        "recorded_start": rows[0]["timestamp"],
        "recorded_end": rows[-1]["timestamp"],
        "focus_session": {
            "session_id": gen.session_id,
            "title": SESSION_TITLE,
            # The app writes ISO8601DateFormatter text (MenuBarView.swift).
            "started_at": (START - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "stopped_at": (START + timedelta(seconds=STOPPED_AT)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "row_counts": {
            "events": len(rows),
            "by_kind": dict(sorted(kinds.items())),
            **dict(sorted(counts.items())),
        },
    }


def main() -> None:
    gen = build()
    out = FIXTURES_DIR / NAME
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out, [row for _, row in gen.rows])
    (out / "manifest.json").write_text(
        json.dumps(manifest_for(gen), indent=2) + "\n", encoding="utf-8",
    )
    print(f"wrote {len(gen.rows)} rows to {out}")


if __name__ == "__main__":
    main()
