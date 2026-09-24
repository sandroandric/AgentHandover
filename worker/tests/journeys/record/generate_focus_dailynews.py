"""Generate the daily-news fixtures: focus recordings of one routine on four
mornings, in the exact row format the daemon writes.

These stand in until real sessions are recorded on the test Mac with
``record_focus_session.py`` (same output format, ``source.kind`` becomes
``recorded``).  The routine: read Hacker News in Chrome, open the stories
worth sharing, copy each title and link, paste it into a Notes note with a
one-line summary, copy the finished digest, and post it in a Slack channel
open in another Chrome tab.  The user always shares that day's "Show HN"
post (and no other), so the chosen and the skipped headlines have a
criterion to learn.  The four runs (:data:`RUNS`):

``focus_dailynews`` (Tue 15 Sep)
    Three stories, all shared.
``focus_dailynews_skip`` (Wed 16 Sep)
    The second story opened is a two-hour video with no transcript: the
    user goes back after a look and leaves it out, so the digest has two
    stories.
``focus_dailynews_diverge`` / ``focus_dailynews_diverge2`` (Thu 17, Fri 18 Sep)
    The second story is behind research.example.com's paywall: the user
    opens it through archive.example.net, reads it there and shares the
    original link.  Two runs take this same detour from the same kind of
    screen, which is what the divergence miner needs to make a branch
    (two agreeing instances, ``branch_mining``).

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

    .venv/bin/python worker/tests/journeys/record/generate_focus_dailynews.py [NAME ...]

With no name it writes every run.  The output is deterministic: running it
twice gives identical files.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

JOURNEYS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JOURNEYS_DIR.parent))  # worker/tests, for daemon_db

from daemon_db import schema_version  # noqa: E402
from journeys.fixture_loader import (  # noqa: E402
    FIXTURES_DIR,
    SCREENSHOTS_TOKEN,
    row_counts,
    write_manifest,
    write_rows,
)

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
SLACK_PAGE = {
    "url": "https://app.slack.com/client/T0EXAMPLE/C0DAILYNEWS",
    "title": "daily-news (Channel) - Example Co - Slack",
}
ARCHIVE_HOST = "archive.example.net"

# What the user sees instead of the story.
VIDEO_PAGE = (
    "1:58:24",
    "Watch later | Share",
    "No transcript available for this video.",
)
PAYWALL = (
    "Subscribe to continue reading",
    "You have read your 3 free articles this month.",
    "Subscribe | Sign in",
)

# Per-story timing (seconds).
READ_S = 40.0          # reading the story before copying title + link
NOTES_S = 62.0         # pasting and writing the summary in Notes
LEFT_OUT_S = 14.0      # looking at a story that will not be shared, then back
PAYWALL_S = 12.0       # looking at the paywall, then opening the archived copy
MINUTE_RESET_PHASE = 17.0  # the daemon's per-minute screenshot counter reset


@dataclass(frozen=True)
class Story:
    rank: int
    title: str
    url: str
    points: int
    summary: str = ""
    body: tuple[str, ...] = ()
    # "kept": read and shared; "left_out": looked at (``blocker`` on
    # screen) and not shared; "archived": ``blocker`` on screen, read and
    # shared through its archived copy.
    outcome: str = "kept"
    blocker: tuple[str, ...] = ()


@dataclass(frozen=True)
class Run:
    """One recorded morning: the stories in the order they are opened and
    when (seconds from ``start``) each part of the routine happens."""

    name: str
    seed: int
    start: datetime
    description: str
    stories: tuple[Story, ...]
    other_headlines: tuple[tuple[int, str, int], ...]
    cycle_start: tuple[float, ...]  # story i is opened at cycle_start[i]
    digest_copy: float              # select-all + copy in Notes
    to_chrome_final: float
    slack_tab_switch: float
    slack_click_box: float
    slack_click_send: float
    stopped_at: float
    yesterday: str                  # the channel's previous digest

    @property
    def day(self) -> str:
        return f"{self.start.day} {self.start:%b}"

    @property
    def kept(self) -> list[Story]:
        return [s for s in self.stories if s.outcome != "left_out"]


RUNS: dict[str, Run] = {run.name: run for run in (
    Run(
        name="focus_dailynews",
        seed=20260915,
        start=datetime(2026, 9, 15, 8, 30, 0, tzinfo=timezone.utc),
        description=(
            "10-minute focus recording: read Hacker News in Chrome, copy three "
            "stories' titles and links into a Notes note with a summary each, "
            "copy the digest and post it in a Slack channel in Chrome."
        ),
        stories=(
            Story(
                rank=3,
                title="SQLite's WAL mode, explained with diagrams",
                url="https://blog.example.org/sqlite-wal-explained",
                points=412,
                summary="Readers keep going while one writer appends; checkpoints fold the log back.",
                body=(
                    "Write-ahead logging keeps a separate log of changed pages.",
                    "Readers see the database as of the last commit they started with.",
                    "A checkpoint copies committed pages back into the main file.",
                ),
            ),
            Story(
                rank=7,
                title="A field guide to on-device vision models",
                url="https://research.example.com/on-device-vision-guide",
                points=288,
                summary="Small quantised models now read screenshots well enough for UI tasks.",
                body=(
                    "Quantisation-aware training closes most of the accuracy gap.",
                    "Context length, not parameter count, limits screenshot reading.",
                    "Latency on a laptop GPU is now under two seconds per frame.",
                ),
            ),
            Story(
                rank=12,
                title="Show HN: A tiny HTTP cassette recorder for tests",
                url="https://github.com/example/cassette",
                points=156,
                summary="Records real API answers once and replays them byte for byte.",
                body=(
                    "Record mode proxies to the real service and appends answers.",
                    "Replay mode fails the test on any request it has not seen.",
                    "Keys hash the request body, so option order does not matter.",
                ),
            ),
        ),
        other_headlines=(
            (1, "The quiet death of the RSS reader, and its return", 530),
            (2, "Designing a scheduler that respects laptop batteries", 498),
            (4, "Ask HN: How do you keep a daily reading habit?", 377),
            (5, "Postmortem: a leap second in our job queue", 341),
            (6, "Typst 1.0 released", 310),
            (8, "Why most dashboards are ignored", 250),
            (9, "An illustrated guide to CRDTs", 233),
            (10, "The economics of small open-source libraries", 201),
            (11, "Reverse-engineering a 1990s pager protocol", 177),
        ),
        cycle_start=(40.6, 190.6, 340.6),
        digest_copy=452.0,
        to_chrome_final=456.2,
        slack_tab_switch=458.4,
        slack_click_box=464.1,
        slack_click_send=482.7,
        stopped_at=600.0,
        yesterday="Daily news - 14 Sep (3 links)",
    ),
    Run(
        name="focus_dailynews_skip",
        seed=20260916,
        start=datetime(2026, 9, 16, 8, 30, 0, tzinfo=timezone.utc),
        description=(
            "8-minute focus recording of the same routine the next morning: "
            "the second story opened is a two-hour video with no transcript, "
            "so it is left out and the digest posted in Slack has two stories."
        ),
        stories=(
            Story(
                rank=4,
                title="Why SQLite's query planner ignores your index",
                url="https://blog.example.org/sqlite-planner-index",
                points=367,
                summary="The planner scans when its statistics say a scan is cheaper; run ANALYZE.",
                body=(
                    "The query planner estimates the cost of every plan it considers.",
                    "Without statistics it assumes each index narrows the rows tenfold.",
                    "ANALYZE records the real distribution, and the plan changes.",
                ),
            ),
            Story(
                rank=6,
                title="Building a tiny database from scratch [video]",
                url="https://video.example.com/watch/tiny-database",
                points=305,
                outcome="left_out",
                blocker=VIDEO_PAGE,
            ),
            Story(
                rank=9,
                title="Show HN: Diffable snapshots of SQLite databases",
                url="https://github.com/example/sqlite-snapshots",
                points=190,
                summary="Dumps a database as sorted text so a diff shows which rows changed.",
                body=(
                    "Each table is written as sorted rows, one per line.",
                    "Schema changes appear as their own section of the diff.",
                    "Snapshots are small enough to commit next to the code.",
                ),
            ),
        ),
        other_headlines=(
            (1, "Lessons from ten years of on-call", 512),
            (2, "The case for boring technology, revisited", 487),
            (3, "Ask HN: What are you reading this month?", 399),
            (5, "A visual history of keyboard layouts", 342),
            (7, "Rust 1.95 released", 277),
            (8, "How we cut our cloud bill in half", 246),
            (10, "Understanding the Linux page cache", 205),
            (11, "The hidden cost of feature flags", 181),
            (12, "A printable guide to regular expressions", 150),
        ),
        cycle_start=(35.2, 186.4, 212.8),
        digest_copy=324.0,
        to_chrome_final=328.3,
        slack_tab_switch=330.5,
        slack_click_box=336.2,
        slack_click_send=351.9,
        stopped_at=480.0,
        yesterday="Daily news - 15 Sep (3 links)",
    ),
    Run(
        name="focus_dailynews_diverge",
        seed=20260917,
        start=datetime(2026, 9, 17, 8, 30, 0, tzinfo=timezone.utc),
        description=(
            "10.5-minute focus recording of the same routine: the second story "
            "is behind research.example.com's paywall, so the user reads it "
            "through archive.example.net and shares the original link."
        ),
        stories=(
            Story(
                rank=2,
                title="Backpressure, explained with queues at a bakery",
                url="https://blog.example.org/backpressure-bakery",
                points=455,
                summary="A queue that never says no hides overload until it falls over.",
                body=(
                    "A bakery with a bounded queue turns customers away at the door.",
                    "An unbounded queue accepts everyone and serves nobody on time.",
                    "Rejecting early keeps latency flat for the requests you accept.",
                ),
            ),
            Story(
                rank=5,
                title="Measuring how people really read on screens",
                url="https://research.example.com/articles/48213",
                points=330,
                outcome="archived",
                blocker=PAYWALL,
                summary="Most readers scan the first two lines, then jump between headings.",
                body=(
                    "Eye tracking of 400 readers shows an F-shaped scanning pattern.",
                    "Headings that state a claim hold attention longer than labels.",
                    "Long paragraphs are skipped unless their first line is concrete.",
                ),
            ),
            Story(
                rank=11,
                title="Show HN: A terminal reader for saved articles",
                url="https://github.com/example/termreader",
                points=142,
                summary="Saves pages as clean text and reads them offline in the terminal.",
                body=(
                    "Pages are fetched once and stored as readable plain text.",
                    "Keyboard navigation follows the article's own headings.",
                    "Everything stays in one SQLite file you can sync.",
                ),
            ),
        ),
        other_headlines=(
            (1, "The end of the free tier", 541),
            (3, "Ask HN: Which tools made you faster this year?", 421),
            (4, "Linear types for the working programmer", 388),
            (6, "A postmortem of our DNS migration", 317),
            (7, "The printer that wouldn't die", 289),
            (8, "Designing error messages people read", 260),
            (9, "A brief history of the tab character", 231),
            (10, "Profiling Python without changing a line", 198),
            (12, "Maps of every public stairway in Lisbon", 133),
        ),
        cycle_start=(42.1, 193.5, 359.7),
        digest_copy=471.3,
        to_chrome_final=475.6,
        slack_tab_switch=477.9,
        slack_click_box=483.4,
        slack_click_send=501.8,
        stopped_at=630.0,
        yesterday="Daily news - 16 Sep (2 links)",
    ),
    Run(
        name="focus_dailynews_diverge2",
        seed=20260918,
        start=datetime(2026, 9, 18, 8, 30, 0, tzinfo=timezone.utc),
        description=(
            "10-minute focus recording of the same routine: again the second "
            "story is behind research.example.com's paywall and the user takes "
            "the same detour through archive.example.net."
        ),
        stories=(
            Story(
                rank=3,
                title="What a write-ahead log buys you",
                url="https://blog.example.org/what-a-wal-buys-you",
                points=402,
                summary="Crash safety and concurrent readers, paid for with a checkpoint step.",
                body=(
                    "Appending to a log is cheaper than rewriting pages in place.",
                    "After a crash the log is replayed up to the last commit.",
                    "Checkpoints keep the log from growing without bound.",
                ),
            ),
            Story(
                rank=6,
                title="Energy use of local language models, measured",
                url="https://research.example.com/articles/51877",
                points=297,
                outcome="archived",
                blocker=PAYWALL,
                summary="A 4B model on a laptop uses about as much energy per answer as a web search.",
                body=(
                    "Power was measured at the wall for 2,000 prompts per model.",
                    "Quantised 4B models used a fifth of the energy of 14B ones.",
                    "Idle time between prompts dominated the total on laptops.",
                ),
            ),
            Story(
                rank=10,
                title="Show HN: Replay HTTP traffic in unit tests",
                url="https://github.com/example/http-replay",
                points=171,
                summary="Captures HTTP exchanges once and serves them to tests without a network.",
                body=(
                    "A proxy records each request and its answer to a file.",
                    "Tests replay the file and fail on any request not in it.",
                    "Secrets in headers are masked before anything is written.",
                ),
            ),
        ),
        other_headlines=(
            (1, "Why we went back to a monolith", 566),
            (2, "The surprising physics of coffee rings", 470),
            (4, "Ask HN: How do you take notes on what you read?", 380),
            (5, "An interactive guide to B-trees", 352),
            (7, "Postgres 19 beta is out", 268),
            (8, "The quiet success of RSS in podcasting", 244),
            (9, "Notes on writing a small compiler", 219),
            (11, "Reverse-engineering a thermal printer", 160),
            (12, "A field report from a four-day week", 128),
        ),
        cycle_start=(38.9, 187.3, 352.5),
        digest_copy=463.8,
        to_chrome_final=468.0,
        slack_tab_switch=470.2,
        slack_click_box=476.0,
        slack_click_send=494.6,
        stopped_at=612.0,
        yesterday="Daily news - 17 Sep (3 links)",
    ),
)}

# The first run, as the other generators import it.
SEED = RUNS["focus_dailynews"].seed
START = RUNS["focus_dailynews"].start


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
    def __init__(self, seed: int = SEED, start: datetime = START) -> None:
        self.rng = random.Random(seed)
        self.start = start
        self.session_id = str(self._uuid())
        self.rows: list[tuple[float, dict]] = []

    def _uuid(self) -> uuid.UUID:
        return uuid.UUID(int=self.rng.getrandbits(128), version=4)

    def _time(self, offset: float) -> datetime:
        return self.start + timedelta(seconds=offset, microseconds=self.rng.randrange(1, 999))

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


def host_of(url: str) -> str:
    return url.split("/")[2]


def archive_url(story: Story) -> str:
    # The archive's lookup form submits the story's link as a query.
    return f"https://{ARCHIVE_HOST}/?url={story.url}"


def front_page_headlines(run: Run) -> list[tuple[int, str, int, str]]:
    rows = [(r, t, p, "") for r, t, p in run.other_headlines]
    rows += [(s.rank, s.title, s.points, s.url) for s in run.stories]
    return sorted(rows)


def front_page_ocr(run: Run) -> list[str]:
    lines = ["Hacker News  new | past | comments | ask | show | jobs | submit", "login"]
    for rank, title, points, _ in front_page_headlines(run):
        lines.append(f"{rank}. {title}")
        lines.append(f"{points} points | {40 + rank * 7} comments")
    return lines


def story_ocr(story: Story) -> list[str]:
    return [story.url, story.title, host_of(story.url), *story.body]


def blocked_ocr(story: Story) -> list[str]:
    return [story.url, story.title, host_of(story.url), *story.blocker]


def archive_ocr(story: Story) -> list[str]:
    return [archive_url(story), ARCHIVE_HOST, f"Saved from {story.url}",
            story.title, *story.body]


def note_lines(run: Run, stories_done: int, typing_current: bool) -> list[str]:
    lines = ["Notes", f"Daily news - {run.day}"]
    for i in range(stories_done):
        story = run.kept[i]
        lines.append(f"{i + 1}. {story.title}")
        lines.append(story.url)
        if i < stories_done - 1 or not typing_current:
            lines.append(story.summary)
    return lines


def digest_text(run: Run) -> str:
    parts = [f"Daily news - {run.day}"]
    for i, story in enumerate(run.kept):
        parts.append(f"{i + 1}. {story.title}\n{story.url}\n{story.summary}")
    return "\n".join(parts)


def slack_ocr(run: Run, posted: bool, composing: bool) -> list[str]:
    lines = ["Example Co", "# daily-news", "Yesterday", run.yesterday]
    if posted:
        lines += ["Today", *digest_text(run).splitlines()]
    lines.append("Message #daily-news" if not composing else digest_text(run).splitlines()[0])
    return lines


def dom_node(tag: str, **fields) -> dict:
    node = {"tag": tag}
    node.update({k: v for k, v in fields.items() if v is not None})
    return node


def front_page_dom(run: Run) -> list[dict]:
    rows = []
    y = 60
    for rank, title, points, url in front_page_headlines(run):
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


def _heading(title: str) -> dict:
    return dom_node("header", role="banner", children=[
        dom_node("h1", innerText=title, bbox={"x": 120, "y": 96, "width": 880, "height": 44}),
    ])


def story_dom(story: Story) -> list[dict]:
    return [
        _heading(story.title),
        dom_node("article", role="article", children=[
            dom_node("p", innerText=line) for line in story.body
        ]),
    ]


def blocked_dom(story: Story) -> list[dict]:
    return [
        _heading(story.title),
        dom_node("section", role="region", ariaLabel="Notice", children=[
            dom_node("p", innerText=line) for line in story.blocker
        ]),
    ]


def archive_dom(story: Story) -> list[dict]:
    return [
        dom_node("div", role="banner", children=[
            dom_node("a", innerText=ARCHIVE_HOST),
            dom_node("a", innerText=f"Saved from {story.url}"),
        ]),
        *story_dom(story),
    ]


def slack_dom(run: Run, posted: bool) -> list[dict]:
    messages = [dom_node("div", role="listitem", innerText=run.yesterday)]
    if posted:
        messages.append(dom_node("div", role="listitem", innerText=digest_text(run)[:500]))
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


@dataclass(frozen=True)
class Page:
    """What the news tab shows: its URL, title, OCR lines and DOM."""

    url: str
    title: str
    ocr: list[str]
    dom: list[dict]


def front_page(run: Run) -> Page:
    return Page(FRONT_PAGE["url"], FRONT_PAGE["title"], front_page_ocr(run), front_page_dom(run))


@dataclass(frozen=True)
class Cycle:
    """One story, from the click on its headline to the tab's return to
    the front page (``back``; infinite for the last story, whose tab stays
    open).  ``pages`` are ``(from, to, page)``; ``notes`` is the Notes
    stretch ``(from, to)`` of a shared story."""

    story: Story
    start: float
    pages: tuple[tuple[float, float, Page], ...]
    copy_at: float | None
    notes: tuple[float, float] | None
    back: float


def cycles(run: Run) -> list[Cycle]:
    out = []
    last = len(run.stories) - 1
    for i, (start, story) in enumerate(zip(run.cycle_start, run.stories)):
        story_page = Page(story.url, story.title, story_ocr(story), story_dom(story))
        blocked = Page(story.url, story.title, blocked_ocr(story), blocked_dom(story))
        if story.outcome == "left_out":
            if i == last:
                raise ValueError(f"{run.name}: the last story must be shared")
            back = start + LEFT_OUT_S
            out.append(Cycle(story, start, ((start, back, blocked),), None, None, back))
            continue
        pages: list[tuple[float, float, Page]] = []
        reading = start
        if story.outcome == "archived":
            reading = start + PAYWALL_S
            pages.append((start, reading, blocked))
            story_page = Page(archive_url(story), f"{story.title} - {ARCHIVE_HOST}",
                              archive_ocr(story), archive_dom(story))
        copy_at = reading + READ_S
        notes_from = copy_at + 3
        if i == last:
            notes, back = (notes_from, run.to_chrome_final), float("inf")
        else:
            notes = (notes_from, notes_from + NOTES_S)
            back = notes[1] + 1.4  # Cmd-[ back to the front page
        pages.append((reading, back, story_page))
        out.append(Cycle(story, start, tuple(pages), copy_at, notes, back))
    return out


def news_tab_page(run: Run, plan: list[Cycle], t: float) -> Page:
    """The page open in the news tab at time ``t``."""
    for cycle in plan:
        for start, end, page in cycle.pages:
            if start < t < end:
                return page
    return front_page(run)


def screen_at(run: Run, plan: list[Cycle], t: float) -> tuple[str, str, list[str]]:
    """(app_id, window title, OCR lines) of the focused window at ``t``."""
    shared = [c for c in plan if c.notes is not None]
    for done, cycle in enumerate(shared, start=1):
        notes_from, notes_to = cycle.notes
        if notes_from <= t < notes_to:
            typing = t < notes_from + NOTES_S * 0.6
            return NOTES_APP, "Notes", note_lines(run, done, typing)
    if t >= run.slack_tab_switch:
        posted = t > run.slack_click_send
        composing = run.slack_click_box < t <= run.slack_click_send
        return CHROME_APP, SLACK_PAGE["title"], slack_ocr(run, posted, composing)
    page = news_tab_page(run, plan, t)
    return CHROME_APP, page.title, page.ocr


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


def build(run: Run) -> Generator:
    gen = Generator(run.seed, run.start)
    plan = cycles(run)

    def page_seen(offset: float, page: Page) -> None:
        # The tab loads the page, the window title follows, the extension
        # reports the dwell.
        gen.extension(offset + 0.8, {"type": "FocusChange"}, NEWS_TAB, page.url, None, {})
        gen.native(offset + 1.3, {"type": "WindowTitleChange"}, CHROME_APP, page.title)
        gen.extension(offset + 8.6, {"type": "DwellSnapshot"}, NEWS_TAB, page.url,
                      page.title, {"title": page.title,
                                   "timestamp": js_iso(run.start + timedelta(seconds=offset + 8.6))})

    def back_to_front(offset: float) -> None:
        gen.extension(offset, {"type": "FocusChange"}, NEWS_TAB, FRONT_PAGE["url"], None, {})
        gen.native(offset + 0.4, {"type": "WindowTitleChange"}, CHROME_APP, FRONT_PAGE["title"])

    # Chrome extension: scrolling and reading on the front page.
    gen.extension(14.3, {"type": "ScrollReadSnapshot"}, NEWS_TAB, FRONT_PAGE["url"],
                  FRONT_PAGE["title"], {"title": FRONT_PAGE["title"],
                                        "timestamp": js_iso(run.start + timedelta(seconds=14.3))})

    for cycle in plan:
        story = cycle.story
        gen.click(cycle.start, NEWS_TAB, FRONT_PAGE["url"], {
            "tagName": "a",
            "innerText": story.title,
            "composedPath": ["a", "span.titleline", "td.title", "tr.athing", "tbody",
                             "table", "td", "tr", "tbody", "table#hnmain", "center",
                             "body", "html"],
        })
        page_seen(cycle.start, cycle.pages[0][2])
        if cycle.notes is None:
            back_to_front(cycle.back)
            continue
        if len(cycle.pages) > 1:
            # Paywalled: the user opens the archived copy in the same tab.
            page_seen(cycle.pages[1][0] - 0.8, cycle.pages[1][2])
        reading = cycle.pages[-1][2]
        gen.clipboard(cycle.copy_at, f"{story.title}\n{story.url}")
        gen.native(cycle.notes[0], {"type": "AppSwitch", "from_app": CHROME_APP,
                                    "to_app": NOTES_APP}, NOTES_APP, "Notes")
        if cycle.back != float("inf"):
            gen.native(cycle.notes[1], {"type": "AppSwitch", "from_app": NOTES_APP,
                                        "to_app": CHROME_APP}, CHROME_APP, reading.title)
            gen.extension(cycle.notes[1] + 1.4, {"type": "FocusChange"}, NEWS_TAB,
                          FRONT_PAGE["url"], None, {})
            gen.native(cycle.notes[1] + 1.8, {"type": "WindowTitleChange"}, CHROME_APP,
                       FRONT_PAGE["title"])

    # Copy the finished digest, switch to the Slack tab and post it.
    gen.clipboard(run.digest_copy, digest_text(run))
    gen.native(run.to_chrome_final, {"type": "AppSwitch", "from_app": NOTES_APP,
                                     "to_app": CHROME_APP}, CHROME_APP, plan[-1].pages[-1][2].title)
    gen.native(run.slack_tab_switch, {"type": "WindowTitleChange"}, CHROME_APP,
               SLACK_PAGE["title"])
    gen.click(run.slack_click_box, SLACK_TAB, SLACK_PAGE["url"], {
        "tagName": "div", "role": "textbox", "ariaLabel": "Message #daily-news",
        "composedPath": ["div.ql-editor", "div.ql-container", "div.c-texty_input",
                         "div.p-message_input", "body", "html"],
    })
    gen.click(run.slack_click_send, SLACK_TAB, SLACK_PAGE["url"], {
        "tagName": "button", "ariaLabel": "Send now",
        "composedPath": ["button.c-wysiwyg_container__button--send",
                         "div.c-wysiwyg_container__footer", "div.p-message_input",
                         "body", "html"],
    })

    # Focus frames: every 2 s, at most 20 per daemon minute window.
    per_window: Counter[int] = Counter()
    t = 0.9
    while t < run.stopped_at:
        window_index = int((t + MINUTE_RESET_PHASE) // 60)
        if per_window[window_index] < 20:
            per_window[window_index] += 1
            app_id, title, lines = screen_at(run, plan, t)
            gen.focus_frame(t, app_id, title, lines)
        t += 2.0

    # DOM snapshots: the daemon asks the last active tab every 2 s.
    t = 1.3
    while t < run.stopped_at:
        if t > run.slack_click_box:
            tab, url, nodes = SLACK_TAB, SLACK_PAGE["url"], slack_dom(run, t > run.slack_click_send)
        else:
            page = news_tab_page(run, plan, t)
            tab, url, nodes = NEWS_TAB, page.url, page.dom
        gen.extension(t, {"type": "DwellSnapshot"}, tab, url, None,
                      {"nodes": nodes, "captureReason": "request"})
        t += 2.0

    gen.rows.sort(key=lambda item: item[0])
    return gen


def manifest_for(run: Run, gen: Generator) -> dict:
    rows = [row for _, row in gen.rows]
    return {
        "name": run.name,
        "description": run.description,
        "source": {
            "kind": "generated",
            "by": "worker/tests/journeys/record/generate_focus_dailynews.py",
            "note": "Replace with a real session via record/record_focus_session.py.",
        },
        "schema_version": schema_version(),
        "recorded_start": rows[0]["timestamp"],
        "recorded_end": rows[-1]["timestamp"],
        "focus_session": {
            "session_id": gen.session_id,
            "title": SESSION_TITLE,
            # The app writes ISO8601DateFormatter text (MenuBarView.swift).
            "started_at": (run.start - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "stopped_at": (run.start + timedelta(seconds=run.stopped_at)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
        },
        "row_counts": row_counts(rows),
    }


def write_fixture(run: Run, fixtures_dir: Path = FIXTURES_DIR) -> Path:
    """Write *run*'s ``events.jsonl.gz`` and ``manifest.json``; returns its directory."""
    gen = build(run)
    out = fixtures_dir / run.name
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out, [row for _, row in gen.rows])
    write_manifest(out, manifest_for(run, gen))
    return out


def main(argv: list[str]) -> None:
    for name in argv or list(RUNS):
        print(f"wrote {write_fixture(RUNS[name])}")


if __name__ == "__main__":
    main(sys.argv[1:])
