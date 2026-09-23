"""WP-P3: evidence into the prompt and the packet.

Unit pins for the clipboard text and verbatim fields in the SOP prompt, the
authoritative WINDOW header, app display names, the DOM host join, the
anchor chosen for a clicked element, and the per-step anchor and evidence
carried into the procedure and the rendered Skill.  The fixture pins drive
the focus path over the ``focus_dailynews`` session rows (the journey with
recorded model output, J-focus, needs a cassette recorded for it).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agenthandover_worker.db import WorkerDB, _url_match
from agenthandover_worker.dom import flatten_dom, implicit_role, node_tag, select_anchor, url_host
from agenthandover_worker.event_helpers import display_name, is_bundle_id, parse_json_field, parse_timestamp
from agenthandover_worker.focus_processor import FocusProcessor
from agenthandover_worker.procedure_schema import sop_to_procedure, validate_procedure
from agenthandover_worker.scene_annotator import build_annotation_prompt, window_header
from agenthandover_worker.skill_render import render_packet
from agenthandover_worker.sop_generator import (
    _build_focus_prompt,
    _format_timeline_entry,
    _vlm_sop_to_template,
)
from daemon_db import create_daemon_db

# ---------------------------------------------------------------------------
# Clipboard text and verbatim fields in the SOP prompt
# ---------------------------------------------------------------------------


class TestClipboardPreview:
    def test_preview_is_kept_from_kind_json(self) -> None:
        event = {"kind_json": json.dumps({
            "type": "ClipboardChange", "content_types": ["public.utf8-plain-text"],
            "byte_size": 88, "high_entropy": False, "content_hash": "abc",
            "content_preview": "SQLite's WAL mode\nhttps://blog.example.org/wal",
        })}
        meta = FocusProcessor._extract_clipboard_meta(event)
        assert meta["content_preview"] == "SQLite's WAL mode\nhttps://blog.example.org/wal"

    def test_copied_text_is_one_line_in_the_prompt(self) -> None:
        entry = _format_timeline_entry(
            0, {"app": "Google Chrome"}, None, "2026-09-15T08:31:20+00:00",
            clipboard_context={
                "byte_size": 88, "content_types": ["public.utf8-plain-text"],
                "content_preview": "SQLite's WAL mode\nhttps://blog.example.org/wal",
                "source_app": "Google Chrome",
            },
        )
        assert '>>> Copied text: "SQLite\'s WAL mode | https://blog.example.org/wal"' in entry

    def test_typed_text_urls_items_and_screen_text_are_rendered_and_capped(self) -> None:
        annotation = {
            "app": "Google Chrome",
            "key_text": {
                "typed_text": "Top news:\n" + "x" * 400,
                "urls": ["https://news.ycombinator.com/", "https://blog.example.org/wal"],
                "emails": [],
            },
            "visible_items": [
                {"label": "SQLite's WAL mode", "chosen": True},
                {"label": "Typst 1.0 released", "chosen": False},
            ],
            "_screen_text": "To: Jane Doe (example.com)\nSubject: Daily news\n" + "y" * 400,
        }
        entry = _format_timeline_entry(0, annotation, None, "2026-09-15T08:31:20+00:00")
        typed = next(line for line in entry.splitlines() if line.startswith("  Typed:"))
        assert typed.startswith('  Typed: "Top news: | xxx')
        assert len(typed) <= len('  Typed: ""') + 200
        assert "  URLs: https://news.ycombinator.com/, https://blog.example.org/wal" in entry
        assert "  Items: SQLite's WAL mode [chosen]; Typst 1.0 released" in entry
        screen = next(line for line in entry.splitlines() if line.startswith("  Screen text:"))
        assert screen.startswith('  Screen text: "To: Jane Doe (example.com) | Subject: Daily news | yyy')
        assert len(screen) <= len('  Screen text: ""') + 300

    def test_legacy_compose_fields_become_screen_text(self) -> None:
        """Rows annotated before the compact schema keep their compose
        fields and visible values in the SOP prompt, as screen text."""
        annotation = {
            "app": "Google Chrome",
            "visible_content": {"headings": ["Inbox"], "labels": [], "values": ["3 unread"]},
            "compose": {
                "is_compose_window_open": True, "recipient": "Jane Doe (example.com)",
                "subject": "Daily news", "body_first_line": "Hi Jane,",
                "send_button_state": "visible",
            },
        }
        entry = _format_timeline_entry(0, annotation, None, "")
        assert ('  Screen text: "Inbox | 3 unread | To: Jane Doe (example.com) | '
                'Subject: Daily news | Body: Hi Jane,"') in entry

    def test_closed_compose_window_is_not_rendered(self) -> None:
        annotation = {"app": "Mail", "compose": {"is_compose_window_open": False, "subject": "stale"}}
        assert "stale" not in _format_timeline_entry(0, annotation, None, "")

    def test_frame_app_line_never_shows_a_bundle_id(self) -> None:
        entry = _format_timeline_entry(0, {"app": "com.google.Chrome"}, None, "")
        assert "Frame 1: Google Chrome" in entry


# ---------------------------------------------------------------------------
# WINDOW header and app names
# ---------------------------------------------------------------------------


class TestWindowHeader:
    def test_header_is_authoritative_with_name_id_title_and_url(self) -> None:
        prompt = build_annotation_prompt(
            window={"app_id": "com.google.Chrome", "title": "Hacker News"},
            app_state={"app_id": "com.google.Chrome", "properties": {"url": "https://news.ycombinator.com/"}},
            page_url="https://ignored.example/",
        )
        assert prompt.startswith(
            "WINDOW (authoritative; copy, do not infer):\n"
            "app: Google Chrome\n"
            "app_id: com.google.Chrome\n"
            "title: Hacker News\n"
            "url: https://news.ycombinator.com/\n"
        )

    def test_url_falls_back_to_the_page_url(self) -> None:
        header = window_header({"app_id": "pid:812:Google Chrome", "title": "HN"}, None, "https://a.example/")
        assert header == "app: Google Chrome\ntitle: HN\nurl: https://a.example/"

    def test_no_window_no_header(self) -> None:
        assert "WINDOW" not in build_annotation_prompt(ocr_text="hello")

    @pytest.mark.parametrize(("app_id", "name"), [
        ("pid:812:Google Chrome", "Google Chrome"),
        ("com.google.Chrome", "Google Chrome"),
        ("com.apple.Notes", "Notes"),
        ("com.tinyspeck.slackmacgap", "Slack"),
        ("com.example.SomeTool", "SomeTool"),
        ("Google Chrome", "Google Chrome"),
        ("zoom.us", "zoom.us"),
        ("", ""),
    ])
    def test_display_name(self, app_id: str, name: str) -> None:
        assert display_name(app_id) == name
        assert not is_bundle_id(display_name(app_id)) or not name

    def test_parse_json_field(self) -> None:
        assert parse_json_field({"w": '{"a": 1}'}, "w") == {"a": 1}
        assert parse_json_field({"w": {"a": 1}}, "w") == {"a": 1}
        assert parse_json_field({"w": "not json"}, "w") == {}
        assert parse_json_field({}, "w") == {}


# ---------------------------------------------------------------------------
# DOM: hosts, flattening, anchors
# ---------------------------------------------------------------------------


class TestHosts:
    @pytest.mark.parametrize(("text", "host"), [
        ("https://news.ycombinator.com/item?id=1", "news.ycombinator.com"),
        ("news.ycombinator.com/item?id=1", "news.ycombinator.com"),
        ("www.example.com", "example.com"),
        ("HTTPS://App.Slack.com:443/client", "app.slack.com"),
        ("localhost:8080/admin", "localhost"),
        ("Hacker News", ""),
        ("/Users/test/Documents", ""),
        ("Notes > Daily digest", ""),
        ("", ""),
    ])
    def test_url_host(self, text: str, host: str) -> None:
        assert url_host(text) == host

    def test_url_match_compares_hosts_with_or_without_scheme(self) -> None:
        assert _url_match("news.ycombinator.com", "https://news.ycombinator.com/")
        assert _url_match("https://www.example.com/a", "https://example.com/b")
        assert not _url_match("https://example.com.evil.test/", "https://example.com/")
        assert not _url_match("Hacker News", "https://news.ycombinator.com/")


class TestFlattenAndAnchor:
    TREE = [
        {"tag": "nav", "role": "navigation", "ariaLabel": "Main", "children": [
            {"tag": "a", "innerText": "Inbox", "name": "nav-link"},
            {"tag": "input[type=email]", "name": "Work email"},
        ]},
        {"tag": "button", "testId": "send-btn", "innerText": "Send"},
        {"tag": "#truncated", "innerText": "[capped]"},
    ]

    def test_flatten_walks_trees_in_document_order_without_markers(self) -> None:
        assert [n["tag"] for n in flatten_dom(self.TREE)] == [
            "nav", "a", "input[type=email]", "button",
        ]
        assert list(flatten_dom(None)) == []

    def test_input_type_is_parsed_out_of_the_tag(self) -> None:
        assert node_tag({"tag": "input[type=email]"}) == ("input", "email")
        assert node_tag({"tag": "input", "type": "checkbox"}) == ("input", "checkbox")
        assert implicit_role({"tag": "input[type=email]"}) == "textbox"
        assert implicit_role({"tag": "input[type=checkbox]"}) == "checkbox"
        assert implicit_role({"tag": "a"}) == "link"
        assert implicit_role({"tag": "div", "role": "treeitem"}) == "treeitem"

    def test_test_id_wins(self) -> None:
        anchor = select_anchor(self.TREE, {"tagName": "button", "innerText": "Send", "testId": "send-btn"})
        assert anchor == {"role": "button", "name": "Send", "selector": '[data-testid="send-btn"]'}

    def test_aria_label_before_role_and_name(self) -> None:
        nodes = [{"tag": "button", "ariaLabel": "Send now"}]
        anchor = select_anchor(nodes, {"tagName": "button", "ariaLabel": "Send now", "composedPath": ["button"]})
        assert anchor == {"role": "button", "name": "Send now", "selector": '[aria-label="Send now"]'}

    def test_role_and_name_from_the_visible_text_not_the_class_hint(self) -> None:
        anchor = select_anchor(self.TREE, {"tagName": "a", "innerText": "Inbox"})
        assert anchor == {"role": "link", "name": "Inbox", "selector": 'role=link[name="Inbox"]'}

    def test_composed_path_when_nothing_else_identifies_it(self) -> None:
        anchor = select_anchor([], {"tagName": "div", "composedPath": ["div.cell", "div.row", "main", "body"]})
        # The target and its three nearest ancestors, outermost first.
        assert anchor == {"role": "", "name": "", "selector": "body > main > div.row > div.cell"}

    def test_anchor_set_records_use_their_selector_and_interacted_flag(self) -> None:
        anchors = [
            {"tag": "span", "text": "Inbox", "name": "Inbox", "selector": "nav > ul > li > a > span", "interacted": True},
            {"tag": "a", "name": "Inbox", "testId": "nav-inbox", "href": "https://mail.example/inbox",
             "selector": "nav > ul > li > a"},
            {"tag": "nav", "name": "Main", "ariaLabel": "Main", "selector": "nav"},
        ]
        assert select_anchor(anchors, None) == {
            "role": "", "name": "Inbox", "selector": "nav > ul > li > a > span",
        }
        clicked = {"tagName": "a", "innerText": "Inbox", "testId": "nav-inbox"}
        assert select_anchor(anchors, clicked)["selector"] == '[data-testid="nav-inbox"]'
        assert select_anchor([], None) is None


# ---------------------------------------------------------------------------
# DB: snapshots (both shapes) and clicks
# ---------------------------------------------------------------------------


def _insert(db_path: Path, event_id: str, timestamp: str, metadata: dict, kind: dict | None = None) -> None:
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO events (id, timestamp, kind_json, window_json, display_topology_json, "
        "primary_display_id, metadata_json) VALUES (?, ?, ?, ?, '[]', '', ?)",
        (
            event_id, timestamp, json.dumps(kind or {"type": "DwellSnapshot"}),
            json.dumps({"window_id": "chrome-tab-1", "app_id": "com.google.Chrome", "title": ""}),
            json.dumps(metadata),
        ),
    )
    conn.commit()
    conn.close()


class TestDbJoin:
    def test_snapshots_of_both_shapes_and_clicks(self, tmp_path: Path) -> None:
        db_path = create_daemon_db(tmp_path / "events.db")
        _insert(db_path, "tree", "2026-09-15T08:30:01.300050+00:00",
                {"url": "https://news.ycombinator.com/", "nodes": [{"tag": "a", "innerText": "x"}]})
        _insert(db_path, "anchors", "2026-09-15T08:30:03.000000+00:00",
                {"url": "https://news.ycombinator.com/item", "anchors": [], "digest": "Hacker News"})
        _insert(db_path, "other-host", "2026-09-15T08:30:02.000000+00:00",
                {"url": "https://app.slack.com/", "anchors": []})
        _insert(db_path, "click", "2026-09-15T08:30:04.000000+00:00",
                {"url": "https://news.ycombinator.com/", "target": {"tagName": "a", "innerText": "x"}},
                kind={"type": "ClickIntent", "target_description": "x<a>"})
        with WorkerDB(db_path) as db:
            snaps = db.get_dom_snapshots_near_timestamp(
                "2026-09-15T08:30:02.500Z", "news.ycombinator.com", tolerance_sec=5.0,
            )
            assert [s["event_id"] for s in snaps] == ["anchors", "tree"]
            assert snaps[0]["digest"] == "Hacker News"
            assert snaps[1]["nodes"] == [{"tag": "a", "innerText": "x"}]
            assert db.get_dom_snapshots_near_timestamp("2026-09-15T08:31:00Z", "news.ycombinator.com") == []

            clicks = db.get_click_intents("2026-09-15T08:30:00+00:00", "2026-09-15T08:30:05+00:00")
            assert [(c["event_id"], c["target"]["tagName"]) for c in clicks] == [("click", "a")]


# ---------------------------------------------------------------------------
# Procedure and packet
# ---------------------------------------------------------------------------


class TestProcedureAndPacket:
    def _template(self) -> dict:
        return {
            "slug": "daily-news", "title": "Daily news digest",
            "steps": [{
                "step": "Open the SQLite story",
                "parameters": {"app": "com.google.Chrome", "location": "https://news.ycombinator.com/"},
                "anchor": {"role": "link", "name": "SQLite's WAL mode", "selector": 'role=link[name="SQLite\'s WAL mode"]'},
                "evidence": {"frames": ["f1"], "dom_snapshot_ids": ["dom-1"]},
            }, {
                "step": "Read it",
            }],
        }

    def test_procedure_carries_anchor_and_evidence(self) -> None:
        proc = sop_to_procedure(self._template())
        first, second = proc["steps"]
        assert first["anchor"]["role"] == "link"
        assert first["evidence"] == {"frames": ["f1"], "dom_snapshot_ids": ["dom-1"]}
        assert second["anchor"] is None
        assert second["evidence"] == {"frames": [], "dom_snapshot_ids": []}
        assert validate_procedure(proc) == []

    def test_validation_rejects_malformed_anchor_or_evidence(self) -> None:
        proc = sop_to_procedure(self._template())
        proc["steps"][0]["anchor"] = "a.link"
        proc["steps"][1]["evidence"] = {"frames": "f1"}
        errors = validate_procedure(proc)
        assert any("anchor" in e for e in errors)
        assert any("evidence" in e for e in errors)

    def test_packet_renders_the_anchor_and_the_app_name(self) -> None:
        body = render_packet(sop_to_procedure(self._template()), target="generic")
        assert "   App: Google Chrome" in body
        assert "com.google.Chrome" not in body
        assert "   Anchor: link \"SQLite's WAL mode\" (`role=link[name=\"SQLite's WAL mode\"]`)" in body


# ---------------------------------------------------------------------------
# The focus path over the focus_dailynews session rows
# ---------------------------------------------------------------------------


def _annotate_frames_like_a_reader(db_path: Path) -> list[dict]:
    """Complete the focus frames' annotations with the app and location a
    reader of each frame would write (the Chrome frames' location is the
    tab's URL at that moment, as the address bar shows it).  Returns the
    focus session's events."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM events ORDER BY julianday(timestamp)").fetchall()
    tab_url = ""
    for row in rows:
        meta = json.loads(row["metadata_json"] or "{}")
        window = json.loads(row["window_json"] or "null") or {}
        if window.get("app_id") == "com.google.Chrome" and meta.get("url"):
            tab_url = meta["url"]
        if not meta.get("focus_session_id") or not meta.get("screenshot_path"):
            continue
        app = display_name(window.get("app_id"))
        location = tab_url if app == "Google Chrome" else window.get("title", "")
        annotation = {"app": app, "location": location, "task_context": {"what_doing": window.get("title", "")}}
        conn.execute(
            "UPDATE events SET scene_annotation_json = ?, annotation_status = 'completed' WHERE id = ?",
            (json.dumps(annotation), row["id"]),
        )
    conn.commit()
    conn.close()


@pytest.fixture()
def dailynews(tmp_path: Path):
    from journeys.fixture_loader import load_fixture

    fixture = load_fixture("focus_dailynews", tmp_path / "data")
    _annotate_frames_like_a_reader(fixture.db_path)
    with WorkerDB(fixture.db_path) as db:
        session_id = fixture.manifest["focus_session"]["session_id"]
        events = db.get_focus_session_events(session_id)
        processor = FocusProcessor(annotator=None, differ=None, sop_generator=None)  # type: ignore[arg-type]
        events = processor._filter_own_events(events)
        events = processor._attach_clipboard_context(events)
        timeline = processor._collect_timeline(db, events)
        yield timeline


class TestFocusDailynewsFixture:
    def test_clicks_join_the_frame_after_them_with_their_pages_dom(self, dailynews) -> None:
        """Each click joins the first frame at or after it, the screen that
        shows its result (WP-F3: on the list's frame a story's link became
        the anchor of the step before the click); the element is looked up
        in the DOM of the page clicked on."""
        clicked = [f for f in dailynews if f["click_targets"]]
        assert sum(len(f["click_targets"]) for f in clicked) == 5
        times = [parse_timestamp(f["timestamp"]) for f in dailynews]
        for frame in clicked:
            for click in frame["click_targets"]:
                when = parse_timestamp(click["timestamp"])
                first_after = next(f for f, t in zip(dailynews, times) if t >= when)
                assert first_after is frame
                assert click["page_dom_nodes"]
        story = next(
            f for f in clicked
            if any("SQLite" in c["target"].get("innerText", "") for c in f["click_targets"])
        )
        assert story["host"] == "blog.example.org"

    def test_story_click_anchors_to_the_nested_link_node(self, dailynews) -> None:
        idx, frame = next(
            (i, f) for i, f in enumerate(dailynews)
            if any("SQLite" in c["target"].get("innerText", "") for c in f["click_targets"])
        )
        click = next(c for c in frame["click_targets"] if "SQLite" in c["target"].get("innerText", ""))
        assert frame["host"] == "blog.example.org"  # the story, shown after the click
        root_texts = {r.get("innerText") for r in click["page_dom_nodes"]}
        vlm_sop = {"steps": [{"action": "Open the SQLite WAL story on Hacker News", "frames": [idx + 1]}]}
        step = _vlm_sop_to_template(vlm_sop, timeline=dailynews)["steps"][0]
        anchor = step["anchor"]
        assert anchor["role"] == "link"
        assert anchor["name"] == "SQLite's WAL mode, explained with diagrams"
        assert anchor["name"] not in root_texts  # drawn from a nested node
        assert step["evidence"]["frames"] == [frame["event_id"]]
        assert step["evidence"]["dom_snapshot_ids"] == [frame["dom_snapshot_id"]]

    def test_slack_send_anchors_to_its_aria_label(self, dailynews) -> None:
        idx = next(
            i for i, f in enumerate(dailynews)
            if any(c["target"].get("ariaLabel") == "Send now" for c in f["click_targets"])
        )
        vlm_sop = {"steps": [{"action": "Send the digest", "frames": [idx + 1]}]}
        anchor = _vlm_sop_to_template(vlm_sop, timeline=dailynews)["steps"][0]["anchor"]
        assert anchor == {"role": "button", "name": "Send now", "selector": '[aria-label="Send now"]'}

    def test_copied_text_reaches_the_prompt(self, dailynews) -> None:
        prompt = _build_focus_prompt("Daily news digest", dailynews)
        assert (
            '>>> Copied text: "SQLite\'s WAL mode, explained with diagrams | '
            'https://blog.example.org/sqlite-wal-explained"'
        ) in prompt
        assert "COPY ACTION: User copied content FROM Google Chrome" in prompt
