"""J-annotate: text-first annotation of the focus_dailynews recording with
the real annotation model (qwen3.5:4b answers from the cassette).

One frame per distinct screen of the recording (12 screens: the Hacker
News front page, three stories, the Notes digest while typing and after
each summary, the Slack channel while pasting and after posting) is
annotated the way the focus processor does it: the browser page from the
Chrome extension's DOM snapshot, stale-skip off.  The journey checks

* the call pattern: a text-only call for every screen with enough text,
  the screenshot only for short-text screens or when the model asks;
* the token bounds: every call's ``prompt_eval_count`` <= 1,700 and
  ``eval_count`` <= the 300-token ``ann_num_predict``;
* extraction quality against hand labels from the fixture's own text
  (app, location, emails, typed text, what the user is doing).

The labels are shared with the one-off old-versus-new comparison run when
the schema changed (reported in the WP-C3b PR), so they are written from
the fixture's text, not from any model's answer.

Re-record with ``AH_RECORD=1 AH_RECORD_UPSTREAM=<ollama>``.
"""

from __future__ import annotations

import json

from agenthandover_worker.dom import url_host
from agenthandover_worker.scene_annotator import as_compact, page_for_event

MODEL = "qwen3.5:4b"

_NOTES = ("note", "digest", "summar")
_SLACK = ("slack", "daily-news", "daily news", "channel", "post", "message")

# One label per distinct screen, in recording order.  ``typed``: "empty"
# (nothing is being typed), "screen" (empty, or text shown on the screen),
# or a string the typed text must contain.
LABELS: list[dict] = [
    {"screen": "HN front page", "app": "chrome", "host": "news.ycombinator.com",
     "typed": "empty", "doing": ("hacker news",)},
    {"screen": "story: SQLite WAL", "app": "chrome", "host": "blog.example.org",
     "typed": "empty", "doing": ("sqlite", "wal")},
    {"screen": "Notes: story 1 pasted, typing", "app": "notes", "host": None,
     "typed": "screen", "doing": _NOTES},
    {"screen": "Notes: story 1 summary", "app": "notes", "host": None,
     "typed": "screen", "doing": _NOTES},
    {"screen": "story: on-device vision", "app": "chrome", "host": "research.example.com",
     "typed": "empty", "doing": ("vision", "on-device")},
    {"screen": "Notes: story 2 pasted, typing", "app": "notes", "host": None,
     "typed": "screen", "doing": _NOTES},
    {"screen": "Notes: story 2 summary", "app": "notes", "host": None,
     "typed": "screen", "doing": _NOTES},
    {"screen": "story: HTTP cassette", "app": "chrome", "host": "github.com",
     "typed": "empty", "doing": ("cassette",)},
    {"screen": "Notes: story 3 pasted, typing", "app": "notes", "host": None,
     "typed": "screen", "doing": _NOTES},
    {"screen": "Notes: story 3 summary", "app": "notes", "host": None,
     "typed": "screen", "doing": _NOTES},
    {"screen": "Slack: digest pasted", "app": "chrome", "host": "app.slack.com",
     "typed": "daily news - 15 sep", "doing": _SLACK},
    {"screen": "Slack: digest posted", "app": "chrome", "host": "app.slack.com",
     "typed": "screen", "doing": _SLACK},
]

FIELDS = ("app", "location", "emails", "typed_text", "what_doing")


def labelled_frames(db) -> list[tuple[dict, dict]]:
    """The first focus frame of each distinct screen (window title and OCR
    text), paired with its label."""
    rows = db._conn.execute(
        "SELECT * FROM events WHERE json_extract(metadata_json, '$.focus_session_id') "
        "IS NOT NULL AND json_extract(metadata_json, '$.ocr') IS NOT NULL "
        "ORDER BY timestamp"
    ).fetchall()
    frames: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        event = dict(row)
        ocr = json.loads(event["metadata_json"])["ocr"]
        text = "\n".join(e["text"] for e in ocr["elements"])
        key = (json.loads(event["window_json"]).get("title", ""), text)
        if key not in seen:
            seen.add(key)
            frames.append(event)
    assert len(frames) == len(LABELS), len(frames)
    return list(zip(frames, LABELS))


def score(annotation: dict, label: dict, event: dict) -> dict[str, bool]:
    """Per-field agreement of *annotation* (either schema) with *label*."""
    ann = as_compact(annotation)
    kt = ann.get("key_text") or {}
    location = str(ann.get("location") or "")
    typed = " ".join(str(kt.get("typed_text") or "").split()).lower()
    screen = " ".join(
        e["text"] for e in json.loads(event["metadata_json"])["ocr"]["elements"]
    ).lower()
    if label["typed"] == "empty":
        typed_ok = typed == ""
    elif label["typed"] == "screen":
        typed_ok = typed == "" or typed[:40] in screen
    else:
        typed_ok = label["typed"] in typed
    if label["host"]:
        location_ok = url_host(location) == label["host"]
    else:
        location_ok = any(k in location.lower() for k in ("notes", "daily news"))
    doing = str((ann.get("task_context") or {}).get("what_doing") or "").lower()
    return {
        "app": label["app"] in str(ann.get("app") or "").lower(),
        "location": location_ok,
        "emails": list(kt.get("emails") or []) == [],
        "typed_text": typed_ok,
        "what_doing": any(k in doing for k in label["doing"]),
    }


def test_text_first_annotation_on_the_focus_recording(
    temp_home, events_db, cassette, worker,
) -> None:
    tape = cassette("text_first_annotation")
    temp_home.config_file.write_text(
        f'[vlm]\nollama_host = "{tape.host}"\nannotation_model = "{MODEL}"\n\n'
        "[knowledge]\nquery_api_enabled = false\n",
        encoding="utf-8",
    )
    fixture = events_db("focus_dailynews")
    w = worker(fixture)
    annotator = w.scene_annotator
    assert annotator is not None and annotator.config.model == MODEL

    # As FocusProcessor._ensure_annotations does it: the sliding-window
    # context from the annotations saved so far, stale-skip off, and the
    # browser page from the extension's DOM snapshot.
    results = []
    for event, label in labelled_frames(w.db):
        recent = w.db.get_recent_annotations(
            before_timestamp=event["timestamp"],
            limit=annotator.config.sliding_window_size,
            max_age_seconds=annotator.config.sliding_window_max_age_sec,
        )
        before = len(tape.served)
        result = annotator.annotate_event(
            event, recent_annotations=recent, artifact_dir=str(fixture.screenshots_dir),
            page=page_for_event(w.db, event),
        )
        calls = [
            e for e in tape.served[before:]
            if e["path"] == "/api/generate" and e["request"].get("format")
        ]
        results.append((event, label, result, calls))
        if result.status == "completed":
            w.db.save_annotation(event["id"], json.dumps(result.annotation), status="completed")

    agreement = {f: 0 for f in FIELDS}
    for event, label, result, calls in results:
        assert result.status == "completed", (label["screen"], result.error)
        ann = result.annotation
        # Call pattern: text first; the screenshot only when needed.
        first = calls[0]["request"]
        if ann.get("_vision") in ("short_text", "canvas_app"):
            assert len(calls) == 1 and first.get("images"), label["screen"]
        else:
            assert "images" not in first, label["screen"]
            assert len(calls) == (2 if ann.get("_vision") == "model" else 1)
        # Token bounds, from Ollama's own counts.
        for call in calls:
            response = call["response"]
            assert response["prompt_eval_count"] <= 1700, (label["screen"], response["prompt_eval_count"])
            assert response["eval_count"] <= 300, (label["screen"], response["eval_count"])
            assert response.get("done_reason") == "stop", label["screen"]
        for field, ok in score(ann, label, event).items():
            agreement[field] += ok

    # The compact schema's answers are short (the old schema's were
    # 360-510 tokens on these screens).
    eval_counts = sorted(c["response"]["eval_count"] for *_, calls in results for c in calls)
    assert eval_counts[len(eval_counts) // 2] <= 120, eval_counts

    # The static instructions are a cached prefix: every call after the
    # first reuses them from Ollama's prompt cache.
    all_calls = [c["response"] for *_, calls in results for c in calls]
    for response in all_calls[1:]:
        assert response["prompt_eval_cached_count"] >= 400, response["prompt_eval_cached_count"]

    # Short screens go to vision: the Notes digest with one pasted story
    # and the Slack channel while pasting.
    vision = {label["screen"]: r.annotation.get("_vision") for _, label, r, _ in results}
    assert vision["Notes: story 1 pasted, typing"] == "short_text"
    assert vision["Slack: digest pasted"] == "short_text"
    text_only = [label["screen"] for _, label, r, _ in results if "_vision" not in r.annotation]
    assert len(text_only) >= 9, text_only

    # The browser page's URL reaches every Chrome screen's location.
    n = len(results)
    assert agreement["app"] == n, agreement
    assert agreement["location"] == n, agreement
    assert agreement["emails"] == n, agreement
    assert agreement["typed_text"] >= n - 1, agreement
    assert agreement["what_doing"] >= n - 1, agreement
