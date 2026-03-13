"""Shared event-field extraction helpers.

Canonical functions for parsing ``scene_annotation_json``,
``window_json``, and extracting common fields (app, what_doing,
location, timestamp) from event dicts.  All modules that work with
raw event rows should import from here instead of reimplementing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone


def parse_annotation(event_or_json: dict | str | None) -> dict | None:
    """Parse a scene annotation from an event dict or raw JSON string.

    Accepts either:
    - A full event dict with a ``scene_annotation_json`` key.
    - A raw JSON string.
    - ``None`` (returns ``None``).

    Returns the parsed annotation dict, or ``None`` if missing/invalid.
    """
    if event_or_json is None:
        return None

    if isinstance(event_or_json, dict):
        raw = event_or_json.get("scene_annotation_json")
        if not raw:
            return None
        if isinstance(raw, dict):
            return raw
        value = raw
    else:
        value = event_or_json

    try:
        ann = json.loads(value) if isinstance(value, str) else value
        return ann if isinstance(ann, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def extract_app(ann: dict, event: dict | None = None) -> str:
    """Extract the active application name from an annotation.

    Checks (in order):
    1. ``ann["app"]`` (v1 flat format)
    2. ``ann["visual_context"]["active_app"]`` (v2 nested format)
    3. ``event["window_json"]["app"]`` (fallback)

    Returns an empty string if no app name is found.
    """
    app = ann.get("app", "")
    if app:
        return app

    vc = ann.get("visual_context")
    if isinstance(vc, dict):
        app = vc.get("active_app", "")
        if app:
            return app

    if event is not None:
        wj = event.get("window_json")
        if wj:
            try:
                wdata = json.loads(wj) if isinstance(wj, str) else wj
                if isinstance(wdata, dict):
                    return wdata.get("app", wdata.get("app_name", ""))
            except (json.JSONDecodeError, TypeError):
                pass

    return ""


def extract_app_from_event(event: dict) -> str:
    """Extract app name directly from an event's ``window_json``."""
    wj = event.get("window_json")
    if not wj:
        return ""
    try:
        w = json.loads(wj) if isinstance(wj, str) else wj
        return w.get("app", "") if isinstance(w, dict) else ""
    except (json.JSONDecodeError, TypeError):
        return ""


def extract_what_doing(ann: dict) -> str:
    """Extract the ``what_doing`` string from an annotation.

    Checks both ``task_context.what_doing`` (nested) and top-level
    ``what_doing`` (flat format).
    """
    tc = ann.get("task_context")
    if isinstance(tc, dict):
        wd = tc.get("what_doing", "")
        if wd:
            return wd
    return ann.get("what_doing", "")


def extract_location(ann: dict) -> str:
    """Extract a URL or location string from an annotation.

    Checks both top-level ``location`` and ``visual_context.location``.
    """
    loc = ann.get("location", "")
    if loc:
        return loc
    vc = ann.get("visual_context")
    if isinstance(vc, dict):
        return vc.get("location", "")
    return ""


def parse_timestamp(ts: str | int | float | None) -> datetime | None:
    """Parse a timestamp value to a timezone-aware datetime.

    Handles:
    - ISO-8601 strings (with ``Z`` or ``+HH:MM`` suffix)
    - Numeric epoch values (int or float)
    - ``None`` or empty string (returns ``None``)
    """
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if not ts:
        return None
    try:
        cleaned = str(ts).replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except (ValueError, TypeError):
        return None
