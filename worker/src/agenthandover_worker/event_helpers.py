"""Shared event-field extraction helpers.

Canonical functions for parsing ``scene_annotation_json``,
``window_json``, and extracting common fields (app, what_doing,
location, timestamp) from event dicts.  All modules that work with
raw event rows should import from here instead of reimplementing.
"""

from __future__ import annotations

import json
import re
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


def parse_json_field(event: dict, key: str) -> dict:
    """``event[key]`` as a dict (the column's JSON text parsed), or ``{}``."""
    raw = event.get(key)
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


# Bundle id -> display name.  The first block mirrors the daemon's
# AppleScript-queryable apps (``KNOWN_APPS`` in
# ``crates/daemon/src/platform/applescript_bridge.rs``); the rest are common
# apps whose bundle id reaches ``window_json.app_id`` (the Chrome extension
# writes ``com.google.Chrome``) or ``metadata.app_state.app_id``.
KNOWN_APPS: dict[str, str] = {
    "com.apple.finder": "Finder",
    "com.apple.mail": "Mail",
    "com.apple.Notes": "Notes",
    "com.apple.iCal": "Calendar",
    "com.apple.MobileSMS": "Messages",
    "com.apple.Safari": "Safari",
    "com.apple.Preview": "Preview",
    "com.apple.TextEdit": "TextEdit",
    "com.apple.reminders": "Reminders",
    "com.apple.AddressBook": "Contacts",
    # Beyond the AppleScript set.
    "com.google.Chrome": "Google Chrome",
    "com.google.Chrome.canary": "Google Chrome Canary",
    "company.thebrowser.Browser": "Arc",
    "com.brave.Browser": "Brave Browser",
    "com.microsoft.edgemac": "Microsoft Edge",
    "org.mozilla.firefox": "Firefox",
    "com.tinyspeck.slackmacgap": "Slack",
    "com.microsoft.VSCode": "Visual Studio Code",
    "com.apple.Terminal": "Terminal",
    "com.googlecode.iterm2": "iTerm2",
    "com.apple.dt.Xcode": "Xcode",
    "com.apple.systempreferences": "System Settings",
    "com.apple.iWork.Numbers": "Numbers",
    "com.apple.iWork.Pages": "Pages",
    "com.apple.iWork.Keynote": "Keynote",
    "com.microsoft.Excel": "Microsoft Excel",
    "com.microsoft.Word": "Microsoft Word",
    "com.microsoft.Powerpoint": "Microsoft PowerPoint",
    "com.microsoft.Outlook": "Microsoft Outlook",
    "com.microsoft.teams2": "Microsoft Teams",
    "us.zoom.xos": "zoom.us",
    "notion.id": "Notion",
    "com.figma.Desktop": "Figma",
    "com.hnc.Discord": "Discord",
    "com.spotify.client": "Spotify",
}
_KNOWN_APPS_LOWER = {k.lower(): v for k, v in KNOWN_APPS.items()}

# A bundle id: three or more dot-separated components, no spaces.
_BUNDLE_ID = re.compile(r"^[A-Za-z][\w-]*(\.[\w-]+){2,}$")


def is_bundle_id(text: str) -> bool:
    return bool(_BUNDLE_ID.match(str(text or "").strip()))


def display_name(app_id: str | None) -> str:
    """The human name of an app from the ids the daemon writes.

    ``window_json.app_id`` is ``pid:<pid>:<owner name>`` for native windows
    and a bundle id for Chrome extension events.  A known bundle id maps
    through :data:`KNOWN_APPS`; an unknown one falls back to its last
    component, so a bundle id never reaches a prompt or a rendered Skill as
    the app's name.  Anything else (already a name) comes back unchanged.
    """
    text = str(app_id or "").strip()
    if text.startswith("pid:"):
        parts = text.split(":", 2)
        return parts[2].strip() if len(parts) == 3 else ""
    known = _KNOWN_APPS_LOWER.get(text.lower())
    if known:
        return known
    if is_bundle_id(text):
        return text.rsplit(".", 1)[-1]
    return text
