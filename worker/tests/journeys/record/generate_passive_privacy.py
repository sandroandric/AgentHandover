"""Generate the ``passive_privacy`` fixture: one minute of passive capture
that touch everything J-privacy guards, in the row format the daemon writes.

The user opens a GitHub login in 1Password, copies the password, pastes it
into a Gmail draft to a colleague, then opens a password-reset link whose
URL carries a ``?token=`` query.  Rows follow the daemon as of WP-V1:

* native passive frames (``DwellSnapshot`` with ``screenshot_path`` and the
  Vision OCR result, no ``focus_session_id``) and ``AppSwitch`` /
  ``WindowTitleChange`` rows with ``app_id`` ``pid:<pid>:<owner name>``
  (``crates/daemon/src/observer/event_loop.rs``);
* the ``ClipboardChange`` row has no ``content_preview``: the preview is
  off unless ``[privacy] enable_clipboard_preview`` is set
  (``crates/daemon/src/capture/clipboard.rs``), and the copied bytes are
  high-entropy;
* Chrome extension rows store the page URL reduced to origin + path
  (``strip_url_to_origin_and_path`` in ``crates/daemon/src/ipc/native_messaging.rs``);
  the full URL with its token is still on screen, in the address bar the
  native frame's OCR reads.

Usage (from the repository root)::

    .venv/bin/python worker/tests/journeys/record/generate_passive_privacy.py

The output is deterministic.
"""

from __future__ import annotations

import hashlib
import random
import sys
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
from journeys.record.generate_focus_dailynews import (  # noqa: E402
    DISPLAY,
    Generator as _FocusGenerator,
    compact,
    metadata_text,
    rfc3339,
)

NAME = "passive_privacy"
SEED = 20260916
START = datetime(2026, 9, 15, 10, 0, 0, tzinfo=timezone.utc)

PASSWORD = "Tr0ub4dor&3-horse-battery"
RESET_TOKEN = "8f3a9c2e71b04d55a0c4"
RESET_PATH = "https://accounts.example.com/reset"
RESET_URL = f"{RESET_PATH}?token={RESET_TOKEN}"

ONEPASSWORD_APP = "pid:944:1Password"
CHROME_APP = "pid:812:Google Chrome"
WINDOWS = {
    ONEPASSWORD_APP: {"window_id": "6120", "bounds": [220, 120, 900, 640]},
    CHROME_APP: {"window_id": "4127", "bounds": [0, 33, 1512, 949]},
}
RESET_TAB = 1907


class Generator(_FocusGenerator):
    """The focus generator's row builders, for passive (untagged) rows."""

    def __init__(self) -> None:
        super().__init__()
        self.rng = random.Random(SEED)
        self.windows = WINDOWS

    def _time(self, offset: float) -> datetime:
        return START + timedelta(seconds=offset, microseconds=self.rng.randrange(1, 999))

    def native(self, offset: float, kind: dict, app_id: str, title: str,
               metadata: dict | None = None) -> None:
        window = self.windows[app_id]
        cx, cy = self._cursor(window)
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
            "metadata_json": metadata_text(metadata or {}),
            "display_ids_spanned_json": None,
        })

    def passive_frame(self, offset: float, app_id: str, title: str, lines: list[str]) -> None:
        elements = []
        top = 0.06
        for text in lines:
            elements.append({
                "text": text,
                "confidence": 1.0,
                "bbox_normalized": [0.05, round(top, 4), round(min(0.9, 0.012 * len(text) + 0.05), 4), 0.018],
            })
            top += 0.028
        self.native(offset, {"type": "DwellSnapshot"}, app_id, title, {
            "ocr": {
                "elements": elements,
                "full_text": "\n".join(lines),
                "processing_time_ms": self.rng.randrange(180, 420),
            },
            "screenshot_path": f"{SCREENSHOTS_TOKEN}/{self._uuid()}.jpg",
        })

    def clipboard_without_preview(self, offset: float, text: str) -> None:
        data = text.encode("utf-8")
        self._add(offset, {
            "id": str(self._uuid()),
            "timestamp": rfc3339(self._time(offset)),
            "kind_json": compact({
                "type": "ClipboardChange",
                "content_types": ["public.utf8-plain-text"],
                "byte_size": len(data),
                "high_entropy": True,
                "content_hash": hashlib.sha256(data).hexdigest(),
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


def build() -> Generator:
    gen = Generator()
    gen.native(0.4, {"type": "AppSwitch", "from_app": CHROME_APP, "to_app": ONEPASSWORD_APP},
               ONEPASSWORD_APP, "1Password")
    gen.passive_frame(2.1, ONEPASSWORD_APP, "1Password", [
        "1Password", "Personal", "All Items", "GitHub",
        "username  sandro@example.com", f"password  {PASSWORD}",
        "website  github.com/login", "Copy",
    ])
    gen.clipboard_without_preview(9.6, PASSWORD)
    gen.native(12.0, {"type": "AppSwitch", "from_app": ONEPASSWORD_APP, "to_app": CHROME_APP},
               CHROME_APP, "Inbox (3) - sandro@example.com - Gmail")
    gen.passive_frame(30.2, CHROME_APP, "Inbox (3) - sandro@example.com - Gmail", [
        "mail.google.com/mail/u/0/#inbox", "Gmail", "Compose", "Inbox 3",
        "New Message", "To  ops@example.com", "Subject  staging login",
        f"The staging GitHub password is {PASSWORD}", "Send",
    ])
    gen.native(48.0, {"type": "WindowTitleChange"}, CHROME_APP, "Reset your password")
    gen.extension(48.9, {"type": "FocusChange"}, RESET_TAB, RESET_PATH, None, {})
    gen.extension(56.3, {"type": "DwellSnapshot"}, RESET_TAB, RESET_PATH, "Reset your password",
                  {"title": "Reset your password", "timestamp": "2026-09-15T10:00:56.300Z"})
    gen.passive_frame(58.0, CHROME_APP, "Reset your password", [
        RESET_URL.removeprefix("https://"), "Example Accounts", "Reset your password",
        "New password", "Confirm new password", "Reset password",
    ])
    gen.rows.sort(key=lambda item: item[0])
    return gen


def manifest_for(gen: Generator) -> dict:
    rows = [row for _, row in gen.rows]
    return {
        "name": NAME,
        "description": (
            "One minute of passive capture: a GitHub login open in 1Password, "
            "its password copied and pasted into a Gmail draft, then a "
            "password-reset page whose URL carries ?token=."
        ),
        "source": {
            "kind": "generated",
            "by": "worker/tests/journeys/record/generate_passive_privacy.py",
        },
        "schema_version": schema_version(),
        "recorded_start": rows[0]["timestamp"],
        "recorded_end": rows[-1]["timestamp"],
        "secrets": {"password": PASSWORD, "reset_token": RESET_TOKEN},
        "row_counts": row_counts(rows),
    }


def main() -> None:
    gen = build()
    out = FIXTURES_DIR / NAME
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out, [row for _, row in gen.rows])
    write_manifest(out, manifest_for(gen))
    print(f"wrote {len(gen.rows)} rows to {out}")


if __name__ == "__main__":
    main()
