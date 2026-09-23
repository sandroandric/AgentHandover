"""Generate the ``passive_volatile_hour`` fixture: one hour of passive
capture at the daemon's ceiling (20 frames a minute, 1,200 frames) while
the user hops between Slack, a terminal, an editor and a few browser tabs.

Rows follow the daemon's passive capture as in ``generate_passive_privacy``:
native ``DwellSnapshot`` frames with ``screenshot_path`` and Vision OCR, and
an ``AppSwitch`` or ``WindowTitleChange`` row on every switch.  The screen
changes the way a busy hour does: the context switches every 20-60 s, the
terminal scrolls, Slack gains messages, so consecutive frames rarely match
exactly while most of them repeat what an earlier frame already showed.

Usage (from the repository root)::

    .venv/bin/python worker/tests/journeys/record/generate_passive_volatile_hour.py

The output is deterministic.
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

JOURNEYS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JOURNEYS_DIR.parent))  # worker/tests, for daemon_db

from daemon_db import schema_version  # noqa: E402
from journeys.fixture_loader import (  # noqa: E402
    FIXTURES_DIR,
    row_counts,
    write_manifest,
    write_rows,
)
from journeys.record import generate_passive_privacy as passive  # noqa: E402

NAME = "passive_volatile_hour"
SEED = 20260917
START = datetime(2026, 9, 15, 11, 0, 0, tzinfo=timezone.utc)
FRAMES = 1200
FRAME_INTERVAL_S = 3.0  # 20 a minute, the daemon's screenshot_max_per_minute

SLACK = "pid:733:Slack"
TERMINAL = "pid:655:Terminal"
EDITOR = "pid:1201:Code"
CHROME = "pid:812:Google Chrome"
WINDOWS = {
    SLACK: {"window_id": "3301", "bounds": [0, 33, 1280, 900]},
    TERMINAL: {"window_id": "2210", "bounds": [80, 80, 1100, 760]},
    EDITOR: {"window_id": "5122", "bounds": [0, 33, 1512, 949]},
    CHROME: {"window_id": "4127", "bounds": [0, 33, 1512, 949]},
}
TABS = [
    ("Grafana - worker latency", ["grafana.example.com/d/worker", "p95 annotation latency", "queue depth"]),
    ("PR #412 - fix retention - GitHub", ["github.com/example/agent/pull/412", "Files changed 6", "Checks passing"]),
    ("Incident 2291 - PagerDuty", ["example.pagerduty.com/incidents/2291", "Triggered", "worker backlog"]),
]


class Generator(passive.Generator):
    def __init__(self) -> None:
        super().__init__()
        self.rng = random.Random(SEED)
        self.windows = WINDOWS

    def _time(self, offset: float) -> datetime:
        return START + timedelta(seconds=offset, microseconds=self.rng.randrange(1, 999))


def screen(context: tuple[str, int], tick: int) -> tuple[str, str, list[str]]:
    """(app_id, title, OCR lines) for a context at frame number *tick*."""
    app, variant = context
    if app == SLACK:
        channel = ["eng-oncall", "deploys", "random"][variant % 3]
        count = 12 + tick // 40
        return app, f"#{channel} - Example Co - Slack", [
            "Example Co", f"# {channel}", *[f"message {i}: backlog at {80 + i} frames" for i in range(count - 3, count)],
        ]
    if app == TERMINAL:
        return app, "zsh - worker - 120x40", [
            "$ tail -f worker.log", *[f"INFO annotated event {tick - i:05d}" for i in range(4)],
        ]
    if app == EDITOR:
        name = ["main.py", "scene_annotator.py", "db.py"][variant % 3]
        return app, f"{name} - agent - Visual Studio Code", [
            "EXPLORER", name, f"Ln {100 + (tick * 7) % 400}, Col 1", "def process_annotations(db):",
        ]
    title, lines = TABS[variant % len(TABS)]
    return CHROME, title, [*lines, f"updated {tick % 60}s ago"]


def build() -> Generator:
    gen = Generator()
    rng = random.Random(SEED + 1)
    contexts = [(SLACK, 0), (TERMINAL, 0), (EDITOR, 0), (CHROME, 0), (SLACK, 1),
                (EDITOR, 1), (CHROME, 1), (CHROME, 2), (EDITOR, 2), (SLACK, 2)]
    current = contexts[0]
    switch_at = 0.0
    for tick in range(FRAMES):
        offset = 0.5 + tick * FRAME_INTERVAL_S
        if offset >= switch_at:
            previous, current = current, rng.choice([c for c in contexts if c != current])
            switch_at = offset + rng.uniform(20, 60)
            app, title, _ = screen(current, tick)
            if app != previous[0]:
                gen.native(offset - 0.3, {"type": "AppSwitch", "from_app": previous[0], "to_app": app},
                           app, title)
            else:
                gen.native(offset - 0.3, {"type": "WindowTitleChange"}, app, title)
        app, title, lines = screen(current, tick)
        gen.passive_frame(offset, app, title, lines)
    gen.rows.sort(key=lambda item: item[0])
    return gen


def manifest_for(gen: Generator) -> dict:
    rows = [row for _, row in gen.rows]
    return {
        "name": NAME,
        "description": (
            "One hour of passive capture at 20 frames a minute (1,200 frames) "
            "while switching between Slack, Terminal, VS Code and three Chrome "
            "tabs every 20-60 s."
        ),
        "source": {
            "kind": "generated",
            "by": "worker/tests/journeys/record/generate_passive_volatile_hour.py",
        },
        "schema_version": schema_version(),
        "recorded_start": rows[0]["timestamp"],
        "recorded_end": rows[-1]["timestamp"],
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
