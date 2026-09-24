"""The passive_3weeks fixture, its labels, and the replay that scores
passive discovery against them (WP-M4)."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from agenthandover_worker.daily_digest import DigestGenerator
from agenthandover_worker.knowledge_base import KnowledgeBase
from daemon_db import kb_db
from journeys.cassette import CASSETTE_DIR
from journeys.fixture_loader import (
    ANNOTATIONS_FILE,
    EVENTS_FILE,
    FIXTURES_DIR,
    LABELS_FILE,
    read_annotations,
    read_labels,
    read_manifest,
    read_rows,
)
from journeys.replay import Candidate, _daily_batch, replay, score

NAME = "passive_3weeks"


def _units(labels: dict) -> Counter:
    ids: Counter = Counter()
    for routine in labels["routines"]:
        for inst in routine["instances"]:
            ids.update(inst["event_ids"] + inst["extra_event_ids"])
    for group in labels["one_off_tasks"] + labels["noise"]:
        ids.update(group["event_ids"])
    return ids


def test_every_annotated_frame_has_exactly_one_label() -> None:
    labelled = _units(read_labels(NAME))
    annotated = {a["id"] for a in read_annotations(NAME)}
    assert set(labelled) == annotated
    assert max(labelled.values()) == 1
    assert read_manifest(NAME)["annotated_rows"] == len(annotated)


def test_no_row_carries_matched_procedure() -> None:
    """Which procedure a task matches is the production matcher's answer."""
    for name in (EVENTS_FILE, ANNOTATIONS_FILE):
        with gzip.open(FIXTURES_DIR / NAME / name, "rt", encoding="utf-8") as fh:
            assert "matched_procedure" not in fh.read()
    assert "matched_procedure" not in (FIXTURES_DIR / NAME / LABELS_FILE).read_text()


def test_annotations_follow_the_annotators_schema() -> None:
    from agenthandover_worker.scene_annotator import ANNOTATION_JSON_SCHEMA

    required = set(ANNOTATION_JSON_SCHEMA["required"])
    for ann in read_annotations(NAME):
        assert ann["annotation_status"] == "completed"
        stored = json.loads(ann["scene_annotation_json"])
        # Keys starting with "_" are the worker's own (e.g. _screen_text).
        assert {k for k in stored if not k.startswith("_")} == required


def test_planted_variation() -> None:
    labels = read_labels(NAME)
    routines = {r["slug"]: r for r in labels["routines"]}
    rows = {r["id"]: r for r in read_rows(NAME)}
    days = {datetime.fromisoformat(r["timestamp"]).weekday() for r in rows.values()}
    assert days <= {0, 1, 2, 3, 4}  # weekends off

    daily = routines["stripe-payout-reconciliation"]["instances"]
    assert len(daily) == 15
    variants = Counter(v for inst in daily for v in inst["variant"])
    assert variants["evening"] == 1 and variants["sheet_first"] >= 1 and variants["extra_tab"] >= 1
    evening = next(i for i in daily if "evening" in i["variant"])
    assert datetime.fromisoformat(evening["start"]).hour == 21
    texts = [
        json.loads(a["scene_annotation_json"])["location"] for a in read_annotations(NAME)
        if "/payouts/po_" in json.loads(a["scene_annotation_json"])["location"]
    ]
    assert len(set(texts)) == len(daily)  # a different payout id each day

    weekly = routines["weekly-expense-report"]["instances"]
    assert [datetime.fromisoformat(i["start"]).weekday() for i in weekly] == [0, 0, 0]
    interrupted = next(i for i in weekly if "interrupted_40min_call" in i["variant"])
    stamps = sorted(datetime.fromisoformat(rows[e]["timestamp"]) for e in interrupted["event_ids"])
    assert max(b - a for a, b in zip(stamps, stamps[1:])) >= timedelta(minutes=40)
    assert len(labels["one_off_tasks"]) == len({t["task"] for t in labels["one_off_tasks"]})


def _cand(segments: list[list[str]], at: str = "2026-09-02T10:00:00+00:00") -> Candidate:
    return Candidate(sop_id="s", slug="x", title="x", source_id="k", generated_at=at, segments=segments)


def test_score_verdicts_and_metrics() -> None:
    labels = {
        "routines": [
            {"slug": "a", "instances": [
                {"start": "2026-09-01T09:00:00+00:00", "event_ids": ["a1", "a2"], "extra_event_ids": ["ax"]},
                {"start": "2026-09-02T09:00:00+00:00", "event_ids": ["a3"], "extra_event_ids": []},
            ]},
            {"slug": "b", "instances": [
                {"start": "2026-09-01T10:00:00+00:00", "event_ids": ["b1"], "extra_event_ids": []},
            ]},
        ],
        "one_off_tasks": [{"event_ids": ["t1"]}],
        "noise": [{"event_ids": ["n1"]}],
    }
    good = _cand([["a1", "a2", "ax", "n1"], ["a3", "unlabelled"]])
    chimera = _cand([["a1"], ["t1"]])
    one_off = _cand([["t1"], ["n1"]])
    precision, recall, chimeras, routines = score(
        [good, chimera, one_off], labels, datetime.fromisoformat("2026-09-01T00:00:00+00:00"),
    )
    assert (good.verdict, chimera.verdict, one_off.verdict) == (
        "true_positive", "chimera", "false_positive",
    )
    assert good.routine == "a"
    assert precision == pytest.approx(1 / 3)
    assert recall == 0.5
    assert chimeras == 1
    assert routines["a"]["days_to_discovery"] == pytest.approx(1 + 1 / 24, abs=0.01)
    assert routines["b"]["days_to_discovery"] is None


CASSETTE = CASSETTE_DIR / f"{NAME}.jsonl"


@pytest.mark.skipif(not CASSETTE.exists(), reason="passive_3weeks cassette not recorded")
def test_replay_on_passive_3weeks() -> None:
    """Three weeks through run_tick on the virtual clock, models from the
    cassette: the routines are found, nothing else is, no chimera, and
    every day's digest has content."""
    manifest = read_manifest(NAME)
    start = datetime.fromisoformat(manifest["recorded_start"]).replace(
        hour=0, minute=0, second=0, microsecond=0)
    report = replay(
        fixture=NAME, labels=read_labels(NAME),
        start=start, end=start + timedelta(days=21),
        cassette=CASSETTE, record=None,
        # The tier the cassettes are recorded on (qwen3.5:4b for both roles),
        # not the RAM tier of the machine running the test (conftest's rule).
        vlm={"tier": "recommended"},
    )
    assert report.failures(min_recall=1.0, min_precision=0.9, max_chimeras=0) == []


class _Day:
    """``_daily_batch``'s collaborators for one day: its events, and its
    summary with one task per entry of *minutes*."""

    def __init__(self, kb, minutes: list[int]) -> None:
        self.kb = kb
        self.minutes = minutes

    def get_annotated_events_for_date(self, date: str) -> list[dict]:
        return [{"id": "e1"}]

    def process_day(self, date: str, events: list[dict]) -> SimpleNamespace:
        tasks = [
            {"intent": f"Task {n}", "duration_minutes": m, "apps": ["Slack"]}
            for n, m in enumerate(self.minutes)
        ]
        self.kb.save_daily_summary(date, {"tasks": tasks})
        return SimpleNamespace(active_hours=sum(self.minutes) / 60, tasks=tasks)


@pytest.mark.parametrize(("minutes", "failures"), [
    ([32], []),
    ([2], []),  # a two-minute day has recorded work, and a highlight
    ([0], ["digest active_hours is 0", "task durations sum to 0 minutes"]),
])
def test_the_digest_check_asks_what_the_digest_promises(
    tmp_path: Path, minutes: list[int], failures: list[str],
) -> None:
    """The replay's "digest has no highlights" check and the production
    digest agree: a highlight on every day with recorded work, and a day
    without it is not reported for its missing highlight."""
    kb = KnowledgeBase(root=tmp_path / "knowledge", db_path=kb_db(tmp_path / "knowledge"))
    kb.ensure_structure()
    day = _Day(kb, minutes)
    (check,) = _daily_batch(day, day, DigestGenerator(kb), date(2026, 8, 31))
    assert check.failures == failures
    assert (check.highlights > 0) == (sum(minutes) > 0)
