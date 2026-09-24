"""The passive_3weeks fixture, its labels, and the replay that scores
passive discovery against them (WP-M4)."""

from __future__ import annotations

import gzip
import json
import os
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from agenthandover_worker.daily_digest import DigestGenerator
from agenthandover_worker.knowledge_base import KnowledgeBase
from daemon_db import kb_db
from journeys.cassette import CASSETTE_DIR, load_entries
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
from journeys.replay import Candidate, _day_checks, replay, score

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
# The calls the replay makes that the cassette lacks since it runs the
# worker's own daily pass (stages.daily), found with AH_RECORD=1
# AH_RECORD_UPSTREAM at a closed port: the ones whose request depends on no
# other missing answer (5,860 distinct requests at the closed port).
# Recording them unblocks the session linker's other embeddings (it embeds
# each intent once; at a closed port it retried every pair) and its YES/NO
# checks (they depend on the vectors); record with AH_RECORD=1 on the test
# Mac until the replay misses nothing.
DAILY_PASS_MISSING_KEYS: tuple[str, ...] = (
    # The daily summary's embedding, one per day with work.
    "c8fc20b77844f04cd4f85e7b8bf38307e9b696045b329fd10515608ac16a0da3",  # 2026-08-31
    "1ee5d88c81a32ba356951a14a890bf5387ff5b7bae5ba035420a9a33fd5c5da3",  # 2026-09-01
    "ba64b47a097a9f3adcb5c95fdcbd8d07194d58a3bb928a25395b388ae6746313",  # 2026-09-02
    "d9600a55ffffdecd5e17019a858be4e753499dec6aac68fee40dcd5c88218a99",  # 2026-09-03
    "034105419c8d12db13b40e4ee1628f2aaf02faf623dbb5e5fa88f74f97b5eb8b",  # 2026-09-04
    "74410f4a07c147c131349fbd5b5e366edb054d37bde1cac04c62f62e906a2690",  # 2026-09-07
    "5c3440624e5bb8874cc7f8da96c91b3e4e8c900df9fb1d413a795290084c0c68",  # 2026-09-08
    "f720cd90f0e4ec7a829cb5cf5ab437c6f178c01a9aa403143406dbd37a41496c",  # 2026-09-09
    "32b2ce96027da9a8f2f28c304d8fb648b2efa83787f1ddb8cf82f5460b1bc660",  # 2026-09-10
    "e7d7b2292eb2fae335d16c7e569069aebfd2d193e8eeb8869f0052fc6d38f86b",  # 2026-09-11
    "6d8626e0f1609575c06f2ec727a405654b1b619ef0f498e1224536ed08caf1f8",  # 2026-09-14
    "29e607e87c3e73567ec314bfbbfc935ba9af1d6d7621538e80fba92a9daee0fa",  # 2026-09-15
    "5caf1fb4dc0a3834e8f7b8067ede5477d8fef09b03096abc4afb209424a8b769",  # 2026-09-16
    "ec62ca70fe0ab296d8549e7eb0f59e1e4c7954f9050e48d477b2e2732dfb09aa",  # 2026-09-17
    "76bfe871abb86b57e4b25636494aa518917fa44276e3f9ef7263e67cea553c87",  # 2026-09-18
    # The profile's embedding, from the third summary on (no procedure is
    # approved in the replay, so it carries no voice profile).
    "b3d99754909c847b4c5ed40c36cdb5a2dcfae8e9334cadd55cea61134ed86d1e",  # 2026-09-02
    "3c6d501fd93162728acfb723d18cdc59f476039e26319382accecac7e7f81995",  # 2026-09-03
    "791b20e2e8f00120b60e97b43650190bf5181e060393df94e13b714b991a6cac",  # 2026-09-04
    "1daf3a0681117c619361711da31af8d0a9db097916d64fa1e1fd1204c31deba7",  # 2026-09-07
    "31d1d551e73bb5c0809a3a82eac2478d679e9b7a7bcaab2f8b8b6de83d9a52f6",  # 2026-09-08
    "95c8476235dcb8ecc61a15680905a1e1061b731202f60b5c57494df5283af51d",  # 2026-09-09
    "26ac105e0a62cc5d3f57b2bb83e006f579c518f7219a7470ea973012e593bd97",  # 2026-09-10
    "1714af954f356a4c4cf5180bda4d824dcdee7ec795f9bfa82dcb3121d947c153",  # 2026-09-11
    "8b6518a3da480abe5d7804a55aeb31e19e83e4dd6a1b88385d8e12b386403291",  # 2026-09-14
    "ac2afb49de4766ed27ba01e8fc06890b836c80aa33482336ed1ec0b5d3897840",  # 2026-09-15
    "86511cbc66dd6600c4d08a66bc2c2f0bfa4b2079098660693d04f0218e7c5d83",  # 2026-09-16
    "5fe84b91b7dd60ff7104659f5675eb77ea676ab38601d15be26e5ae13ec364e9",  # 2026-09-17
    "192aaf0f39f1a1240a4259b91dc4431e311ead21e6accfdb8e78e66d2be7a5fb",  # 2026-09-18
    # The session linker's first intent embedding (day 3).
    "783426732123516105e760924f9131a055e04ba39190a7b17a0805d68a9a8a85",
)
_ENTRIES = load_entries(CASSETTE) if CASSETTE.exists() else {}
needs_daily_recording = pytest.mark.skipif(
    any(key not in _ENTRIES for key in DAILY_PASS_MISSING_KEYS)
    and os.environ.get("AH_RECORD") != "1",
    reason=(
        f"cassettes/{NAME}.jsonl lacks the daily pass's model calls: "
        + ", ".join(k for k in DAILY_PASS_MISSING_KEYS if k not in _ENTRIES)
        + " (and the calls they unblock); record with AH_RECORD=1 on the test Mac"
    ),
)


@needs_daily_recording
def test_replay_on_passive_3weeks() -> None:
    """Three weeks through run_tick on the virtual clock, with the worker's
    daily pass, models from the cassette: the routines are found, nothing
    else is, no chimera, and every weekday's digest has content.
    (Discovery alone is pinned now by test_j_passive.py's drafts test.)"""
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
    assert len(report.days) == 15  # every weekday of the three weeks


def _save_day(kb: KnowledgeBase, day: str, minutes: list[int]) -> None:
    """What the daily pass saves for *day*: its summary, with one task per
    entry of *minutes*, then the production digest of it."""
    tasks = [
        {"intent": f"Task {n}", "duration_minutes": m, "apps": ["Slack"]}
        for n, m in enumerate(minutes)
    ]
    kb.save_daily_summary(day, {"tasks": tasks, "active_hours": sum(minutes) / 60})
    digests = DigestGenerator(kb)
    digests.save_digest(digests.generate(day))


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
    _save_day(kb, "2026-08-31", minutes)
    (check,) = _day_checks(kb, date(2026, 8, 31))
    assert check.failures == failures
    assert (check.highlights > 0) == (sum(minutes) > 0)


def test_a_day_without_its_digest_fails_the_check(tmp_path: Path) -> None:
    """A summary without a digest is the daily pass stopping between them
    (as it did on every day with work while it read ``TaskBoundary``
    records as dicts); a day without a summary had no work and is not
    checked."""
    kb = KnowledgeBase(root=tmp_path / "knowledge", db_path=kb_db(tmp_path / "knowledge"))
    kb.ensure_structure()
    kb.save_daily_summary("2026-08-31", {"tasks": [{"duration_minutes": 30}], "active_hours": 0.5})
    (check,) = _day_checks(kb, date(2026, 8, 31))
    assert check.failures == ["no digest was saved"]
    assert _day_checks(kb, date(2026, 9, 1)) == []
