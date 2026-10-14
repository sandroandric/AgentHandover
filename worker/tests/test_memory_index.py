"""The Memory index's RAM (WP-E11): 8-bit vectors, float32 re-score,
loaded on the first search and released when idle.

Every ranking is compared with a float32 brute force (the index as it was
before WP-E11: every row's float32 vector against the query, the keyword
boost over the best 500, rows of one window grouped into moments) on
random clustered data and on the real embeddings recorded in the journey
cassettes.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from agenthandover_worker import memory
from agenthandover_worker.memory import (
    KEYWORD_BOOST,
    MOMENT_GAP_SECONDS,
    SHORTLIST,
    MemoryIndex,
    _epoch,
    _quantize,
    _words,
    utc_text,
)
from daemon_db import create_daemon_db
from journeys.cassette import CASSETTE_DIR

MODEL = "embeddinggemma-2:440m"
START = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
APPS = ["Google Chrome", "Slack", "Xcode", "Terminal", "Figma", "Mail"]
VOCAB = ["invoice", "acme", "build", "design", "landing", "terminal", "report", "lunch"]


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def _db(tmp_path: Path) -> Path:
    return create_daemon_db(tmp_path / "events.db")


def _insert(db_path: Path, rows: list[tuple], *, model: str = MODEL) -> None:
    """*rows*: (id, first, last, app, title, description, vector)."""
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT INTO memory_frames (id, first_ts, last_ts, frames, app, window_title, "
        "description, embedding, model, dim) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?)",
        [
            (rid, utc_text(first), utc_text(last), app, title, description,
             np.asarray(vector, dtype="<f4").tobytes(), model, len(vector))
            for rid, first, last, app, title, description, vector in rows
        ],
    )
    conn.commit()
    conn.close()


def _random_rows(rng: np.random.Generator, n: int, dim: int, *, words: bool) -> list[tuple]:
    """Clustered vectors (near-duplicates, as a day of screens is), one
    window per stretch of a few rows, gaps that split and join moments."""
    centers = _unit_rows(rng.normal(size=(max(4, n // 60), dim)))
    labels = rng.integers(0, len(centers), n)
    vectors = _unit_rows(centers[labels] + rng.normal(scale=0.35 / np.sqrt(dim) * 4, size=(n, dim)))
    rows = []
    at = START
    window = 0
    for i in range(n):
        if rng.random() < 0.3:
            window = int(rng.integers(0, 40))
        at += timedelta(seconds=float(rng.choice([30, 120, 900])))
        app = APPS[window % len(APPS)]
        description = " ".join(rng.choice(VOCAB, 3)) if words else f"row {i}"
        rows.append((f"r{i:05d}", at, at + timedelta(seconds=20), app, f"window {window}",
                     description, vectors[i]))
    return rows


def float32_search(
    db_path: Path, q: np.ndarray, query: str, *, dim: int, limit: int,
    since: datetime | None = None, until: datetime | None = None, app: str | None = None,
) -> list[tuple[str, list[str], float]]:
    """The float32 brute force: (best row, rows, score) per moment, best first."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = [
        r for r in conn.execute(
            "SELECT id, first_ts, last_ts, app, window_title, description, embedding "
            "FROM memory_frames WHERE model = ? AND dim = ?", (MODEL, dim),
        )
    ]
    conn.close()
    matrix = np.stack([np.frombuffer(r["embedding"], dtype="<f4") for r in rows])
    first = np.array([_epoch(r["first_ts"]) for r in rows])
    last = np.array([_epoch(r["last_ts"]) for r in rows])
    windows = {}
    window = np.array([windows.setdefault((r["app"], r["window_title"]), len(windows)) for r in rows])
    mask = np.ones(len(rows), dtype=bool)
    if since is not None:
        mask &= last >= since.timestamp()
    if until is not None:
        mask &= first <= until.timestamp()
    if app:
        mask &= np.array([r["app"].casefold() == app.casefold() for r in rows])
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return []
    scores = matrix @ q
    shortlist = idx if idx.size <= SHORTLIST else idx[np.argsort(-scores[idx])[:SHORTLIST]]
    words = _words(query)
    if words:
        for i in shortlist:
            if all(w in rows[i]["description"].lower() for w in words):
                scores[i] += KEYWORD_BOOST
    order = np.argsort(first, kind="stable")
    order = order[mask[order]]
    moments: list[list[int]] = []
    for i in order:
        if moments and window[moments[-1][-1]] == window[i] and first[i] - last[moments[-1][-1]] < MOMENT_GAP_SECONDS:
            moments[-1].append(int(i))
        else:
            moments.append([int(i)])
    best = [max(m, key=lambda i: scores[i]) for m in moments]
    ranked = sorted(range(len(moments)), key=lambda g: -scores[best[g]])[:limit]
    return [
        (rows[best[g]]["id"], [rows[i]["id"] for i in moments[g]], float(scores[best[g]]))
        for g in ranked
    ]


def _assert_same(hits: list[dict], expected: list[tuple[str, list[str], float]]) -> None:
    assert [(h["id"], h["rows"]) for h in hits] == [(e[0], e[1]) for e in expected]
    for hit, (_id, _rows, score) in zip(hits, expected):
        assert hit["score"] == pytest.approx(score, abs=1.5e-4)


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def test_quantize_is_symmetric_per_row_and_within_half_a_step():
    rng = np.random.default_rng(1)
    vectors = _unit_rows(rng.normal(size=(200, 512)))
    vectors[7] = 0.0
    q8, scale = _quantize(vectors)
    assert q8.dtype == np.int8 and scale.dtype == np.float32
    assert np.abs(q8).max() == 127 and q8.min() >= -127
    restored = q8.astype(np.float32) * scale[:, None]
    assert np.all(np.abs(restored - vectors) <= scale[:, None] / 2 + 1e-7)
    assert not q8[7].any() and np.isfinite(scale).all()


# ---------------------------------------------------------------------------
# Rankings: identical to float32
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_rankings_match_float32_on_random_clustered_rows(tmp_path, seed):
    rng = np.random.default_rng(seed)
    dim, n = 512, 3000
    rows = _random_rows(rng, n, dim, words=seed % 2 == 1)
    db_path = _db(tmp_path)
    _insert(db_path, rows)
    vectors = np.stack([r[6] for r in rows])
    queries: dict[str, np.ndarray] = {}
    for k in range(12):
        # Near a stored row: the hard case, many rows score alike.
        base = vectors[int(rng.integers(0, n))]
        text = " ".join(rng.choice(VOCAB, 2)) if seed % 2 else f"q{k}"
        queries[f"{text} #{k}"] = _unit_rows(
            (base + rng.normal(scale=0.02, size=dim))[None, :]
        )[0]
    index = MemoryIndex(db_path, model=MODEL, dim=dim, embed_query=lambda t: queries[t])
    try:
        middle = START + (rows[-1][1] - START) / 2
        for text, q in queries.items():
            for limit, filters in [
                (1, {}), (5, {}), (10, {}), (50, {}),
                (10, {"since": middle}), (10, {"until": middle}), (10, {"app": "slack"}),
            ]:
                hits = index.search(text, limit=limit, **filters)
                _assert_same(hits, float32_search(
                    db_path, q, text, dim=dim, limit=limit, **filters,
                ))
    finally:
        index.close()


def _recorded_vectors(model: str, dim: int) -> np.ndarray:
    """Every distinct embedding of *model* at *dim* in the journey cassettes."""
    seen: dict[bytes, np.ndarray] = {}
    for path in sorted(CASSETTE_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry["path"] != "/api/embed" or entry["request"].get("model") != model:
                continue
            for vector in entry["response"].get("embeddings") or []:
                v = np.asarray(vector, dtype=np.float32)
                if v.size == dim and np.linalg.norm(v) > 0:
                    v = v / np.linalg.norm(v)
                    seen.setdefault(v.tobytes(), v)
    return np.stack(list(seen.values()))


@pytest.mark.parametrize(("model", "least"), [("nomic-embed-text", 400), (MODEL, 60)])
def test_rankings_match_float32_on_the_recorded_embeddings(tmp_path, model, least):
    """The real model answers in the journey cassettes (screens, Skills,
    queries): each one as the query against all of them."""
    vectors = _recorded_vectors(model, 768)
    assert len(vectors) >= least
    rows = [
        (f"v{i:04d}", START + timedelta(minutes=30 * i), START + timedelta(minutes=30 * i, seconds=20),
         APPS[i % len(APPS)], f"window {i}", f"recorded {i}", v)
        for i, v in enumerate(vectors)
    ]
    db_path = _db(tmp_path)
    _insert(db_path, [(*r[:6], r[6]) for r in rows], model=MODEL)
    index = MemoryIndex(db_path, model=MODEL, dim=768, embed_query=lambda t: vectors[int(t[1:])])
    try:
        for i in range(len(vectors)):
            hits = index.search(f"q{i}", limit=10)
            _assert_same(hits, float32_search(db_path, vectors[i], f"q{i}", dim=768, limit=10))
    finally:
        index.close()


# ---------------------------------------------------------------------------
# RAM: what the index holds, released when idle
# ---------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _small_index(tmp_path, *, n: int = 300, dim: int = 64, idle: float = 600.0):
    rng = np.random.default_rng(5)
    rows = _random_rows(rng, n, dim, words=False)
    db_path = _db(tmp_path)
    _insert(db_path, rows)
    clock = Clock()
    vectors = {f"q{i}": rows[i][6] for i in range(n)}
    index = MemoryIndex(
        db_path, model=MODEL, dim=dim, embed_query=lambda t: vectors[t],
        idle_seconds=idle, clock=clock,
    )
    return index, clock, db_path, rows


def test_the_index_holds_a_byte_per_value(tmp_path):
    index, _clock, _db_path, _rows = _small_index(tmp_path, n=2000, dim=512)
    try:
        assert len(index) == 0 and index.nbytes == 0, "nothing is loaded before a search"
        index.search("q1")
        assert len(index) == 2000
        assert index._q8.dtype == np.int8 and index._q8.shape == (2000, 512)
        # int8 rows, a float32 scale, two float64 times, two int64 codes.
        assert index.nbytes == 2000 * (512 + 4 + 8 + 8 + 8 + 8)
        assert index.nbytes < 2000 * 512 * 4 / 3
    finally:
        index.close()


def test_the_index_is_released_after_the_idle_time_and_reloaded_by_a_search(tmp_path):
    index, clock, db_path, rows = _small_index(tmp_path)
    try:
        before = index.search("q3", limit=5)
        assert len(index) == len(rows)
        assert not index.release_if_idle(clock.now + 599)
        clock.now += 300
        index.search("q4")  # a search restarts the idle time
        assert not index.release_if_idle(clock.now + 599)
        assert index.release_if_idle(clock.now + 600)
        assert len(index) == 0 and index.nbytes == 0
        assert not index.release_if_idle(clock.now + 10_000), "nothing left to release"

        # Rows written while released are there when the next search loads.
        late = START + timedelta(days=30)
        _insert(db_path, [("late", late, late, "Notes", "late", "late", rows[3][6])])
        after = index.search("q3", limit=6)
        assert len(index) == len(rows) + 1
        assert {h["id"] for h in after} >= {h["id"] for h in before} | {"late"}
        index.reload()
        assert len(index) == 0
    finally:
        index.close()


def test_a_release_never_waits_for_a_search(tmp_path):
    index, clock, _db_path, _rows = _small_index(tmp_path)
    try:
        index.search("q1")
        with index._lock:  # a search in progress
            started = time.monotonic()
            assert not index.release_if_idle(clock.now + 3600)
            assert time.monotonic() - started < 0.5
        assert len(index) > 0
        assert index.release_if_idle(clock.now + 3600)
    finally:
        index.close()


def test_the_idle_timer_releases_the_index(tmp_path):
    index, _clock, _db_path, rows = _small_index(tmp_path)
    try:
        index._clock = time.monotonic
        index._idle_seconds = 0.2
        index.search("q1")
        assert len(index) == len(rows) and index._timer is not None
        deadline = time.monotonic() + 5
        while len(index) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(index) == 0, "released by the timer"
        deadline = time.monotonic() + 5
        while index._timer is not None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert index._timer is None, "no timer while nothing is held"
        index.search("q2")
        assert len(index) == len(rows) and index._timer is not None
    finally:
        index.close()
    assert index._timer is None, "close stops the timer"


def test_the_timer_waits_for_a_search_that_holds_the_index(tmp_path):
    index, _clock, _db_path, rows = _small_index(tmp_path)
    try:
        index._clock = time.monotonic
        index._idle_seconds = 0.1
        index.search("q1")
        with index._lock:
            time.sleep(0.4)  # the timer fires while a search runs
            assert len(index) == len(rows)
        index._used = time.monotonic()
        deadline = time.monotonic() + 5
        while len(index) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(index) == 0, "re-armed, then released"
    finally:
        index.close()


def test_a_replaced_vector_is_quantized_again_in_its_own_row(tmp_path):
    index, _clock, db_path, rows = _small_index(tmp_path)
    try:
        index.search("q1")
        at = index._pos["r00010"]
        others = index._q8.copy()
        replacement = rows[200][6]
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE memory_frames SET embedding = ?, last_ts = ? WHERE id = 'r00010'",
            (np.asarray(replacement, dtype="<f4").tobytes(), utc_text(START + timedelta(days=40))),
        )
        conn.commit()
        conn.close()
        index.search("q1")
        restored = index._q8[at].astype(np.float32) * index._scale[at]
        assert np.allclose(restored, replacement, atol=float(index._scale[at]))
        changed = np.nonzero((index._q8 != others).any(axis=1))[0]
        assert changed.tolist() == [at]
        assert len(index) == len(rows)
    finally:
        index.close()


def test_new_rows_are_quantized_by_the_incremental_refresh(tmp_path):
    index, _clock, db_path, rows = _small_index(tmp_path)
    try:
        index.search("q1")
        late = START + timedelta(days=50)
        vector = rows[42][6]
        _insert(db_path, [("new", late, late, "Notes", "new", "new", vector)])
        assert index.refresh() >= 1
        at = index._pos["new"]
        assert at == len(rows)
        restored = index._q8[at].astype(np.float32) * index._scale[at]
        assert np.allclose(restored, vector, atol=float(index._scale[at]))
    finally:
        index.close()


def test_rows_are_read_in_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "SCORE_BLOCK_ROWS", 7)
    index, _clock, db_path, rows = _small_index(tmp_path, n=100)
    try:
        hits = index.search("q9", limit=10)
        assert len(index) == 100
        _assert_same(hits, float32_search(db_path, rows[9][6], "q9", dim=64, limit=10))
    finally:
        index.close()


def test_concurrent_searches_and_releases_agree(tmp_path):
    index, clock, db_path, rows = _small_index(tmp_path)
    expected = float32_search(db_path, rows[5][6], "q5", dim=64, limit=5)
    errors: list[BaseException] = []

    def releaser() -> None:
        for _ in range(200):
            index.release_if_idle(clock.now + 10_000)

    thread = threading.Thread(target=releaser)
    thread.start()
    try:
        for _ in range(30):
            try:
                _assert_same(index.search("q5", limit=5), expected)
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)
    finally:
        thread.join()
        index.close()
    assert not errors


# -- browsing: an empty query lists the newest moments (2.0.3) -------------


def test_an_empty_query_lists_the_newest_moments_without_the_model(tmp_path):
    """Commonplace opened with nothing typed: the newest moments first,
    rows of one window less than MOMENT_GAP_SECONDS apart as one moment,
    no query embedded."""
    db_path = _db(tmp_path)
    v = [1.0, 0.0, 0.0, 0.0]
    gap = timedelta(seconds=MOMENT_GAP_SECONDS + 5)
    t = START
    _insert(db_path, [
        ("a1", t, t + timedelta(seconds=10), "Mail", "Inbox", "inbox with the invoice", v),
        ("a2", t + timedelta(seconds=20), t + timedelta(seconds=30), "Mail", "Inbox", "", v),
        ("b1", t + timedelta(seconds=40), t + timedelta(seconds=50), "Xcode", "App.swift", "code", v),
        ("c1", t + timedelta(seconds=50) + gap, t + timedelta(seconds=60) + gap, "Xcode", "App.swift", "later", v),
        ("other", t + timedelta(hours=1), t + timedelta(hours=1), "Mail", "Inbox", "another model", v),
    ])
    sqlite3.connect(db_path).execute("UPDATE memory_frames SET model = 'x' WHERE id = 'other'").connection.commit()

    def no_model(text):
        raise AssertionError("browsing embeds nothing")

    index = MemoryIndex(db_path, model=MODEL, dim=4, embed_query=no_model)
    try:
        found = index.search("  ", limit=10)
        assert [m["id"] for m in found] == ["c1", "b1", "a2"]
        # The two Mail rows are one moment: its newest row, the kept description.
        mail = found[2]
        assert mail["rows"] == ["a1", "a2"] and mail["frames"] == 2
        assert mail["first_ts"] == utc_text(t) and mail["description"] == "inbox with the invoice"
        assert all(m["score"] == 0.0 for m in found)
        # Filters and the limit, as for a search.
        assert [m["id"] for m in index.search("", limit=1)] == ["c1"]
        assert [m["id"] for m in index.search("", app="mail")] == ["a2"]
        assert [m["id"] for m in index.search("", until=t + timedelta(seconds=45))] == ["b1", "a2"]
        assert index.search("", since=t + timedelta(hours=2)) == []
    finally:
        index.close()
