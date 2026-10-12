"""Tests for the VectorKB module."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from daemon_db import create_daemon_db
from agenthandover_worker.model_profiles import EMBEDDING_PROFILES, MEMORY_MODEL
from agenthandover_worker.vector_kb import (
    VectorKB,
    VectorKBConfig,
    VectorSearchResult,
    _decode_embedding,
    _encode_embedding,
    _text_hash,
    _vector_id,
)


# nomic-embed-text's task prefixes: a model without a profile of its own
# ("test-model") embeds with them.
_NOMIC = EMBEDDING_PROFILES["nomic-embed-text"]
DOCUMENT_PREFIX = _NOMIC.doc_prefix
QUERY_PREFIX = _NOMIC.query_prefix
CLUSTERING_PREFIX = _NOMIC.cluster_prefix
# The fake vectors' size.
_DIM = 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """A database the daemon created (it owns ``vector_store``)."""
    return create_daemon_db(tmp_path / "test.db")


@pytest.fixture
def config() -> VectorKBConfig:
    return VectorKBConfig(
        embedding_model="test-model",
        batch_size=2,
        max_embeds_per_cycle=10,
    )


def _stable_hash(text: str) -> int:
    """32 bits of the text's SHA-256: the same in every process (``hash()``
    is salted per process, which made scores differ between runs)."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")


def _fake_embed_response(texts: list[str], dim: int = 4) -> bytes:
    """Build a fake Ollama /api/embed JSON response."""
    embeddings = []
    for i, t in enumerate(texts):
        # Deterministic embeddings based on text hash
        h = _stable_hash(t)
        vec = [(h >> (j * 8) & 0xFF) / 255.0 for j in range(dim)]
        norm = math.sqrt(sum(x * x for x in vec))
        vec = [x / norm if norm > 0 else 0.0 for x in vec]
        embeddings.append(vec)
    return json.dumps({"embeddings": embeddings}).encode("utf-8")


@pytest.fixture
def kb(tmp_db: Path, config: VectorKBConfig) -> VectorKB:
    """VectorKB with mocked Ollama that returns deterministic embeddings.

    The vectors depend on the text only; ``vkb.embed_calls`` records each
    call's ``(model, prefix, texts)``.
    """
    vkb = VectorKB(tmp_db, config)
    vkb.embed_calls = []

    def mock_compute(texts, *, prefix=None):
        if prefix is None:
            prefix = vkb.profile.doc_prefix
        vkb.embed_calls.append((vkb.config.embedding_model, prefix, list(texts)))
        embeddings = []
        for t in texts:
            h = _stable_hash(t)
            vec = [(h >> (j * 8) & 0xFF) / 255.0 for j in range(_DIM)]
            norm = math.sqrt(sum(x * x for x in vec))
            vec = [x / norm if norm > 0 else 0.0 for x in vec]
            embeddings.append(vec)
        return embeddings

    vkb.compute_embeddings = mock_compute
    return vkb


# ---------------------------------------------------------------------------
# BLOB helpers
# ---------------------------------------------------------------------------

class TestBlobHelpers:

    def test_encode_decode_roundtrip(self):
        vec = [0.1, 0.2, 0.3, 0.4]
        blob = _encode_embedding(vec)
        assert len(blob) == 16  # 4 floats * 4 bytes
        decoded = _decode_embedding(blob, 4)
        for a, b in zip(vec, decoded):
            assert abs(a - b) < 1e-6

    def test_encode_empty(self):
        blob = _encode_embedding([])
        assert blob == b""

    def test_text_hash_deterministic(self):
        assert _text_hash("hello") == _text_hash("hello")
        assert _text_hash("hello") != _text_hash("world")

    def test_vector_id_deterministic(self):
        assert _vector_id("a", "b") == _vector_id("a", "b")
        assert _vector_id("a", "b") != _vector_id("a", "c")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:

    def test_opens_the_daemons_table(self, tmp_db: Path, config: VectorKBConfig):
        with VectorKB(tmp_db, config) as vkb:
            assert vkb.count() == 0
        with VectorKB(tmp_db, config) as vkb:
            assert vkb.count() == 0


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:

    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0, 0.0]
        assert VectorKB.cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0, 0.0]
        assert VectorKB.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert VectorKB.cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert VectorKB.cosine_similarity([], []) == 0.0

    def test_mismatched_dimensions(self):
        assert VectorKB.cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_zero_norm(self):
        assert VectorKB.cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

class TestUpsert:

    def test_upsert_stores_vector(self, kb: VectorKB):
        assert kb.upsert("annotation", "evt1", "debugging code in VS Code")
        assert kb.count() == 1

    def test_upsert_idempotent_same_text(self, kb: VectorKB):
        kb.upsert("annotation", "evt1", "debugging code")
        assert kb.count() == 1
        # Same text — should skip
        result = kb.upsert("annotation", "evt1", "debugging code")
        assert result is False
        assert kb.count() == 1

    def test_upsert_updates_on_text_change(self, kb: VectorKB):
        kb.upsert("annotation", "evt1", "debugging code")
        # Different text — should update
        result = kb.upsert("annotation", "evt1", "writing tests")
        assert result is True
        assert kb.count() == 1  # same source, updated in place

    def test_upsert_with_precomputed_embedding(self, kb: VectorKB):
        emb = [0.1, 0.2, 0.3, 0.4]
        assert kb.upsert("procedure", "my-slug", "text", embedding=emb)
        assert kb.count() == 1

    def test_upsert_budget_exhaustion(self, kb: VectorKB):
        kb._cycle_embed_count = kb.config.max_embeds_per_cycle
        result = kb.upsert("annotation", "evt99", "some text")
        assert result is False

    def test_upsert_precomputed_bypasses_budget(self, kb: VectorKB):
        """Pre-computed embeddings don't count against embed budget."""
        kb._cycle_embed_count = kb.config.max_embeds_per_cycle
        emb = [0.1, 0.2, 0.3, 0.4]
        result = kb.upsert("annotation", "evt99", "text", embedding=emb)
        assert result is True


# ---------------------------------------------------------------------------
# Batch upsert
# ---------------------------------------------------------------------------

class TestUpsertBatch:

    def test_batch_stores_all(self, kb: VectorKB):
        items = [
            ("annotation", "e1", "task one"),
            ("annotation", "e2", "task two"),
            ("annotation", "e3", "task three"),
        ]
        count = kb.upsert_batch(items)
        assert count == 3
        assert kb.count() == 3

    def test_batch_skips_unchanged(self, kb: VectorKB):
        items = [("annotation", "e1", "task one")]
        kb.upsert_batch(items)
        # Re-batch with same text
        count = kb.upsert_batch(items)
        assert count == 0

    def test_model_change_reembeds_unchanged_text(self, kb: VectorKB):
        """A stored vector from another model is stale even for the same text."""
        items = [("annotation", "e1", "task one")]
        assert kb.upsert_batch(items) == 1
        assert kb.upsert("annotation", "e1", "task one") is False

        kb.config.embedding_model = "other-model"
        assert kb.upsert_batch(items) == 1
        row = kb._conn.execute(
            "SELECT model FROM vector_store WHERE id = ?",
            (_vector_id("annotation", "e1"),),
        ).fetchone()
        assert row[0] == "other-model"
        assert kb.upsert_batch(items) == 0

        kb.config.embedding_model = "third-model"
        assert kb.upsert("annotation", "e1", "task one") is True

    def test_batch_with_precomputed_embeddings(self, kb: VectorKB):
        items = [
            ("annotation", "e1", "task one"),
            ("annotation", "e2", "task two"),
        ]
        embs = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        count = kb.upsert_batch(items, embeddings=embs)
        assert count == 2

    def test_batch_respects_budget(self, kb: VectorKB):
        kb.config.max_embeds_per_cycle = 2
        kb._cycle_embed_count = 0
        items = [
            ("annotation", f"e{i}", f"task {i}") for i in range(5)
        ]
        count = kb.upsert_batch(items)
        assert count == 2  # capped at budget


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearch:

    def test_search_returns_results(self, kb: VectorKB):
        kb.upsert("annotation", "e1", "debugging python code")
        kb.upsert("annotation", "e2", "writing unit tests")
        kb.upsert("procedure", "deploy", "deploy to production server")

        results = kb.search("debugging python code", top_k=10)
        assert len(results) > 0
        # First result should be the exact match
        assert results[0].source_id == "e1"
        assert results[0].score == pytest.approx(1.0, abs=0.01)

    def test_search_filters_by_source_type(self, kb: VectorKB):
        kb.upsert("annotation", "e1", "task one")
        kb.upsert("procedure", "p1", "task one")

        results = kb.search("task one", source_types=["procedure"])
        assert all(r.source_type == "procedure" for r in results)

    def test_search_respects_min_score(self, kb: VectorKB):
        kb.upsert("annotation", "e1", "apples and oranges")
        kb.upsert("annotation", "e2", "quantum physics lecture")

        results = kb.search("apples and oranges", min_score=0.99)
        # Only the near-exact match should survive
        assert len(results) <= 1

    def test_search_empty_store(self, kb: VectorKB):
        results = kb.search("anything")
        assert results == []

    def test_search_by_vector(self, kb: VectorKB):
        vec = [0.5, 0.5, 0.5, 0.5]
        kb.upsert("annotation", "e1", "text", embedding=vec)

        results = kb.search_by_vector(vec, top_k=5)
        assert len(results) == 1
        assert results[0].score == pytest.approx(1.0, abs=0.01)

    def test_search_top_k(self, kb: VectorKB):
        for i in range(20):
            kb.upsert("annotation", f"e{i}", f"unique text {i}")

        results = kb.search("unique text 0", top_k=5)
        assert len(results) <= 5


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

class TestMaintenance:

    def test_delete_by_source(self, kb: VectorKB):
        kb.upsert("annotation", "e1", "text one")
        assert kb.count() == 1
        assert kb.delete_by_source("annotation", "e1")
        assert kb.count() == 0

    def test_delete_nonexistent(self, kb: VectorKB):
        assert kb.delete_by_source("annotation", "nope") is False

    def test_count_by_type(self, kb: VectorKB):
        kb.upsert("annotation", "e1", "text one")
        kb.upsert("procedure", "p1", "text two")
        assert kb.count("annotation") == 1
        assert kb.count("procedure") == 1
        assert kb.count() == 2

    def test_purge_stale(self, kb: VectorKB):
        kb.upsert("annotation", "e1", "old text")
        # Manually backdate the updated_at
        kb._conn.execute(
            "UPDATE vector_store SET updated_at = "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-100 days')"
        )
        kb._conn.commit()
        purged = kb.purge_stale(max_age_days=90)
        assert purged == 1
        assert kb.count() == 0

    def test_purge_drops_vectors_from_another_model(self, kb: VectorKB):
        kb.upsert("annotation", "e1", "task one")
        # The removed SigLIP embedder's model is no longer exempt.
        kb.upsert(
            "annotation", "e2", "task two", embedding=[0.1, 0.2, 0.3, 0.4],
            prefix="", model="mlx-community/siglip-so400m-patch14-384",
        )
        kb.config.embedding_model = "new-model"
        kb.upsert("annotation", "e3", "task three")

        assert kb.purge_stale() == 2
        kept = {
            row[0] for row in kb._conn.execute("SELECT source_id FROM vector_store")
        }
        assert kept == {"e3"}

    def test_reset_cycle_budget(self, kb: VectorKB):
        kb._cycle_embed_count = 100
        kb.reset_cycle_budget()
        assert kb._cycle_embed_count == 0


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestLifecycle:

    def test_context_manager(self, tmp_db: Path, config: VectorKBConfig):
        with VectorKB(tmp_db, config) as vkb:
            assert vkb.count() == 0
        # Connection should be closed — verify by trying to use it
        with pytest.raises(Exception):
            vkb.count()


# ---------------------------------------------------------------------------
# Unit vectors, the row cap, and search at scale
# ---------------------------------------------------------------------------

def _random_unit(rng, dim: int) -> list[float]:
    vec = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def _fill(kb: VectorKB, rows: int, dim: int, *, distinct: int, seed: int = 7) -> list[list[float]]:
    """Insert *rows* vectors cycling through *distinct* random unit vectors,
    written the way ``_store`` writes them.  Returns the distinct vectors."""
    import random

    from agenthandover_worker.vector_kb import _quantize

    rng = random.Random(seed)
    base = [_random_unit(rng, dim) for _ in range(distinct)]
    packed = [(_encode_embedding(v), *_quantize(v)) for v in base]
    kb._conn.executemany(
        "INSERT INTO vector_store "
        "(id, source_type, source_id, embedding, model, dim, text_hash, qvec, qscale) "
        "VALUES (?, 'annotation', ?, ?, 'test-model', ?, 'h', ?, ?)",
        (
            (f"id{i}", f"e{i}", packed[i % distinct][0], dim,
             packed[i % distinct][1], packed[i % distinct][2])
            for i in range(rows)
        ),
    )
    kb._conn.commit()
    return base


class TestUnitVectorsAndCap:

    def test_upsert_stores_a_unit_vector(self, kb: VectorKB):
        kb.upsert("annotation", "e1", "t", embedding=[3.0, 4.0, 0.0, 0.0])
        blob, dim = kb._conn.execute(
            "SELECT embedding, dim FROM vector_store"
        ).fetchone()
        assert list(_decode_embedding(blob, dim)) == pytest.approx([0.6, 0.8, 0.0, 0.0])

    def test_old_rows_are_normalised_on_open(self, tmp_db: Path, config: VectorKBConfig):
        conn = sqlite3.connect(tmp_db)
        # Rows as the daemon's migration leaves v0.4's: qvec NULL, prefix ''.
        conn.execute(
            "INSERT INTO vector_store (id, source_type, source_id, embedding, model, dim, text_hash)"
            " VALUES ('v', 'procedure', 'p', ?, 'test-model', 4, 'h')",
            (_encode_embedding([0.0, 2.0, 0.0, 0.0]),),
        )
        conn.commit()
        conn.close()
        with VectorKB(tmp_db, config) as vkb:
            hits = vkb.search_by_vector([0.0, 1.0, 0.0, 0.0])
            assert hits[0].source_id == "p"
            assert hits[0].score == pytest.approx(1.0)

    def test_siglip_visual_rows_are_deleted_on_open(
        self, tmp_db: Path, config: VectorKBConfig, caplog,
    ):
        # What the removed SigLIP image embedder left: 1152d 'visual' rows
        # (written normalised with their 8-bit copy, as v0.5 did).
        siglip = [1.0] + [0.0] * 1151
        with VectorKB(tmp_db, config) as vkb:
            for i in range(3):
                vkb.upsert(
                    "visual", f"e{i}", f"image:e{i}", embedding=siglip,
                    prefix="", model="mlx-community/siglip-so400m-patch14-384",
                )
            vkb.upsert("annotation", "e0", "t", embedding=[1.0, 0.0, 0.0, 0.0])
            assert vkb.count("visual") == 3

        with caplog.at_level("INFO", logger="agenthandover_worker.vector_kb"):
            with VectorKB(tmp_db, config) as vkb:
                assert vkb.count("visual") == 0
                assert vkb.count("annotation") == 1
        assert sum("screenshot vector" in r.message for r in caplog.records) == 1
        assert "Deleted 3 screenshot vector(s)" in caplog.text

        # Idempotent: nothing left to delete, nothing logged again.
        caplog.clear()
        with caplog.at_level("INFO", logger="agenthandover_worker.vector_kb"):
            with VectorKB(tmp_db, config) as vkb:
                assert vkb.count() == 1
        assert "screenshot vector" not in caplog.text

    def test_cap_holds_after_an_over_filling_upsert(self, tmp_db: Path):
        kb = VectorKB(tmp_db, VectorKBConfig(embedding_model="test-model", max_rows=5))
        _fill(kb, 5, 4, distinct=5)
        kb._conn.execute(
            "UPDATE vector_store SET updated_at = "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-1 day')"
        )
        kb._conn.commit()
        stored = kb.upsert_batch(
            [("procedure", f"p{i}", f"text {i}") for i in range(3)],
            embeddings=[[1.0, 0.0, 0.0, float(i)] for i in range(3)],
        )
        assert stored == 3
        assert kb.count() == 5
        # The three new rows survive; the two oldest were evicted.
        assert kb.count("procedure") == 3
        kb.close()

    def test_purge_stale_defaults_to_configured_retention(self, tmp_db: Path):
        kb = VectorKB(tmp_db, VectorKBConfig(embedding_model="test-model", retention_days=7))
        kb.upsert("annotation", "old", "t", embedding=[1.0, 0.0])
        kb.upsert("annotation", "new", "t", embedding=[0.0, 1.0])
        kb._conn.execute(
            "UPDATE vector_store SET updated_at = "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-8 days') WHERE source_id = 'old'"
        )
        kb._conn.commit()
        assert kb.purge_stale() == 1
        assert [h.source_id for h in kb.search_by_vector([1.0, 1.0])] == ["new"]
        kb.close()

    def test_search_sees_rows_written_by_another_connection(self, tmp_db: Path, config):
        kb = VectorKB(tmp_db, config)
        kb.upsert("annotation", "a", "t", embedding=[1.0, 0.0, 0.0, 0.0])
        assert [h.source_id for h in kb.search_by_vector([1.0, 0.0, 0.0, 0.0])] == ["a"]
        other = VectorKB(tmp_db, config)
        other.upsert("annotation", "b", "t", embedding=[0.0, 1.0, 0.0, 0.0])
        other.delete_by_source("annotation", "a")
        other.close()
        assert [h.source_id for h in kb.search_by_vector([0.0, 1.0, 0.0, 0.0])] == ["b"]
        kb.close()


class TestSearchAtScale:

    def test_matches_exact_ranking(self, tmp_db: Path):
        import random

        kb = VectorKB(tmp_db, VectorKBConfig(embedding_model="test-model"))
        base = _fill(kb, 2000, 64, distinct=2000, seed=3)
        rng = random.Random(11)
        for _ in range(5):
            query = _random_unit(rng, 64)
            exact = sorted(
                ((sum(a * b for a, b in zip(query, v)), f"e{i}") for i, v in enumerate(base)),
                reverse=True,
            )[:10]
            hits = kb.search_by_vector(query, top_k=10)
            assert [h.source_id for h in hits] == [sid for _, sid in exact]
            for hit, (score, _sid) in zip(hits, exact):
                assert hit.score == pytest.approx(score, abs=1e-4)
        kb.close()

    def test_fifty_thousand_vectors_under_250_ms(self, tmp_db: Path):
        import random
        import time

        kb = VectorKB(tmp_db, VectorKBConfig(embedding_model="test-model", max_rows=60_000))
        _fill(kb, 50_000, 768, distinct=500)
        query = _random_unit(random.Random(5), 768)
        kb.search_by_vector(query, top_k=10)  # builds the in-memory index once
        # The search's own CPU time, best of five: a single wall-clock run
        # also counted time spent waiting for a CPU and failed whenever the
        # machine was busy (measured: 181 ms quiet; with 14 busy processes
        # ~300 ms wall but a fastest CPU time of ~230 ms).
        timings = []
        for _ in range(5):
            began = time.process_time()
            hits = kb.search_by_vector(query, top_k=10)
            timings.append(time.process_time() - began)
        assert len(hits) == 10
        assert min(timings) < 0.25, f"fastest search used {min(timings) * 1000:.0f} ms of CPU"
        kb.close()


# ---------------------------------------------------------------------------
# Embedding model and task prefix (WP-C6)
# ---------------------------------------------------------------------------

class TestModelAndPrefix:

    def test_model_change_re_embeds_the_row(self, kb: VectorKB):
        """Embed one row, change the configured model: the row is embedded
        again by the new model instead of being skipped as unchanged."""
        assert kb.upsert("procedure", "p1", "file the expense report") is True
        assert kb.upsert("procedure", "p1", "file the expense report") is False
        assert len(kb.embed_calls) == 1

        kb.config.embedding_model = "nomic-embed-text-v2"
        assert kb.upsert("procedure", "p1", "file the expense report") is True
        assert kb.embed_calls[-1] == (
            "nomic-embed-text-v2", DOCUMENT_PREFIX, ["file the expense report"],
        )
        row = kb._conn.execute(
            "SELECT model, prefix FROM vector_store WHERE id = ?",
            (_vector_id("procedure", "p1"),),
        ).fetchone()
        assert tuple(row) == ("nomic-embed-text-v2", DOCUMENT_PREFIX)

    def test_prefix_change_re_embeds_the_row(self, kb: VectorKB):
        items = [("annotation", "e1", "reading hacker news")]
        assert kb.upsert_batch(items, prefix=CLUSTERING_PREFIX) == 1
        assert kb.upsert_batch(items, prefix=CLUSTERING_PREFIX) == 0
        assert kb.upsert_batch(items) == 1
        assert [c[1] for c in kb.embed_calls] == [CLUSTERING_PREFIX, DOCUMENT_PREFIX]

    def test_documents_and_queries_use_their_prefixes(self, kb: VectorKB):
        kb.upsert("procedure", "p1", "task one")
        kb.search("task one")
        assert [c[1] for c in kb.embed_calls] == [DOCUMENT_PREFIX, QUERY_PREFIX]

    def test_rows_from_before_prefixes_are_migrated_and_re_embedded(
        self, tmp_db: Path, config: VectorKBConfig,
    ):
        conn = sqlite3.connect(tmp_db)
        # Rows as the daemon's migration leaves v0.4's: qvec NULL, prefix ''.
        conn.execute(
            "INSERT INTO vector_store (id, source_type, source_id, embedding, model,"
            " dim, text_hash) VALUES (?, 'procedure', 'p1', ?, ?, 4, ?)",
            (_vector_id("procedure", "p1"), _encode_embedding([0.5] * 4),
             config.embedding_model, _text_hash("task one")),
        )
        conn.commit()
        conn.close()

        vkb = VectorKB(tmp_db, config)
        try:
            with patch.object(vkb, "compute_embeddings", return_value=[[0.1] * 4]) as embed:
                assert vkb.upsert("procedure", "p1", "task one") is True
                embed.assert_called_once_with(["task one"], prefix=DOCUMENT_PREFIX)
        finally:
            vkb.close()

    def test_get_current_matches_what_upsert_stores(self, kb: VectorKB):
        """get_current tests the same (text hash, model, prefix) that
        upsert skips on; the hashed text is the unprefixed text."""
        items = [("annotation", "e1", "reading hacker news")]
        assert kb.upsert_batch(items, prefix=CLUSTERING_PREFIX) == 1
        vec = kb.get_current(
            "annotation", "e1", "reading hacker news", prefix=CLUSTERING_PREFIX,
        )
        assert vec is not None and len(vec) == 4
        # Another prefix, another text or another model is not current.
        assert kb.get_current("annotation", "e1", "reading hacker news") is None
        assert kb.get_current(
            "annotation", "e1", CLUSTERING_PREFIX + "reading hacker news",
            prefix=CLUSTERING_PREFIX,
        ) is None
        kb.config.embedding_model = "nomic-embed-text-v2"
        assert kb.get_current(
            "annotation", "e1", "reading hacker news", prefix=CLUSTERING_PREFIX,
        ) is None

    def test_vectors_from_before_the_prefix_column_are_stale(
        self, tmp_db: Path, config: VectorKBConfig,
    ):
        """A vector stored before prefixes were recorded is never reused:
        neither a plain row nor a segmenter row whose hashed text carried
        the ``clustering:`` prefix inline (the pre-WP-C6 format)."""
        conn = sqlite3.connect(tmp_db)
        # Rows as the daemon's migration leaves v0.4's: qvec NULL, prefix ''.
        for sid, hashed in (
            ("plain", "reading hacker news"),
            ("inline", CLUSTERING_PREFIX + "reading hacker news"),
        ):
            conn.execute(
                "INSERT INTO vector_store (id, source_type, source_id, embedding, model,"
                " dim, text_hash) VALUES (?, 'annotation', ?, ?, ?, 4, ?)",
                (_vector_id("annotation", sid), sid, _encode_embedding([0.5] * 4),
                 config.embedding_model, _text_hash(hashed)),
            )
        conn.commit()
        conn.close()

        vkb = VectorKB(tmp_db, config)
        try:
            for sid in ("plain", "inline"):
                for prefix in (DOCUMENT_PREFIX, CLUSTERING_PREFIX):
                    assert vkb.get_current(
                        "annotation", sid, "reading hacker news", prefix=prefix,
                    ) is None
            with patch.object(
                vkb, "compute_embeddings", return_value=[[0.1] * 4, [0.2] * 4],
            ) as embed:
                stored = vkb.upsert_batch(
                    [("annotation", "plain", "reading hacker news"),
                     ("annotation", "inline", "reading hacker news")],
                    prefix=CLUSTERING_PREFIX,
                )
            assert stored == 2
            embed.assert_called_once()
            assert vkb.get_current(
                "annotation", "inline", "reading hacker news", prefix=CLUSTERING_PREFIX,
            ) is not None
        finally:
            vkb.close()

    def test_purge_by_model_resets_the_search_index(self, kb: VectorKB):
        kb.upsert("procedure", "p1", "task one")
        kb.upsert("annotation", "e1", "task two", embedding=[1.0, 0.0, 0.0, 0.0],
                  model="nomic-embed-text-v2")
        kb.search_by_vector([1.0, 0.0, 0.0, 0.0])  # builds the index
        kb.config.embedding_model = "nomic-embed-text-v2"
        assert kb.purge_stale() == 1
        assert kb.count("procedure") == 0 and kb.count("annotation") == 1
        assert [h.source_id for h in kb.search_by_vector([1.0, 0.0, 0.0, 0.0])] == ["e1"]

    def test_compute_embeddings_sends_the_prefix(self, tmp_db: Path, config: VectorKBConfig):
        vkb = VectorKB(tmp_db, config)
        try:
            with patch("agenthandover_worker.vector_kb.OllamaClient") as client:
                client.return_value.embed.return_value = [[0.0] * 4]
                vkb.compute_embeddings(["x"], prefix=QUERY_PREFIX)
            client.return_value.embed.assert_called_once_with(
                "test-model", ["x"], prefix=QUERY_PREFIX, dimensions=None,
                batch_size=config.batch_size, timeout=config.timeout_seconds,
            )
        finally:
            vkb.close()


# ---------------------------------------------------------------------------
# Embedding models: one space per model (WP-E2)
# ---------------------------------------------------------------------------

class TestOneSpacePerModel:

    def test_rows_of_another_model_are_never_ranked_with_the_query(self, kb: VectorKB):
        # Same size, and the other model's row is the closer one by the
        # numbers: it is still not a result, its space is not the query's.
        kb.upsert("procedure", "mine", "t", embedding=[0.6, 0.8, 0.0, 0.0])
        kb.upsert("procedure", "theirs", "t2", embedding=[1.0, 0.0, 0.0, 0.0],
                  model="other-model")
        hits = kb.search_by_vector([1.0, 0.0, 0.0, 0.0])
        assert [(h.source_id, h.model) for h in hits] == [("mine", "test-model")]
        # Asked for the other model's space, only its row answers.
        hits = kb.search_by_vector([1.0, 0.0, 0.0, 0.0], model="other-model")
        assert [(h.source_id, h.model, h.score) for h in hits] == [
            ("theirs", "other-model", 1.0),
        ]

    def test_search_ranks_only_the_configured_models_rows(self, kb: VectorKB):
        query = kb.compute_embeddings(["deploy to production"])[0]
        kb.upsert("procedure", "p1", "deploy to production")
        kb.upsert("procedure", "p2", "x", embedding=query, model=MEMORY_MODEL)
        assert [h.source_id for h in kb.search("deploy to production")] == ["p1"]

    def test_a_row_that_changes_model_leaves_the_querys_space(self, kb: VectorKB):
        kb.upsert("procedure", "p1", "t", embedding=[1.0, 0.0, 0.0, 0.0])
        assert [h.source_id for h in kb.search_by_vector([1.0, 0.0, 0.0, 0.0])] == ["p1"]
        kb.upsert("procedure", "p1", "t2", embedding=[1.0, 0.0, 0.0, 0.0],
                  model="other-model")
        assert kb.search_by_vector([1.0, 0.0, 0.0, 0.0]) == []

    def test_purge_keeps_the_embedding_and_the_memory_models_rows(self, kb: VectorKB):
        kb.upsert("procedure", "text", "t", embedding=[1.0, 0.0, 0.0, 0.0])
        kb.upsert("procedure", "memory", "t", embedding=[0.0, 1.0, 0.0, 0.0],
                  model=MEMORY_MODEL)
        kb.upsert("procedure", "gone", "t", embedding=[0.0, 0.0, 1.0, 0.0],
                  model="retired-model")
        assert kb.purge_stale() == 1
        kept = {row[0] for row in kb._conn.execute("SELECT source_id FROM vector_store")}
        assert kept == {"text", "memory"}

    def test_purge_keeps_the_configured_memory_model(self, tmp_db: Path):
        with VectorKB(tmp_db, VectorKBConfig(
            embedding_model="test-model", memory_model="memory-x",
        )) as kb:
            kb.upsert("procedure", "a", "t", embedding=[1.0, 0.0], model="memory-x")
            kb.upsert("procedure", "b", "t", embedding=[1.0, 0.0], model=MEMORY_MODEL)
            assert kb.purge_stale() == 1
            assert [r[0] for r in kb._conn.execute(
                "SELECT source_id FROM vector_store")] == ["a"]

    def test_age_still_purges_the_memory_models_rows(self, kb: VectorKB):
        kb.upsert("procedure", "m", "t", embedding=[1.0, 0.0, 0.0, 0.0],
                  model=MEMORY_MODEL)
        kb._conn.execute(
            "UPDATE vector_store SET updated_at = "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-100 days')"
        )
        kb._conn.commit()
        assert kb.purge_stale(max_age_days=90) == 1


class TestProfile:

    def test_nomic_requests_are_todays_bytes(self, tmp_db: Path):
        """On nomic-embed-text the /api/embed body is exactly v0.5's:
        model and prefixed input, no ``dimensions``."""
        bodies = []

        def fake_request(self, method, path, payload, timeout):
            bodies.append(json.dumps(payload))
            return {"embeddings": [[0.0, 1.0]] * len(payload["input"])}

        with patch("agenthandover_worker.ollama.OllamaClient._request", fake_request):
            with VectorKB(tmp_db, VectorKBConfig()) as kb:
                kb.compute_embeddings(["health check"])
                kb.search("deploy")
        assert bodies == [
            '{"model": "nomic-embed-text", "input": ["search_document: health check"]}',
            '{"model": "nomic-embed-text", "input": ["search_query: deploy"]}',
        ]

    def test_embeddinggemma_is_asked_at_its_native_size_with_its_prefixes(
        self, tmp_db: Path,
    ):
        payloads = []

        def fake_request(self, method, path, payload, timeout):
            payloads.append(payload)
            return {"embeddings": [[3.0, 4.0]] * len(payload["input"])}

        with patch("agenthandover_worker.ollama.OllamaClient._request", fake_request):
            with VectorKB(tmp_db, VectorKBConfig(embedding_model=MEMORY_MODEL)) as kb:
                assert kb.profile.dimensions is None
                vectors = kb.compute_embeddings(["a"])
                kb.search("b")
        assert payloads == [
            {"model": MEMORY_MODEL, "input": ["title: none | text: a"]},
            {"model": MEMORY_MODEL, "input": ["task: search result | query: b"]},
        ]
        assert vectors == [[3.0, 4.0]], "native size: as Ollama answers, like nomic"

    def test_documents_are_stored_with_the_models_prefix(self, tmp_db: Path):
        with patch("agenthandover_worker.ollama.OllamaClient._request",
                   lambda self, m, p, payload, t: {"embeddings": [[1.0, 0.0]]}):
            with VectorKB(tmp_db, VectorKBConfig(embedding_model=MEMORY_MODEL)) as kb:
                assert kb.upsert("procedure", "p", "deploy")
                (prefix,) = kb._conn.execute("SELECT prefix FROM vector_store").fetchone()
        assert prefix == "title: none | text: "

    def test_an_uncalibrated_model_has_no_thresholds(self, tmp_db: Path, monkeypatch):
        from dataclasses import replace

        from agenthandover_worker import model_profiles as mp

        monkeypatch.setitem(mp.EMBEDDING_PROFILES, "uncal-embed", replace(
            mp.EMBEDDING_PROFILES[MEMORY_MODEL], model="uncal-embed", thresholds=None,
        ))
        with VectorKB(tmp_db, VectorKBConfig(embedding_model="uncal-embed")) as kb:
            with pytest.raises(ValueError, match="no calibrated thresholds"):
                kb.thresholds
        with VectorKB(tmp_db, VectorKBConfig()) as kb:
            assert kb.thresholds is _NOMIC.thresholds

    def test_a_model_without_a_profile_is_used_as_nomic_was(self, tmp_db: Path):
        with VectorKB(tmp_db, VectorKBConfig(embedding_model="mxbai-embed-large")) as kb:
            assert kb.profile.model == "mxbai-embed-large"
            assert kb.profile.doc_prefix == DOCUMENT_PREFIX
            assert kb.profile.dimensions is None
            assert kb.thresholds == _NOMIC.thresholds
