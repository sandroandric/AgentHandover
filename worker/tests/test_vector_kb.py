"""Tests for the VectorKB module."""

from __future__ import annotations

import json
import math
import sqlite3
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agenthandover_worker.vector_kb import (
    VectorKB,
    VectorKBConfig,
    VectorSearchResult,
    _decode_embedding,
    _encode_embedding,
    _text_hash,
    _vector_id,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def config() -> VectorKBConfig:
    return VectorKBConfig(
        embedding_model="test-model",
        embedding_dim=4,
        batch_size=2,
        max_embeds_per_cycle=10,
    )


def _fake_embed_response(texts: list[str], dim: int = 4) -> bytes:
    """Build a fake Ollama /api/embed JSON response."""
    embeddings = []
    for i, t in enumerate(texts):
        # Deterministic embeddings based on text hash
        h = hash(t) & 0xFFFFFFFF
        vec = [(h >> (j * 8) & 0xFF) / 255.0 for j in range(dim)]
        norm = math.sqrt(sum(x * x for x in vec))
        vec = [x / norm if norm > 0 else 0.0 for x in vec]
        embeddings.append(vec)
    return json.dumps({"embeddings": embeddings}).encode("utf-8")


@pytest.fixture
def kb(tmp_db: Path, config: VectorKBConfig) -> VectorKB:
    """VectorKB with mocked Ollama that returns deterministic embeddings."""
    vkb = VectorKB(tmp_db, config)

    # Patch compute_embeddings to avoid real Ollama calls
    original_compute = vkb.compute_embeddings

    def mock_compute(texts, *, model=None):
        embeddings = []
        for t in texts:
            h = hash(t) & 0xFFFFFFFF
            vec = [(h >> (j * 8) & 0xFF) / 255.0 for j in range(config.embedding_dim)]
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

    def test_table_creation(self, tmp_db: Path, config: VectorKBConfig):
        vkb = VectorKB(tmp_db, config)
        conn = sqlite3.connect(str(tmp_db))
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "vector_store" in tables
        vkb.close()
        conn.close()

    def test_table_creation_idempotent(self, tmp_db: Path, config: VectorKBConfig):
        vkb1 = VectorKB(tmp_db, config)
        vkb1.close()
        vkb2 = VectorKB(tmp_db, config)
        assert vkb2.count() == 0
        vkb2.close()


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
