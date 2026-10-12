"""Vector Knowledge Base — embedding, storage, and semantic search.

Embeds text via Ollama (nomic-embed-text by default), stores unit vectors
as BLOB in SQLite, and provides brute-force cosine similarity search.

Design choices:
  * BLOB storage (struct.pack float32) — half the size of JSON, zero parse cost
  * Unit vectors plus an 8-bit copy — search is a coarse byte dot product
    over every row (``math.sumprod``) and an exact re-score of the best
  * Retention and a row cap — ``purge_stale`` runs daily, ``max_rows`` holds
  * Own SQLite connection — same isolation pattern as ActivitySearcher
  * Idempotent upsert — a row is reused only when its text_hash, embedding
    model and task prefix all match; a model or prefix change re-embeds
  * One embedding profile per model (``model_profiles.EMBEDDING_PROFILES``):
    its task prefixes (documents, queries, the task segmenter's frame
    vectors), its stored size and its thresholds; a search ranks only rows
    of the query's model
  * Budget cap — max_embeds_per_cycle prevents Ollama overload on backlog catch-up
"""

from __future__ import annotations

import hashlib
import heapq
import logging
import math
import sqlite3
import struct
from dataclasses import dataclass, field
from pathlib import Path

from agenthandover_worker.model_profiles import (
    EMBEDDING_MODEL,
    MEMORY_MODEL,
    EmbeddingProfile,
    EmbeddingThresholds,
    embedding_profile,
)
from agenthandover_worker.ollama import OllamaClient, OllamaError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VectorKBConfig:
    """Configuration for the vector knowledge base."""

    embedding_model: str = EMBEDDING_MODEL
    # ``[memory] model``: ``purge_stale`` keeps its rows too, so vectors a
    # background re-embed writes with it survive the daily maintenance
    # until the switch.
    memory_model: str = MEMORY_MODEL
    ollama_host: str = "http://localhost:11434"
    batch_size: int = 32
    timeout_seconds: float = 60.0
    max_embeds_per_cycle: int = 200
    # ``[embedding] retention_days``: rows not updated for this long are
    # purged by the daily maintenance.
    retention_days: int = 90
    # ``[embedding] max_rows``: hard cap; the least recently updated rows
    # beyond it are evicted after each upsert.
    max_rows: int = 50_000


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class VectorSearchResult:
    """A single search result from the vector store."""

    source_type: str
    source_id: str
    score: float
    model: str


# ---------------------------------------------------------------------------
# BLOB helpers
# ---------------------------------------------------------------------------

# One ``struct.Struct`` per dimension, built once instead of per row.
_STRUCTS: dict[int, struct.Struct] = {}

# The search candidates re-scored exactly: at least this many, and at
# least this many times ``top_k``.
_RERANK_MIN = 40
_RERANK_FACTOR = 4


def _struct(dim: int) -> struct.Struct:
    packer = _STRUCTS.get(dim)
    if packer is None:
        packer = _STRUCTS[dim] = struct.Struct(f"{dim}f")
    return packer


def _encode_embedding(vec: list[float]) -> bytes:
    """Pack a float vector into raw float32 bytes."""
    return _struct(len(vec)).pack(*vec)


def _decode_embedding(blob: bytes, dim: int) -> tuple[float, ...]:
    """Unpack raw float32 bytes into a float vector."""
    return _struct(dim).unpack(blob)


def _normalize(vec) -> list[float]:
    """*vec* scaled to unit length (a zero vector stays zero)."""
    norm = math.sqrt(math.fsum(x * x for x in vec))
    if norm == 0.0:
        return [0.0] * len(vec)
    return [x / norm for x in vec]


def _quantize(unit: list[float]) -> tuple[bytes, float]:
    """An 8-bit copy of a unit vector for the coarse search pass.

    Each component becomes ``round(x * scale) + 128`` in one byte, with
    ``scale = 127 / max|x|``, so a dot product of two such byte strings
    runs on ``math.sumprod``'s small-integer fast path.  Returns the bytes
    and the scale.
    """
    peak = max((abs(x) for x in unit), default=0.0)
    scale = 127.0 / peak if peak > 0.0 else 1.0
    return bytes(round(x * scale) + 128 for x in unit), scale


def _dot(a, b) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


_sumprod = getattr(math, "sumprod", _dot)  # math.sumprod is Python 3.12+


def _text_hash(text: str) -> str:
    """SHA-256 hash of input text for idempotent skip."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vector_id(source_type: str, source_id: str) -> str:
    """Deterministic ID for a vector entry."""
    raw = f"{source_type}:{source_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Row:
    """One vector in the in-memory search index."""

    source_type: str
    source_id: str
    model: str
    dim: int
    qvec: bytes
    scale: float
    offset_sum: int  # sum of the signed components, sum(qvec) - 128 * dim


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class VectorKB:
    """Embedding storage and semantic search over SQLite.

    Vectors are stored L2-normalised, so a similarity is a dot product.
    Each row also carries an 8-bit copy (``qvec``); search scores every
    row against it from an in-memory index kept in step with the table
    (rows changed since the last search are read by ``updated_at``), then
    re-scores the best candidates exactly from the float32 vectors.

    Manages its own SQLite connection (read-write).  Call :meth:`close`
    or use as a context manager when done.
    """

    def __init__(
        self,
        db_path: str | Path,
        config: VectorKBConfig | None = None,
    ) -> None:
        self.config = config or VectorKBConfig()
        self._db_path = str(db_path)
        # Shared by the control, inference and query-API threads
        # (inference_runner); SQLite itself is serialized.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA busy_timeout = 5000;")
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._quantize_old_rows()
        self._drop_visual_rows()
        self._cycle_embed_count = 0
        # Search index: vector id -> _Row, and the newest updated_at read.
        self._index: dict[str, _Row] | None = None
        self._index_mark = ""

    # ------------------------------------------------------------------
    # Rows from v0.4
    # ------------------------------------------------------------------

    def _quantize_old_rows(self) -> None:
        """Rows written before vectors were stored normalised with their
        8-bit copy (``qvec`` NULL, as v0.4 left them): convert them once.
        The table itself is the daemon's (``vector_store``, migration v006).
        """
        stale = self._conn.execute(
            "SELECT id, embedding, dim FROM vector_store WHERE qvec IS NULL"
        ).fetchall()
        for vid, blob, dim in stale:
            unit = _normalize(_decode_embedding(blob, dim))
            qvec, scale = _quantize(unit)
            self._conn.execute(
                "UPDATE vector_store SET embedding = ?, qvec = ?, qscale = ? "
                "WHERE id = ?",
                (_encode_embedding(unit), qvec, scale, vid),
            )
        if stale:
            logger.info("Normalised %d stored vector(s)", len(stale))
        self._conn.commit()

    def _drop_visual_rows(self) -> None:
        """Screenshot vectors of the SigLIP image embedder, which earlier
        versions wrote with ``[embedding] image_embeddings = true``
        (``source_type 'visual'``, 1152d): nothing ever searched them, and
        the embedder is gone.  Delete them once; afterwards none are left.
        """
        cur = self._conn.execute(
            "DELETE FROM vector_store WHERE source_type = 'visual'"
        )
        self._conn.commit()
        if cur.rowcount:
            logger.info("Deleted %d screenshot vector(s) of the removed "
                        "image embedder", cur.rowcount)

    @property
    def profile(self) -> EmbeddingProfile:
        """How the configured model embeds: task prefixes, stored size,
        thresholds."""
        return embedding_profile(self.config.embedding_model)

    @property
    def thresholds(self) -> EmbeddingThresholds:
        """The cosine thresholds measured on the configured model.

        Raises ValueError for a model not calibrated yet: its vectors can
        be stored and searched, but no threshold measured on another model
        means anything for them.
        """
        if self.profile.thresholds is None:
            raise ValueError(f"{self.profile.model} has no calibrated thresholds")
        return self.profile.thresholds

    # ------------------------------------------------------------------
    # Embedding computation (Ollama /api/embed)
    # ------------------------------------------------------------------

    def compute_embeddings(
        self,
        texts: list[str],
        *,
        prefix: str | None = None,
    ) -> list[list[float]]:
        """Embed *texts* with the task *prefix* (default: the profile's
        document prefix) at the profile's size, batched by
        config.batch_size.

        Raises ConnectionError if Ollama is unreachable.
        """

        return OllamaClient(self.config.ollama_host).embed(
            self.config.embedding_model,
            texts,
            prefix=self.profile.doc_prefix if prefix is None else prefix,
            dimensions=self.profile.dimensions,
            batch_size=self.config.batch_size,
            timeout=self.config.timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Cosine similarity
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Returns 0.0 for empty, zero-norm, or dimension-mismatched vectors.
        """
        if not a or not b or len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Storage: upsert
    # ------------------------------------------------------------------

    def _is_current(self, vid: str, thash: str, model: str, prefix: str) -> bool:
        """True if *vid* holds a vector of this text, model and prefix.

        A vector from another embedding model, or embedded with another
        task prefix, lives in a different space: either change re-embeds
        even when the text is unchanged.
        """
        row = self._conn.execute(
            "SELECT text_hash, model, prefix FROM vector_store WHERE id = ?",
            (vid,),
        ).fetchone()
        return row is not None and tuple(row) == (thash, model, prefix)

    def _store(
        self, source_type: str, source_id: str, text_hash: str,
        embedding: list[float], model: str, prefix: str,
    ) -> None:
        """Write one normalised vector with its 8-bit copy (no commit)."""
        unit = _normalize(embedding)
        qvec, scale = _quantize(unit)
        self._conn.execute(
            "INSERT INTO vector_store "
            "(id, source_type, source_id, embedding, model, dim, text_hash, "
            " prefix, qvec, qscale) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            " embedding = excluded.embedding,"
            " model = excluded.model,"
            " dim = excluded.dim,"
            " text_hash = excluded.text_hash,"
            " prefix = excluded.prefix,"
            " qvec = excluded.qvec,"
            " qscale = excluded.qscale,"
            " updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
            (_vector_id(source_type, source_id), source_type, source_id,
             _encode_embedding(unit), model, len(unit), text_hash, prefix,
             qvec, scale),
        )

    def get_current(
        self, source_type: str, source_id: str, text: str,
        *, prefix: str | None = None,
    ) -> list[float] | None:
        """The stored vector of *text* for this source, or None when the row
        is missing or is not current for this text, the configured model
        and *prefix* (default: the profile's document prefix; the same test
        ``upsert`` skips on).  *text* is the unprefixed text: the prefix is
        recorded in its own column."""
        if prefix is None:
            prefix = self.profile.doc_prefix
        vid = _vector_id(source_type, source_id)
        if not self._is_current(
            vid, _text_hash(text), self.config.embedding_model, prefix,
        ):
            return None
        row = self._conn.execute(
            "SELECT embedding, dim FROM vector_store WHERE id = ?", (vid,),
        ).fetchone()
        return _decode_embedding(row[0], row[1]) if row else None

    def upsert(
        self,
        source_type: str,
        source_id: str,
        text: str,
        *,
        embedding: list[float] | None = None,
        prefix: str | None = None,
        model: str | None = None,
    ) -> bool:
        """Store a single vector of *text* embedded with *prefix* (default:
        the document prefix of the model that made it).

        Skips when the stored row has the same text, model and prefix.
        If *embedding* is provided, uses it directly (no Ollama call);
        *model* then names the model that made it when that is not the
        configured embedding model.
        Returns True if the vector was stored/updated.
        """
        vid = _vector_id(source_type, source_id)
        thash = _text_hash(text)
        model = model or self.config.embedding_model
        if prefix is None:
            prefix = embedding_profile(model).doc_prefix

        if self._is_current(vid, thash, model, prefix):
            return False  # unchanged

        # Compute embedding if not provided
        if embedding is None:
            if self._cycle_embed_count >= self.config.max_embeds_per_cycle:
                logger.debug("Embed budget exhausted, skipping %s:%s", source_type, source_id)
                return False
            try:
                embeddings = self.compute_embeddings([text], prefix=prefix)
                if not embeddings or not embeddings[0]:
                    return False
                embedding = embeddings[0]
                self._cycle_embed_count += 1
            except (ConnectionError, OllamaError):
                logger.debug("Ollama unavailable for embedding", exc_info=True)
                return False

        self._store(source_type, source_id, thash, embedding, model, prefix)
        self._conn.commit()
        self.enforce_row_cap()
        return True

    def upsert_batch(
        self,
        items: list[tuple[str, str, str]],
        *,
        embeddings: list[list[float]] | None = None,
        prefix: str | None = None,
    ) -> int:
        """Bulk upsert.  Each item is (source_type, source_id, text).

        If *embeddings* is provided (matching length), uses them directly;
        they must have been computed from the texts with *prefix* (default:
        the profile's document prefix) by the configured model.  Otherwise,
        filters unchanged items, then embeds the rest with *prefix* in one
        batched call.  Returns count of vectors stored/updated.
        """
        if not items:
            return 0

        model = self.config.embedding_model
        if prefix is None:
            prefix = self.profile.doc_prefix
        to_store: list[tuple[str, str, str, list[float]]] = []

        if embeddings and len(embeddings) == len(items):
            # Caller provided embeddings
            for (stype, sid, text), emb in zip(items, embeddings):
                thash = _text_hash(text)
                if emb and not self._is_current(
                    _vector_id(stype, sid), thash, model, prefix,
                ):
                    to_store.append((stype, sid, thash, emb))
        else:
            # Filter unchanged items (same text, model and prefix)
            to_embed: list[tuple[str, str, str, str]] = []  # (stype, sid, hash, text)
            for stype, sid, text in items:
                thash = _text_hash(text)
                if not self._is_current(_vector_id(stype, sid), thash, model, prefix):
                    to_embed.append((stype, sid, thash, text))
            if not to_embed:
                return 0

            budget_left = self.config.max_embeds_per_cycle - self._cycle_embed_count
            if budget_left <= 0:
                logger.debug("Embed budget exhausted, skipping batch of %d", len(to_embed))
                return 0
            to_embed = to_embed[:budget_left]

            try:
                computed = self.compute_embeddings(
                    [t[3] for t in to_embed], prefix=prefix,
                )
            except (ConnectionError, OllamaError):
                logger.debug("Ollama unavailable for batch embedding", exc_info=True)
                return 0
            self._cycle_embed_count += len(computed)
            to_store = [
                (stype, sid, thash, emb)
                for (stype, sid, thash, _text), emb in zip(to_embed, computed)
                if emb
            ]

        for stype, sid, thash, emb in to_store:
            self._store(stype, sid, thash, emb, model, prefix)
        self._conn.commit()
        if to_store:
            self.enforce_row_cap()
        return len(to_store)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        source_types: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Embed *query_text* with the profile's query prefix and find the
        closest vectors of the configured model by cosine similarity."""
        try:
            embeddings = self.compute_embeddings(
                [query_text], prefix=self.profile.query_prefix,
            )
        except (ConnectionError, OllamaError):
            logger.debug("Ollama unavailable for search embedding")
            return []

        if not embeddings or not embeddings[0]:
            return []

        return self.search_by_vector(
            embeddings[0],
            top_k=top_k,
            source_types=source_types,
            min_score=min_score,
        )

    def _sync_index(self, *, reload: bool = True) -> dict[str, _Row]:
        """Bring the in-memory index up to date with the table.

        Rows written since the last sync are read through the
        ``updated_at`` index; a row count that then disagrees with the
        table (rows deleted, possibly by another process) reloads it all,
        once (a writer racing the reload is caught by the next search).
        """
        if self._index is None:
            self._index, self._index_mark = {}, ""
        rows = self._conn.execute(
            "SELECT id, source_type, source_id, model, dim, qvec, qscale, "
            "updated_at FROM vector_store WHERE updated_at >= ?",
            (self._index_mark,),
        ).fetchall()
        for vid, stype, sid, model, dim, qvec, scale, updated in rows:
            self._index[vid] = _Row(
                stype, sid, model, dim, qvec, scale, sum(qvec) - 128 * dim,
            )
            if updated > self._index_mark:
                self._index_mark = updated
        (total,) = self._conn.execute("SELECT COUNT(*) FROM vector_store").fetchone()
        if total != len(self._index) and reload:
            self._index = None
            return self._sync_index(reload=False)
        return self._index

    def search_by_vector(
        self,
        query_vector: list[float],
        *,
        top_k: int = 10,
        source_types: list[str] | None = None,
        min_score: float = 0.0,
        model: str | None = None,
    ) -> list[VectorSearchResult]:
        """Find closest vectors to a pre-computed vector made by *model*
        (default: the configured embedding model).

        Only rows of that model and size are ranked: a vector of another
        model lives in another space whatever its size, so its cosine with
        the query means nothing.  A coarse pass ranks every such row by its
        8-bit copy; the best ``max(40, 4 * top_k)`` are re-scored from their
        float32 vectors, and those exact scores are what is returned.
        """
        if not query_vector or top_k <= 0:
            return []
        model = model or self.config.embedding_model
        dim = len(query_vector)
        unit = _normalize(query_vector)
        qvec, qscale = _quantize(unit)
        q_offset = sum(qvec) - 128 * dim
        base = 128 * q_offset + 16384 * dim
        wanted = set(source_types) if source_types else None

        coarse: list[tuple[float, str]] = []
        for vid, row in self._sync_index().items():
            if (
                row.model != model or row.dim != dim
                or (wanted is not None and row.source_type not in wanted)
            ):
                continue
            # sum((q+128)(r+128)) minus the offset terms = sum(q * r)
            approx = (
                _sumprod(qvec, row.qvec) - base - 128 * row.offset_sum
            ) / (qscale * row.scale)
            coarse.append((approx, vid))
        if not coarse:
            return []
        candidates = heapq.nlargest(
            max(_RERANK_MIN, _RERANK_FACTOR * top_k), coarse,
        )

        packer = _struct(dim)
        results: list[VectorSearchResult] = []
        for _approx, vid in candidates:
            found = self._conn.execute(
                "SELECT embedding FROM vector_store WHERE id = ?", (vid,),
            ).fetchone()
            if found is None:
                continue
            score = _sumprod(unit, packer.unpack(found[0]))
            if score >= min_score:
                row = self._index[vid]
                results.append(VectorSearchResult(
                    source_type=row.source_type,
                    source_id=row.source_id,
                    score=round(score, 4),
                    model=row.model,
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def delete_by_source(self, source_type: str, source_id: str) -> bool:
        """Delete a vector by source type and ID."""
        vid = _vector_id(source_type, source_id)
        cur = self._conn.execute("DELETE FROM vector_store WHERE id = ?", (vid,))
        self._conn.commit()
        if self._index is not None:
            self._index.pop(vid, None)
        return cur.rowcount > 0

    def forgotten(self) -> None:
        """Rows were deleted elsewhere (``memory_forget``): the next search
        reads the index again."""
        self._index = None

    def count(self, source_type: str | None = None) -> int:
        """Count vectors in the store, optionally filtered by source_type."""
        if source_type:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM vector_store WHERE source_type = ?",
                (source_type,),
            )
        else:
            cur = self._conn.execute("SELECT COUNT(*) FROM vector_store")
        return cur.fetchone()[0]

    def purge_stale(self, max_age_days: int | None = None) -> int:
        """Remove entries not updated for *max_age_days* (default
        ``config.retention_days``), and every vector made by a model that is
        neither the configured embedding model nor the memory model: after
        a model change those are in a space nothing searches.  The memory
        model's rows stay (a background re-embed with it is kept until the
        switch)."""
        days = self.config.retention_days if max_age_days is None else max_age_days
        kept = (self.config.embedding_model, self.config.memory_model)
        cur = self._conn.execute(
            "DELETE FROM vector_store "
            "WHERE updated_at < strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?) "
            "OR model NOT IN (?, ?)",
            (f"-{days} days", *kept),
        )
        self._conn.commit()
        deleted = cur.rowcount
        if deleted:
            self._index = None
            logger.info(
                "Purged %d vectors (>%dd old, or from neither %s nor %s)",
                deleted, days, *kept,
            )
        return deleted

    def enforce_row_cap(self, max_rows: int | None = None) -> int:
        """Evict the least recently updated rows beyond *max_rows*
        (default ``config.max_rows``).  Returns the number evicted."""
        cap = self.config.max_rows if max_rows is None else max_rows
        if self.count() <= cap:
            return 0
        cur = self._conn.execute(
            "DELETE FROM vector_store WHERE id NOT IN ("
            "  SELECT id FROM vector_store ORDER BY updated_at DESC LIMIT ?"
            ")",
            (cap,),
        )
        self._conn.commit()
        self._index = None
        if cur.rowcount:
            logger.info("Evicted %d vector(s) over the %d-row cap", cur.rowcount, cap)
        return cur.rowcount

    def reset_cycle_budget(self) -> None:
        """Reset the per-cycle embedding budget counter."""
        self._cycle_embed_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> VectorKB:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
