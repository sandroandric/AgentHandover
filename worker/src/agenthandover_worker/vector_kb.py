"""Vector Knowledge Base — embedding, storage, and semantic search.

Embeds text via Ollama (nomic-embed-text by default), stores vectors as
BLOB in SQLite, and provides brute-force cosine similarity search.

Design choices:
  * BLOB storage (struct.pack float32) — half the size of JSON, zero parse cost
  * Own SQLite connection — same isolation pattern as ActivitySearcher
  * Idempotent upsert — SHA-256 text_hash skips re-embedding unchanged content
  * Budget cap — max_embeds_per_cycle prevents Ollama overload on backlog catch-up
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VectorKBConfig:
    """Configuration for the vector knowledge base."""

    embedding_model: str = "nomic-embed-text"
    ollama_host: str = "http://localhost:11434"
    embedding_dim: int = 768
    batch_size: int = 32
    timeout_seconds: float = 60.0
    max_embeds_per_cycle: int = 200


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

def _encode_embedding(vec: list[float]) -> bytes:
    """Pack a float vector into raw float32 bytes."""
    return struct.pack(f"{len(vec)}f", *vec)


def _decode_embedding(blob: bytes, dim: int) -> list[float]:
    """Unpack raw float32 bytes into a float vector."""
    return list(struct.unpack(f"{dim}f", blob))


def _text_hash(text: str) -> str:
    """SHA-256 hash of input text for idempotent skip."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vector_id(source_type: str, source_id: str) -> str:
    """Deterministic ID for a vector entry."""
    raw = f"{source_type}:{source_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class VectorKB:
    """Embedding storage and semantic search over SQLite.

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
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA busy_timeout = 5000;")
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._ensure_table()
        self._cycle_embed_count = 0

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vector_store ("
            "  id          TEXT PRIMARY KEY,"
            "  source_type TEXT NOT NULL,"
            "  source_id   TEXT NOT NULL,"
            "  embedding   BLOB NOT NULL,"
            "  model       TEXT NOT NULL,"
            "  dim         INTEGER NOT NULL,"
            "  text_hash   TEXT NOT NULL,"
            "  created_at  TEXT NOT NULL DEFAULT "
            "    (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),"
            "  updated_at  TEXT NOT NULL DEFAULT "
            "    (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
            ")"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vs_source "
            "ON vector_store(source_type, source_id)"
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Embedding computation (Ollama /api/embed)
    # ------------------------------------------------------------------

    def compute_embeddings(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Compute text embeddings via Ollama, batched by config.batch_size.

        Raises ConnectionError if Ollama is unreachable.
        """
        if not texts:
            return []

        model = model or self.config.embedding_model
        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            payload = json.dumps({"model": model, "input": batch}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.config.ollama_host}/api/embed",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    req, timeout=self.config.timeout_seconds,
                ) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
            except urllib.error.URLError as exc:
                raise ConnectionError(
                    f"Ollama not reachable at {self.config.ollama_host}: {exc}"
                ) from exc

            embeddings = result.get("embeddings", [])
            # Pad with empty vectors if count doesn't match
            while len(embeddings) < len(batch):
                embeddings.append([])
            all_embeddings.extend(embeddings[: len(batch)])

        return all_embeddings

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

    def upsert(
        self,
        source_type: str,
        source_id: str,
        text: str,
        *,
        embedding: list[float] | None = None,
    ) -> bool:
        """Store a single vector.  Skips if text_hash is unchanged.

        If *embedding* is provided, uses it directly (no Ollama call).
        Returns True if the vector was stored/updated.
        """
        vid = _vector_id(source_type, source_id)
        thash = _text_hash(text)

        # Check if unchanged
        cur = self._conn.execute(
            "SELECT text_hash FROM vector_store WHERE id = ?", (vid,),
        )
        row = cur.fetchone()
        if row and row[0] == thash:
            return False  # unchanged

        # Compute embedding if not provided
        if embedding is None:
            if self._cycle_embed_count >= self.config.max_embeds_per_cycle:
                logger.debug("Embed budget exhausted, skipping %s:%s", source_type, source_id)
                return False
            try:
                embeddings = self.compute_embeddings([text])
                if not embeddings or not embeddings[0]:
                    return False
                embedding = embeddings[0]
                self._cycle_embed_count += 1
            except ConnectionError:
                logger.debug("Ollama unavailable for embedding", exc_info=True)
                return False

        blob = _encode_embedding(embedding)
        self._conn.execute(
            "INSERT INTO vector_store "
            "(id, source_type, source_id, embedding, model, dim, text_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            " embedding = excluded.embedding,"
            " model = excluded.model,"
            " dim = excluded.dim,"
            " text_hash = excluded.text_hash,"
            " updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
            (vid, source_type, source_id, blob,
             self.config.embedding_model, len(embedding), thash),
        )
        self._conn.commit()
        return True

    def upsert_batch(
        self,
        items: list[tuple[str, str, str]],
        *,
        embeddings: list[list[float]] | None = None,
    ) -> int:
        """Bulk upsert.  Each item is (source_type, source_id, text).

        If *embeddings* is provided (matching length), uses them directly.
        Otherwise, filters unchanged items, then embeds remaining in one
        batched call.  Returns count of vectors stored/updated.
        """
        if not items:
            return 0

        count = 0

        if embeddings and len(embeddings) == len(items):
            # Caller provided embeddings — upsert directly
            for (stype, sid, text), emb in zip(items, embeddings):
                if emb and self.upsert(stype, sid, text, embedding=emb):
                    count += 1
            return count

        # Filter unchanged items by text_hash
        to_embed: list[tuple[int, str, str, str]] = []  # (idx, stype, sid, text)
        for i, (stype, sid, text) in enumerate(items):
            vid = _vector_id(stype, sid)
            thash = _text_hash(text)
            cur = self._conn.execute(
                "SELECT text_hash FROM vector_store WHERE id = ?", (vid,),
            )
            row = cur.fetchone()
            if row and row[0] == thash:
                continue  # unchanged
            to_embed.append((i, stype, sid, text))

        if not to_embed:
            return 0

        # Budget check
        budget_left = self.config.max_embeds_per_cycle - self._cycle_embed_count
        if budget_left <= 0:
            logger.debug("Embed budget exhausted, skipping batch of %d", len(to_embed))
            return 0
        to_embed = to_embed[:budget_left]

        # Compute embeddings
        texts = [t[3] for t in to_embed]
        try:
            computed = self.compute_embeddings(texts)
        except ConnectionError:
            logger.debug("Ollama unavailable for batch embedding", exc_info=True)
            return 0

        self._cycle_embed_count += len(computed)

        for (_, stype, sid, text), emb in zip(to_embed, computed):
            if not emb:
                continue
            vid = _vector_id(stype, sid)
            thash = _text_hash(text)
            blob = _encode_embedding(emb)
            self._conn.execute(
                "INSERT INTO vector_store "
                "(id, source_type, source_id, embedding, model, dim, text_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                " embedding = excluded.embedding,"
                " model = excluded.model,"
                " dim = excluded.dim,"
                " text_hash = excluded.text_hash,"
                " updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
                (vid, stype, sid, blob,
                 self.config.embedding_model, len(emb), thash),
            )
            count += 1

        self._conn.commit()
        return count

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
        """Embed *query_text* and find closest vectors by cosine similarity."""
        try:
            embeddings = self.compute_embeddings([query_text])
        except ConnectionError:
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

    def search_by_vector(
        self,
        query_vector: list[float],
        *,
        top_k: int = 10,
        source_types: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Find closest vectors to a pre-computed vector."""
        if not query_vector:
            return []

        # Load all vectors (filtered by source_types if given)
        if source_types:
            placeholders = ",".join("?" for _ in source_types)
            cur = self._conn.execute(
                f"SELECT source_type, source_id, embedding, dim, model "
                f"FROM vector_store WHERE source_type IN ({placeholders})",
                source_types,
            )
        else:
            cur = self._conn.execute(
                "SELECT source_type, source_id, embedding, dim, model "
                "FROM vector_store",
            )

        results: list[VectorSearchResult] = []
        for row in cur:
            stype, sid, blob, dim, model = row
            if dim != len(query_vector):
                continue  # dimension mismatch — skip
            vec = _decode_embedding(blob, dim)
            score = self.cosine_similarity(query_vector, vec)
            if score >= min_score:
                results.append(VectorSearchResult(
                    source_type=stype,
                    source_id=sid,
                    score=round(score, 4),
                    model=model,
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
        return cur.rowcount > 0

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

    def purge_stale(self, max_age_days: int = 90) -> int:
        """Remove entries older than *max_age_days*."""
        cur = self._conn.execute(
            "DELETE FROM vector_store "
            "WHERE datetime(updated_at) < datetime('now', ?)",
            (f"-{max_age_days} days",),
        )
        self._conn.commit()
        deleted = cur.rowcount
        if deleted:
            logger.info("Purged %d stale vectors (>%dd old)", deleted, max_age_days)
        return deleted

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


# ---------------------------------------------------------------------------
# Image embedder (SigLIP via mlx-embeddings)
# ---------------------------------------------------------------------------

class ImageEmbedder:
    """Embed images via SigLIP on Apple Silicon (mlx-embeddings).

    Lazily loads the model on first call.  Produces 1152-dim vectors
    in a separate vector space from text embeddings (768d nomic).
    Requires ``pip install mlx-embeddings`` — gracefully unavailable
    if the dependency is missing.

    Usage::

        embedder = ImageEmbedder()
        vec = embedder.embed_image("/path/to/screenshot.jpg")
        if vec:
            vector_kb.upsert("visual", event_id, "image", embedding=vec)
    """

    MODEL_NAME = "mlx-community/siglip-so400m-patch14-384"
    DIM = 1152

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        """Check if mlx-embeddings + SigLIP are importable."""
        if self._available is None:
            try:
                import mlx.core  # noqa: F401
                from mlx_embeddings.utils import load  # noqa: F401
                from PIL import Image  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
                logger.info(
                    "Image embeddings unavailable — "
                    "install mlx-embeddings: pip install mlx-embeddings"
                )
        return self._available

    def _ensure_model(self) -> bool:
        """Load model on first use.  Returns True if ready."""
        if self._model is not None:
            return True
        if not self.available:
            return False
        try:
            from mlx_embeddings.utils import load
            self._model, self._processor = load(self.MODEL_NAME)
            logger.info("SigLIP model loaded: %s", self.MODEL_NAME)
            return True
        except Exception:
            logger.warning("Failed to load SigLIP model", exc_info=True)
            self._available = False
            return False

    def embed_image(self, image_path: str) -> list[float] | None:
        """Embed a single image file.  Returns 1152-dim vector or None."""
        if not self._ensure_model():
            return None
        try:
            import mlx.core as mx
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=image, return_tensors="np")
            pixel_values = mx.array(inputs.pixel_values)
            # Transpose from NCHW to NHWC if needed
            if pixel_values.shape[1] == 3:
                pixel_values = pixel_values.transpose(0, 2, 3, 1)
            pixel_values = pixel_values.astype(mx.float32)

            outputs = self._model.vision_model(pixel_values=pixel_values)

            # Extract pooled embedding — shape varies by model version
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output[0]
            elif hasattr(outputs, "last_hidden_state"):
                # Mean pool over spatial tokens
                emb = outputs.last_hidden_state[0].mean(axis=0)
            else:
                logger.debug("Unknown SigLIP output format")
                return None

            vec = emb.tolist()
            if isinstance(vec, list) and len(vec) > 0:
                return vec
            return None
        except Exception:
            logger.debug("Image embedding failed for %s", image_path, exc_info=True)
            return None
