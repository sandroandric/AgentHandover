"""Annotation cache: a frame that shows what an annotated frame showed
reuses that annotation instead of calling the model.

A frame's key is its app, a hash of its window title, and a 64-bit simhash
of its screen text (the OCR lines that go into the annotation prompt).  A
lookup matches a stored row with the same app and title whose simhash is
within ``MAX_HAMMING`` bits, so a clock tick or a cursor blink in the OCR
still hits.  Frames with too little screen text (``MIN_OCR_CHARS``) are
never cached: app and title alone do not say what is on screen.

The table lives in the events database next to the rows it summarises and
is pruned with the same retention (``prune``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_OCR_CHARS = 40
MAX_HAMMING = 3
# Rows compared per (app, title): the most recently used first.
_CANDIDATES = 64

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_MASK64 = (1 << 64) - 1


# Screen text that changes while the screen shows the same thing: a relative
# time ("5s ago", "updated 3 min ago", "just now"), a clock, an editor's
# cursor position.  Each becomes one token, then every run of digits
# becomes "0", before the simhash.  The window title is hashed as is, so
# two screens that differ only in a number still need the same title to
# share an annotation.
_UNIT = (
    r"(?:s|secs?|seconds?|m|mins?|minutes?|h|hrs?|hours?|d|days?|w|wks?|weeks?"
    r"|mo|mos|months?|y|yrs?|years?)"
)
_RELATIVE_TIME_RE = re.compile(
    rf"\b(?:(?:\d+|an?|a few)\s*{_UNIT}\s+ago|just now|moments? ago)\b", re.IGNORECASE,
)
_CLOCK_RE = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\b\.?)?", re.IGNORECASE,
)
_TEXT_POSITION_RE = re.compile(
    r"\b(?:Ln|Line)\s*\d+\s*,\s*(?:Col|Column)\s*\d+\b", re.IGNORECASE,
)
_DIGITS_RE = re.compile(r"\d+")


def normalise_screen_text(text: str) -> str:
    """*text* with its volatile parts folded, for the cache key only (the
    prompt gets the real text)."""
    text = _RELATIVE_TIME_RE.sub(" _reltime_ ", text)
    text = _CLOCK_RE.sub(" _clock_ ", text)
    text = _TEXT_POSITION_RE.sub(" _textpos_ ", text)
    return _DIGITS_RE.sub("0", text)


def simhash64(text: str) -> int:
    """64-bit simhash of *text*'s lowercased word tokens (unsigned)."""
    weights = [0] * 64
    for token in _TOKEN_RE.findall(text.lower()):
        h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if (h >> bit) & 1 else -1
    return sum(1 << bit for bit in range(64) if weights[bit] > 0)


def _signed(value: int) -> int:
    """An unsigned 64-bit value as SQLite's signed INTEGER."""
    return value - (1 << 64) if value >= 1 << 63 else value


@dataclass(frozen=True)
class CacheKey:
    app: str
    title_hash: str
    ocr_simhash: int  # unsigned


def cache_key(app: str, title: str, ocr_text: str) -> CacheKey | None:
    """The key for a frame, or None when it must not be cached."""
    if len(ocr_text.strip()) < MIN_OCR_CHARS:
        return None
    return CacheKey(
        app=app,
        title_hash=hashlib.sha256(title.encode()).hexdigest()[:32],
        ocr_simhash=simhash64(normalise_screen_text(ocr_text)),
    )


class AnnotationCache:
    """``annotation_cache(app, title_hash, ocr_simhash) -> annotation_json``."""

    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA busy_timeout = 5000;")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS annotation_cache ("
            "  app             TEXT NOT NULL,"
            "  title_hash      TEXT NOT NULL,"
            "  ocr_simhash     INTEGER NOT NULL,"
            "  annotation_json TEXT NOT NULL,"
            "  used_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),"
            "  PRIMARY KEY (app, title_hash, ocr_simhash)"
            ")"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_annotation_cache_used "
            "ON annotation_cache(used_at)"
        )
        self._conn.commit()

    def lookup(self, key: CacheKey) -> dict | None:
        """The cached annotation for *key*, or None (also when the database
        is busy: a miss only costs a model call)."""
        try:
            rows = self._conn.execute(
                "SELECT ocr_simhash, annotation_json FROM annotation_cache "
                "WHERE app = ? AND title_hash = ? ORDER BY used_at DESC LIMIT ?",
                (key.app, key.title_hash, _CANDIDATES),
            ).fetchall()
            for stored, annotation_json in rows:
                if ((stored & _MASK64) ^ key.ocr_simhash).bit_count() > MAX_HAMMING:
                    continue
                try:
                    annotation = json.loads(annotation_json)
                except (json.JSONDecodeError, TypeError):
                    continue
                self._conn.execute(
                    "UPDATE annotation_cache SET used_at = "
                    "strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                    "WHERE app = ? AND title_hash = ? AND ocr_simhash = ?",
                    (key.app, key.title_hash, stored),
                )
                self._conn.commit()
                return annotation
        except sqlite3.Error:
            logger.debug("Annotation cache lookup failed", exc_info=True)
        return None

    def store(self, key: CacheKey, annotation: dict) -> None:
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO annotation_cache "
                "(app, title_hash, ocr_simhash, annotation_json) VALUES (?, ?, ?, ?)",
                (key.app, key.title_hash, _signed(key.ocr_simhash), json.dumps(annotation)),
            )
            self._conn.commit()
        except sqlite3.Error:
            logger.debug("Annotation cache store failed", exc_info=True)

    def prune(self, *, max_age_days: int = 14, max_rows: int = 20_000) -> int:
        """Drop rows unused for *max_age_days* and the least recently used
        beyond *max_rows*.  Returns the number removed."""
        removed = self._conn.execute(
            "DELETE FROM annotation_cache "
            "WHERE used_at < strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?)",
            (f"-{max_age_days} days",),
        ).rowcount
        removed += self._conn.execute(
            "DELETE FROM annotation_cache WHERE rowid NOT IN ("
            "  SELECT rowid FROM annotation_cache ORDER BY used_at DESC LIMIT ?"
            ")",
            (max_rows,),
        ).rowcount
        self._conn.commit()
        if removed:
            logger.info("Pruned %d annotation cache row(s)", removed)
        return removed

    def close(self) -> None:
        self._conn.close()
