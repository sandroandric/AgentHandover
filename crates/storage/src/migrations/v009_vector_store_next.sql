-- v009: "Upgrade everything" (2.0), the switch of `[embedding] model` to
-- EmbeddingGemma 2 with every stored vector re-embedded first.
--
-- `vector_store`'s key is the source (`id` = sha256 of
-- "source_type:source_id"), so one source cannot hold a vector of the old
-- model and one of the new at once.  The worker's background re-embed
-- (`embedding_upgrade.py`) writes the new model's vectors here, one row
-- per `vector_store` row and with the same `id`, while every search, link
-- and match keeps using `vector_store` and the old model.  `from_hash` is
-- the `text_hash` of the `vector_store` row a row re-embeds: when that row
-- changes, the re-embed does it again.  `embedding` NULL marks a source
-- whose text can no longer be rebuilt (its event or procedure is gone):
-- it is dropped at the switch.  The switch is one transaction:
-- `vector_store`'s rows are replaced by these, and this table is emptied.

CREATE TABLE IF NOT EXISTS vector_store_next (
    id          TEXT PRIMARY KEY NOT NULL,
    source_type TEXT NOT NULL,
    source_id   TEXT NOT NULL,
    from_hash   TEXT NOT NULL,
    embedding   BLOB,
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL DEFAULT 0,
    text_hash   TEXT NOT NULL DEFAULT '',
    prefix      TEXT NOT NULL DEFAULT '',
    qvec        BLOB,
    qscale      REAL,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
