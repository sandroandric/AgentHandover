-- v007: procedures live in this database.
--
-- One row per procedure: `procedure_json` is the procedure document, the
-- exact text the worker exports to `<knowledge>/procedures/<slug>.json`
-- for the agents that read files.  `title` and `lifecycle_state` repeat
-- two of its fields for queries that should not parse every document.
-- The worker imports the procedure files a v0.4 knowledge base holds
-- once, on its first start against this schema (marker in worker_state:
-- `procedures_imported_at`).

CREATE TABLE IF NOT EXISTS procedures (
    slug TEXT PRIMARY KEY NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    lifecycle_state TEXT NOT NULL DEFAULT '',
    procedure_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_procedures_lifecycle ON procedures(lifecycle_state);
