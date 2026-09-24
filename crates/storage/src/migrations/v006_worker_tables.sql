-- v006: the tables the worker used to create and alter itself.
--
-- Up to v0.4 the worker ran its own CREATE/ALTER TABLE statements on this
-- database; the daemon is now the one schema owner and the worker checks
-- `user_version` at startup.  Every statement is IF NOT EXISTS: a database
-- the v0.4 worker already wrote keeps its tables and rows.  The column
-- changes a CREATE cannot express on an existing table (columns added
-- since v0.4, columns no longer written) run in code after this file
-- (`reconcile_columns`, crates/storage/src/lib.rs).

-- Small persisted worker values (dates, counters, one-time markers).
CREATE TABLE IF NOT EXISTS worker_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- The segmenter's output, kept for 14 days.
CREATE TABLE IF NOT EXISTS task_segments (
    segment_id TEXT PRIMARY KEY,
    cluster_id INTEGER NOT NULL,
    task_label TEXT NOT NULL DEFAULT '',
    event_ids_json TEXT NOT NULL,
    frame_count INTEGER NOT NULL,
    apps_json TEXT NOT NULL DEFAULT '[]',
    start_time TEXT,
    end_time TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Every finished work session, outliving the events it came from.
-- user_label / user_is_routine / labelled_at: the user's answer to a
-- labelling card (1 routine, 0 not, NULL unsure or never asked).
CREATE TABLE IF NOT EXISTS workflow_observations (
    obs_id TEXT PRIMARY KEY,
    workflow_key TEXT NOT NULL,
    started_utc TEXT NOT NULL,
    started_local_date TEXT NOT NULL,
    started_local_weekday INTEGER NOT NULL,
    started_local_minute INTEGER NOT NULL,
    duration_s REAL NOT NULL,
    apps_json TEXT NOT NULL DEFAULT '[]',
    hosts_json TEXT NOT NULL DEFAULT '[]',
    step_sig_json TEXT NOT NULL DEFAULT '[]',
    evidence_json TEXT NOT NULL DEFAULT '{}',
    source TEXT NOT NULL DEFAULT 'passive',
    task_label TEXT NOT NULL DEFAULT '',
    event_ids_json TEXT NOT NULL DEFAULT '[]',
    promoted_slug TEXT,
    promoted_at TEXT,
    user_label TEXT,
    user_is_routine INTEGER,
    labelled_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_workflow_observations_started
    ON workflow_observations(started_utc);

-- Every micro-question asked, so each is asked once.
CREATE TABLE IF NOT EXISTS micro_questions (
    question_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    subject TEXT NOT NULL,
    slot TEXT,
    channel TEXT NOT NULL,
    question_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    answer TEXT,
    asked_at TEXT NOT NULL,
    answered_at TEXT
);

-- Generated SOPs and their review status.
CREATE TABLE IF NOT EXISTS generated_sops (
    sop_id TEXT PRIMARY KEY,
    slug TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT 'Untitled',
    source TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    sop_json TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    source_id TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    reviewed_at TEXT DEFAULT NULL
);

-- Generation failures kept for retry.
CREATE TABLE IF NOT EXISTS failed_generations (
    failure_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    title TEXT DEFAULT '',
    error TEXT NOT NULL,
    context_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    retried INTEGER DEFAULT 0
);

-- OCR / visual text kept past the events' retention ('ocr', 'visual_proxy').
CREATE TABLE IF NOT EXISTS preserved_text (
    event_id TEXT NOT NULL,
    source TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    text_content TEXT NOT NULL,
    app TEXT DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (event_id, source)
);

CREATE INDEX IF NOT EXISTS idx_preserved_text_ts ON preserved_text(timestamp);

-- Embeddings: the float32 vector normalised, its 8-bit copy (qvec, qscale)
-- and the task prefix it was embedded with.
CREATE TABLE IF NOT EXISTS vector_store (
    id          TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_id   TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    text_hash   TEXT NOT NULL,
    prefix      TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    qvec        BLOB,
    qscale      REAL
);

CREATE INDEX IF NOT EXISTS idx_vs_source ON vector_store(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_vs_updated ON vector_store(updated_at);

-- Scene annotations reused for an unchanged screen.
CREATE TABLE IF NOT EXISTS annotation_cache (
    app             TEXT NOT NULL,
    title_hash      TEXT NOT NULL,
    ocr_simhash     INTEGER NOT NULL,
    annotation_json TEXT NOT NULL,
    used_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (app, title_hash, ocr_simhash)
);

CREATE INDEX IF NOT EXISTS idx_annotation_cache_used ON annotation_cache(used_at);

-- Activity search: the full-text index and, per indexed event, its rowid.
CREATE VIRTUAL TABLE IF NOT EXISTS events_fts
    USING fts5(event_id, what_doing, app, location);

CREATE TABLE IF NOT EXISTS events_fts_ids (
    event_id TEXT PRIMARY KEY,
    fts_rowid INTEGER NOT NULL
);

-- An index the v0.4 worker built has no rowid map yet: record its rows.
INSERT OR IGNORE INTO events_fts_ids (event_id, fts_rowid)
    SELECT event_id, rowid FROM events_fts
    WHERE NOT EXISTS (SELECT 1 FROM events_fts_ids LIMIT 1);

-- Agent runs of procedures (reported through the MCP server).
CREATE TABLE IF NOT EXISTS runs (
    execution_id    TEXT PRIMARY KEY,
    procedure_slug  TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    status          TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    completed_at    TEXT,
    error           TEXT,
    steps_json      TEXT NOT NULL,
    deviations_json TEXT NOT NULL,
    outcomes_json   TEXT NOT NULL,
    metadata_json   TEXT
);

CREATE INDEX IF NOT EXISTS runs_by_slug ON runs (procedure_slug, started_at);

-- The removed v1 miner's episode store.
DROP TABLE IF EXISTS translated_episodes;
