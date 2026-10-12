//! v006/v007: the daemon owns every table in the database, including the
//! ones the v0.4 worker created itself, and upgrades a v0.4 database to
//! the schema a new database gets.

use agenthandover_storage::EventStore;
use rusqlite::Connection;
use tempfile::TempDir;

/// The tables, columns and indexes of a database, comparable across files.
fn schema_of(conn: &Connection) -> Vec<String> {
    let mut out = Vec::new();
    let tables: Vec<(String, String)> = conn
        .prepare(
            "SELECT name, type FROM sqlite_master \
             WHERE type IN ('table', 'index') AND name NOT LIKE 'sqlite_%' \
               AND (name NOT LIKE 'events_fts_%' OR name = 'events_fts_ids') \
             ORDER BY type, name",
        )
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .map(|r| r.unwrap())
        .collect();
    for (name, kind) in tables {
        if kind == "index" {
            let columns: Vec<String> = conn
                .prepare(&format!("PRAGMA index_info({name})"))
                .unwrap()
                .query_map([], |r| r.get::<_, String>(2))
                .unwrap()
                .map(|r| r.unwrap())
                .collect();
            out.push(format!("index {name} ({})", columns.join(", ")));
            continue;
        }
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({name})")).unwrap();
        let columns = stmt
            .query_map([], |r| {
                Ok(format!(
                    "{} {} notnull={} default={:?} pk={}",
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, i64>(3)?,
                    r.get::<_, Option<String>>(4)?,
                    r.get::<_, i64>(5)?,
                ))
            })
            .unwrap()
            .map(|r| r.unwrap())
            .collect::<Vec<_>>();
        // Columns compare by name: ALTER TABLE ADD COLUMN appends, so an
        // upgraded table may list them in another order.
        let mut columns = columns;
        columns.sort();
        out.push(format!("table {name}: {}", columns.join("; ")));
    }
    out
}

/// A database as v0.4.0 left it: the daemon's schema 3 and the tables the
/// v0.4.0 worker created (statements copied from its `db.py`,
/// `vector_kb.py` and `activity_search.py`), with a row in each.
fn v040_database(path: &std::path::Path) {
    let conn = Connection::open(path).unwrap();
    for sql in [
        include_str!("../src/migrations/v001_initial.sql"),
        include_str!("../src/migrations/v002_add_display_ids_spanned.sql"),
        include_str!("../src/migrations/v003_add_scene_annotation.sql"),
    ] {
        conn.execute_batch(sql).unwrap();
    }
    conn.pragma_update(None, "user_version", 3).unwrap();
    conn.execute_batch(
        "CREATE TABLE translated_episodes (episode_id TEXT PRIMARY KEY, \
           thread_id TEXT NOT NULL, steps_json TEXT NOT NULL, step_count INTEGER NOT NULL, \
           created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')));
         INSERT INTO translated_episodes (episode_id, thread_id, steps_json, step_count) \
           VALUES ('ep', 't', '[]', 0);
         CREATE TABLE task_segments (segment_id TEXT PRIMARY KEY, cluster_id INTEGER NOT NULL, \
           task_label TEXT NOT NULL DEFAULT '', event_ids_json TEXT NOT NULL, \
           frame_count INTEGER NOT NULL, apps_json TEXT NOT NULL DEFAULT '[]', \
           start_time TEXT, end_time TEXT, sop_generated INTEGER NOT NULL DEFAULT 0, \
           embeddings_json TEXT, \
           created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')));
         INSERT INTO task_segments (segment_id, cluster_id, task_label, event_ids_json, \
           frame_count, embeddings_json) VALUES ('seg-1', 3, 'Expenses', '[\"e1\"]', 1, '[[0.1]]');
         CREATE TABLE generated_sops (sop_id TEXT PRIMARY KEY, slug TEXT NOT NULL, \
           title TEXT NOT NULL DEFAULT 'Untitled', source TEXT NOT NULL, \
           status TEXT NOT NULL DEFAULT 'draft', sop_json TEXT NOT NULL, \
           confidence REAL DEFAULT 0.0, source_id TEXT DEFAULT '', created_at TEXT NOT NULL, \
           reviewed_at TEXT DEFAULT NULL);
         INSERT INTO generated_sops (sop_id, slug, source, sop_json, created_at) \
           VALUES ('sop-1', 'file-expenses', 'focus', '{}', '2026-05-01T00:00:00Z');
         CREATE TABLE failed_generations (failure_id TEXT PRIMARY KEY, source TEXT NOT NULL, \
           source_id TEXT NOT NULL, title TEXT DEFAULT '', error TEXT NOT NULL, \
           context_json TEXT DEFAULT '{}', created_at TEXT NOT NULL, retried INTEGER DEFAULT 0);
         CREATE TABLE preserved_text (event_id TEXT NOT NULL, source TEXT NOT NULL, \
           timestamp TEXT NOT NULL, text_content TEXT NOT NULL, app TEXT DEFAULT '', \
           created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')), \
           PRIMARY KEY (event_id, source));
         CREATE INDEX idx_preserved_text_ts ON preserved_text(timestamp);
         INSERT INTO preserved_text (event_id, source, timestamp, text_content) \
           VALUES ('e1', 'ocr', '2026-05-01T00:00:00Z', 'Invoice 42');
         CREATE TABLE vector_store (id TEXT PRIMARY KEY, source_type TEXT NOT NULL, \
           source_id TEXT NOT NULL, embedding BLOB NOT NULL, model TEXT NOT NULL, \
           dim INTEGER NOT NULL, text_hash TEXT NOT NULL, \
           created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')), \
           updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')));
         CREATE INDEX idx_vs_source ON vector_store(source_type, source_id);
         INSERT INTO vector_store (id, source_type, source_id, embedding, model, dim, text_hash) \
           VALUES ('v1', 'event', 'e1', x'0000803f', 'nomic', 1, 'h');
         CREATE VIRTUAL TABLE events_fts USING fts5(event_id, what_doing, app, location);
         INSERT INTO events_fts (event_id, what_doing, app, location) \
           VALUES ('e1', 'Filing an expense', 'Chrome', 'expenses.example');",
    )
    .unwrap();
}

#[test]
fn a_new_database_has_every_table_the_worker_uses() {
    let tmp = TempDir::new().unwrap();
    let store = EventStore::open(&tmp.path().join("events.db")).unwrap();
    assert_eq!(store.schema_version(), 9);
    let names: Vec<String> = store
        .connection()
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table'")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .map(|r| r.unwrap())
        .collect();
    for table in [
        "worker_state", "task_segments", "workflow_observations", "micro_questions",
        "generated_sops", "failed_generations", "preserved_text", "vector_store",
        "annotation_cache", "events_fts", "events_fts_ids", "runs", "procedures",
        "memory_frames", "vector_store_next",
    ] {
        assert!(names.iter().any(|n| n == table), "{table} missing from {names:?}");
    }
    assert!(!names.iter().any(|n| n == "translated_episodes"));
}

#[test]
fn a_v040_database_upgrades_to_the_new_schema_and_keeps_its_rows() {
    let tmp = TempDir::new().unwrap();
    let old = tmp.path().join("events.db");
    v040_database(&old);

    let upgraded = EventStore::open(&old).unwrap();
    let fresh = EventStore::open(&tmp.path().join("fresh.db")).unwrap();

    assert_eq!(upgraded.schema_version(), 9);
    let fresh_schema = schema_of(fresh.connection());
    assert!(fresh_schema.len() > 25, "compares the whole schema: {fresh_schema:?}");
    assert_eq!(schema_of(upgraded.connection()), fresh_schema);

    let conn = upgraded.connection();
    let one = |sql: &str| -> String { conn.query_row(sql, [], |r| r.get(0)).unwrap() };
    assert_eq!(one("SELECT task_label FROM task_segments WHERE segment_id = 'seg-1'"), "Expenses");
    assert_eq!(one("SELECT slug FROM generated_sops"), "file-expenses");
    assert_eq!(one("SELECT text_content FROM preserved_text"), "Invoice 42");
    // Old vectors get the columns; the worker quantises them (qvec NULL)
    // and re-embeds them (prefix '') itself.
    assert_eq!(one("SELECT prefix FROM vector_store WHERE id = 'v1'"), "");
    let qvec_null: bool = conn
        .query_row("SELECT qvec IS NULL FROM vector_store WHERE id = 'v1'", [], |r| r.get(0))
        .unwrap();
    assert!(qvec_null);
    // The old full-text index is kept and gets its rowid map.
    assert_eq!(one("SELECT event_id FROM events_fts_ids"), "e1");
    assert_eq!(
        one("SELECT event_id FROM events_fts WHERE events_fts MATCH 'expense'"),
        "e1"
    );

    let backups = std::fs::read_dir(tmp.path())
        .unwrap()
        .filter(|e| e.as_ref().unwrap().file_name().to_string_lossy().contains(".bak-"))
        .count();
    assert_eq!(backups, 1, "the upgrade backs the old database up first");
}

#[test]
fn reopening_an_upgraded_database_changes_nothing() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("events.db");
    v040_database(&path);
    let before = {
        let store = EventStore::open(&path).unwrap();
        schema_of(store.connection())
    };
    let store = EventStore::open(&path).unwrap();
    assert_eq!(store.schema_version(), 9);
    assert_eq!(schema_of(store.connection()), before);
}

#[test]
fn a_v5_database_with_pre_labelling_observations_gains_the_label_columns() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("events.db");
    {
        let conn = Connection::open(&path).unwrap();
        for sql in [
            include_str!("../src/migrations/v001_initial.sql"),
            include_str!("../src/migrations/v002_add_display_ids_spanned.sql"),
            include_str!("../src/migrations/v003_add_scene_annotation.sql"),
            include_str!("../src/migrations/v004_drop_old_dom_trees.sql"),
            include_str!("../src/migrations/v005_add_focused_element.sql"),
        ] {
            conn.execute_batch(sql).unwrap();
        }
        conn.pragma_update(None, "user_version", 5).unwrap();
        conn.execute_batch(
            "CREATE TABLE workflow_observations (obs_id TEXT PRIMARY KEY, \
               workflow_key TEXT NOT NULL, started_utc TEXT NOT NULL, \
               started_local_date TEXT NOT NULL, started_local_weekday INTEGER NOT NULL, \
               started_local_minute INTEGER NOT NULL, duration_s REAL NOT NULL, \
               apps_json TEXT NOT NULL DEFAULT '[]', hosts_json TEXT NOT NULL DEFAULT '[]', \
               step_sig_json TEXT NOT NULL DEFAULT '[]', evidence_json TEXT NOT NULL DEFAULT '{}', \
               source TEXT NOT NULL DEFAULT 'passive', task_label TEXT NOT NULL DEFAULT '', \
               event_ids_json TEXT NOT NULL DEFAULT '[]', promoted_slug TEXT, promoted_at TEXT);
             INSERT INTO workflow_observations (obs_id, workflow_key, started_utc, \
               started_local_date, started_local_weekday, started_local_minute, duration_s) \
               VALUES ('o1', 'k', '2026-09-01T09:00:00Z', '2026-09-01', 1, 540, 60);",
        )
        .unwrap();
    }
    let store = EventStore::open(&path).unwrap();
    let label: Option<String> = store
        .connection()
        .query_row("SELECT user_label FROM workflow_observations WHERE obs_id = 'o1'", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(label, None);
}

#[test]
fn a_failed_migration_leaves_the_database_at_its_old_version() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("events.db");
    {
        let conn = Connection::open(&path).unwrap();
        for sql in [
            include_str!("../src/migrations/v001_initial.sql"),
            include_str!("../src/migrations/v002_add_display_ids_spanned.sql"),
            include_str!("../src/migrations/v003_add_scene_annotation.sql"),
            include_str!("../src/migrations/v004_drop_old_dom_trees.sql"),
            include_str!("../src/migrations/v005_add_focused_element.sql"),
        ] {
            conn.execute_batch(sql).unwrap();
        }
        conn.pragma_update(None, "user_version", 5).unwrap();
        // A view where v006 creates a table: its CREATE INDEX fails.
        conn.execute_batch("CREATE VIEW runs AS SELECT 1 AS procedure_slug, 2 AS started_at")
            .unwrap();
    }
    assert!(EventStore::open(&path).is_err());

    let conn = Connection::open(&path).unwrap();
    let version: u32 = conn.pragma_query_value(None, "user_version", |r| r.get(0)).unwrap();
    assert_eq!(version, 5);
    let worker_state: i64 = conn
        .query_row(
            "SELECT count(*) FROM sqlite_master WHERE name = 'worker_state'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(worker_state, 0, "nothing of v006 is left behind");
}

#[test]
fn a_v7_database_gains_the_memory_table_and_keeps_its_rows() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("events.db");
    {
        let conn = Connection::open(&path).unwrap();
        for sql in [
            include_str!("../src/migrations/v001_initial.sql"),
            include_str!("../src/migrations/v002_add_display_ids_spanned.sql"),
            include_str!("../src/migrations/v003_add_scene_annotation.sql"),
            include_str!("../src/migrations/v004_drop_old_dom_trees.sql"),
            include_str!("../src/migrations/v005_add_focused_element.sql"),
            include_str!("../src/migrations/v006_worker_tables.sql"),
            include_str!("../src/migrations/v007_procedures.sql"),
        ] {
            conn.execute_batch(sql).unwrap();
        }
        conn.pragma_update(None, "user_version", 7).unwrap();
        conn.execute_batch("INSERT INTO procedures (slug, procedure_json) VALUES ('p', '{}');")
            .unwrap();
    }
    let store = EventStore::open(&path).unwrap();
    assert_eq!(store.schema_version(), 9);
    let conn = store.connection();
    let kept: i64 = conn.query_row("SELECT count(*) FROM procedures", [], |r| r.get(0)).unwrap();
    assert_eq!(kept, 1);
    conn.execute(
        "INSERT INTO memory_frames (id, first_ts, last_ts, embedding, model, dim) \
         VALUES ('e1', '2026-10-12T09:00:00Z', '2026-10-12T09:00:00Z', x'0000803f', 'm', 1)",
        [],
    )
    .unwrap();
    let (frames, created): (i64, String) = conn
        .query_row("SELECT frames, created_at FROM memory_frames WHERE id = 'e1'", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(frames, 1);
    assert!(created.ends_with('Z'));
}

#[test]
fn a_v8_database_gains_the_re_embed_staging_table_and_keeps_its_vectors() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("events.db");
    {
        let conn = Connection::open(&path).unwrap();
        for sql in [
            include_str!("../src/migrations/v001_initial.sql"),
            include_str!("../src/migrations/v002_add_display_ids_spanned.sql"),
            include_str!("../src/migrations/v003_add_scene_annotation.sql"),
            include_str!("../src/migrations/v004_drop_old_dom_trees.sql"),
            include_str!("../src/migrations/v005_add_focused_element.sql"),
            include_str!("../src/migrations/v006_worker_tables.sql"),
            include_str!("../src/migrations/v007_procedures.sql"),
            include_str!("../src/migrations/v008_memory_frames.sql"),
        ] {
            conn.execute_batch(sql).unwrap();
        }
        conn.pragma_update(None, "user_version", 8).unwrap();
        conn.execute_batch(
            "INSERT INTO vector_store (id, source_type, source_id, embedding, model, dim, text_hash) \
             VALUES ('v1', 'procedure', 'p', x'0000803f', 'nomic-embed-text', 1, 'h');",
        )
        .unwrap();
    }
    let store = EventStore::open(&path).unwrap();
    assert_eq!(store.schema_version(), 9);
    let conn = store.connection();
    let kept: i64 = conn.query_row("SELECT count(*) FROM vector_store", [], |r| r.get(0)).unwrap();
    assert_eq!(kept, 1);
    // A source whose text is gone is staged without a vector.
    conn.execute(
        "INSERT INTO vector_store_next (id, source_type, source_id, from_hash, model) \
         VALUES ('v1', 'procedure', 'p', 'h', 'embeddinggemma-2:440m')",
        [],
    )
    .unwrap();
    let (dim, created): (i64, String) = conn
        .query_row("SELECT dim, created_at FROM vector_store_next WHERE id = 'v1'", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(dim, 0);
    assert!(created.ends_with('Z'));
}
