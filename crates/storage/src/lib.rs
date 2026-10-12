mod schema;
mod migrations;
pub mod artifact_store;
pub mod maintenance;

use anyhow::Result;
use agenthandover_common::event::*;
use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};
use tracing::info;
use uuid::Uuid;

pub struct EventStore {
    conn: Connection,
    db_path: PathBuf,
}

impl EventStore {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;

        // Enable WAL mode for concurrent read/write
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        // 30s to accommodate maintenance VACUUM operations on large databases
        conn.pragma_update(None, "busy_timeout", 30000)?;

        let store = Self {
            conn,
            db_path: path.to_path_buf(),
        };
        store.run_migrations()?;
        Ok(store)
    }

    /// Create a backup of the database using rusqlite's backup API.
    /// The backup is written to `{db_path}.bak-{timestamp}`.
    fn backup_before_migrate(&self) -> Result<PathBuf> {
        let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ");
        let backup_path = PathBuf::from(format!(
            "{}.bak-{}",
            self.db_path.display(),
            timestamp
        ));

        let mut dst = Connection::open(&backup_path)?;
        let backup = rusqlite::backup::Backup::new(&self.conn, &mut dst)?;
        backup.run_to_completion(100, std::time::Duration::from_millis(10), None)?;

        info!(
            backup_path = %backup_path.display(),
            "Created pre-migration backup"
        );

        Ok(backup_path)
    }

    /// Bring the database to [`schema::CURRENT_SCHEMA_VERSION`].
    ///
    /// Every pending migration and the `user_version` bump run in one
    /// immediate transaction: a failure leaves the database at its old
    /// version, and a writer (the worker) never sees half a schema.  A
    /// database with data is backed up first.  A database newer than this
    /// daemon is left alone (the worker refuses to run against a version
    /// it does not expect).
    fn run_migrations(&self) -> Result<()> {
        let current_version: u32 = self
            .conn
            .pragma_query_value(None, "user_version", |row| row.get(0))?;

        if current_version >= schema::CURRENT_SCHEMA_VERSION {
            return Ok(());
        }

        // Create a backup before applying migrations on any non-empty database
        // that already has a schema (i.e., not a brand-new database).
        // We check for the events table to distinguish a freshly-created DB
        // (which has no user tables yet) from one with existing data.
        let has_existing_schema: bool = self
            .conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='events'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0)
            > 0;
        if has_existing_schema && self.db_path.exists() && std::fs::metadata(&self.db_path).map(|m| m.len()).unwrap_or(0) > 0 {
            self.backup_before_migrate()?;
        }

        let tx = rusqlite::Transaction::new_unchecked(
            &self.conn,
            rusqlite::TransactionBehavior::Immediate,
        )?;
        for (version, sql, then) in MIGRATIONS {
            if current_version < *version {
                tx.execute_batch(sql)?;
                if let Some(then) = then {
                    then(&tx)?;
                }
                info!(version, "Applied schema migration");
            }
        }
        tx.pragma_update(None, "user_version", schema::CURRENT_SCHEMA_VERSION)?;
        tx.commit()?;
        Ok(())
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    pub fn schema_version(&self) -> u32 {
        self.conn
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .unwrap_or(0)
    }

    pub fn is_wal_mode(&self) -> bool {
        let mode: String = self
            .conn
            .pragma_query_value(None, "journal_mode", |row| row.get(0))
            .unwrap_or_default();
        mode.to_lowercase() == "wal"
    }

    pub fn insert_event(&self, event: &Event) -> Result<()> {
        let cursor_x = event.cursor_global_px.as_ref().map(|c| c.x);
        let cursor_y = event.cursor_global_px.as_ref().map(|c| c.y);
        let display_ids_spanned_json = event
            .display_ids_spanned
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        let focused_element_json = event
            .focused_element
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;

        self.conn.execute(
            "INSERT INTO events (id, timestamp, kind_json, window_json, display_topology_json, \
             primary_display_id, cursor_x, cursor_y, ui_scale, artifact_ids_json, metadata_json, \
             display_ids_spanned_json, focused_element_json) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                event.id.to_string(),
                event.timestamp.to_rfc3339(),
                serde_json::to_string(&event.kind)?,
                event
                    .window
                    .as_ref()
                    .map(serde_json::to_string)
                    .transpose()?,
                serde_json::to_string(&event.display_topology)?,
                event.primary_display_id,
                cursor_x,
                cursor_y,
                event.ui_scale,
                serde_json::to_string(&event.artifact_ids)?,
                event.metadata.to_string(),
                display_ids_spanned_json,
                focused_element_json,
            ],
        )?;
        Ok(())
    }

    pub fn get_event(&self, id: Uuid) -> Result<Option<Event>> {
        use rusqlite::OptionalExtension;

        let mut stmt = self
            .conn
            .prepare(&format!("SELECT {EVENT_COLUMNS} FROM events WHERE id = ?1"))?;
        let row = stmt
            .query_row(params![id.to_string()], EventRow::read)
            .optional()?;
        row.map(EventRow::into_event).transpose()
    }

    pub fn get_unprocessed_events(&self, limit: u32) -> Result<Vec<Event>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {EVENT_COLUMNS} FROM events WHERE processed = 0 \
             ORDER BY timestamp ASC LIMIT ?1"
        ))?;
        let rows = stmt.query_map(params![limit], EventRow::read)?;
        rows.map(|row| row?.into_event()).collect()
    }

    /// True when a `ClipboardChange` with `content_hash` was recorded at or
    /// after `since`: the copy a paste of that content came from.
    pub fn has_copy_since(
        &self,
        content_hash: &str,
        since: chrono::DateTime<chrono::Utc>,
    ) -> Result<bool> {
        let found: i64 = self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM events \
             WHERE json_extract(kind_json, '$.type') = 'ClipboardChange' \
               AND json_extract(kind_json, '$.content_hash') = ?1 \
               AND julianday(timestamp) >= julianday(?2))",
            params![content_hash, since.to_rfc3339()],
            |row| row.get(0),
        )?;
        Ok(found != 0)
    }
}

/// A step that runs after a migration's SQL, for changes SQL cannot make
/// conditionally.
type MigrationStep = fn(&Connection) -> Result<()>;

/// Every migration, in the order it applies: the version it brings the
/// database to, its SQL, and the step that follows the SQL.
/// `worker/tests/daemon_db.py` reads the `include_str!` names below to
/// build test databases the same way (on a new database the steps change
/// nothing).
const MIGRATIONS: &[(u32, &str, Option<MigrationStep>)] = &[
    (1, include_str!("migrations/v001_initial.sql"), None),
    (2, include_str!("migrations/v002_add_display_ids_spanned.sql"), None),
    (3, include_str!("migrations/v003_add_scene_annotation.sql"), None),
    (4, include_str!("migrations/v004_drop_old_dom_trees.sql"), None),
    (5, include_str!("migrations/v005_add_focused_element.sql"), None),
    (6, include_str!("migrations/v006_worker_tables.sql"), Some(reconcile_columns)),
    (7, include_str!("migrations/v007_procedures.sql"), None),
    (8, include_str!("migrations/v008_memory_frames.sql"), None),
    (9, include_str!("migrations/v009_vector_store_next.sql"), None),
];

/// Columns v006's `CREATE TABLE IF NOT EXISTS` cannot give a table the
/// worker created before (SQLite has no `ADD COLUMN IF NOT EXISTS`).
const V006_ADDED_COLUMNS: &[(&str, &str, &str)] = &[
    // v0.4's vector store: the 8-bit copy and the embedding prefix.  Rows
    // without a prefix were embedded without one; '' makes them stale for
    // every current prefix, so they re-embed on their next upsert.  The
    // worker quantises rows with a NULL `qvec` when it opens the store.
    ("vector_store", "qvec", "BLOB"),
    ("vector_store", "qscale", "REAL"),
    ("vector_store", "prefix", "TEXT NOT NULL DEFAULT ''"),
    // Labelling-card answers, on observation tables from before they existed.
    ("workflow_observations", "user_label", "TEXT"),
    ("workflow_observations", "user_is_routine", "INTEGER"),
    ("workflow_observations", "labelled_at", "TEXT"),
];

/// Columns v0.4's worker wrote that nothing reads any more: per-frame
/// vectors (they live in `vector_store`) and the v1 miner's flag.
const V006_DROPPED_COLUMNS: &[(&str, &str)] = &[
    ("task_segments", "embeddings_json"),
    ("task_segments", "sop_generated"),
];

/// Give tables that existed before v006 the columns v006 creates them with.
fn reconcile_columns(conn: &Connection) -> Result<()> {
    let columns = |table: &str| -> Result<Vec<String>> {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let names = stmt
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(names)
    };
    for (table, column, decl) in V006_ADDED_COLUMNS {
        if !columns(table)?.iter().any(|c| c == column) {
            conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {column} {decl}"))?;
        }
    }
    for (table, column) in V006_DROPPED_COLUMNS {
        if columns(table)?.iter().any(|c| c == column) {
            conn.execute_batch(&format!("ALTER TABLE {table} DROP COLUMN {column}"))?;
        }
    }
    Ok(())
}

/// The columns [`EventRow::read`] reads, in its order.
const EVENT_COLUMNS: &str = "id, timestamp, kind_json, window_json, display_topology_json, \
     primary_display_id, cursor_x, cursor_y, ui_scale, artifact_ids_json, metadata_json, \
     display_ids_spanned_json, focused_element_json";

/// One `events` row as stored, before its JSON columns are parsed.
struct EventRow {
    id: String,
    timestamp: String,
    kind_json: String,
    window_json: Option<String>,
    display_json: String,
    primary_display: String,
    cursor_x: Option<i32>,
    cursor_y: Option<i32>,
    ui_scale: Option<f64>,
    artifact_ids_json: String,
    metadata_json: String,
    display_ids_spanned_json: Option<String>,
    focused_element_json: Option<String>,
}

impl EventRow {
    fn read(row: &rusqlite::Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            id: row.get(0)?,
            timestamp: row.get(1)?,
            kind_json: row.get(2)?,
            window_json: row.get(3)?,
            display_json: row.get(4)?,
            primary_display: row.get(5)?,
            cursor_x: row.get(6)?,
            cursor_y: row.get(7)?,
            ui_scale: row.get(8)?,
            artifact_ids_json: row.get(9)?,
            metadata_json: row.get(10)?,
            display_ids_spanned_json: row.get(11)?,
            focused_element_json: row.get(12)?,
        })
    }

    fn into_event(self) -> Result<Event> {
        Ok(Event {
            id: Uuid::parse_str(&self.id)?,
            timestamp: chrono::DateTime::parse_from_rfc3339(&self.timestamp)
                .map_err(|e| anyhow::anyhow!(e))?
                .with_timezone(&chrono::Utc),
            kind: serde_json::from_str(&self.kind_json)?,
            window: self
                .window_json
                .map(|j| serde_json::from_str(&j))
                .transpose()?,
            display_topology: serde_json::from_str(&self.display_json)?,
            primary_display_id: self.primary_display,
            cursor_global_px: match (self.cursor_x, self.cursor_y) {
                (Some(x), Some(y)) => Some(CursorPosition { x, y }),
                _ => None,
            },
            ui_scale: self.ui_scale,
            artifact_ids: serde_json::from_str(&self.artifact_ids_json)?,
            metadata: serde_json::from_str(&self.metadata_json)?,
            display_ids_spanned: self
                .display_ids_spanned_json
                .map(|j| serde_json::from_str(&j))
                .transpose()?,
            focused_element: self
                .focused_element_json
                .map(|j| serde_json::from_str(&j))
                .transpose()?,
        })
    }
}
