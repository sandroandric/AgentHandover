mod schema;
mod migrations;
pub mod artifact_store;
pub mod maintenance;

use anyhow::Result;
use oc_apprentice_common::event::*;
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

    fn run_migrations(&self) -> Result<()> {
        let current_version: u32 = self
            .conn
            .pragma_query_value(None, "user_version", |row| row.get(0))?;

        if current_version < schema::CURRENT_SCHEMA_VERSION {
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

            if current_version < 1 {
                self.conn
                    .execute_batch(include_str!("migrations/v001_initial.sql"))?;
            }

            if current_version < 2 {
                self.conn
                    .execute_batch(include_str!("migrations/v002_add_display_ids_spanned.sql"))?;
            }

            if current_version < 3 {
                self.conn
                    .execute_batch(include_str!("migrations/v003_add_scene_annotation.sql"))?;
            }

            self.conn
                .pragma_update(None, "user_version", schema::CURRENT_SCHEMA_VERSION)?;
        }

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
            .map(|ids| serde_json::to_string(ids))
            .transpose()?;

        self.conn.execute(
            "INSERT INTO events (id, timestamp, kind_json, window_json, display_topology_json, \
             primary_display_id, cursor_x, cursor_y, ui_scale, artifact_ids_json, metadata_json, \
             display_ids_spanned_json) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                event.id.to_string(),
                event.timestamp.to_rfc3339(),
                serde_json::to_string(&event.kind)?,
                event
                    .window
                    .as_ref()
                    .map(|w| serde_json::to_string(w))
                    .transpose()?,
                serde_json::to_string(&event.display_topology)?,
                event.primary_display_id,
                cursor_x,
                cursor_y,
                event.ui_scale,
                serde_json::to_string(&event.artifact_ids)?,
                event.metadata.to_string(),
                display_ids_spanned_json,
            ],
        )?;
        Ok(())
    }

    /// Insert an artifact record into the `artifacts` table so maintenance
    /// (retention, size-based eviction) can find and clean up real files.
    pub fn insert_artifact(
        &self,
        artifact_id: &str,
        event_id: &str,
        artifact_type: &str,
        file_path: &str,
        original_size_bytes: u64,
        stored_size_bytes: u64,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO artifacts \
             (id, event_id, artifact_type, file_path, original_size_bytes, stored_size_bytes) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                artifact_id,
                event_id,
                artifact_type,
                file_path,
                original_size_bytes as i64,
                stored_size_bytes as i64,
            ],
        )?;
        Ok(())
    }

    pub fn get_event(&self, id: Uuid) -> Result<Option<Event>> {
        use rusqlite::OptionalExtension;

        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp, kind_json, window_json, display_topology_json, \
             primary_display_id, cursor_x, cursor_y, ui_scale, artifact_ids_json, metadata_json, \
             display_ids_spanned_json \
             FROM events WHERE id = ?1",
        )?;

        let result = stmt
            .query_row(params![id.to_string()], |row| {
                let id_str: String = row.get(0)?;
                let ts_str: String = row.get(1)?;
                let kind_json: String = row.get(2)?;
                let window_json: Option<String> = row.get(3)?;
                let display_json: String = row.get(4)?;
                let primary_display: String = row.get(5)?;
                let cursor_x: Option<i32> = row.get(6)?;
                let cursor_y: Option<i32> = row.get(7)?;
                let ui_scale: Option<f64> = row.get(8)?;
                let artifact_ids_json: String = row.get(9)?;
                let metadata_json: String = row.get(10)?;
                let display_ids_spanned_json: Option<String> = row.get(11)?;

                Ok((
                    id_str,
                    ts_str,
                    kind_json,
                    window_json,
                    display_json,
                    primary_display,
                    cursor_x,
                    cursor_y,
                    ui_scale,
                    artifact_ids_json,
                    metadata_json,
                    display_ids_spanned_json,
                ))
            })
            .optional();

        match result {
            Ok(Some((
                id_str,
                ts_str,
                kind_json,
                window_json,
                display_json,
                primary_display,
                cursor_x,
                cursor_y,
                ui_scale,
                artifact_ids_json,
                metadata_json,
                display_ids_spanned_json,
            ))) => {
                let event = Self::row_to_event(
                    id_str,
                    ts_str,
                    kind_json,
                    window_json,
                    display_json,
                    primary_display,
                    cursor_x,
                    cursor_y,
                    ui_scale,
                    artifact_ids_json,
                    metadata_json,
                    display_ids_spanned_json,
                )?;
                Ok(Some(event))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(anyhow::anyhow!(e)),
        }
    }

    pub fn get_unprocessed_events(&self, limit: u32) -> Result<Vec<Event>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp, kind_json, window_json, display_topology_json, \
             primary_display_id, cursor_x, cursor_y, ui_scale, artifact_ids_json, metadata_json, \
             display_ids_spanned_json \
             FROM events WHERE processed = 0 ORDER BY timestamp ASC LIMIT ?1",
        )?;

        let rows = stmt.query_map(params![limit], |row| {
            let id_str: String = row.get(0)?;
            let ts_str: String = row.get(1)?;
            let kind_json: String = row.get(2)?;
            let window_json: Option<String> = row.get(3)?;
            let display_json: String = row.get(4)?;
            let primary_display: String = row.get(5)?;
            let cursor_x: Option<i32> = row.get(6)?;
            let cursor_y: Option<i32> = row.get(7)?;
            let ui_scale: Option<f64> = row.get(8)?;
            let artifact_ids_json: String = row.get(9)?;
            let metadata_json: String = row.get(10)?;
            let display_ids_spanned_json: Option<String> = row.get(11)?;
            Ok((
                id_str,
                ts_str,
                kind_json,
                window_json,
                display_json,
                primary_display,
                cursor_x,
                cursor_y,
                ui_scale,
                artifact_ids_json,
                metadata_json,
                display_ids_spanned_json,
            ))
        })?;

        let mut events = Vec::new();
        for row in rows {
            let (
                id_str,
                ts_str,
                kind_json,
                window_json,
                display_json,
                primary_display,
                cursor_x,
                cursor_y,
                ui_scale,
                artifact_ids_json,
                metadata_json,
                display_ids_spanned_json,
            ) = row?;
            let event = Self::row_to_event(
                id_str,
                ts_str,
                kind_json,
                window_json,
                display_json,
                primary_display,
                cursor_x,
                cursor_y,
                ui_scale,
                artifact_ids_json,
                metadata_json,
                display_ids_spanned_json,
            )?;
            events.push(event);
        }
        Ok(events)
    }

    /// Convert a raw row tuple into an Event struct.
    fn row_to_event(
        id_str: String,
        ts_str: String,
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
    ) -> Result<Event> {
        Ok(Event {
            id: Uuid::parse_str(&id_str)?,
            timestamp: chrono::DateTime::parse_from_rfc3339(&ts_str)
                .map_err(|e| anyhow::anyhow!(e))?
                .with_timezone(&chrono::Utc),
            kind: serde_json::from_str(&kind_json)?,
            window: window_json
                .map(|j| serde_json::from_str(&j))
                .transpose()?,
            display_topology: serde_json::from_str(&display_json)?,
            primary_display_id: primary_display,
            cursor_global_px: match (cursor_x, cursor_y) {
                (Some(x), Some(y)) => Some(CursorPosition { x, y }),
                _ => None,
            },
            ui_scale,
            artifact_ids: serde_json::from_str(&artifact_ids_json)?,
            metadata: serde_json::from_str(&metadata_json)?,
            display_ids_spanned: display_ids_spanned_json
                .map(|j| serde_json::from_str(&j))
                .transpose()?,
        })
    }
}
