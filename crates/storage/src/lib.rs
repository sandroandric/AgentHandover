mod schema;
mod migrations;
pub mod artifact_store;

use anyhow::Result;
use oc_apprentice_common::event::*;
use rusqlite::{params, Connection};
use std::path::Path;
use uuid::Uuid;

pub struct EventStore {
    conn: Connection,
}

impl EventStore {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;

        // Enable WAL mode for concurrent read/write
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.pragma_update(None, "busy_timeout", 5000)?;

        let store = Self { conn };
        store.run_migrations()?;
        Ok(store)
    }

    fn run_migrations(&self) -> Result<()> {
        let current_version: u32 = self
            .conn
            .pragma_query_value(None, "user_version", |row| row.get(0))?;

        if current_version < 1 {
            self.conn
                .execute_batch(include_str!("migrations/v001_initial.sql"))?;
            self.conn
                .pragma_update(None, "user_version", schema::CURRENT_SCHEMA_VERSION)?;
        }

        Ok(())
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

        self.conn.execute(
            "INSERT INTO events (id, timestamp, kind_json, window_json, display_topology_json, \
             primary_display_id, cursor_x, cursor_y, ui_scale, artifact_ids_json, metadata_json) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
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
            ],
        )?;
        Ok(())
    }

    pub fn get_event(&self, id: Uuid) -> Result<Option<Event>> {
        use rusqlite::OptionalExtension;

        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp, kind_json, window_json, display_topology_json, \
             primary_display_id, cursor_x, cursor_y, ui_scale, artifact_ids_json, metadata_json \
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
             primary_display_id, cursor_x, cursor_y, ui_scale, artifact_ids_json, metadata_json \
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
        })
    }
}
