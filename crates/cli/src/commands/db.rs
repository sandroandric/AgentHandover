//! `agenthandover db migrate`: bring the database to the schema this
//! release uses, with the daemon's own migrations
//! (`agenthandover_storage::EventStore::open`: a backup first, then every
//! pending migration in one transaction).
//!
//! The daemon migrates when it starts, but it starts only while capture
//! runs. After an upgrade with capture paused, the worker waited for it
//! forever and Memory, search and Commonplace stayed down (seen on a 0.4.0
//! database upgraded to 2.0.0 while paused). The worker now runs this
//! command when it finds an older database.

use std::path::Path;

use anyhow::{Context, Result};

/// `(before, after)` schema versions of the database at `path`, migrated;
/// None when there is no database yet (the daemon creates it on its first
/// start, with nothing to migrate).
pub fn migrate_at(path: &Path) -> Result<Option<(u32, u32)>> {
    if !path.exists() {
        return Ok(None);
    }
    let before = schema_version(path)?;
    let store = agenthandover_storage::EventStore::open(path)
        .with_context(|| format!("migrating {}", path.display()))?;
    Ok(Some((before, store.schema_version())))
}

/// The database's `user_version`, read without migrating it.
fn schema_version(path: &Path) -> Result<u32> {
    let conn = rusqlite::Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )?;
    Ok(conn.pragma_query_value(None, "user_version", |row| row.get(0))?)
}

pub fn migrate(path: Option<std::path::PathBuf>) -> Result<()> {
    let path = path.unwrap_or_else(|| agenthandover_common::status::data_dir().join("events.db"));
    match migrate_at(&path)? {
        None => println!("No database yet at {}; the daemon creates it on its first start.", path.display()),
        Some((before, after)) if before == after => {
            println!("Database already at schema {after}: {}", path.display())
        }
        Some((before, after)) => {
            println!("Database migrated from schema {before} to {after}: {}", path.display())
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_database_is_left_for_the_daemon_to_create() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.db");
        assert_eq!(migrate_at(&path).unwrap(), None);
        assert!(!path.exists());
    }

    /// An older database (here an empty one, version 0) reaches the
    /// current schema; a second run changes nothing. The migrations
    /// themselves, from 0.4.0's version 3 with its rows, are proved in
    /// crates/storage/tests/migration_test.rs.
    #[test]
    fn an_older_database_reaches_the_current_schema_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.db");
        std::fs::File::create(&path).unwrap();
        let (before, after) = migrate_at(&path).unwrap().unwrap();
        assert_eq!(before, 0);
        let current = agenthandover_storage::EventStore::open(&path).unwrap().schema_version();
        assert_eq!(after, current);
        assert!(after >= 9);
        assert_eq!(migrate_at(&path).unwrap(), Some((after, after)));
    }
}
