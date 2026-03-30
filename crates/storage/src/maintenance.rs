use anyhow::Result;
use rusqlite::Connection;
use std::path::Path;
use tracing::{info, warn};

pub struct MaintenanceRunner<'a> {
    conn: &'a Connection,
}

impl<'a> MaintenanceRunner<'a> {
    pub fn new(conn: &'a Connection) -> Self {
        Self { conn }
    }

    /// Delete events older than `retention_days` that have been processed.
    /// Returns the number of rows deleted.
    pub fn purge_old_events(&self, retention_days: u32) -> Result<usize> {
        let deleted = self.conn.execute(
            "DELETE FROM events WHERE processed = 1 AND datetime(timestamp) < datetime('now', ?1)",
            [format!("-{} days", retention_days)],
        )?;
        info!(deleted, retention_days, "Purged old processed events");
        Ok(deleted)
    }

    /// Delete old artifact records and return their file paths for cleanup.
    pub fn purge_old_artifacts(&self, retention_days: u32) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT a.file_path FROM artifacts a \
             INNER JOIN events e ON a.event_id = e.id \
             WHERE e.processed = 1 AND datetime(e.timestamp) < datetime('now', ?1)",
        )?;

        let paths: Vec<String> = stmt
            .query_map([format!("-{} days", retention_days)], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        if !paths.is_empty() {
            self.conn.execute(
                "DELETE FROM artifacts WHERE event_id IN (\
                    SELECT e.id FROM events e \
                    WHERE e.processed = 1 AND datetime(e.timestamp) < datetime('now', ?1)\
                )",
                [format!("-{} days", retention_days)],
            )?;
            info!(count = paths.len(), "Purged old artifact records");
        }

        Ok(paths)
    }

    /// Delete episodes older than `retention_days` that are closed.
    /// Returns the number of rows deleted.
    pub fn purge_old_episodes(&self, retention_days: u32) -> Result<usize> {
        let deleted = self.conn.execute(
            "DELETE FROM episodes WHERE status = 'closed' AND datetime(start_time) < datetime('now', ?1)",
            [format!("-{} days", retention_days)],
        )?;
        if deleted > 0 {
            info!(deleted, retention_days, "Purged old closed episodes");
        }
        Ok(deleted)
    }

    /// Delete expired VLM queue entries.
    pub fn purge_expired_vlm_jobs(&self) -> Result<usize> {
        let deleted = self.conn.execute(
            "DELETE FROM vlm_queue WHERE datetime(ttl_expires_at) < datetime('now')",
            [],
        )?;
        if deleted > 0 {
            info!(deleted, "Purged expired VLM queue entries");
        }
        Ok(deleted)
    }

    /// Run WAL checkpoint (TRUNCATE mode -- reclaims WAL file space).
    pub fn wal_checkpoint(&self) -> Result<()> {
        self.conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
        info!("WAL checkpoint completed (TRUNCATE)");
        Ok(())
    }

    /// Check if there's enough free disk space before VACUUM.
    /// Returns true if safe to proceed (free space > db_size * safety_multiplier).
    pub fn is_vacuum_safe(
        db_path: &Path,
        min_free_gb: u64,
        safety_multiplier: f64,
    ) -> Result<bool> {
        let db_size = std::fs::metadata(db_path).map(|m| m.len()).unwrap_or(0);

        // Get free space using statvfs on Unix
        let free_bytes = get_free_disk_space(db_path)?;
        let free_gb = free_bytes / (1024 * 1024 * 1024);

        // safety_multiplier should be at least 2.5 to account for concurrent writes
        // during VACUUM, which temporarily doubles the database file.
        let required = (db_size as f64 * safety_multiplier) as u64;
        let safe = free_gb >= min_free_gb && free_bytes > required;

        if !safe {
            warn!(
                free_gb,
                min_free_gb,
                db_size_bytes = db_size,
                "Insufficient disk space for VACUUM"
            );
        }

        Ok(safe)
    }

    /// Run VACUUM if disk space permits.
    pub fn vacuum_if_safe(
        &self,
        db_path: &Path,
        min_free_gb: u64,
        safety_multiplier: f64,
    ) -> Result<bool> {
        if Self::is_vacuum_safe(db_path, min_free_gb, safety_multiplier)? {
            self.conn.execute_batch("VACUUM;")?;
            info!("VACUUM completed successfully");
            Ok(true)
        } else {
            warn!("Skipping VACUUM due to insufficient disk space");
            Ok(false)
        }
    }

    /// Evict the oldest artifact records (by event timestamp) until the total
    /// on-disk folder size drops below `max_bytes`.
    ///
    /// Returns the file paths of records removed from the DB; the caller is
    /// responsible for deleting the actual files.
    pub fn evict_artifacts_by_size(
        &self,
        artifact_dir: &Path,
        max_bytes: u64,
    ) -> Result<Vec<String>> {
        let current_size = get_dir_size_shallow(artifact_dir);
        if current_size <= max_bytes {
            return Ok(vec![]);
        }

        let excess = current_size - max_bytes;
        info!(
            current_mb = current_size / (1024 * 1024),
            max_mb = max_bytes / (1024 * 1024),
            excess_mb = excess / (1024 * 1024),
            "Artifact directory exceeds size cap, evicting oldest artifacts"
        );

        // Fetch oldest artifacts first (by event timestamp ascending).
        let mut stmt = self.conn.prepare(
            "SELECT a.file_path FROM artifacts a \
             INNER JOIN events e ON a.event_id = e.id \
             ORDER BY e.timestamp ASC",
        )?;

        let all_paths: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        let mut freed: u64 = 0;
        let mut evict_paths: Vec<String> = Vec::new();

        for path_str in &all_paths {
            if freed >= excess {
                break;
            }
            let p = Path::new(path_str);
            let file_size = p.metadata().map(|m| m.len()).unwrap_or(0);
            freed += file_size;
            evict_paths.push(path_str.clone());
        }

        if !evict_paths.is_empty() {
            // Delete evicted records from the DB
            let placeholders: String = evict_paths.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!("DELETE FROM artifacts WHERE file_path IN ({})", placeholders);
            let params: Vec<&dyn rusqlite::types::ToSql> = evict_paths
                .iter()
                .map(|s| s as &dyn rusqlite::types::ToSql)
                .collect();
            self.conn.execute(&sql, params.as_slice())?;
            info!(
                evicted = evict_paths.len(),
                freed_mb = freed / (1024 * 1024),
                "Evicted oldest artifacts to meet size cap"
            );
        }

        Ok(evict_paths)
    }

    /// Run the full nightly maintenance cycle.
    pub fn run_full_maintenance(
        &self,
        db_path: &Path,
        retention_days_raw: u32,
        retention_days_episodes: u32,
        min_free_gb: u64,
        vacuum_safety_multiplier: f64,
        artifact_dir: Option<&Path>,
        artifact_max_bytes: Option<u64>,
    ) -> Result<MaintenanceReport> {
        let events_purged = self.purge_old_events(retention_days_raw)?;
        let mut artifact_paths = self.purge_old_artifacts(retention_days_raw)?;

        // Size-based eviction: evict oldest artifacts if folder exceeds cap
        let mut artifact_size_evicted: usize = 0;
        if let (Some(dir), Some(max_bytes)) = (artifact_dir, artifact_max_bytes) {
            let evicted = self.evict_artifacts_by_size(dir, max_bytes)?;
            artifact_size_evicted = evicted.len();
            artifact_paths.extend(evicted);
        }

        let episodes_purged = self.purge_old_episodes(retention_days_episodes)?;
        let vlm_purged = self.purge_expired_vlm_jobs()?;
        self.wal_checkpoint()?;
        let vacuumed = self.vacuum_if_safe(db_path, min_free_gb, vacuum_safety_multiplier)?;

        Ok(MaintenanceReport {
            events_purged,
            artifact_paths_to_delete: artifact_paths,
            artifact_size_evicted,
            episodes_purged,
            vlm_jobs_purged: vlm_purged,
            vacuumed,
        })
    }
}

#[derive(Debug)]
pub struct MaintenanceReport {
    pub events_purged: usize,
    pub artifact_paths_to_delete: Vec<String>,
    /// Number of artifacts evicted due to folder size exceeding the cap
    /// (on top of normal retention-based purge).
    pub artifact_size_evicted: usize,
    pub episodes_purged: usize,
    pub vlm_jobs_purged: usize,
    pub vacuumed: bool,
}

/// Sum file sizes in a directory, traversing one level of subdirectories.
fn get_dir_size_shallow(dir: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                total += path.metadata().map(|m| m.len()).unwrap_or(0);
            } else if path.is_dir() {
                if let Ok(sub_entries) = std::fs::read_dir(&path) {
                    for sub_entry in sub_entries.flatten() {
                        if sub_entry.path().is_file() {
                            total += sub_entry.path().metadata().map(|m| m.len()).unwrap_or(0);
                        }
                    }
                }
            }
        }
    }
    total
}

#[cfg(unix)]
fn get_free_disk_space(path: &Path) -> Result<u64> {
    use std::ffi::CString;
    use std::mem;

    let path_str = path.parent().unwrap_or(path).to_string_lossy();
    let c_path = CString::new(path_str.as_bytes())?;

    unsafe {
        let mut stat: libc::statvfs = mem::zeroed();
        if libc::statvfs(c_path.as_ptr(), &mut stat) == 0 {
            Ok(stat.f_bavail as u64 * stat.f_frsize as u64)
        } else {
            anyhow::bail!("Failed to get disk space info")
        }
    }
}
