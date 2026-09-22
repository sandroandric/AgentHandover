use anyhow::Result;
use rusqlite::Connection;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
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
    ///
    /// Joins on `events`, so it must run before [`Self::purge_old_events`].
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

    /// Evict the oldest artifact records until the total on-disk size of
    /// `artifact_dir` (walked recursively) drops below `max_bytes`.
    ///
    /// Returns the file paths of records removed from the DB; the caller is
    /// responsible for deleting the actual files.
    pub fn evict_artifacts_by_size(
        &self,
        artifact_dir: &Path,
        max_bytes: u64,
    ) -> Result<Vec<String>> {
        let current_size = dir_size(artifact_dir);
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

        // Oldest first. LEFT JOIN so a row whose event is already gone is
        // still evictable (ordered by its own creation time).
        let mut stmt = self.conn.prepare(
            "SELECT a.file_path FROM artifacts a \
             LEFT JOIN events e ON a.event_id = e.id \
             ORDER BY COALESCE(e.timestamp, a.created_at) ASC",
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

    /// Remove artifact state that no retention rule can reach any more:
    /// rows whose event is gone (left behind by releases that purged events
    /// before artifacts), and `.bin` / `.bin.tmp` files under `artifact_dir`
    /// that no row references. Empty date directories are removed too.
    ///
    /// Returns `(files_deleted, bytes_freed)`.
    pub fn sweep_orphan_artifacts(&self, artifact_dir: &Path) -> Result<(usize, u64)> {
        let orphan_paths: Vec<String> = {
            let mut stmt = self.conn.prepare(
                "SELECT file_path FROM artifacts \
                 WHERE event_id NOT IN (SELECT id FROM events)",
            )?;
            let rows = stmt.query_map([], |row| row.get(0))?;
            rows.filter_map(|r| r.ok()).collect()
        };
        if !orphan_paths.is_empty() {
            self.conn.execute(
                "DELETE FROM artifacts WHERE event_id NOT IN (SELECT id FROM events)",
                [],
            )?;
        }
        let (mut deleted, mut bytes) = delete_files(&orphan_paths);

        let referenced: HashSet<PathBuf> = {
            let mut stmt = self.conn.prepare("SELECT file_path FROM artifacts")?;
            let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
            rows.filter_map(|r| r.ok()).map(PathBuf::from).collect()
        };
        let mut unreferenced = Vec::new();
        collect_files(artifact_dir, &mut |path| {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if (name.ends_with(".bin") || name.ends_with(".bin.tmp"))
                && !referenced.contains(path)
            {
                unreferenced.push(path.to_string_lossy().into_owned());
            }
        });
        let (d, b) = delete_files(&unreferenced);
        deleted += d;
        bytes += b;
        remove_empty_dirs(artifact_dir);

        if deleted > 0 {
            info!(deleted, bytes, "Swept orphan artifact files");
        }
        Ok((deleted, bytes))
    }

    /// Run the full daily maintenance cycle and delete the files it releases.
    pub fn run_full_maintenance(&self, opts: &MaintenanceOptions) -> Result<MaintenanceReport> {
        let mut artifacts_purged = 0usize;
        let mut bytes_freed = 0u64;

        if let Some(dir) = opts.artifact_dir {
            let (d, b) = self.sweep_orphan_artifacts(dir)?;
            artifacts_purged += d;
            bytes_freed += b;
        }

        // Artifact paths must be collected while their events still exist:
        // purge_old_artifacts joins on events.
        let expired = self.purge_old_artifacts(opts.retention_days_raw)?;
        let (d, b) = delete_files(&expired);
        artifacts_purged += d;
        bytes_freed += b;

        let events_purged = self.purge_old_events(opts.retention_days_raw)?;

        let mut artifact_size_evicted = 0usize;
        if let (Some(dir), Some(max_bytes)) = (opts.artifact_dir, opts.artifact_max_bytes) {
            let evicted = self.evict_artifacts_by_size(dir, max_bytes)?;
            artifact_size_evicted = evicted.len();
            let (d, b) = delete_files(&evicted);
            artifacts_purged += d;
            bytes_freed += b;
        }
        if let Some(dir) = opts.artifact_dir {
            remove_empty_dirs(dir);
        }

        let mut screenshots_purged = 0usize;
        if let Some(dir) = opts.screenshots_dir {
            let (d, b) = sweep_expired_screenshots(dir, opts.screenshot_ttl_days);
            screenshots_purged = d;
            bytes_freed += b;
        }

        let episodes_purged = self.purge_old_episodes(opts.retention_days_episodes)?;
        let vlm_purged = self.purge_expired_vlm_jobs()?;
        self.wal_checkpoint()?;
        let vacuumed =
            self.vacuum_if_safe(opts.db_path, opts.min_free_gb, opts.vacuum_safety_multiplier)?;

        Ok(MaintenanceReport {
            events_purged,
            artifacts_purged,
            artifact_size_evicted,
            screenshots_purged,
            bytes_freed,
            episodes_purged,
            vlm_jobs_purged: vlm_purged,
            vacuumed,
        })
    }
}

/// Inputs for [`MaintenanceRunner::run_full_maintenance`].
#[derive(Debug, Clone)]
pub struct MaintenanceOptions<'p> {
    pub db_path: &'p Path,
    pub retention_days_raw: u32,
    pub retention_days_episodes: u32,
    pub min_free_gb: u64,
    pub vacuum_safety_multiplier: f64,
    /// Legacy encrypted artifact store (`artifacts/yyyy/mm/dd/*.bin`).
    pub artifact_dir: Option<&'p Path>,
    pub artifact_max_bytes: Option<u64>,
    /// Plain JPEGs written for the VLM annotation pipeline.
    pub screenshots_dir: Option<&'p Path>,
    /// Screenshots older than this are deleted even if never annotated
    /// (the VLM job TTL: an unannotated frame past it will never be used).
    pub screenshot_ttl_days: u32,
}

#[derive(Debug)]
pub struct MaintenanceReport {
    pub events_purged: usize,
    /// Artifact files deleted from disk (retention, size cap and orphans).
    pub artifacts_purged: usize,
    /// Number of artifacts evicted due to folder size exceeding the cap
    /// (on top of normal retention-based purge).
    pub artifact_size_evicted: usize,
    /// Screenshot images deleted because they outlived the job TTL.
    pub screenshots_purged: usize,
    /// Bytes released by deleted artifact and screenshot files.
    pub bytes_freed: u64,
    pub episodes_purged: usize,
    pub vlm_jobs_purged: usize,
    pub vacuumed: bool,
}

/// Total size in bytes of all regular files under `dir`, at any depth.
/// Symlinks are not followed. A missing directory has size 0.
pub fn dir_size(dir: &Path) -> u64 {
    let mut total = 0u64;
    collect_files(dir, &mut |path| {
        total += std::fs::symlink_metadata(path).map(|m| m.len()).unwrap_or(0);
    });
    total
}

/// Delete screenshot images (`.jpg`, `.jpeg`, `.png`) directly under `dir`
/// whose modification time is more than `ttl_days` days ago.
///
/// Returns `(files_deleted, bytes_freed)`.
pub fn sweep_expired_screenshots(dir: &Path, ttl_days: u32) -> (usize, u64) {
    let cutoff = std::time::SystemTime::now()
        .checked_sub(std::time::Duration::from_secs(u64::from(ttl_days) * 86_400))
        .unwrap_or(std::time::UNIX_EPOCH);
    let mut expired = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let is_image = path
                .extension()
                .map(|e| {
                    let e = e.to_string_lossy().to_ascii_lowercase();
                    e == "jpg" || e == "jpeg" || e == "png"
                })
                .unwrap_or(false);
            let Ok(meta) = entry.metadata() else { continue };
            if !is_image || !meta.is_file() {
                continue;
            }
            if meta.modified().map(|m| m < cutoff).unwrap_or(false) {
                expired.push(path.to_string_lossy().into_owned());
            }
        }
    }
    let (deleted, bytes) = delete_files(&expired);
    if deleted > 0 {
        info!(deleted, bytes, ttl_days, "Deleted screenshots past the job TTL");
    }
    (deleted, bytes)
}

/// Delete each path; missing files are skipped. Returns `(deleted, bytes)`.
fn delete_files(paths: &[String]) -> (usize, u64) {
    let mut deleted = 0usize;
    let mut bytes = 0u64;
    for path_str in paths {
        let path = Path::new(path_str);
        let size = match std::fs::symlink_metadata(path) {
            Ok(m) => m.len(),
            Err(_) => continue,
        };
        match std::fs::remove_file(path) {
            Ok(()) => {
                deleted += 1;
                bytes += size;
            }
            Err(e) => warn!(path = %path.display(), error = %e, "Failed to delete file"),
        }
    }
    (deleted, bytes)
}

/// Call `f` for every regular file under `dir`, recursively, without
/// following symlinks.
fn collect_files(dir: &Path, f: &mut dyn FnMut(&Path)) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let Ok(ft) = entry.file_type() else { continue };
        let path = entry.path();
        if ft.is_dir() {
            collect_files(&path, f);
        } else if ft.is_file() {
            f(&path);
        }
    }
}

/// Remove empty subdirectories below `dir` (bottom-up); `dir` itself stays.
fn remove_empty_dirs(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            let path = entry.path();
            remove_empty_dirs(&path);
            // Fails (harmlessly) when the directory is not empty.
            let _ = std::fs::remove_dir(&path);
        }
    }
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
