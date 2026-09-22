//! Once-per-day storage maintenance, scheduled on a persisted date marker.
//!
//! The marker file records the local calendar date of the last completed
//! run plus what it removed. Maintenance runs whenever the marker is older
//! than today — at daemon startup and on each hourly check — so a machine
//! that is asleep or shut down overnight still gets its daily run.

use agenthandover_storage::maintenance::MaintenanceReport;
use anyhow::Result;
use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Marker file name inside the daemon's state directory.
pub const MARKER_FILE: &str = "maintenance-state.json";

/// Contents of the marker file: the last completed run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaintenanceRecord {
    /// Local calendar date of the last completed run.
    pub date: NaiveDate,
    pub finished_at: DateTime<Utc>,
    pub events_purged: u64,
    pub artifacts_purged: u64,
    pub bytes_freed: u64,
}

pub fn marker_path(state_dir: &Path) -> PathBuf {
    state_dir.join(MARKER_FILE)
}

/// Read the marker; a missing or unreadable marker means "never ran".
pub fn read_record(state_dir: &Path) -> Option<MaintenanceRecord> {
    let raw = std::fs::read_to_string(marker_path(state_dir)).ok()?;
    serde_json::from_str(&raw).ok()
}

/// True when no run has completed on `today` (local date) or later.
pub fn is_due(record: Option<&MaintenanceRecord>, today: NaiveDate) -> bool {
    record.map_or(true, |r| r.date < today)
}

/// Run `maintenance` if it is due for `today`, then advance the marker.
///
/// Returns the new record when a run happened, `None` when it was not due.
/// A failed run leaves the marker untouched so the next check retries.
pub fn run_if_due(
    state_dir: &Path,
    today: NaiveDate,
    maintenance: impl FnOnce() -> Result<MaintenanceReport>,
) -> Result<Option<MaintenanceRecord>> {
    if !is_due(read_record(state_dir).as_ref(), today) {
        return Ok(None);
    }
    let report = maintenance()?;
    let record = MaintenanceRecord {
        date: today,
        finished_at: Utc::now(),
        events_purged: report.events_purged as u64,
        artifacts_purged: (report.artifacts_purged + report.screenshots_purged) as u64,
        bytes_freed: report.bytes_freed,
    };
    write_record(state_dir, &record)?;
    Ok(Some(record))
}

/// Atomically write the marker (tmp file + fsync + rename).
fn write_record(state_dir: &Path, record: &MaintenanceRecord) -> Result<()> {
    std::fs::create_dir_all(state_dir)?;
    let target = marker_path(state_dir);
    let tmp = state_dir.join(format!(".{}.tmp", MARKER_FILE));
    let mut file = std::fs::File::create(&tmp)?;
    file.write_all(serde_json::to_string_pretty(record)?.as_bytes())?;
    file.sync_all()?;
    std::fs::rename(&tmp, &target)?;
    Ok(())
}
