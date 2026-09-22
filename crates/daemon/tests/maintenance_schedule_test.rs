use agenthandover_daemon::maintenance::{
    is_due, marker_path, read_record, run_if_due, MaintenanceRecord,
};
use agenthandover_storage::maintenance::MaintenanceReport;
use chrono::{Duration, NaiveDate, Utc};
use std::cell::Cell;
use tempfile::TempDir;

fn report() -> MaintenanceReport {
    MaintenanceReport {
        events_purged: 12,
        artifacts_purged: 3,
        artifact_size_evicted: 0,
        screenshots_purged: 2,
        bytes_freed: 4096,
        episodes_purged: 0,
        vlm_jobs_purged: 0,
        vacuumed: false,
    }
}

fn write_marker(dir: &std::path::Path, date: NaiveDate) {
    let record = MaintenanceRecord {
        date,
        finished_at: Utc::now() - Duration::days(2),
        events_purged: 0,
        artifacts_purged: 0,
        bytes_freed: 0,
    };
    std::fs::write(marker_path(dir), serde_json::to_string(&record).unwrap()).unwrap();
}

#[test]
fn stale_marker_runs_once_and_advances() {
    let tmp = TempDir::new().unwrap();
    let today = NaiveDate::from_ymd_opt(2026, 9, 22).unwrap();
    write_marker(tmp.path(), today - Duration::days(2));

    let runs = Cell::new(0);
    let first = run_if_due(tmp.path(), today, || {
        runs.set(runs.get() + 1);
        Ok(report())
    })
    .unwrap();
    // A second check the same day (next hourly tick, or a daemon restart).
    let second = run_if_due(tmp.path(), today, || {
        runs.set(runs.get() + 1);
        Ok(report())
    })
    .unwrap();

    assert_eq!(runs.get(), 1, "maintenance runs exactly once per day");
    assert!(second.is_none());
    let record = first.expect("stale marker makes maintenance due");
    assert_eq!(record.date, today);
    assert_eq!(record.events_purged, 12);
    assert_eq!(record.artifacts_purged, 5, "artifact and screenshot files");
    assert_eq!(record.bytes_freed, 4096);
    assert_eq!(read_record(tmp.path()), Some(record), "marker advanced on disk");
}

#[test]
fn missing_marker_is_due() {
    let tmp = TempDir::new().unwrap();
    let today = NaiveDate::from_ymd_opt(2026, 9, 22).unwrap();
    assert!(is_due(read_record(tmp.path()).as_ref(), today));

    let ran = run_if_due(tmp.path(), today, || Ok(report())).unwrap();
    assert!(ran.is_some());
    assert!(!is_due(read_record(tmp.path()).as_ref(), today));
    assert!(is_due(read_record(tmp.path()).as_ref(), today + Duration::days(1)));
}

#[test]
fn failed_run_leaves_marker_for_retry() {
    let tmp = TempDir::new().unwrap();
    let today = NaiveDate::from_ymd_opt(2026, 9, 22).unwrap();
    let stale = today - Duration::days(2);
    write_marker(tmp.path(), stale);

    let result = run_if_due(tmp.path(), today, || anyhow::bail!("database is locked"));

    assert!(result.is_err());
    assert_eq!(read_record(tmp.path()).unwrap().date, stale);
    assert!(is_due(read_record(tmp.path()).as_ref(), today));
}

#[test]
fn corrupt_marker_is_treated_as_never_run() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(marker_path(tmp.path()), "not json").unwrap();
    let today = NaiveDate::from_ymd_opt(2026, 9, 22).unwrap();
    assert!(read_record(tmp.path()).is_none());
    assert!(run_if_due(tmp.path(), today, || Ok(report())).unwrap().is_some());
}
