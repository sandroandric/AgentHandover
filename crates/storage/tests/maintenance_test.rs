use agenthandover_storage::EventStore;
use agenthandover_storage::maintenance::{
    dir_size, sweep_expired_screenshots, MaintenanceOptions, MaintenanceRunner,
};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use agenthandover_common::event::*;
use chrono::Utc;
use rusqlite;
use uuid::Uuid;
use tempfile::TempDir;

fn make_test_event() -> Event {
    Event {
        id: Uuid::new_v4(),
        timestamp: Utc::now(),
        kind: EventKind::FocusChange,
        window: None,
        display_topology: vec![],
        primary_display_id: "d1".into(),
        cursor_global_px: None,
        ui_scale: None,
        artifact_ids: vec![],
        metadata: serde_json::json!({}),
        display_ids_spanned: None,
    }
}

#[test]
fn test_purge_old_events_with_no_old_data() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test.db");
    let store = EventStore::open(&db_path).unwrap();

    // Insert fresh events
    for _ in 0..3 {
        store.insert_event(&make_test_event()).unwrap();
    }

    let runner = MaintenanceRunner::new(store.connection());
    let purged = runner.purge_old_events(14).unwrap();
    assert_eq!(purged, 0, "Fresh events should not be purged");
}

#[test]
fn test_wal_checkpoint_succeeds() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test.db");
    let store = EventStore::open(&db_path).unwrap();

    let runner = MaintenanceRunner::new(store.connection());
    runner.wal_checkpoint().unwrap();
}

#[test]
fn test_vacuum_safe_check() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test.db");
    let _store = EventStore::open(&db_path).unwrap();

    // With very low thresholds, should be safe
    let safe = MaintenanceRunner::is_vacuum_safe(&db_path, 0, 1.0).unwrap();
    assert!(safe, "VACUUM should be safe with 0 GB minimum requirement");

    // With impossibly high threshold, should not be safe
    let unsafe_result = MaintenanceRunner::is_vacuum_safe(&db_path, 999999, 1.0).unwrap();
    assert!(!unsafe_result, "VACUUM should not be safe with 999999 GB requirement");
}

#[test]
fn test_purge_old_episodes_with_no_old_data() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test.db");
    let store = EventStore::open(&db_path).unwrap();

    // Insert a fresh, open episode
    store.connection().execute(
        "INSERT INTO episodes (id, segment_id, start_time, status) VALUES (?1, 0, ?2, 'open')",
        rusqlite::params![Uuid::new_v4().to_string(), Utc::now().to_rfc3339()],
    ).unwrap();

    let runner = MaintenanceRunner::new(store.connection());
    let purged = runner.purge_old_episodes(90).unwrap();
    assert_eq!(purged, 0, "Fresh or open episodes should not be purged");
}

#[test]
fn test_purge_old_episodes_deletes_closed_old() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test.db");
    let store = EventStore::open(&db_path).unwrap();

    // Insert an old, closed episode (200 days ago)
    let old_time = Utc::now() - chrono::Duration::days(200);
    store.connection().execute(
        "INSERT INTO episodes (id, segment_id, start_time, status) VALUES (?1, 0, ?2, 'closed')",
        rusqlite::params![Uuid::new_v4().to_string(), old_time.to_rfc3339()],
    ).unwrap();

    // Insert a recent closed episode (5 days ago)
    let recent_time = Utc::now() - chrono::Duration::days(5);
    store.connection().execute(
        "INSERT INTO episodes (id, segment_id, start_time, status) VALUES (?1, 0, ?2, 'closed')",
        rusqlite::params![Uuid::new_v4().to_string(), recent_time.to_rfc3339()],
    ).unwrap();

    let runner = MaintenanceRunner::new(store.connection());
    let purged = runner.purge_old_episodes(90).unwrap();
    assert_eq!(purged, 1, "Only the old closed episode should be purged");
}

#[test]
fn test_purge_expired_vlm_jobs() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test.db");
    let store = EventStore::open(&db_path).unwrap();

    let runner = MaintenanceRunner::new(store.connection());
    let purged = runner.purge_expired_vlm_jobs().unwrap();
    assert_eq!(purged, 0, "Empty table should have 0 purged");
}

// ---------------------------------------------------------------------------
// Retention over the legacy three-level artifact layout (artifacts/yyyy/mm/dd)
// ---------------------------------------------------------------------------

/// Insert an event `age_days` old, marked processed.
fn insert_aged_event(store: &EventStore, age_days: i64) -> String {
    let mut event = make_test_event();
    event.timestamp = Utc::now() - chrono::Duration::days(age_days);
    store.insert_event(&event).unwrap();
    store
        .connection()
        .execute("UPDATE events SET processed = 1 WHERE id = ?1", [event.id.to_string()])
        .unwrap();
    event.id.to_string()
}

/// Write a `.bin` file of `size` bytes under `artifacts/<date>/` and record it.
fn insert_artifact_file(
    store: &EventStore,
    artifact_dir: &Path,
    date: &str,
    id: &str,
    event_id: &str,
    size: usize,
) -> PathBuf {
    let dir = artifact_dir.join(date);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{id}.bin"));
    std::fs::write(&path, vec![7u8; size]).unwrap();
    store
        .connection()
        .execute(
            "INSERT INTO artifacts (id, event_id, artifact_type, file_path, \
             original_size_bytes, stored_size_bytes) VALUES (?1, ?2, 'screenshot', ?3, ?4, ?4)",
            rusqlite::params![id, event_id, path.to_string_lossy(), size as i64],
        )
        .unwrap();
    path
}

fn options<'p>(db_path: &'p Path, artifact_dir: &'p Path) -> MaintenanceOptions<'p> {
    MaintenanceOptions {
        db_path,
        retention_days_raw: 14,
        retention_days_episodes: 90,
        min_free_gb: 0,
        vacuum_safety_multiplier: 1.0,
        artifact_dir: Some(artifact_dir),
        artifact_max_bytes: None,
        screenshots_dir: None,
        screenshot_ttl_days: 7,
    }
}

fn artifact_rows(store: &EventStore) -> i64 {
    store
        .connection()
        .query_row("SELECT COUNT(*) FROM artifacts", [], |r| r.get(0))
        .unwrap()
}

fn set_age(path: &Path, days: u64) {
    std::fs::File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(SystemTime::now() - Duration::from_secs(days * 86_400))
        .unwrap();
}

#[test]
fn test_dir_size_walks_three_levels() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("artifacts");
    std::fs::create_dir_all(root.join("2026/08/01")).unwrap();
    std::fs::create_dir_all(root.join("2026/09/15")).unwrap();
    std::fs::write(root.join("2026/08/01/a.bin"), vec![0u8; 1000]).unwrap();
    std::fs::write(root.join("2026/09/15/b.bin"), vec![0u8; 234]).unwrap();
    std::fs::write(root.join("top.bin"), vec![0u8; 6]).unwrap();

    assert_eq!(dir_size(&root), 1240);
    assert_eq!(dir_size(&tmp.path().join("missing")), 0);
}

#[test]
fn test_retention_deletes_expired_artifact_files() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("events.db");
    let artifact_dir = tmp.path().join("artifacts");
    let store = EventStore::open(&db_path).unwrap();

    let old_event = insert_aged_event(&store, 30);
    let fresh_event = insert_aged_event(&store, 1);
    let old_file =
        insert_artifact_file(&store, &artifact_dir, "2026/08/01", "screenshot_old", &old_event, 500);
    let fresh_file = insert_artifact_file(
        &store, &artifact_dir, "2026/09/20", "screenshot_new", &fresh_event, 300,
    );

    let runner = MaintenanceRunner::new(store.connection());
    let report = runner.run_full_maintenance(&options(&db_path, &artifact_dir)).unwrap();

    assert!(!old_file.exists(), "expired artifact file must be deleted");
    assert!(fresh_file.exists(), "artifact inside retention must survive");
    assert_eq!(report.events_purged, 1);
    assert_eq!(report.artifacts_purged, 1);
    assert_eq!(report.bytes_freed, 500);
    assert_eq!(artifact_rows(&store), 1);
    assert!(!artifact_dir.join("2026/08").exists(), "empty date directories are removed");
}

#[test]
fn test_size_cap_evicts_oldest_artifact_files() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("events.db");
    let artifact_dir = tmp.path().join("artifacts");
    let store = EventStore::open(&db_path).unwrap();

    // Both inside retention; only the size cap can remove them.
    let older = insert_aged_event(&store, 3);
    let newer = insert_aged_event(&store, 1);
    let older_file =
        insert_artifact_file(&store, &artifact_dir, "2026/09/18", "screenshot_a", &older, 600);
    let newer_file =
        insert_artifact_file(&store, &artifact_dir, "2026/09/20", "screenshot_b", &newer, 600);

    let mut opts = options(&db_path, &artifact_dir);
    opts.artifact_max_bytes = Some(1000);
    let runner = MaintenanceRunner::new(store.connection());
    let report = runner.run_full_maintenance(&opts).unwrap();

    assert!(!older_file.exists(), "oldest artifact is evicted to meet the cap");
    assert!(newer_file.exists());
    assert_eq!(report.artifact_size_evicted, 1);
    assert_eq!(report.artifacts_purged, 1);
    assert_eq!(report.bytes_freed, 600);
    assert_eq!(artifact_rows(&store), 1);
    assert_eq!(dir_size(&artifact_dir), 600);
}

#[test]
fn test_orphan_sweep_deletes_unreachable_artifacts() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("events.db");
    let artifact_dir = tmp.path().join("artifacts");
    let store = EventStore::open(&db_path).unwrap();

    let kept_event = insert_aged_event(&store, 1);
    let kept =
        insert_artifact_file(&store, &artifact_dir, "2026/09/20", "screenshot_kept", &kept_event, 10);

    // A row whose event was purged by an older release (events before artifacts).
    store.connection().pragma_update(None, "foreign_keys", "OFF").unwrap();
    let orphan_row = insert_artifact_file(
        &store, &artifact_dir, "2026/07/01", "screenshot_orphan", "gone-event", 20,
    );
    store.connection().pragma_update(None, "foreign_keys", "ON").unwrap();

    // Files no row references at all, including an interrupted write.
    let stray_dir = artifact_dir.join("2026/06/30");
    std::fs::create_dir_all(&stray_dir).unwrap();
    let stray = stray_dir.join("screenshot_stray.bin");
    let tmp_file = stray_dir.join("screenshot_x.bin.tmp");
    std::fs::write(&stray, vec![0u8; 30]).unwrap();
    std::fs::write(&tmp_file, vec![0u8; 40]).unwrap();

    let runner = MaintenanceRunner::new(store.connection());
    let report = runner.run_full_maintenance(&options(&db_path, &artifact_dir)).unwrap();

    assert!(kept.exists());
    assert!(!orphan_row.exists());
    assert!(!stray.exists());
    assert!(!tmp_file.exists());
    assert_eq!(report.artifacts_purged, 3);
    assert_eq!(report.bytes_freed, 90);
    assert_eq!(artifact_rows(&store), 1);
    assert!(!artifact_dir.join("2026/06").exists());
    assert!(!artifact_dir.join("2026/07").exists());
}

#[test]
fn test_screenshot_ttl_sweep_deletes_only_expired_images() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join("screenshots");
    std::fs::create_dir_all(&dir).unwrap();

    let expired = dir.join("a.jpg");
    let fresh = dir.join("b.jpg");
    let other = dir.join("notes.txt");
    for p in [&expired, &fresh, &other] {
        std::fs::write(p, vec![1u8; 100]).unwrap();
    }
    set_age(&expired, 8);
    set_age(&other, 8);

    let (deleted, bytes) = sweep_expired_screenshots(&dir, 7);

    assert_eq!((deleted, bytes), (1, 100));
    assert!(!expired.exists());
    assert!(fresh.exists(), "a frame younger than the TTL waits for annotation");
    assert!(other.exists(), "only screenshot images are swept");
}

#[test]
fn test_full_maintenance_sweeps_screenshots_past_ttl() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("events.db");
    let artifact_dir = tmp.path().join("artifacts");
    let screenshots = tmp.path().join("screenshots");
    std::fs::create_dir_all(&screenshots).unwrap();
    let store = EventStore::open(&db_path).unwrap();

    let jpg = screenshots.join("frame.jpg");
    std::fs::write(&jpg, vec![1u8; 64]).unwrap();
    set_age(&jpg, 10);

    let mut opts = options(&db_path, &artifact_dir);
    opts.screenshots_dir = Some(&screenshots);
    let runner = MaintenanceRunner::new(store.connection());
    let report = runner.run_full_maintenance(&opts).unwrap();

    assert!(!jpg.exists());
    assert_eq!(report.screenshots_purged, 1);
    assert_eq!(report.bytes_freed, 64);
}
