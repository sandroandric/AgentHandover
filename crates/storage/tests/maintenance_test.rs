use oc_apprentice_storage::EventStore;
use oc_apprentice_storage::maintenance::MaintenanceRunner;
use oc_apprentice_common::event::*;
use chrono::Utc;
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
fn test_purge_expired_vlm_jobs() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test.db");
    let store = EventStore::open(&db_path).unwrap();

    let runner = MaintenanceRunner::new(store.connection());
    let purged = runner.purge_expired_vlm_jobs().unwrap();
    assert_eq!(purged, 0, "Empty table should have 0 purged");
}
