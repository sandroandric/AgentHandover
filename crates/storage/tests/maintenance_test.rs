use agenthandover_storage::EventStore;
use agenthandover_storage::maintenance::MaintenanceRunner;
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
