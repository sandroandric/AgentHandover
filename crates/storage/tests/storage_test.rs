use agenthandover_storage::EventStore;
use agenthandover_common::event::*;
use chrono::Utc;
use uuid::Uuid;
use tempfile::TempDir;

fn make_test_event() -> Event {
    Event {
        id: Uuid::new_v4(),
        timestamp: Utc::now(),
        kind: EventKind::FocusChange,
        window: Some(WindowInfo {
            window_id: "win_1".into(),
            app_id: "com.app.Test".into(),
            title: "Test Window".into(),
            bounds_global_px: [0, 0, 800, 600],
            z_order: 0,
            is_fullscreen: false,
        }),
        display_topology: vec![DisplayInfo {
            display_id: "d1".into(),
            bounds_global_px: [0, 0, 2560, 1440],
            scale_factor: 2.0,
            orientation: 0,
        }],
        primary_display_id: "d1".into(),
        cursor_global_px: Some(CursorPosition { x: 100, y: 200 }),
        ui_scale: Some(2.0),
        artifact_ids: vec![],
        metadata: serde_json::json!({}),
        display_ids_spanned: None,
        focused_element: None,
    }
}

#[test]
fn test_create_store_and_insert_event() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test.db");
    let store = EventStore::open(&db_path).unwrap();

    let event = make_test_event();
    let id = event.id;
    store.insert_event(&event).unwrap();

    let fetched = store.get_event(id).unwrap().unwrap();
    assert_eq!(fetched.id, id);
    assert_eq!(fetched.kind, EventKind::FocusChange);
}

#[test]
fn test_wal_mode_enabled() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test_wal.db");
    let store = EventStore::open(&db_path).unwrap();
    assert!(store.is_wal_mode());
}

#[test]
fn test_schema_version() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test_ver.db");
    let store = EventStore::open(&db_path).unwrap();
    assert_eq!(store.schema_version(), 7);
}

#[test]
fn test_get_unprocessed_events() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test_unproc.db");
    let store = EventStore::open(&db_path).unwrap();

    for _ in 0..5 {
        store.insert_event(&make_test_event()).unwrap();
    }

    let unprocessed = store.get_unprocessed_events(10).unwrap();
    assert_eq!(unprocessed.len(), 5);
}

#[test]
fn test_db_path_accessor() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test_path.db");
    let store = EventStore::open(&db_path).unwrap();
    assert_eq!(store.db_path(), db_path.as_path());
}

#[test]
fn test_no_backup_on_fresh_db() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("fresh.db");
    let _store = EventStore::open(&db_path).unwrap();

    // A fresh database should not create a backup file
    let backup_files: Vec<_> = std::fs::read_dir(tmp.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(".bak-"))
        .collect();
    assert_eq!(backup_files.len(), 0, "No backup should be created for a brand-new database");
}

#[test]
fn test_v004_drops_dom_trees_older_than_the_retention_window() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("v3.db");
    {
        // A database the previous daemon (schema 3) wrote.
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        for sql in [
            include_str!("../src/migrations/v001_initial.sql"),
            include_str!("../src/migrations/v002_add_display_ids_spanned.sql"),
            include_str!("../src/migrations/v003_add_scene_annotation.sql"),
        ] {
            conn.execute_batch(sql).unwrap();
        }
        conn.pragma_update(None, "user_version", 3).unwrap();
        let insert = |id: &str, age_days: i64, metadata: &str| {
            let ts = (Utc::now() - chrono::Duration::days(age_days)).to_rfc3339();
            conn.execute(
                "INSERT INTO events (id, timestamp, kind_json, display_topology_json, \
                 primary_display_id, metadata_json) \
                 VALUES (?1, ?2, '{\"type\":\"DwellSnapshot\"}', '[]', '', ?3)",
                rusqlite::params![id, ts, metadata],
            )
            .unwrap();
        };
        insert("old-tree", 30, r#"{"nodes":[{"tag":"div"}],"url":"https://a.example/"}"#);
        insert("new-tree", 1, r#"{"nodes":[{"tag":"div"}],"url":"https://a.example/"}"#);
        insert("old-anchors", 30, r#"{"anchors":[],"digest":"x","url":"https://a.example/"}"#);
    }

    let store = EventStore::open(&db_path).unwrap();
    assert_eq!(store.schema_version(), 7);
    let metadata = |id: &str| -> serde_json::Value {
        let text: String = store
            .connection()
            .query_row("SELECT metadata_json FROM events WHERE id = ?1", [id], |r| r.get(0))
            .unwrap();
        serde_json::from_str(&text).unwrap()
    };
    let old = metadata("old-tree");
    assert!(old.get("nodes").is_none(), "tree outside the window is dropped: {old}");
    assert_eq!(old["url"], "https://a.example/", "the rest of the row stays");
    assert!(metadata("new-tree")["nodes"].is_array(), "tree inside the window stays");
    assert_eq!(metadata("old-anchors")["digest"], "x");
}

#[test]
fn test_focused_element_round_trips_through_its_column() {
    let tmp = TempDir::new().unwrap();
    let store = EventStore::open(&tmp.path().join("focused.db")).unwrap();

    let mut event = make_test_event();
    event.kind = EventKind::DwellSnapshot;
    event.focused_element = Some(FocusedElement {
        role: "AXCell".into(),
        value: Some("1,250.00".into()),
        column_header: Some("Amount".into()),
        document: Some("file:///Users/me/Q3.xlsx".into()),
        ancestors: vec![AxAncestor {
            role: "AXRow".into(),
            title: Some("Row 4".into()),
            index: Some(3),
            ..Default::default()
        }],
        ..Default::default()
    });
    store.insert_event(&event).unwrap();

    let column: String = store
        .connection()
        .query_row(
            "SELECT focused_element_json FROM events WHERE id = ?1",
            [event.id.to_string()],
            |r| r.get(0),
        )
        .unwrap();
    let stored: serde_json::Value = serde_json::from_str(&column).unwrap();
    assert_eq!(stored["role"], "AXCell");
    assert_eq!(stored["column_header"], "Amount");
    assert_eq!(stored["ancestors"][0]["index"], 3);
    assert!(stored.get("subrole").is_none(), "absent fields are not written");

    let fetched = store.get_event(event.id).unwrap().unwrap();
    assert_eq!(fetched.focused_element, event.focused_element);

    // An event without one stores NULL.
    let plain = make_test_event();
    store.insert_event(&plain).unwrap();
    let null: Option<String> = store
        .connection()
        .query_row(
            "SELECT focused_element_json FROM events WHERE id = ?1",
            [plain.id.to_string()],
            |r| r.get(0),
        )
        .unwrap();
    assert!(null.is_none());
}

#[test]
fn test_v005_adds_the_column_to_a_v4_database() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("v4.db");
    {
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        for sql in [
            include_str!("../src/migrations/v001_initial.sql"),
            include_str!("../src/migrations/v002_add_display_ids_spanned.sql"),
            include_str!("../src/migrations/v003_add_scene_annotation.sql"),
            include_str!("../src/migrations/v004_drop_old_dom_trees.sql"),
        ] {
            conn.execute_batch(sql).unwrap();
        }
        conn.pragma_update(None, "user_version", 4).unwrap();
    }
    let store = EventStore::open(&db_path).unwrap();
    assert_eq!(store.schema_version(), 7);
    store.insert_event(&make_test_event()).unwrap();
}

#[test]
fn test_has_copy_since_matches_a_recent_clipboard_change() {
    let tmp = TempDir::new().unwrap();
    let store = EventStore::open(&tmp.path().join("copy.db")).unwrap();

    let mut copy = make_test_event();
    copy.timestamp = Utc::now() - chrono::Duration::minutes(5);
    copy.kind = EventKind::ClipboardChange {
        content_types: vec!["public.utf8-plain-text".into()],
        byte_size: 11,
        high_entropy: false,
        content_hash: "hash-a".into(),
        content_preview: None,
    };
    store.insert_event(&copy).unwrap();

    let since = Utc::now() - chrono::Duration::minutes(30);
    assert!(store.has_copy_since("hash-a", since).unwrap());
    assert!(!store.has_copy_since("hash-b", since).unwrap());
    // A copy older than the window does not match.
    assert!(!store
        .has_copy_since("hash-a", Utc::now() - chrono::Duration::minutes(1))
        .unwrap());
}
