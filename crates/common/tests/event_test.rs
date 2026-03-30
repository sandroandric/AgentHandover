use agenthandover_common::event::{
    Event, EventKind, DisplayInfo, WindowInfo, CursorPosition,
};
use chrono::Utc;
use uuid::Uuid;

#[test]
fn test_event_creation_and_serialization() {
    let event = Event {
        id: Uuid::new_v4(),
        timestamp: Utc::now(),
        kind: EventKind::FocusChange,
        window: Some(WindowInfo {
            window_id: "win_123".into(),
            app_id: "com.apple.Safari".into(),
            title: "Google - Safari".into(),
            bounds_global_px: [0, 0, 1920, 1080],
            z_order: 0,
            is_fullscreen: false,
        }),
        display_topology: vec![DisplayInfo {
            display_id: "display_1".into(),
            bounds_global_px: [0, 0, 2560, 1440],
            scale_factor: 2.0,
            orientation: 0,
        }],
        primary_display_id: "display_1".into(),
        cursor_global_px: Some(CursorPosition { x: 500, y: 300 }),
        ui_scale: Some(2.0),
        artifact_ids: vec![],
        metadata: serde_json::json!({}),
        display_ids_spanned: None,
    };

    let json = serde_json::to_string(&event).unwrap();
    let deserialized: Event = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.id, event.id);
    assert_eq!(deserialized.kind, EventKind::FocusChange);
    assert_eq!(deserialized.window.as_ref().unwrap().app_id, "com.apple.Safari");
}

#[test]
fn test_event_kind_variants() {
    let kinds = vec![
        EventKind::FocusChange,
        EventKind::WindowTitleChange,
        EventKind::ClickIntent { target_description: "Export CSV button".into() },
        EventKind::DwellSnapshot,
        EventKind::ScrollReadSnapshot,
        EventKind::ClipboardChange { content_types: vec!["text/plain".into()], byte_size: 42, high_entropy: false, content_hash: "abc123".into(), content_preview: Some("hello world".into()) },
        EventKind::PasteDetected { matched_copy_hash: Some("abc123".into()) },
        EventKind::SecureFieldFocus,
        EventKind::AppSwitch { from_app: "Safari".into(), to_app: "Terminal".into() },
    ];
    for kind in &kinds {
        let json = serde_json::to_string(kind).unwrap();
        let _: EventKind = serde_json::from_str(&json).unwrap();
    }
    assert_eq!(kinds.len(), 9);
}
