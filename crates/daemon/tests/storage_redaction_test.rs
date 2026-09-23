//! Redaction is a storage invariant: whatever producer built an event, the
//! row that lands in SQLite is redacted.  These tests push a clipboard event
//! (built exactly as the daemon's clipboard forwarder builds it) and Chrome
//! events (decoded by the real native-messaging server) through
//! `run_storage_writer`, then read every stored column back as raw text.

use agenthandover_common::event::{Event, EventKind};
use agenthandover_daemon::ipc::native_messaging::{encode_frame, DaemonCommand, NativeMessageServer};
use agenthandover_daemon::observer::event_loop::{run_storage_writer, ObserverMessage};
use serde_json::json;
use tokio::sync::mpsc;

const ANTHROPIC_KEY: &str = "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789";
const GITHUB_TOKEN: &str = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij";
const JWT: &str = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U";
const STRIPE_KEY: &str = concat!("sk_", "live_51HqLyjWDarjtT1zdp7dcXYZ");
const RESET_TOKEN: &str = "reset-4f1c9e";

fn clipboard_event(preview: &str) -> Event {
    Event {
        id: uuid::Uuid::new_v4(),
        timestamp: chrono::Utc::now(),
        kind: EventKind::ClipboardChange {
            content_types: vec!["public.utf8-plain-text".into()],
            byte_size: preview.len() as u64,
            high_entropy: false,
            content_hash: "0".repeat(64),
            content_preview: Some(preview.to_string()),
        },
        window: None,
        display_topology: vec![],
        primary_display_id: "unknown".to_string(),
        cursor_global_px: None,
        ui_scale: None,
        artifact_ids: vec![],
        metadata: json!({}),
        display_ids_spanned: None,
        focused_element: None,
    }
}

/// Decode Chrome frames with the real native-messaging server.
async fn chrome_events(messages: Vec<serde_json::Value>) -> Vec<Event> {
    let mut bytes = Vec::new();
    for m in &messages {
        bytes.extend_from_slice(&encode_frame(m).unwrap());
    }
    let mut writer: Vec<u8> = Vec::new();
    let server = NativeMessageServer::new(std::io::Cursor::new(bytes), &mut writer);
    let (tx, mut rx) = mpsc::channel(16);
    let (_cmd_tx, cmd_rx) = mpsc::channel::<DaemonCommand>(1);
    server.run(tx, cmd_rx).await.unwrap();
    let mut events = Vec::new();
    while let Ok(e) = rx.try_recv() {
        events.push(e);
    }
    assert_eq!(events.len(), messages.len());
    events
}

/// Every text column of every stored event, concatenated.
fn stored_text(db_path: &std::path::Path) -> String {
    let conn = rusqlite::Connection::open(db_path).unwrap();
    let mut stmt = conn
        .prepare("SELECT kind_json, COALESCE(window_json, ''), metadata_json FROM events")
        .unwrap();
    let rows = stmt
        .query_map([], |r| {
            Ok(format!(
                "{}\n{}\n{}",
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?
            ))
        })
        .unwrap();
    rows.map(|r| r.unwrap()).collect::<Vec<_>>().join("\n")
}

#[tokio::test]
async fn clipboard_and_chrome_events_are_redacted_before_insert() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("events.db");
    let (tx, rx) = mpsc::channel(16);
    let writer = tokio::spawn(run_storage_writer(db_path.clone(), rx, None));

    // Clipboard: the user copied an API key and a labelled password.
    let clip = clipboard_event(&format!("key {ANTHROPIC_KEY} password: hunter2"));
    tx.send(ObserverMessage::Event { event: clip }).await.unwrap();

    // Chrome: a token in the tab title, a JWT in a DOM node, a Stripe key in
    // a click target, and a reset token in the URL query.
    let chrome = chrome_events(vec![
        json!({
            "type": "content_ready",
            "seq": 1,
            "timestamp": "2026-01-01T00:00:00Z",
            "payload": {
                "url": format!("https://app.example.com/reset?token={RESET_TOKEN}#top"),
                "title": format!("Settings {GITHUB_TOKEN}"),
                "tabId": 4
            }
        }),
        json!({
            "type": "dom_snapshot",
            "seq": 2,
            "timestamp": "2026-01-01T00:00:01Z",
            "payload": {
                "url": "https://app.example.com/debug",
                "tabId": 4,
                "captureReason": "request",
                "nodes": [{"tag": "pre", "innerText": format!("Authorization: {JWT}")}]
            }
        }),
        json!({
            "type": "click_intent",
            "seq": 3,
            "timestamp": "2026-01-01T00:00:02Z",
            "payload": {
                "x": 1, "y": 2,
                "url": "https://dashboard.example.com/keys",
                "tabId": 4,
                "target": {"tagName": "BUTTON", "innerText": format!("Copy {STRIPE_KEY}"), "composedPath": []}
            }
        }),
    ])
    .await;
    for event in chrome {
        tx.send(ObserverMessage::Event { event }).await.unwrap();
    }

    drop(tx);
    writer.await.unwrap().unwrap();

    let stored = stored_text(&db_path);
    for secret in [ANTHROPIC_KEY, "hunter2", GITHUB_TOKEN, JWT, STRIPE_KEY, RESET_TOKEN] {
        assert!(!stored.contains(secret), "{secret} reached storage:\n{stored}");
    }
    for marker in [
        "[REDACTED_ANTHROPIC_KEY]",
        "[REDACTED_PASSWORD]",
        "[REDACTED_GITHUB_TOKEN]",
        "[REDACTED_JWT]",
        "[REDACTED_STRIPE_KEY]",
    ] {
        assert!(stored.contains(marker), "{marker} missing:\n{stored}");
    }
    // The URL keeps origin + pathname so workflows stay recognisable.
    assert!(stored.contains("https://app.example.com/reset"));
    // Four events stored, none dropped by redaction.
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM events", [], |r| r.get(0)).unwrap();
    assert_eq!(count, 4);
}
