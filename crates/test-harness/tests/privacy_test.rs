use agenthandover_common::event::*;
use agenthandover_common::redaction::Redactor;
use agenthandover_test_harness::recorder::EventRecorder;
use agenthandover_test_harness::replayer::EventReplayer;
use chrono::Utc;
use uuid::Uuid;
use tempfile::TempDir;

fn make_event_with_title(title: &str) -> Event {
    Event {
        id: Uuid::new_v4(),
        timestamp: Utc::now(),
        kind: EventKind::FocusChange,
        window: Some(WindowInfo {
            window_id: "win_1".into(),
            app_id: "com.test.App".into(),
            title: title.into(),
            bounds_global_px: [0, 0, 800, 600],
            z_order: 0,
            is_fullscreen: false,
        }),
        display_topology: vec![],
        primary_display_id: "d1".into(),
        cursor_global_px: None,
        ui_scale: None,
        artifact_ids: vec![],
        metadata: serde_json::json!({}),
        display_ids_spanned: None,
    }
}

fn make_event_with_metadata(key: &str, value: &str) -> Event {
    Event {
        id: Uuid::new_v4(),
        timestamp: Utc::now(),
        kind: EventKind::DwellSnapshot,
        window: None,
        display_topology: vec![],
        primary_display_id: "d1".into(),
        cursor_global_px: None,
        ui_scale: None,
        artifact_ids: vec![],
        metadata: serde_json::json!({ key: value }),
        display_ids_spanned: None,
    }
}

/// Helper: apply redaction to all text fields of an event.
fn redact_event(event: &Event, redactor: &Redactor) -> Event {
    let mut redacted = event.clone();

    if let Some(ref mut w) = redacted.window {
        w.title = redactor.redact(&w.title);
    }

    // Redact metadata values
    if let serde_json::Value::Object(ref mut map) = redacted.metadata {
        for value in map.values_mut() {
            if let serde_json::Value::String(s) = value {
                *s = redactor.redact(s);
            }
        }
    }

    redacted
}

// ===== Privacy Tests =====

#[test]
fn test_aws_key_never_reaches_storage() {
    let redactor = Redactor::new();
    let event = make_event_with_title("export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE");
    let redacted = redact_event(&event, &redactor);

    let title = &redacted.window.unwrap().title;
    assert!(!title.contains("AKIAIOSFODNN7EXAMPLE"), "AWS key should be redacted from title");
    assert!(title.contains("[REDACTED_AWS_KEY]"));
}

#[test]
fn test_aws_secret_never_reaches_storage() {
    let redactor = Redactor::new();
    let event = make_event_with_metadata(
        "env",
        "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    );
    let redacted = redact_event(&event, &redactor);

    let meta_str = serde_json::to_string(&redacted.metadata).unwrap();
    assert!(!meta_str.contains("wJalrXUtnFEMI"), "AWS secret should be redacted from metadata");
    assert!(meta_str.contains("[REDACTED_SECRET]"));
}

#[test]
fn test_credit_card_never_reaches_storage() {
    let redactor = Redactor::new();
    let event = make_event_with_title("Payment: 4111-1111-1111-1111");
    let redacted = redact_event(&event, &redactor);

    let title = &redacted.window.unwrap().title;
    assert!(!title.contains("4111-1111-1111-1111"), "CC number should be redacted");
    assert!(title.contains("[REDACTED_CC]"));
}

#[test]
fn test_private_key_never_reaches_storage() {
    let redactor = Redactor::new();
    let pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQ...\n-----END RSA PRIVATE KEY-----";
    let event = make_event_with_metadata("key_file", pem);
    let redacted = redact_event(&event, &redactor);

    let meta_str = serde_json::to_string(&redacted.metadata).unwrap();
    assert!(!meta_str.contains("MIIEowIBAAKCAQ"), "Private key should be redacted");
    assert!(meta_str.contains("[REDACTED_PRIVATE_KEY]"));
}

#[test]
fn test_github_token_never_reaches_storage() {
    let redactor = Redactor::new();
    let event = make_event_with_title("GITHUB_TOKEN=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl");
    let redacted = redact_event(&event, &redactor);

    let title = &redacted.window.unwrap().title;
    assert!(!title.contains("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "GitHub token should be redacted");
}

#[test]
fn test_high_entropy_hex_never_reaches_storage() {
    let redactor = Redactor::new();
    // 80+ hex chars to trigger redaction (shorter hashes like SHA-256 are not redacted)
    let hex_token = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0";
    let event = make_event_with_title(&format!("token: {}", hex_token));
    let redacted = redact_event(&event, &redactor);

    let title = &redacted.window.unwrap().title;
    assert!(!title.contains(hex_token), "High-entropy hex should be redacted");
    assert!(title.contains("[REDACTED_HIGH_ENTROPY]"));
}

#[test]
fn test_normal_text_passes_through() {
    let redactor = Redactor::new();
    let event = make_event_with_title("Google Chrome - Search Results");
    let redacted = redact_event(&event, &redactor);

    assert_eq!(
        redacted.window.unwrap().title,
        "Google Chrome - Search Results",
        "Normal text should not be modified"
    );
}

#[test]
fn test_redacted_events_round_trip_through_harness() {
    let redactor = Redactor::new();
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("privacy.jsonl");

    // Create events with secrets
    let events = vec![
        make_event_with_title("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"),
        make_event_with_title("Card: 4111-1111-1111-1111"),
        make_event_with_title("Normal window title"),
    ];

    // Redact and record
    let mut recorder = EventRecorder::new(&path).unwrap();
    for event in &events {
        let redacted = redact_event(event, &redactor);
        recorder.record(&redacted).unwrap();
    }
    recorder.flush().unwrap();

    // Replay and verify no secrets leaked
    let replayer = EventReplayer::from_file(&path).unwrap();
    let file_content = std::fs::read_to_string(&path).unwrap();

    assert!(!file_content.contains("AKIAIOSFODNN7EXAMPLE"), "AWS key leaked to file");
    assert!(!file_content.contains("4111-1111-1111-1111"), "CC number leaked to file");
    assert!(file_content.contains("Normal window title"), "Normal text should persist");
    assert_eq!(replayer.event_count(), 3);
}

#[test]
fn test_multiple_secrets_in_one_event() {
    let redactor = Redactor::new();
    let event = make_event_with_title(
        "env: AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE card=4111-1111-1111-1111"
    );
    let redacted = redact_event(&event, &redactor);

    let title = &redacted.window.unwrap().title;
    assert!(!title.contains("AKIAIOSFODNN7EXAMPLE"), "First secret should be redacted");
    assert!(!title.contains("4111-1111-1111-1111"), "Second secret should be redacted");
}
