//! Integration tests for the Chrome Native Messaging framing protocol.
//!
//! Tests cover:
//!   - Frame encoding: verify 4-byte LE length prefix + UTF-8 JSON payload
//!   - Frame decoding: parse a raw byte buffer back to JSON
//!   - Roundtrip: encode then decode
//!   - Oversized message rejection (> 1 MB)
//!   - Malformed JSON handling (error, no panic)
//!   - Async read/write through NativeMessageServer

use agenthandover_daemon::ipc::native_messaging::{
    decode_frame, encode_frame, DaemonCommand, NativeMessageServer, MAX_MESSAGE_SIZE,
};
use serde_json::json;
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------
// Frame encoding tests
// ---------------------------------------------------------------------------

#[test]
fn encode_produces_correct_length_prefix() {
    let msg = json!({"type": "content_ready", "url": "https://example.com"});
    let frame = encode_frame(&msg).unwrap();

    // First 4 bytes are the LE length prefix
    let len = u32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
    // Remaining bytes are the JSON payload
    assert_eq!(len, frame.len() - 4);

    // Payload should be valid JSON
    let payload_str = std::str::from_utf8(&frame[4..]).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(payload_str).unwrap();
    assert_eq!(parsed["type"], "content_ready");
}

#[test]
fn encode_small_message() {
    let msg = json!({"a": 1});
    let frame = encode_frame(&msg).unwrap();
    let expected_json = serde_json::to_vec(&msg).unwrap();
    assert_eq!(frame.len(), 4 + expected_json.len());
}

// ---------------------------------------------------------------------------
// Frame decoding tests
// ---------------------------------------------------------------------------

#[test]
fn decode_valid_frame() {
    let original = json!({"key": "value", "num": 42});
    let json_bytes = serde_json::to_vec(&original).unwrap();
    let len = json_bytes.len() as u32;

    let mut frame = Vec::new();
    frame.extend_from_slice(&len.to_le_bytes());
    frame.extend_from_slice(&json_bytes);

    let decoded = decode_frame(&frame).unwrap();
    assert_eq!(decoded["key"], "value");
    assert_eq!(decoded["num"], 42);
}

#[test]
fn decode_frame_too_short_errors() {
    // Less than 4 bytes
    let result = decode_frame(&[0x01, 0x02]);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("too short"));
}

#[test]
fn decode_frame_incomplete_payload_errors() {
    // Header says 100 bytes but only 3 bytes of payload
    let mut frame = Vec::new();
    frame.extend_from_slice(&100u32.to_le_bytes());
    frame.extend_from_slice(b"abc");

    let result = decode_frame(&frame);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("incomplete"));
}

// ---------------------------------------------------------------------------
// Roundtrip tests
// ---------------------------------------------------------------------------

#[test]
fn encode_then_decode_roundtrip() {
    let messages = vec![
        json!({"type": "click_intent", "x": 100, "y": 200}),
        json!({"type": "dom_snapshot", "nodes": [{"tag": "div"}]}),
        json!({"type": "secure_field_status", "isSecure": true}),
        json!(null),
        json!(42),
        json!("hello"),
        json!([1, 2, 3]),
    ];

    for msg in messages {
        let frame = encode_frame(&msg).unwrap();
        let decoded = decode_frame(&frame).unwrap();
        assert_eq!(msg, decoded, "roundtrip failed for: {}", msg);
    }
}

// ---------------------------------------------------------------------------
// Oversized message rejection
// ---------------------------------------------------------------------------

#[test]
fn encode_rejects_oversized_message() {
    // Create a message larger than 1 MB
    let big_string = "x".repeat(MAX_MESSAGE_SIZE as usize + 100);
    let msg = json!({"data": big_string});

    let result = encode_frame(&msg);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("exceeds maximum"));
}

#[test]
fn decode_rejects_oversized_length_prefix() {
    // Craft a frame with a length prefix exceeding the max
    let oversized_len = MAX_MESSAGE_SIZE + 1;
    let mut frame = Vec::new();
    frame.extend_from_slice(&oversized_len.to_le_bytes());
    frame.extend_from_slice(b"{}"); // minimal payload (won't be read)

    let result = decode_frame(&frame);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("exceeds maximum"));
}

// ---------------------------------------------------------------------------
// Malformed JSON handling
// ---------------------------------------------------------------------------

#[test]
fn decode_invalid_json_returns_error() {
    let bad_json = b"this is not json{{{";
    let len = bad_json.len() as u32;

    let mut frame = Vec::new();
    frame.extend_from_slice(&len.to_le_bytes());
    frame.extend_from_slice(bad_json);

    let result = decode_frame(&frame);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("not valid JSON"));
}

#[test]
fn decode_invalid_utf8_returns_error() {
    // 0xFF 0xFE is not valid UTF-8
    let bad_bytes: &[u8] = &[0xFF, 0xFE, 0x00, 0x01];
    let len = bad_bytes.len() as u32;

    let mut frame = Vec::new();
    frame.extend_from_slice(&len.to_le_bytes());
    frame.extend_from_slice(bad_bytes);

    let result = decode_frame(&frame);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("UTF-8"));
}

// ---------------------------------------------------------------------------
// Async NativeMessageServer tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn server_read_write_roundtrip() {
    let msg = json!({"type": "ping", "seq": 1});
    let frame = encode_frame(&msg).unwrap();

    // Use the frame bytes as the reader; collect written bytes
    let reader = tokio::io::BufReader::new(&frame[..]);
    let mut writer = Vec::new();

    let mut server = NativeMessageServer::new(reader, &mut writer);

    // Read the message
    let received = server.read_message().await.unwrap();
    assert_eq!(received["type"], "ping");
    assert_eq!(received["seq"], 1);

    // Write a response
    let response = json!({"type": "ack", "ackSeq": 1});
    server.write_message(&response).await.unwrap();

    // Verify the written frame
    let decoded = decode_frame(&writer).unwrap();
    assert_eq!(decoded["type"], "ack");
    assert_eq!(decoded["ackSeq"], 1);
}

#[tokio::test]
async fn server_read_eof_returns_error() {
    // Empty reader — should hit EOF immediately
    let reader = tokio::io::BufReader::new(&[][..]);
    let mut writer = Vec::new();

    let mut server = NativeMessageServer::new(reader, &mut writer);

    let result = server.read_message().await;
    assert!(result.is_err());
}

#[tokio::test]
async fn server_read_oversized_returns_error() {
    // Craft a frame with oversized length prefix
    let oversized_len = MAX_MESSAGE_SIZE + 1;
    let mut frame = Vec::new();
    frame.extend_from_slice(&oversized_len.to_le_bytes());

    let reader = tokio::io::BufReader::new(&frame[..]);
    let mut writer = Vec::new();

    let mut server = NativeMessageServer::new(reader, &mut writer);

    let result = server.read_message().await;
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("exceeds maximum"));
}

#[tokio::test]
async fn server_run_processes_messages_until_eof() {
    // Create two extension messages
    let msg1 = json!({
        "type": "content_ready",
        "seq": 1,
        "timestamp": "2026-01-01T00:00:00Z",
        "payload": {
            "url": "https://example.com",
            "title": "Example",
            "tabId": 1
        }
    });
    let msg2 = json!({
        "type": "click_intent",
        "seq": 2,
        "timestamp": "2026-01-01T00:00:01Z",
        "payload": {
            "x": 100,
            "y": 200,
            "target": {
                "tagName": "BUTTON",
                "ariaLabel": "Submit",
                "composedPath": ["BUTTON", "FORM"]
            },
            "tabId": 1,
            "url": "https://example.com"
        }
    });

    let mut input_bytes = Vec::new();
    input_bytes.extend_from_slice(&encode_frame(&msg1).unwrap());
    input_bytes.extend_from_slice(&encode_frame(&msg2).unwrap());

    let reader = tokio::io::BufReader::new(&input_bytes[..]);
    let mut writer = Vec::new();

    let mut server = NativeMessageServer::new(reader, &mut writer);

    let (tx, mut rx) = mpsc::channel(10);
    let (_dummy_cmd_tx, dummy_cmd_rx) = mpsc::channel::<DaemonCommand>(1);

    // Run should process both messages then exit on EOF
    server.run(tx, dummy_cmd_rx).await.unwrap();

    // Collect events
    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event);
    }

    assert_eq!(events.len(), 2, "expected 2 events, got {}", events.len());

    // First event should be FocusChange (from content_ready)
    assert_eq!(
        events[0].kind,
        agenthandover_common::event::EventKind::FocusChange
    );

    // Second event should be ClickIntent
    match &events[1].kind {
        agenthandover_common::event::EventKind::ClickIntent {
            target_description, ..
        } => {
            assert!(
                target_description.contains("Submit"),
                "expected target_description to contain 'Submit', got: {}",
                target_description
            );
        }
        other => panic!("expected ClickIntent, got {:?}", other),
    }
}

#[tokio::test]
async fn server_run_skips_malformed_json() {
    // First: a valid message
    let valid_msg = json!({
        "type": "content_ready",
        "seq": 1,
        "timestamp": "2026-01-01T00:00:00Z",
        "payload": {
            "url": "https://example.com",
            "title": "Test",
            "tabId": 1
        }
    });

    // Second: malformed JSON (but valid framing)
    let bad_json = b"{not valid json!!!}";
    let bad_len = bad_json.len() as u32;

    // Third: another valid message
    let valid_msg2 = json!({
        "type": "scroll_snapshot",
        "seq": 3,
        "timestamp": "2026-01-01T00:00:02Z",
        "payload": {
            "nodes": [],
            "url": "https://example.com",
            "tabId": 1,
            "captureReason": "scroll"
        }
    });

    let mut input_bytes = Vec::new();
    input_bytes.extend_from_slice(&encode_frame(&valid_msg).unwrap());
    // Manually add bad frame
    input_bytes.extend_from_slice(&bad_len.to_le_bytes());
    input_bytes.extend_from_slice(bad_json);
    input_bytes.extend_from_slice(&encode_frame(&valid_msg2).unwrap());

    let reader = tokio::io::BufReader::new(&input_bytes[..]);
    let mut writer = Vec::new();

    let mut server = NativeMessageServer::new(reader, &mut writer);

    let (tx, mut rx) = mpsc::channel(10);
    let (_dummy_cmd_tx, dummy_cmd_rx) = mpsc::channel::<DaemonCommand>(1);

    server.run(tx, dummy_cmd_rx).await.unwrap();

    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event);
    }

    // Should have processed 2 events (the malformed one was skipped)
    assert_eq!(
        events.len(),
        2,
        "expected 2 events (malformed skipped), got {}",
        events.len()
    );
}
