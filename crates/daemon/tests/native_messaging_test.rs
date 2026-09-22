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
    decode_frame, encode_frame, strip_url_to_origin_and_path, DaemonCommand, DaemonMessageType,
    FrameError, NativeMessageServer, MAX_INBOUND_FRAME_SIZE, MAX_MESSAGE_SIZE,
};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
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
    let mut writer: Vec<u8> = Vec::new();

    let mut server = NativeMessageServer::new(reader, &mut writer);

    let err = server.read_message().await.unwrap_err();
    assert!(matches!(err, FrameError::Closed(_)), "got {err:?}");
    assert!(err.is_fatal());
}

/// An oversized frame is consumed whole, so the stream stays aligned and
/// the next frame decodes (coverage audit #116 (a)).
#[tokio::test]
async fn server_read_oversized_skips_payload_then_reads_next_frame() {
    let oversized_len = MAX_MESSAGE_SIZE + 1;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&oversized_len.to_le_bytes());
    bytes.extend(std::iter::repeat(b'x').take(oversized_len as usize));
    bytes.extend_from_slice(&encode_frame(&json!({"type": "ping", "seq": 7})).unwrap());

    let mut writer: Vec<u8> = Vec::new();
    let mut server = NativeMessageServer::new(std::io::Cursor::new(bytes), &mut writer);

    let err = server.read_message().await.unwrap_err();
    assert!(matches!(err, FrameError::Oversized { len } if len == oversized_len), "got {err:?}");
    assert!(!err.is_fatal());
    assert!(format!("{err}").contains("exceeds maximum"));

    let next = server.read_message().await.unwrap();
    assert_eq!(next["seq"], 7);
}

/// An oversized header whose payload never arrives is a closed stream, not
/// a skippable frame.
#[tokio::test]
async fn server_read_oversized_header_without_payload_is_closed() {
    let frame = (MAX_MESSAGE_SIZE + 1).to_le_bytes().to_vec();
    let mut writer: Vec<u8> = Vec::new();
    let mut server = NativeMessageServer::new(std::io::Cursor::new(frame), &mut writer);

    let err = server.read_message().await.unwrap_err();
    assert!(matches!(err, FrameError::Closed(_)), "got {err:?}");
}

/// A header beyond Chrome's own 64 MiB cap means the stream is out of sync;
/// the server must not try to read or skip gigabytes.
#[tokio::test]
async fn server_read_header_beyond_chrome_cap_is_desynced() {
    let frame = (MAX_INBOUND_FRAME_SIZE + 1).to_le_bytes().to_vec();
    let mut writer: Vec<u8> = Vec::new();
    let mut server = NativeMessageServer::new(std::io::Cursor::new(frame), &mut writer);

    let err = server.read_message().await.unwrap_err();
    assert!(matches!(err, FrameError::Desynced(_)), "got {err:?}");
    assert!(err.is_fatal());
}

/// Through `run()`: an oversized frame followed by a valid one yields the
/// valid event, and the server keeps going (coverage audit #116 (a)).
#[tokio::test]
async fn server_run_recovers_after_oversized_frame() {
    let oversized_len = MAX_MESSAGE_SIZE + 10;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&oversized_len.to_le_bytes());
    bytes.extend(std::iter::repeat(b'{').take(oversized_len as usize));
    bytes.extend_from_slice(
        &encode_frame(&json!({
            "type": "content_ready",
            "seq": 1,
            "timestamp": "2026-01-01T00:00:00Z",
            "payload": {"url": "https://example.com/after", "title": "After", "tabId": 3}
        }))
        .unwrap(),
    );

    let mut writer: Vec<u8> = Vec::new();
    let server = NativeMessageServer::new(std::io::Cursor::new(bytes), &mut writer);
    let (tx, mut rx) = mpsc::channel(10);
    let (_cmd_tx, cmd_rx) = mpsc::channel::<DaemonCommand>(1);
    server.run(tx, cmd_rx).await.unwrap();

    let event = rx.try_recv().expect("valid frame after the oversized one must decode");
    assert_eq!(event.metadata["url"], "https://example.com/after");
    assert!(rx.try_recv().is_err());
}

/// A command that wins the `select!` while an inbound frame is only half
/// written must not lose the bytes already read (coverage audit #116 (b)).
#[tokio::test]
async fn server_run_command_mid_frame_loses_no_bytes() {
    let frame = encode_frame(&json!({
        "type": "content_ready",
        "seq": 1,
        "timestamp": "2026-01-01T00:00:00Z",
        "payload": {"url": "https://example.com/split", "title": "Split", "tabId": 5}
    }))
    .unwrap();
    let second = encode_frame(&json!({
        "type": "content_ready",
        "seq": 2,
        "timestamp": "2026-01-01T00:00:01Z",
        "payload": {"url": "https://example.com/second", "title": "Second", "tabId": 5}
    }))
    .unwrap();
    let (head, tail) = frame.split_at(frame.len() / 2);

    let (mut chrome_wr, host_rd) = tokio::io::duplex(1 << 16);
    let (host_wr, mut chrome_rd) = tokio::io::duplex(1 << 16);
    let (event_tx, mut event_rx) = mpsc::channel(10);
    let (cmd_tx, cmd_rx) = mpsc::channel::<DaemonCommand>(4);
    let server = NativeMessageServer::new(host_rd, host_wr);
    let run = tokio::spawn(server.run(event_tx, cmd_rx));

    // Half a frame (header plus part of the payload) is now in flight.
    chrome_wr.write_all(head).await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // An outbound command is processed while the read is mid-frame.
    cmd_tx
        .send(DaemonCommand::SendMessage {
            msg_type: DaemonMessageType::Ping,
            payload: json!({}),
        })
        .await
        .unwrap();
    let mut len_buf = [0u8; 4];
    chrome_rd.read_exact(&mut len_buf).await.unwrap();
    let mut out = vec![0u8; u32::from_le_bytes(len_buf) as usize];
    chrome_rd.read_exact(&mut out).await.unwrap();
    let ping: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(ping["type"], "ping");

    // The rest of the frame, then another whole frame, then EOF.
    chrome_wr.write_all(tail).await.unwrap();
    chrome_wr.write_all(&second).await.unwrap();
    drop(chrome_wr);
    drop(cmd_tx);
    run.await.unwrap().unwrap();

    let first = event_rx.try_recv().expect("split frame must decode");
    assert_eq!(first.metadata["url"], "https://example.com/split");
    let next = event_rx.try_recv().expect("following frame must decode");
    assert_eq!(next.metadata["url"], "https://example.com/second");
}

/// Stored Chrome URLs keep origin + pathname only.
#[tokio::test]
async fn server_run_strips_query_and_fragment_from_url() {
    let bytes = encode_frame(&json!({
        "type": "content_ready",
        "seq": 1,
        "timestamp": "2026-01-01T00:00:00Z",
        "payload": {
            "url": "https://user:pw@mail.example.com/inbox/42?token=abc&q=salary#msg-9",
            "tabId": 1
        }
    }))
    .unwrap();
    let mut writer: Vec<u8> = Vec::new();
    let server = NativeMessageServer::new(std::io::Cursor::new(bytes), &mut writer);
    let (tx, mut rx) = mpsc::channel(10);
    let (_cmd_tx, cmd_rx) = mpsc::channel::<DaemonCommand>(1);
    server.run(tx, cmd_rx).await.unwrap();

    let event = rx.try_recv().unwrap();
    assert_eq!(event.metadata["url"], "https://mail.example.com/inbox/42");
    // No title in the payload: the window title falls back to the stripped URL.
    assert_eq!(event.window.unwrap().title, "https://mail.example.com/inbox/42");
}

#[test]
fn strip_url_to_origin_and_path_cases() {
    assert_eq!(strip_url_to_origin_and_path("https://a.com/p?q=1#f"), "https://a.com/p");
    assert_eq!(strip_url_to_origin_and_path("https://a.com#f"), "https://a.com");
    assert_eq!(strip_url_to_origin_and_path("https://u:p@a.com:8443/x"), "https://a.com:8443/x");
    // An @ in the path is not userinfo.
    assert_eq!(strip_url_to_origin_and_path("https://a.com/@me?x"), "https://a.com/@me");
    assert_eq!(strip_url_to_origin_and_path("chrome://newtab/"), "chrome://newtab/");
    assert_eq!(strip_url_to_origin_and_path("unknown"), "unknown");
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

    let reader = std::io::Cursor::new(input_bytes);
    let mut writer = Vec::new();

    let server = NativeMessageServer::new(reader, &mut writer);

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

    let reader = std::io::Cursor::new(input_bytes);
    let mut writer = Vec::new();

    let server = NativeMessageServer::new(reader, &mut writer);

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
