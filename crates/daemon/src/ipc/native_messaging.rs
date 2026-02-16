//! Chrome Native Messaging stdio server.
//!
//! Implements the Chrome Native Messaging framing protocol:
//! - Each message is prefixed with a 4-byte little-endian length header.
//! - The payload is JSON-encoded UTF-8.
//! - Maximum message size: 1 MB (1_048_576 bytes).
//!
//! The server reads messages from stdin, parses them into browser events,
//! and forwards them through a `tokio::sync::mpsc` channel. Outbound
//! messages are written to stdout using the same framing.

use anyhow::{bail, Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use oc_apprentice_common::event::*;

/// Maximum allowed message size per the Chrome Native Messaging spec (1 MB).
pub const MAX_MESSAGE_SIZE: u32 = 1_048_576;

/// Chrome Native Messaging host name for the OpenClaw Apprentice bridge.
pub const NATIVE_HOST_NAME: &str = "com.openclaw.apprentice";

// ---------------------------------------------------------------------------
// Inbound message types (extension -> daemon)
// ---------------------------------------------------------------------------

/// Discriminator values for messages the extension sends to the daemon.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtensionMessageType {
    ContentReady,
    DomSnapshot,
    ClickIntent,
    DwellSnapshot,
    ScrollSnapshot,
    SecureFieldStatus,
}

/// Top-level envelope for an inbound native message from the browser extension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionMessage {
    #[serde(rename = "type")]
    pub msg_type: ExtensionMessageType,
    pub seq: u64,
    pub timestamp: String,
    pub payload: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Outbound message types (daemon -> extension)
// ---------------------------------------------------------------------------

/// Discriminator values for messages the daemon sends to the extension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonMessageType {
    Ping,
    RequestSnapshot,
    ConfigUpdate,
    Ack,
}

/// Top-level envelope for an outbound native message to the browser extension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonMessage {
    #[serde(rename = "type")]
    pub msg_type: DaemonMessageType,
    pub seq: u64,
    pub timestamp: String,
    pub payload: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Frame encoding / decoding (public for testing)
// ---------------------------------------------------------------------------

/// Encode a JSON value into Chrome Native Messaging wire format.
///
/// Returns a `Vec<u8>` containing a 4-byte little-endian length prefix
/// followed by the UTF-8 JSON payload.
pub fn encode_frame(msg: &serde_json::Value) -> Result<Vec<u8>> {
    let json_bytes = serde_json::to_vec(msg).context("failed to serialize message to JSON")?;
    let len = json_bytes.len() as u32;
    if len > MAX_MESSAGE_SIZE {
        bail!(
            "message size {} bytes exceeds maximum {} bytes",
            len,
            MAX_MESSAGE_SIZE,
        );
    }
    let mut buf = Vec::with_capacity(4 + json_bytes.len());
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&json_bytes);
    Ok(buf)
}

/// Decode a Chrome Native Messaging frame from a byte slice.
///
/// Expects the first 4 bytes to be a little-endian length prefix followed
/// by exactly that many bytes of UTF-8 JSON.
pub fn decode_frame(data: &[u8]) -> Result<serde_json::Value> {
    if data.len() < 4 {
        bail!("frame too short: need at least 4 bytes, got {}", data.len());
    }
    let len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    if len > MAX_MESSAGE_SIZE as usize {
        bail!(
            "message size {} bytes exceeds maximum {} bytes",
            len,
            MAX_MESSAGE_SIZE,
        );
    }
    let payload_start = 4;
    let payload_end = payload_start + len;
    if data.len() < payload_end {
        bail!(
            "incomplete frame: header says {} bytes but only {} available",
            len,
            data.len() - 4,
        );
    }
    let json_str = std::str::from_utf8(&data[payload_start..payload_end])
        .context("message payload is not valid UTF-8")?;
    serde_json::from_str(json_str).context("message payload is not valid JSON")
}

// ---------------------------------------------------------------------------
// NativeMessageServer
// ---------------------------------------------------------------------------

/// A Chrome Native Messaging server that reads from stdin and writes to stdout.
///
/// The server uses the 4-byte little-endian length-prefixed framing protocol
/// specified by Chrome. Messages are JSON-encoded UTF-8.
pub struct NativeMessageServer<R, W> {
    reader: R,
    writer: W,
    outbound_seq: u64,
}

impl<R, W> NativeMessageServer<R, W>
where
    R: AsyncReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
{
    /// Create a new server with the given reader (stdin) and writer (stdout).
    pub fn new(reader: R, writer: W) -> Self {
        Self {
            reader,
            writer,
            outbound_seq: 0,
        }
    }

    /// Read a single framed message from the reader.
    ///
    /// Returns `Ok(value)` on success. Returns an error on EOF, oversized
    /// messages, or malformed JSON.
    pub async fn read_message(&mut self) -> Result<serde_json::Value> {
        // Read 4-byte length prefix
        let mut len_buf = [0u8; 4];
        self.reader
            .read_exact(&mut len_buf)
            .await
            .context("failed to read message length prefix (EOF?)")?;

        let len = u32::from_le_bytes(len_buf);

        if len > MAX_MESSAGE_SIZE {
            bail!(
                "message size {} bytes exceeds maximum {} bytes",
                len,
                MAX_MESSAGE_SIZE,
            );
        }

        // Read JSON payload
        let mut payload_buf = vec![0u8; len as usize];
        self.reader
            .read_exact(&mut payload_buf)
            .await
            .context("failed to read message payload")?;

        let json_str = std::str::from_utf8(&payload_buf)
            .context("message payload is not valid UTF-8")?;

        serde_json::from_str(json_str).context("message payload is not valid JSON")
    }

    /// Write a single framed message to the writer.
    pub async fn write_message(&mut self, msg: &serde_json::Value) -> Result<()> {
        let frame = encode_frame(msg)?;
        self.writer
            .write_all(&frame)
            .await
            .context("failed to write message frame")?;
        self.writer
            .flush()
            .await
            .context("failed to flush message")?;
        Ok(())
    }

    /// Send a typed daemon message to the extension.
    pub async fn send_daemon_message(
        &mut self,
        msg_type: DaemonMessageType,
        payload: serde_json::Value,
    ) -> Result<()> {
        self.outbound_seq += 1;
        let msg = DaemonMessage {
            msg_type,
            seq: self.outbound_seq,
            timestamp: Utc::now().to_rfc3339(),
            payload,
        };
        let value = serde_json::to_value(&msg).context("failed to serialize daemon message")?;
        self.write_message(&value).await
    }

    /// Main event loop: read messages, parse them, convert to `Event`s, and
    /// send through the provided channel.
    ///
    /// Runs until the reader reaches EOF or an unrecoverable error occurs.
    /// Malformed JSON messages are logged and skipped.
    pub async fn run(&mut self, event_tx: mpsc::Sender<Event>) -> Result<()> {
        info!("Native messaging server starting");

        loop {
            let raw = match self.read_message().await {
                Ok(v) => v,
                Err(e) => {
                    // Check if this is an EOF (normal shutdown)
                    let msg = format!("{:#}", e);
                    if msg.contains("EOF")
                        || msg.contains("UnexpectedEof")
                        || msg.contains("unexpected eof")
                    {
                        info!("Native messaging: stdin closed (extension disconnected)");
                        break;
                    }
                    // For malformed JSON, log and continue
                    if msg.contains("not valid JSON") || msg.contains("not valid UTF-8") {
                        warn!("Native messaging: malformed message, skipping: {}", msg);
                        continue;
                    }
                    // For other errors (e.g., oversized), log and continue
                    warn!("Native messaging: read error, skipping: {}", msg);
                    continue;
                }
            };

            debug!("Native messaging: received raw message");

            // Try to parse as a typed ExtensionMessage
            let ext_msg: ExtensionMessage = match serde_json::from_value(raw.clone()) {
                Ok(m) => m,
                Err(e) => {
                    warn!(
                        "Native messaging: failed to parse extension message: {}",
                        e
                    );
                    continue;
                }
            };

            // Convert to an Event based on message type
            match convert_to_event(&ext_msg) {
                Some(event) => {
                    debug!(
                        kind = ?event.kind,
                        "Native messaging: converted to event"
                    );
                    if event_tx.send(event).await.is_err() {
                        info!("Native messaging: event channel closed, shutting down");
                        break;
                    }
                    // Send acknowledgement
                    if let Err(e) = self
                        .send_daemon_message(
                            DaemonMessageType::Ack,
                            serde_json::json!({ "ackSeq": ext_msg.seq }),
                        )
                        .await
                    {
                        error!("Native messaging: failed to send ack: {}", e);
                    }
                }
                None => {
                    debug!(
                        msg_type = ?ext_msg.msg_type,
                        "Native messaging: message type does not map to an event"
                    );
                }
            }
        }

        info!("Native messaging server stopped");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Event conversion
// ---------------------------------------------------------------------------

/// Convert a parsed `ExtensionMessage` into an `Event`.
///
/// Returns `None` for message types that do not directly map to an event
/// (currently all defined types map, but this keeps the API extensible).
fn convert_to_event(msg: &ExtensionMessage) -> Option<Event> {
    let kind = match &msg.msg_type {
        ExtensionMessageType::ContentReady => EventKind::FocusChange,
        ExtensionMessageType::DomSnapshot => EventKind::DwellSnapshot,
        ExtensionMessageType::ClickIntent => {
            let target_desc = msg
                .payload
                .get("target")
                .and_then(|t| {
                    let tag = t.get("tagName").and_then(|v| v.as_str()).unwrap_or("unknown");
                    let label = t
                        .get("ariaLabel")
                        .and_then(|v| v.as_str())
                        .or_else(|| t.get("innerText").and_then(|v| v.as_str()))
                        .unwrap_or("");
                    Some(format!("{}<{}>", label, tag))
                })
                .unwrap_or_else(|| "unknown".to_string());
            EventKind::ClickIntent {
                target_description: target_desc,
            }
        }
        ExtensionMessageType::DwellSnapshot => EventKind::DwellSnapshot,
        ExtensionMessageType::ScrollSnapshot => EventKind::ScrollReadSnapshot,
        ExtensionMessageType::SecureFieldStatus => EventKind::SecureFieldFocus,
    };

    // Extract window info from the payload if available
    let url = msg
        .payload
        .get("url")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let tab_id = msg
        .payload
        .get("tabId")
        .and_then(|v| v.as_i64())
        .unwrap_or(-1);

    let window = Some(WindowInfo {
        window_id: format!("chrome-tab-{}", tab_id),
        app_id: "com.google.Chrome".to_string(),
        title: msg
            .payload
            .get("title")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or(url.clone()),
        bounds_global_px: [0, 0, 0, 0],
        z_order: 0,
        is_fullscreen: false,
    });

    Some(Event {
        id: Uuid::new_v4(),
        timestamp: Utc::now(),
        kind,
        window,
        display_topology: vec![],
        primary_display_id: String::new(),
        cursor_global_px: None,
        ui_scale: None,
        artifact_ids: vec![],
        metadata: msg.payload.clone(),
    })
}

// ---------------------------------------------------------------------------
// Convenience constructor for stdio
// ---------------------------------------------------------------------------

/// Create a `NativeMessageServer` bound to tokio stdin/stdout.
pub fn stdio_server() -> NativeMessageServer<tokio::io::Stdin, tokio::io::Stdout> {
    NativeMessageServer::new(tokio::io::stdin(), tokio::io::stdout())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let msg = serde_json::json!({"type": "ping", "data": 42});
        let frame = encode_frame(&msg).unwrap();
        let decoded = decode_frame(&frame).unwrap();
        assert_eq!(msg, decoded);
    }

    #[test]
    fn test_encode_frame_length_prefix() {
        let msg = serde_json::json!({"hello": "world"});
        let frame = encode_frame(&msg).unwrap();
        let expected_json = serde_json::to_vec(&msg).unwrap();
        let expected_len = expected_json.len() as u32;
        let actual_len = u32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]);
        assert_eq!(expected_len, actual_len);
        assert_eq!(&frame[4..], &expected_json[..]);
    }

    #[test]
    fn test_decode_frame_too_short() {
        let result = decode_frame(&[0, 1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_frame_incomplete() {
        // Header says 100 bytes but only 5 bytes of payload
        let mut data = vec![100, 0, 0, 0]; // length = 100
        data.extend_from_slice(b"hello");
        let result = decode_frame(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_extension_message_type_serde() {
        let msg = ExtensionMessage {
            msg_type: ExtensionMessageType::ContentReady,
            seq: 1,
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            payload: serde_json::json!({"url": "https://example.com"}),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"content_ready\""));

        let parsed: ExtensionMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.msg_type, ExtensionMessageType::ContentReady);
        assert_eq!(parsed.seq, 1);
    }

    #[test]
    fn test_daemon_message_type_serde() {
        let msg = DaemonMessage {
            msg_type: DaemonMessageType::Ack,
            seq: 5,
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            payload: serde_json::json!({"ackSeq": 4}),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"ack\""));

        let parsed: DaemonMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.msg_type, DaemonMessageType::Ack);
    }

    #[test]
    fn test_convert_click_intent() {
        let msg = ExtensionMessage {
            msg_type: ExtensionMessageType::ClickIntent,
            seq: 1,
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            payload: serde_json::json!({
                "x": 100,
                "y": 200,
                "target": {
                    "tagName": "BUTTON",
                    "ariaLabel": "Submit",
                    "composedPath": ["BUTTON", "FORM", "BODY"]
                },
                "tabId": 42,
                "url": "https://example.com"
            }),
        };
        let event = convert_to_event(&msg).unwrap();
        match &event.kind {
            EventKind::ClickIntent {
                target_description, ..
            } => {
                assert!(target_description.contains("Submit"));
                assert!(target_description.contains("BUTTON"));
            }
            other => panic!("expected ClickIntent, got {:?}", other),
        }
    }

    #[test]
    fn test_convert_secure_field() {
        let msg = ExtensionMessage {
            msg_type: ExtensionMessageType::SecureFieldStatus,
            seq: 1,
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            payload: serde_json::json!({
                "isSecure": true,
                "tabId": 1,
                "url": "https://login.example.com"
            }),
        };
        let event = convert_to_event(&msg).unwrap();
        assert_eq!(event.kind, EventKind::SecureFieldFocus);
    }
}
