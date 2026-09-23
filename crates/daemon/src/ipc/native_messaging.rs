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
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use agenthandover_common::browser_transition::BrowserTransitionKind;
use agenthandover_common::event::*;

// ---------------------------------------------------------------------------
// Outbound command channel (internal tasks -> NM server -> extension)
// ---------------------------------------------------------------------------

/// Commands that can be sent TO the NM server for outbound transmission.
#[derive(Debug, Clone)]
pub enum DaemonCommand {
    /// Send a typed message to the extension.
    SendMessage {
        msg_type: DaemonMessageType,
        payload: serde_json::Value,
    },
}

/// Maximum allowed message size per the Chrome Native Messaging spec (1 MB).
pub const MAX_MESSAGE_SIZE: u32 = 1_048_576;

/// Chrome Native Messaging host name for the AgentHandover bridge.
pub const NATIVE_HOST_NAME: &str = "com.agenthandover.host";

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
    /// A `paste` DOM event (Cmd-V, Edit > Paste, context menu).
    Paste,
    /// The active tab committed a navigation (`tabs.onUpdated`).
    NavigationChange,
    /// A form was submitted in a page (content-script `submit` listener).
    FormSubmit,
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
// Inbound frame reading
// ---------------------------------------------------------------------------

/// Chrome caps a single extension-to-host message at 64 MiB.  A length
/// header above this cannot have come from Chrome, so the stream is out of
/// sync and the only safe recovery is to close the pipe (the extension
/// reconnects and a fresh host process starts with a clean stream).
pub const MAX_INBOUND_FRAME_SIZE: u32 = 64 * 1024 * 1024;

/// Why an inbound frame could not be turned into a JSON value.
#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    /// The pipe closed or failed (EOF, possibly mid-frame).  Nothing more
    /// can be read; the server stops.
    #[error("native messaging stream closed: {0}")]
    Closed(#[source] std::io::Error),
    /// The length header exceeded [`MAX_INBOUND_FRAME_SIZE`]; the stream is
    /// out of sync and the server stops.
    #[error("length header {0} exceeds Chrome's {max} byte limit; stream out of sync", max = MAX_INBOUND_FRAME_SIZE)]
    Desynced(u32),
    /// The frame exceeded [`MAX_MESSAGE_SIZE`].  Its payload was read and
    /// discarded, so the stream stays aligned on the next frame.
    #[error("message size {len} bytes exceeds maximum {max} bytes (payload discarded)", max = MAX_MESSAGE_SIZE)]
    Oversized { len: u32 },
    /// The frame was read completely but is not UTF-8 JSON.  The stream
    /// stays aligned.
    #[error("malformed message: {0}")]
    Malformed(String),
}

impl FrameError {
    /// True when no further frame can be read from the stream.
    pub fn is_fatal(&self) -> bool {
        matches!(self, FrameError::Closed(_) | FrameError::Desynced(_))
    }
}

/// Read one length-prefixed frame.  Every return leaves the stream either
/// aligned on the next frame's header or unusable (`is_fatal()`), never
/// part-way through a payload.
async fn read_frame<R: AsyncRead + Unpin>(
    reader: &mut R,
) -> std::result::Result<serde_json::Value, FrameError> {
    let mut len_buf = [0u8; 4];
    reader
        .read_exact(&mut len_buf)
        .await
        .map_err(FrameError::Closed)?;
    let len = u32::from_le_bytes(len_buf);

    if len > MAX_INBOUND_FRAME_SIZE {
        return Err(FrameError::Desynced(len));
    }
    if len > MAX_MESSAGE_SIZE {
        // Consume exactly `len` bytes so the next read starts on a header.
        let discarded = tokio::io::copy(&mut (&mut *reader).take(u64::from(len)), &mut tokio::io::sink())
            .await
            .map_err(FrameError::Closed)?;
        if discarded < u64::from(len) {
            return Err(FrameError::Closed(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "EOF while discarding an oversized payload",
            )));
        }
        return Err(FrameError::Oversized { len });
    }

    let mut payload = vec![0u8; len as usize];
    reader
        .read_exact(&mut payload)
        .await
        .map_err(FrameError::Closed)?;
    let json_str = std::str::from_utf8(&payload)
        .map_err(|e| FrameError::Malformed(format!("message payload is not valid UTF-8: {e}")))?;
    serde_json::from_str(json_str)
        .map_err(|e| FrameError::Malformed(format!("message payload is not valid JSON: {e}")))
}

/// Reader task body: owns the reader and forwards every frame outcome until
/// the stream becomes unusable or `run()` stops listening.
async fn forward_frames<R: AsyncRead + Unpin>(
    mut reader: R,
    tx: mpsc::Sender<std::result::Result<serde_json::Value, FrameError>>,
) {
    loop {
        let frame = read_frame(&mut reader).await;
        let fatal = frame.as_ref().err().map_or(false, FrameError::is_fatal);
        if tx.send(frame).await.is_err() || fatal {
            break;
        }
    }
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
    // u64 overflow is theoretical only (~centuries at realistic rates). No action needed.
    outbound_seq: u64,
    /// Highest inbound `seq` successfully processed.  Messages with
    /// `seq <= last_processed_seq` are duplicates (e.g. reconnect replays)
    /// and are ack'd but not forwarded as events.
    last_processed_seq: u64,
}

impl<R, W> NativeMessageServer<R, W> {
    /// Create a new server with the given reader (stdin) and writer (stdout).
    pub fn new(reader: R, writer: W) -> Self {
        Self {
            reader,
            writer,
            outbound_seq: 0,
            last_processed_seq: 0,
        }
    }
}

impl<R: AsyncRead + Unpin, W> NativeMessageServer<R, W> {
    /// Read a single framed message from the reader.
    ///
    /// An oversized frame is consumed and reported as
    /// [`FrameError::Oversized`], so the next call reads the next frame.
    pub async fn read_message(&mut self) -> std::result::Result<serde_json::Value, FrameError> {
        read_frame(&mut self.reader).await
    }
}

impl<R, W: AsyncWrite + Unpin> NativeMessageServer<R, W> {
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
}

impl<R, W> NativeMessageServer<R, W>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin,
{
    /// Main event loop: read messages, parse them, convert to `Event`s, and
    /// send through the provided channel.  Also processes outbound commands
    /// received via `cmd_rx` (e.g. `request_snapshot` for DOM capture).
    ///
    /// Runs until the reader reaches EOF or an unrecoverable error occurs.
    /// Malformed and oversized messages are logged and skipped.
    ///
    /// # Cancel-safety
    ///
    /// A spawned task owns the reader and pushes whole decoded frames into
    /// a channel, so the `select!` below only races two `mpsc::recv()`
    /// futures, both cancel-safe.  A command arriving while a frame is only
    /// half-written by Chrome never drops the bytes already read.
    pub async fn run(
        self,
        event_tx: mpsc::Sender<Event>,
        mut cmd_rx: mpsc::Receiver<DaemonCommand>,
    ) -> Result<()> {
        info!("Native messaging server starting");

        let NativeMessageServer { reader, writer, outbound_seq, last_processed_seq } = self;
        // The writer half stays here; the reader moves into its own task.
        let mut out = NativeMessageServer { reader: (), writer, outbound_seq, last_processed_seq };
        let (inbound_tx, mut inbound_rx) = mpsc::channel(64);
        let reader_task = tokio::spawn(forward_frames(reader, inbound_tx));

        // Track whether the command channel is still open.  Once closed
        // (all senders dropped — normal when the DOM request task ends),
        // we stop selecting on it and only read inbound messages.
        let mut cmd_channel_open = true;

        loop {
            tokio::select! {
                biased;

                // Branch 1: Process outbound commands from daemon tasks
                cmd = cmd_rx.recv(), if cmd_channel_open => {
                    match cmd {
                        Some(DaemonCommand::SendMessage { msg_type, payload }) => {
                            debug!(?msg_type, "Native messaging: sending outbound command");
                            if let Err(e) = out.send_daemon_message(msg_type, payload).await {
                                warn!("Failed to send outbound command: {}", e);
                            }
                        }
                        None => {
                            // Command channel closed — all senders dropped.
                            // This is normal (e.g. DOM request task ended).
                            // Continue reading inbound messages from Chrome.
                            info!("Native messaging: command channel closed, continuing reads");
                            cmd_channel_open = false;
                        }
                    }
                }

                // Branch 2: Inbound frame decoded by the reader task
                frame = inbound_rx.recv() => {
                    let raw = match frame {
                        Some(Ok(v)) => v,
                        Some(Err(FrameError::Closed(e))) => {
                            info!("Native messaging: stdin closed (extension disconnected): {}", e);
                            break;
                        }
                        Some(Err(e @ FrameError::Desynced(_))) => {
                            error!("Native messaging: {}; closing so the extension reconnects", e);
                            break;
                        }
                        Some(Err(e)) => {
                            warn!("Native messaging: skipping frame: {}", e);
                            continue;
                        }
                        None => {
                            info!("Native messaging: reader task ended");
                            break;
                        }
                    };

                    debug!("Native messaging: received raw message");

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

                    // Dedup: skip messages already processed (e.g. reconnect replays)
                    if ext_msg.seq <= out.last_processed_seq {
                        debug!(
                            seq = ext_msg.seq,
                            last = out.last_processed_seq,
                            "Native messaging: duplicate seq, ack-ing but not forwarding"
                        );
                        if let Err(e) = out
                            .send_daemon_message(
                                DaemonMessageType::Ack,
                                serde_json::json!({ "ackSeq": ext_msg.seq }),
                            )
                            .await
                        {
                            error!("Native messaging: failed to send dup ack: {}", e);
                        }
                        continue;
                    }

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
                            out.last_processed_seq = ext_msg.seq;
                            if let Err(e) = out
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
                            let payload_summary: String = ext_msg.payload.to_string().chars().take(200).collect();
                            warn!(
                                msg_type = ?ext_msg.msg_type,
                                payload_summary,
                                "Native messaging: message type does not map to an event"
                            );
                        }
                    }
                }
            }
        }

        reader_task.abort();
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
    // Only origin + pathname of the page URL is stored: query strings and
    // fragments carry search terms, session ids, reset tokens and email
    // addresses.  The sanitized URL replaces the payload's `url` so neither
    // the window title fallback nor the metadata keeps the original.
    let mut metadata = msg.payload.clone();
    let url = match metadata.get("url").and_then(|v| v.as_str()) {
        Some(raw) => {
            let stripped = strip_url_to_origin_and_path(raw);
            metadata["url"] = serde_json::Value::String(stripped.clone());
            stripped
        }
        None => "unknown".to_string(),
    };
    let title = msg
        .payload
        .get("title")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

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
        // The pasted content's hash is added by `capture::paste::resolve_paste`.
        ExtensionMessageType::Paste => EventKind::PasteDetected { matched_copy_hash: None },
        ExtensionMessageType::NavigationChange => EventKind::NavigationChange {
            url: url.clone(),
            title: title.clone().unwrap_or_default(),
        },
        ExtensionMessageType::FormSubmit => {
            let mut form_anchor = msg
                .payload
                .get("formAnchor")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            // The form's action URL gets the same origin + path treatment.
            if let Some(action) = form_anchor.get("action").and_then(|v| v.as_str()) {
                form_anchor["action"] =
                    serde_json::Value::String(strip_url_to_origin_and_path(action));
            }
            if let Some(anchor) = metadata.get_mut("formAnchor") {
                *anchor = form_anchor.clone();
            }
            EventKind::FormSubmit {
                url: url.clone(),
                form_anchor,
            }
        }
    };

    let tab_id = msg
        .payload
        .get("tabId")
        .and_then(|v| v.as_i64())
        .unwrap_or(-1);

    let window = Some(WindowInfo {
        window_id: format!("chrome-tab-{}", tab_id),
        app_id: "com.google.Chrome".to_string(),
        title: title.unwrap_or(url),
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
        metadata,
        display_ids_spanned: None,
        focused_element: None,
    })
}

/// The screen transition a browser event stands for, if any: the bridge
/// hands these to the observer (`browser_transition`) so it captures a
/// frame of the page the user just acted on.
pub fn browser_transition_kind(kind: &EventKind) -> Option<BrowserTransitionKind> {
    match kind {
        EventKind::NavigationChange { .. } => Some(BrowserTransitionKind::Navigation),
        EventKind::FormSubmit { .. } => Some(BrowserTransitionKind::FormSubmit),
        EventKind::ClickIntent { .. } => Some(BrowserTransitionKind::Click),
        _ => None,
    }
}

/// Reduce a URL to origin + pathname: drop the query string, the fragment
/// and any `user:password@` userinfo.  Strings that are not URLs lose only
/// what follows a `?` or `#`.
pub fn strip_url_to_origin_and_path(url: &str) -> String {
    let end = url.find(['?', '#']).unwrap_or(url.len());
    let base = &url[..end];
    if let Some(scheme_end) = base.find("://") {
        let after_scheme = &base[scheme_end + 3..];
        let authority_end = after_scheme.find('/').unwrap_or(after_scheme.len());
        if let Some(at) = after_scheme[..authority_end].rfind('@') {
            return format!("{}{}", &base[..scheme_end + 3], &after_scheme[at + 1..]);
        }
    }
    base.to_string()
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
    fn test_convert_paste() {
        let msg: ExtensionMessage = serde_json::from_value(serde_json::json!({
            "type": "paste",
            "seq": 7,
            "timestamp": "2026-01-01T00:00:00Z",
            "payload": {
                "target": {"tagName": "TEXTAREA", "ariaLabel": "Comment"},
                "tabId": 3,
                "url": "https://tracker.example.com/issues/12?draft=1",
            },
        }))
        .unwrap();
        let event = convert_to_event(&msg).unwrap();
        assert_eq!(event.kind, EventKind::PasteDetected { matched_copy_hash: None });
        assert_eq!(event.metadata["url"], "https://tracker.example.com/issues/12");
        assert_eq!(event.metadata["target"]["ariaLabel"], "Comment");
    }

    #[test]
    fn test_convert_navigation_change() {
        let msg: ExtensionMessage = serde_json::from_value(serde_json::json!({
            "type": "navigation_change",
            "seq": 3,
            "timestamp": "2026-01-01T00:00:00Z",
            "payload": {
                "url": "https://mail.example.com/inbox?q=invoice#msg-1",
                "title": "Inbox - Mail",
                "tabId": 7
            }
        }))
        .unwrap();
        let event = convert_to_event(&msg).unwrap();
        assert_eq!(
            event.kind,
            EventKind::NavigationChange {
                url: "https://mail.example.com/inbox".into(),
                title: "Inbox - Mail".into(),
            }
        );
        assert_eq!(event.metadata["url"], "https://mail.example.com/inbox");
        assert_eq!(event.window.as_ref().unwrap().title, "Inbox - Mail");
        assert_eq!(
            browser_transition_kind(&event.kind),
            Some(BrowserTransitionKind::Navigation)
        );
    }

    #[test]
    fn test_convert_form_submit_strips_action_url() {
        let msg: ExtensionMessage = serde_json::from_value(serde_json::json!({
            "type": "form_submit",
            "seq": 4,
            "timestamp": "2026-01-01T00:00:00Z",
            "payload": {
                "url": "https://shop.example.com/checkout?session=abc",
                "tabId": 7,
                "formAnchor": {
                    "tag": "form",
                    "name": "checkout",
                    "action": "https://shop.example.com/pay?token=secret",
                    "method": "post",
                    "submitter": {"tagName": "button", "text": "Pay now"}
                }
            }
        }))
        .unwrap();
        let event = convert_to_event(&msg).unwrap();
        let EventKind::FormSubmit { url, form_anchor } = &event.kind else {
            panic!("expected FormSubmit, got {:?}", event.kind);
        };
        assert_eq!(url, "https://shop.example.com/checkout");
        assert_eq!(form_anchor["action"], "https://shop.example.com/pay");
        assert_eq!(form_anchor["submitter"]["text"], "Pay now");
        assert_eq!(event.metadata["formAnchor"]["action"], "https://shop.example.com/pay");
        assert_eq!(
            browser_transition_kind(&event.kind),
            Some(BrowserTransitionKind::FormSubmit)
        );
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

    #[tokio::test]
    async fn test_run_processes_outbound_command() {
        // Reader returns EOF immediately (empty pipe).
        // Use a duplex stream for the writer so we can inspect output.
        let reader = tokio::io::empty();
        let (wr, mut rd) = tokio::io::duplex(4096);

        let (event_tx, _event_rx) = mpsc::channel::<Event>(16);
        let (cmd_tx, cmd_rx) = mpsc::channel::<DaemonCommand>(16);

        // Send a command before starting the server
        cmd_tx
            .send(DaemonCommand::SendMessage {
                msg_type: DaemonMessageType::RequestSnapshot,
                payload: serde_json::json!({"tabId": 42, "reason": "focus_recording"}),
            })
            .await
            .unwrap();
        // Drop sender so command channel closes after the one message
        drop(cmd_tx);

        let server = NativeMessageServer::new(reader, wr);
        // run() will: process the command (write to writer), then cmd_rx returns None → break
        let result = server.run(event_tx, cmd_rx).await;
        assert!(result.is_ok());

        // Read the framed message from the duplex reader
        let mut len_buf = [0u8; 4];
        tokio::io::AsyncReadExt::read_exact(&mut rd, &mut len_buf)
            .await
            .expect("should read length prefix");
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut payload_buf = vec![0u8; len];
        tokio::io::AsyncReadExt::read_exact(&mut rd, &mut payload_buf)
            .await
            .expect("should read payload");

        let written: serde_json::Value =
            serde_json::from_slice(&payload_buf).expect("should be valid JSON");
        assert_eq!(written["type"], "request_snapshot");
        assert_eq!(written["payload"]["tabId"], 42);
        assert_eq!(written["payload"]["reason"], "focus_recording");
        assert_eq!(written["seq"], 1);
        assert!(written["timestamp"].as_str().is_some());
    }

    #[tokio::test]
    async fn test_run_interleaves_commands_and_reads() {
        // Use a duplex stream for the reader: we write the inbound message
        // into one end and the server reads from the other.
        let inbound_msg = serde_json::json!({
            "type": "content_ready",
            "seq": 1,
            "timestamp": "2026-01-01T00:00:00Z",
            "payload": {"url": "https://example.com", "tabId": 10}
        });
        let inbound_frame = encode_frame(&inbound_msg).unwrap();

        let (mut reader_wr, reader_rd) = tokio::io::duplex(8192);
        let (writer_wr, mut writer_rd) = tokio::io::duplex(8192);

        let (event_tx, mut event_rx) = mpsc::channel::<Event>(16);
        let (cmd_tx, cmd_rx) = mpsc::channel::<DaemonCommand>(16);

        // Send a command that will be processed alongside the inbound read
        cmd_tx
            .send(DaemonCommand::SendMessage {
                msg_type: DaemonMessageType::Ping,
                payload: serde_json::json!({}),
            })
            .await
            .unwrap();
        drop(cmd_tx);

        // Write the inbound message into the reader pipe, then close it (EOF)
        tokio::io::AsyncWriteExt::write_all(&mut reader_wr, &inbound_frame)
            .await
            .unwrap();
        drop(reader_wr); // Close → EOF on the reader side

        let server = NativeMessageServer::new(reader_rd, writer_wr);
        let result = server.run(event_tx, cmd_rx).await;
        assert!(result.is_ok());

        // Verify inbound message was converted to an event
        let event = event_rx.try_recv().expect("should have received an event");
        assert_eq!(event.kind, EventKind::FocusChange);

        // run() consumed the server and dropped the writer, so writer_rd sees EOF.

        // Verify outbound messages were written (ping command + ack for content_ready)
        let mut messages = Vec::new();
        loop {
            let mut len_buf = [0u8; 4];
            match tokio::io::AsyncReadExt::read_exact(&mut writer_rd, &mut len_buf).await {
                Ok(_) => {
                    let len = u32::from_le_bytes(len_buf) as usize;
                    let mut buf = vec![0u8; len];
                    tokio::io::AsyncReadExt::read_exact(&mut writer_rd, &mut buf)
                        .await
                        .unwrap();
                    let val: serde_json::Value = serde_json::from_slice(&buf).unwrap();
                    messages.push(val);
                }
                Err(_) => break,
            }
        }

        // Should have at least a ping and an ack
        assert!(
            messages.len() >= 2,
            "expected at least 2 outbound messages, got {}",
            messages.len()
        );
        let types: Vec<&str> = messages
            .iter()
            .filter_map(|m| m["type"].as_str())
            .collect();
        assert!(types.contains(&"ping"), "missing ping command: {:?}", types);
        assert!(types.contains(&"ack"), "missing ack response: {:?}", types);
    }

    #[tokio::test]
    async fn test_daemon_command_request_snapshot_format() {
        // Verify the wire format of a RequestSnapshot command
        let (wr, mut rd) = tokio::io::duplex(4096);
        let reader = tokio::io::empty();

        let (event_tx, _event_rx) = mpsc::channel::<Event>(16);
        let (cmd_tx, cmd_rx) = mpsc::channel::<DaemonCommand>(16);

        cmd_tx
            .send(DaemonCommand::SendMessage {
                msg_type: DaemonMessageType::RequestSnapshot,
                payload: serde_json::json!({
                    "tabId": 99,
                    "reason": "focus_recording",
                }),
            })
            .await
            .unwrap();
        drop(cmd_tx);

        let server = NativeMessageServer::new(reader, wr);
        server.run(event_tx, cmd_rx).await.unwrap();

        // Parse the framed output
        let mut len_buf = [0u8; 4];
        tokio::io::AsyncReadExt::read_exact(&mut rd, &mut len_buf)
            .await
            .unwrap();
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        tokio::io::AsyncReadExt::read_exact(&mut rd, &mut buf)
            .await
            .unwrap();

        let msg: DaemonMessage = serde_json::from_slice(&buf).unwrap();
        assert_eq!(msg.msg_type, DaemonMessageType::RequestSnapshot);
        assert_eq!(msg.seq, 1);
        assert_eq!(msg.payload["tabId"], 99);
        assert_eq!(msg.payload["reason"], "focus_recording");
        // Timestamp should be a valid RFC3339 string
        chrono::DateTime::parse_from_rfc3339(msg.timestamp.as_str())
            .expect("timestamp should be valid RFC3339");
    }

    #[tokio::test]
    async fn test_run_deduplicates_replayed_seq() {
        // Send two messages with seq 1 and 2, then replay seq 1 again.
        // Only the first two should produce events; the replayed seq 1
        // should be ack'd but NOT forwarded.
        let msg1 = serde_json::json!({
            "type": "content_ready",
            "seq": 1,
            "timestamp": "2026-01-01T00:00:00Z",
            "payload": {"url": "https://a.com", "tabId": 1}
        });
        let msg2 = serde_json::json!({
            "type": "content_ready",
            "seq": 2,
            "timestamp": "2026-01-01T00:00:01Z",
            "payload": {"url": "https://b.com", "tabId": 2}
        });
        // Replay of seq 1 (same seq, simulating reconnect resend)
        let msg1_replay = serde_json::json!({
            "type": "content_ready",
            "seq": 1,
            "timestamp": "2026-01-01T00:00:00Z",
            "payload": {"url": "https://a.com", "tabId": 1}
        });

        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&encode_frame(&msg1).unwrap());
        input_bytes.extend_from_slice(&encode_frame(&msg2).unwrap());
        input_bytes.extend_from_slice(&encode_frame(&msg1_replay).unwrap());

        let reader = std::io::Cursor::new(input_bytes);
        let mut writer = Vec::new();

        let server = NativeMessageServer::new(reader, &mut writer);

        let (tx, mut rx) = mpsc::channel(10);
        let (_dummy_cmd_tx, dummy_cmd_rx) = mpsc::channel::<DaemonCommand>(1);

        server.run(tx, dummy_cmd_rx).await.unwrap();

        // Collect events — should be exactly 2 (seq 1 and seq 2), not 3
        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }

        assert_eq!(
            events.len(),
            2,
            "expected 2 events (duplicate seq 1 should be skipped), got {}",
            events.len()
        );
    }
}
