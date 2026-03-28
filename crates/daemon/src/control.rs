//! Unix domain socket control API for the AgentHandover daemon.
//!
//! Provides a JSON-over-Unix-socket control plane so that the CLI, SwiftUI
//! menu bar app, and future agents can query daemon state and trigger actions
//! without relying on file-based IPC hacks.
//!
//! ## Protocol
//!
//! - Client connects to `{status_dir}/control.sock`
//! - Client sends one JSON command (newline-terminated)
//! - Server responds with one JSON response (newline-terminated)
//! - Server closes the connection after the response

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;
use tokio::sync::watch;
use tracing::{debug, error, info, warn};

/// Socket filename within the status directory.
const CONTROL_SOCKET_FILE: &str = "control.sock";

// ---------------------------------------------------------------------------
// Shared daemon state
// ---------------------------------------------------------------------------

/// Mutable daemon state shared between the control server and the main loop.
#[derive(Debug, Clone)]
pub struct FocusSession {
    pub session_id: String,
    pub title: String,
    pub started_at: String,
}

/// State that the control server can read and mutate.
#[derive(Debug)]
pub struct DaemonState {
    pub focus_session: Option<FocusSession>,
    pub capture_paused: bool,
}

impl DaemonState {
    pub fn new() -> Self {
        Self {
            focus_session: None,
            capture_paused: false,
        }
    }
}

/// All the handles the control server needs to answer queries and trigger
/// actions. Constructed in `main.rs` and passed to [`serve`].
pub struct ControlContext {
    /// Directory where `control.sock` will be created.
    pub status_dir: PathBuf,
    /// Daemon process start time (for uptime calculation).
    pub start_time: DateTime<Utc>,
    /// Shared event counter (incremented by the storage writer).
    pub event_counter: Arc<AtomicU64>,
    /// Mutable daemon state (focus session, capture paused).
    pub state: Arc<tokio::sync::Mutex<DaemonState>>,
    /// Shutdown receiver — when `true`, the server should exit.
    pub shutdown_rx: watch::Receiver<bool>,
}

// ---------------------------------------------------------------------------
// JSON protocol types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
enum Command {
    GetStatus,
    RequestAccessibility,
    RequestScreenRecording,
    StartFocus {
        title: String,
        session_id: String,
    },
    StopFocus,
    PauseCapture,
    ResumeCapture,
}

#[derive(Debug, Serialize)]
struct StatusResponse {
    ok: bool,
    pid: u32,
    version: &'static str,
    uptime_seconds: u64,
    accessibility_permitted: bool,
    screen_recording_permitted: bool,
    capture_active: bool,
    focus_session: Option<FocusSessionResponse>,
    events_today: u64,
}

#[derive(Debug, Serialize)]
struct FocusSessionResponse {
    session_id: String,
    title: String,
}

#[derive(Debug, Serialize)]
struct SimpleOkResponse {
    ok: bool,
}

#[derive(Debug, Serialize)]
struct PermissionResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    accessibility_permitted: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    screen_recording_permitted: Option<bool>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    ok: bool,
    error: String,
}

// ---------------------------------------------------------------------------
// Permission helpers (platform-specific)
// ---------------------------------------------------------------------------

/// Check if accessibility permission is currently granted.
fn check_accessibility() -> bool {
    crate::observation::snapshot()
        .map(|snapshot| snapshot.accessibility_granted)
        .unwrap_or(false)
}

/// Accessibility is owned by the main app process.
/// The daemon keeps this command as a compatibility no-op so older clients
/// receive a sensible current-state response instead of failing outright.
fn request_accessibility_with_prompt() -> bool {
    check_accessibility()
}

/// Check if screen recording permission is currently granted.
fn check_screen_recording() -> bool {
    crate::observation::snapshot()
        .map(|snapshot| snapshot.screen_recording_granted)
        .unwrap_or(false)
}

/// Request screen recording permission.
fn request_screen_recording() -> bool {
    check_screen_recording()
}

// ---------------------------------------------------------------------------
// Command dispatch
// ---------------------------------------------------------------------------

/// Handle a single parsed command and return a JSON response string.
async fn handle_command(cmd: Command, ctx: &ControlContext) -> String {
    match cmd {
        Command::GetStatus => {
            let now = Utc::now();
            let uptime = now
                .signed_duration_since(ctx.start_time)
                .num_seconds()
                .unsigned_abs();
            let events_today = ctx.event_counter.load(Ordering::Relaxed);

            let state = ctx.state.lock().await;
            let focus = state.focus_session.as_ref().map(|fs| FocusSessionResponse {
                session_id: fs.session_id.clone(),
                title: fs.title.clone(),
            });
            let capture_active = !state.capture_paused;
            drop(state);

            let resp = StatusResponse {
                ok: true,
                pid: std::process::id(),
                version: env!("CARGO_PKG_VERSION"),
                uptime_seconds: uptime,
                accessibility_permitted: check_accessibility(),
                screen_recording_permitted: check_screen_recording(),
                capture_active,
                focus_session: focus,
                events_today,
            };
            serde_json::to_string(&resp).unwrap_or_else(|e| {
                format!(r#"{{"ok":false,"error":"serialization error: {}"}}"#, e)
            })
        }

        Command::RequestAccessibility => {
            // AXIsProcessTrustedWithOptions is a synchronous FFI call.
            // Run on a blocking thread so we don't stall the async runtime.
            let permitted =
                tokio::task::spawn_blocking(request_accessibility_with_prompt)
                    .await
                    .unwrap_or(false);
            let resp = PermissionResponse {
                ok: true,
                accessibility_permitted: Some(permitted),
                screen_recording_permitted: None,
            };
            serde_json::to_string(&resp).unwrap_or_else(|e| {
                format!(r#"{{"ok":false,"error":"serialization error: {}"}}"#, e)
            })
        }

        Command::RequestScreenRecording => {
            let permitted =
                tokio::task::spawn_blocking(request_screen_recording)
                    .await
                    .unwrap_or(false);
            let resp = PermissionResponse {
                ok: true,
                accessibility_permitted: None,
                screen_recording_permitted: Some(permitted),
            };
            serde_json::to_string(&resp).unwrap_or_else(|e| {
                format!(r#"{{"ok":false,"error":"serialization error: {}"}}"#, e)
            })
        }

        Command::StartFocus { title, session_id } => {
            let mut state = ctx.state.lock().await;
            if state.focus_session.is_some() {
                let resp = ErrorResponse {
                    ok: false,
                    error: "a focus session is already active".to_string(),
                };
                return serde_json::to_string(&resp).unwrap_or_else(|e| {
                    format!(r#"{{"ok":false,"error":"serialization error: {}"}}"#, e)
                });
            }

            let started_at = Utc::now().to_rfc3339();

            // Write the focus signal file so the observer loop and worker can
            // see it (backwards-compatible with existing file-based consumers).
            let signal = agenthandover_common::focus_session::FocusSessionSignal {
                session_id: session_id.clone(),
                title: title.clone(),
                started_at: started_at.clone(),
                status: agenthandover_common::focus_session::FocusSessionStatus::Recording,
            };
            let state_dir = agenthandover_common::status::data_dir();
            if let Err(e) =
                agenthandover_common::focus_session::write_focus_signal(&state_dir, &signal)
            {
                warn!(error = %e, "Failed to write focus-session.json from control API");
            }

            state.focus_session = Some(FocusSession {
                session_id,
                title,
                started_at,
            });
            drop(state);

            info!("Focus session started via control API");
            let resp = SimpleOkResponse { ok: true };
            serde_json::to_string(&resp).unwrap_or_else(|e| {
                format!(r#"{{"ok":false,"error":"serialization error: {}"}}"#, e)
            })
        }

        Command::StopFocus => {
            let mut state = ctx.state.lock().await;
            if state.focus_session.is_none() {
                let resp = ErrorResponse {
                    ok: false,
                    error: "no focus session is active".to_string(),
                };
                return serde_json::to_string(&resp).unwrap_or_else(|e| {
                    format!(r#"{{"ok":false,"error":"serialization error: {}"}}"#, e)
                });
            }

            // Update signal file to "stopped" so the worker picks it up.
            let state_dir = agenthandover_common::status::data_dir();
            if let Some(mut signal) =
                agenthandover_common::focus_session::read_focus_signal(&state_dir)
            {
                signal.status =
                    agenthandover_common::focus_session::FocusSessionStatus::Stopped;
                if let Err(e) =
                    agenthandover_common::focus_session::write_focus_signal(&state_dir, &signal)
                {
                    warn!(error = %e, "Failed to write stopped focus-session.json from control API");
                }
            }

            state.focus_session = None;
            drop(state);

            info!("Focus session stopped via control API");
            let resp = SimpleOkResponse { ok: true };
            serde_json::to_string(&resp).unwrap_or_else(|e| {
                format!(r#"{{"ok":false,"error":"serialization error: {}"}}"#, e)
            })
        }

        Command::PauseCapture => {
            let mut state = ctx.state.lock().await;
            state.capture_paused = true;
            drop(state);

            info!("Capture paused via control API");
            let resp = SimpleOkResponse { ok: true };
            serde_json::to_string(&resp).unwrap_or_else(|e| {
                format!(r#"{{"ok":false,"error":"serialization error: {}"}}"#, e)
            })
        }

        Command::ResumeCapture => {
            let mut state = ctx.state.lock().await;
            state.capture_paused = false;
            drop(state);

            info!("Capture resumed via control API");
            let resp = SimpleOkResponse { ok: true };
            serde_json::to_string(&resp).unwrap_or_else(|e| {
                format!(r#"{{"ok":false,"error":"serialization error: {}"}}"#, e)
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Connection handler
// ---------------------------------------------------------------------------

/// Handle a single client connection: read one JSON command, respond, close.
async fn handle_connection(
    stream: tokio::net::UnixStream,
    ctx: &ControlContext,
) {
    let (reader, mut writer) = stream.into_split();
    let mut buf_reader = BufReader::new(reader);
    let mut line = String::new();

    // Read one newline-terminated JSON command. Limit to 64 KiB to prevent
    // memory exhaustion from a misbehaving client.
    match tokio::time::timeout(
        std::time::Duration::from_secs(5),
        buf_reader.read_line(&mut line),
    )
    .await
    {
        Ok(Ok(0)) => {
            // Client closed without sending anything.
            debug!("Control client disconnected without sending a command");
            return;
        }
        Ok(Ok(n)) if n > 65536 => {
            let resp = r#"{"ok":false,"error":"request too large"}"#;
            let _ = writer.write_all(resp.as_bytes()).await;
            let _ = writer.write_all(b"\n").await;
            return;
        }
        Ok(Ok(_)) => {
            // Got a line — parse below.
        }
        Ok(Err(e)) => {
            warn!(error = %e, "Control socket read error");
            return;
        }
        Err(_) => {
            let resp = r#"{"ok":false,"error":"read timeout"}"#;
            let _ = writer.write_all(resp.as_bytes()).await;
            let _ = writer.write_all(b"\n").await;
            return;
        }
    }

    let response = match serde_json::from_str::<Command>(line.trim()) {
        Ok(cmd) => {
            debug!(command = ?cmd, "Control API command received");
            handle_command(cmd, ctx).await
        }
        Err(e) => {
            let resp = ErrorResponse {
                ok: false,
                error: format!("invalid command: {}", e),
            };
            serde_json::to_string(&resp).unwrap_or_else(|e| {
                format!(r#"{{"ok":false,"error":"serialization error: {}"}}"#, e)
            })
        }
    };

    if let Err(e) = writer.write_all(response.as_bytes()).await {
        warn!(error = %e, "Control socket write error");
        return;
    }
    if let Err(e) = writer.write_all(b"\n").await {
        warn!(error = %e, "Control socket write newline error");
    }
}

// ---------------------------------------------------------------------------
// Server entry point
// ---------------------------------------------------------------------------

/// Remove a stale socket file if it exists.
fn cleanup_socket(path: &Path) {
    if path.exists() {
        if let Err(e) = std::fs::remove_file(path) {
            warn!(
                path = %path.display(),
                error = %e,
                "Failed to remove stale control socket"
            );
        }
    }
}

/// Run the Unix domain socket control server.
///
/// This function should be spawned as a tokio task from `main.rs`. It listens
/// for connections until the shutdown signal fires, then cleans up the socket
/// file.
pub async fn serve(ctx: ControlContext) {
    let sock_path = ctx.status_dir.join(CONTROL_SOCKET_FILE);

    // Ensure the status directory exists.
    if let Err(e) = std::fs::create_dir_all(&ctx.status_dir) {
        error!(
            path = %ctx.status_dir.display(),
            error = %e,
            "Failed to create status directory for control socket"
        );
        return;
    }

    // Remove stale socket from a previous unclean shutdown.
    cleanup_socket(&sock_path);

    let listener = match UnixListener::bind(&sock_path) {
        Ok(l) => {
            info!(path = %sock_path.display(), "Control socket listening");
            l
        }
        Err(e) => {
            error!(
                path = %sock_path.display(),
                error = %e,
                "Failed to bind control socket"
            );
            return;
        }
    };

    let mut shutdown_rx = ctx.shutdown_rx.clone();

    loop {
        tokio::select! {
            accept_result = listener.accept() => {
                match accept_result {
                    Ok((stream, _addr)) => {
                        handle_connection(stream, &ctx).await;
                    }
                    Err(e) => {
                        warn!(error = %e, "Control socket accept error");
                    }
                }
            }
            _ = shutdown_rx.changed() => {
                info!("Control socket shutting down");
                break;
            }
        }
    }

    // Clean up the socket file on graceful shutdown.
    cleanup_socket(&sock_path);
    info!(path = %sock_path.display(), "Control socket removed");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::UnixStream;

    /// Helper: create a ControlContext with a socket in a temp directory.
    fn make_context(
        tmp_dir: &Path,
        state: Arc<tokio::sync::Mutex<DaemonState>>,
    ) -> (ControlContext, watch::Sender<bool>) {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let ctx = ControlContext {
            status_dir: tmp_dir.to_path_buf(),
            start_time: Utc::now(),
            event_counter: Arc::new(AtomicU64::new(42)),
            state,
            shutdown_rx,
        };
        (ctx, shutdown_tx)
    }

    /// Send a JSON command to the control socket and return the response.
    async fn send_command(sock_path: &Path, json: &str) -> String {
        let stream = UnixStream::connect(sock_path).await.unwrap();
        let (reader, mut writer) = stream.into_split();
        writer.write_all(json.as_bytes()).await.unwrap();
        writer.write_all(b"\n").await.unwrap();
        // Signal we're done writing so the server can detect EOF if needed.
        writer.shutdown().await.unwrap();

        let mut buf_reader = BufReader::new(reader);
        let mut response = String::new();
        buf_reader.read_line(&mut response).await.unwrap();
        response
    }

    #[tokio::test]
    async fn test_get_status() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = Arc::new(tokio::sync::Mutex::new(DaemonState::new()));
        let (ctx, shutdown_tx) = make_context(tmp.path(), state);
        let sock_path = tmp.path().join(CONTROL_SOCKET_FILE);

        let server = tokio::spawn(serve(ctx));

        // Give the server a moment to bind.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let resp = send_command(&sock_path, r#"{"command":"get_status"}"#).await;
        let parsed: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();

        assert_eq!(parsed["ok"], true);
        assert_eq!(parsed["events_today"], 42);
        assert!(parsed["pid"].as_u64().unwrap() > 0);
        assert!(parsed["uptime_seconds"].as_u64().is_some());
        assert_eq!(parsed["capture_active"], true);
        assert!(parsed["focus_session"].is_null());

        let _ = shutdown_tx.send(true);
        let _ = server.await;
    }

    #[tokio::test]
    async fn test_start_stop_focus() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = Arc::new(tokio::sync::Mutex::new(DaemonState::new()));
        let (ctx, shutdown_tx) = make_context(tmp.path(), Arc::clone(&state));
        let sock_path = tmp.path().join(CONTROL_SOCKET_FILE);

        let server = tokio::spawn(serve(ctx));
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Start focus
        let resp = send_command(
            &sock_path,
            r#"{"command":"start_focus","title":"Test Session","session_id":"uuid-123"}"#,
        )
        .await;
        let parsed: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();
        assert_eq!(parsed["ok"], true);

        // Verify state
        {
            let s = state.lock().await;
            assert!(s.focus_session.is_some());
            assert_eq!(s.focus_session.as_ref().unwrap().session_id, "uuid-123");
        }

        // Get status — should show focus session
        let resp = send_command(&sock_path, r#"{"command":"get_status"}"#).await;
        let parsed: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();
        assert_eq!(parsed["focus_session"]["session_id"], "uuid-123");
        assert_eq!(parsed["focus_session"]["title"], "Test Session");

        // Starting another focus should fail
        let resp = send_command(
            &sock_path,
            r#"{"command":"start_focus","title":"Dup","session_id":"uuid-456"}"#,
        )
        .await;
        let parsed: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();
        assert_eq!(parsed["ok"], false);
        assert!(parsed["error"].as_str().unwrap().contains("already active"));

        // Stop focus
        let resp = send_command(&sock_path, r#"{"command":"stop_focus"}"#).await;
        let parsed: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();
        assert_eq!(parsed["ok"], true);

        // Verify state cleared
        {
            let s = state.lock().await;
            assert!(s.focus_session.is_none());
        }

        // Stopping again should fail
        let resp = send_command(&sock_path, r#"{"command":"stop_focus"}"#).await;
        let parsed: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();
        assert_eq!(parsed["ok"], false);

        let _ = shutdown_tx.send(true);
        let _ = server.await;
    }

    #[tokio::test]
    async fn test_pause_resume_capture() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = Arc::new(tokio::sync::Mutex::new(DaemonState::new()));
        let (ctx, shutdown_tx) = make_context(tmp.path(), Arc::clone(&state));
        let sock_path = tmp.path().join(CONTROL_SOCKET_FILE);

        let server = tokio::spawn(serve(ctx));
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Pause
        let resp = send_command(&sock_path, r#"{"command":"pause_capture"}"#).await;
        let parsed: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();
        assert_eq!(parsed["ok"], true);

        // Status should show capture_active = false
        let resp = send_command(&sock_path, r#"{"command":"get_status"}"#).await;
        let parsed: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();
        assert_eq!(parsed["capture_active"], false);

        // Resume
        let resp = send_command(&sock_path, r#"{"command":"resume_capture"}"#).await;
        let parsed: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();
        assert_eq!(parsed["ok"], true);

        // Status should show capture_active = true
        let resp = send_command(&sock_path, r#"{"command":"get_status"}"#).await;
        let parsed: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();
        assert_eq!(parsed["capture_active"], true);

        let _ = shutdown_tx.send(true);
        let _ = server.await;
    }

    #[tokio::test]
    async fn test_invalid_command() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = Arc::new(tokio::sync::Mutex::new(DaemonState::new()));
        let (ctx, shutdown_tx) = make_context(tmp.path(), state);
        let sock_path = tmp.path().join(CONTROL_SOCKET_FILE);

        let server = tokio::spawn(serve(ctx));
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let resp = send_command(&sock_path, r#"{"command":"nonexistent"}"#).await;
        let parsed: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();
        assert_eq!(parsed["ok"], false);
        assert!(parsed["error"].as_str().unwrap().contains("invalid command"));

        let resp = send_command(&sock_path, r#"not json at all"#).await;
        let parsed: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();
        assert_eq!(parsed["ok"], false);

        let _ = shutdown_tx.send(true);
        let _ = server.await;
    }

    #[tokio::test]
    async fn test_socket_cleanup_on_shutdown() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = Arc::new(tokio::sync::Mutex::new(DaemonState::new()));
        let (ctx, shutdown_tx) = make_context(tmp.path(), state);
        let sock_path = tmp.path().join(CONTROL_SOCKET_FILE);

        let server = tokio::spawn(serve(ctx));
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Socket file should exist.
        assert!(sock_path.exists());

        // Trigger shutdown.
        let _ = shutdown_tx.send(true);
        let _ = server.await;

        // Socket file should be removed after shutdown.
        assert!(!sock_path.exists());
    }

    #[tokio::test]
    async fn test_stale_socket_removed_on_startup() {
        let tmp = tempfile::TempDir::new().unwrap();
        let sock_path = tmp.path().join(CONTROL_SOCKET_FILE);

        // Create a stale socket file.
        std::fs::write(&sock_path, b"stale").unwrap();
        assert!(sock_path.exists());

        let state = Arc::new(tokio::sync::Mutex::new(DaemonState::new()));
        let (ctx, shutdown_tx) = make_context(tmp.path(), state);

        let server = tokio::spawn(serve(ctx));
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Server should have replaced the stale file with a real socket.
        // Verify by connecting.
        let resp = send_command(&sock_path, r#"{"command":"get_status"}"#).await;
        let parsed: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();
        assert_eq!(parsed["ok"], true);

        let _ = shutdown_tx.send(true);
        let _ = server.await;
    }
}
