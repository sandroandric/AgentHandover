//! Unix domain socket control API for the AgentHandover daemon.
//!
//! The menu bar app uses this socket to pause and resume passive capture
//! around focus recordings. It accepts exactly two commands:
//! `pause_capture` and `resume_capture`.
//!
//! ## Protocol
//!
//! - Client connects to `{status_dir}/control.sock`
//! - Client sends one JSON command (newline-terminated)
//! - Server responds with one JSON response (newline-terminated)
//! - Server closes the connection after the response

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;
use tokio::sync::watch;
use tracing::{debug, error, info, warn};

/// Socket filename within the status directory.
const CONTROL_SOCKET_FILE: &str = "control.sock";

/// All the handles the control server needs. Constructed in `main.rs` and
/// passed to [`serve`].
pub struct ControlContext {
    /// Directory where `control.sock` will be created.
    pub status_dir: PathBuf,
    /// Atomic capture-paused flag shared with the observer event loop.
    /// The control API sets this; the observer reads it each tick.
    pub capture_paused: Arc<AtomicBool>,
    /// Shutdown receiver — when `true`, the server should exit.
    pub shutdown_rx: watch::Receiver<bool>,
}

// ---------------------------------------------------------------------------
// JSON protocol types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
enum Command {
    PauseCapture,
    ResumeCapture,
}

#[derive(Debug, Serialize)]
struct SimpleOkResponse {
    ok: bool,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    ok: bool,
    error: String,
}

// ---------------------------------------------------------------------------
// Command dispatch
// ---------------------------------------------------------------------------

/// Handle a single parsed command and return a JSON response string.
fn handle_command(cmd: Command, ctx: &ControlContext) -> String {
    let paused = matches!(cmd, Command::PauseCapture);
    // Atomic flag — observer loop reads this each tick without locking
    ctx.capture_paused.store(paused, Ordering::Release);
    if paused {
        info!("Capture paused via control API");
    } else {
        info!("Capture resumed via control API");
    }
    let resp = SimpleOkResponse { ok: true };
    serde_json::to_string(&resp).unwrap_or_else(|e| {
        format!(r#"{{"ok":false,"error":"serialization error: {}"}}"#, e)
    })
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
            handle_command(cmd, ctx)
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
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::UnixStream;

    /// Helper: create a ControlContext with a socket in a temp directory.
    fn make_context(
        tmp_dir: &Path,
        capture_paused: Arc<AtomicBool>,
    ) -> (ControlContext, watch::Sender<bool>) {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let ctx = ControlContext {
            status_dir: tmp_dir.to_path_buf(),
            capture_paused,
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
    async fn test_pause_resume_capture() {
        let tmp = tempfile::TempDir::new().unwrap();
        let flag = Arc::new(AtomicBool::new(false));
        let (ctx, shutdown_tx) = make_context(tmp.path(), Arc::clone(&flag));
        let sock_path = tmp.path().join(CONTROL_SOCKET_FILE);

        let server = tokio::spawn(serve(ctx));
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Pause
        let resp = send_command(&sock_path, r#"{"command":"pause_capture"}"#).await;
        let parsed: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();
        assert_eq!(parsed["ok"], true);
        assert!(flag.load(Ordering::Acquire));

        // Resume
        let resp = send_command(&sock_path, r#"{"command":"resume_capture"}"#).await;
        let parsed: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();
        assert_eq!(parsed["ok"], true);
        assert!(!flag.load(Ordering::Acquire));

        let _ = shutdown_tx.send(true);
        let _ = server.await;
    }

    #[tokio::test]
    async fn test_removed_commands_are_rejected() {
        let tmp = tempfile::TempDir::new().unwrap();
        let flag = Arc::new(AtomicBool::new(false));
        let (ctx, shutdown_tx) = make_context(tmp.path(), Arc::clone(&flag));
        let sock_path = tmp.path().join(CONTROL_SOCKET_FILE);

        let server = tokio::spawn(serve(ctx));
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        for cmd in [
            "get_status",
            "request_accessibility",
            "request_screen_recording",
            "start_focus",
            "stop_focus",
        ] {
            let json = format!(r#"{{"command":"{}"}}"#, cmd);
            let resp = send_command(&sock_path, &json).await;
            let parsed: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();
            assert_eq!(parsed["ok"], false, "{} should be rejected", cmd);
            assert!(parsed["error"].as_str().unwrap().contains("invalid command"));
        }
        assert!(!flag.load(Ordering::Acquire), "rejected commands must not touch capture state");

        let _ = shutdown_tx.send(true);
        let _ = server.await;
    }

    #[tokio::test]
    async fn test_invalid_command() {
        let tmp = tempfile::TempDir::new().unwrap();
        let (ctx, shutdown_tx) = make_context(tmp.path(), Arc::new(AtomicBool::new(false)));
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
        let (ctx, shutdown_tx) = make_context(tmp.path(), Arc::new(AtomicBool::new(false)));
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

        let (ctx, shutdown_tx) = make_context(tmp.path(), Arc::new(AtomicBool::new(false)));

        let server = tokio::spawn(serve(ctx));
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Server should have replaced the stale file with a real socket.
        // Verify by connecting.
        let resp = send_command(&sock_path, r#"{"command":"resume_capture"}"#).await;
        let parsed: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();
        assert_eq!(parsed["ok"], true);

        let _ = shutdown_tx.send(true);
        let _ = server.await;
    }
}
