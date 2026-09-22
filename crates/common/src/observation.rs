//! Client for the menu bar app's observation socket.
//!
//! AgentHandover.app owns the Accessibility and Screen Recording grants and
//! serves what they allow over a Unix socket in the data dir: the daemon
//! asks it for window, display and permission state, and `doctor` asks it
//! whether the grants are in place. One client for both.

use crate::event::{CursorPosition, DisplayInfo, WindowInfo};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Read/write timeout for one snapshot request.
pub const SNAPSHOT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

#[derive(Debug, Clone, Deserialize)]
pub struct ObservationSnapshot {
    pub accessibility_granted: bool,
    pub screen_recording_granted: bool,
    pub secure_field_focused: bool,
    pub focused_window: Option<WindowInfo>,
    pub display_topology: Vec<DisplayInfo>,
    pub cursor_global_px: Option<CursorPosition>,
}

/// `<data dir>/observation.sock`, served by ObservationServer.swift.
pub fn observation_socket_path() -> PathBuf {
    crate::status::data_dir().join("observation.sock")
}

/// `<data dir>/capture.sock`, served by ScreenCaptureServer.swift.
pub fn capture_socket_path() -> PathBuf {
    crate::status::data_dir().join("capture.sock")
}

/// Ask the observation server at `socket_path` for a snapshot.
///
/// Sends `{"command":"snapshot"}\n` and parses one JSON line back, with
/// [`SNAPSHOT_TIMEOUT`] on both directions. The error says which step
/// failed (connect, write, read, empty reply, parse) for diagnostics.
#[cfg(unix)]
pub fn snapshot_at(socket_path: &Path) -> std::io::Result<ObservationSnapshot> {
    use std::io::{BufRead, BufReader, Error, ErrorKind, Write};
    use std::os::unix::net::UnixStream;

    let mut stream = UnixStream::connect(socket_path)?;
    stream.set_read_timeout(Some(SNAPSHOT_TIMEOUT))?;
    stream.set_write_timeout(Some(SNAPSHOT_TIMEOUT))?;
    stream.write_all(b"{\"command\":\"snapshot\"}\n")?;

    let mut response = String::new();
    BufReader::new(stream).read_line(&mut response)?;
    let response = response.trim();
    if response.is_empty() {
        return Err(Error::new(ErrorKind::UnexpectedEof, "empty reply"));
    }
    serde_json::from_str(response).map_err(|e| Error::new(ErrorKind::InvalidData, e))
}

#[cfg(not(unix))]
pub fn snapshot_at(_socket_path: &Path) -> std::io::Result<ObservationSnapshot> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "observation socket requires a Unix platform",
    ))
}

/// Snapshot from the default socket, `None` on any failure.
pub fn snapshot() -> Option<ObservationSnapshot> {
    snapshot_at(&observation_socket_path()).ok()
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixListener;

    fn serve_once(listener: UnixListener, reply: &'static str) -> std::thread::JoinHandle<String> {
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut line = String::new();
            BufReader::new(&stream).read_line(&mut line).unwrap();
            (&stream).write_all(reply.as_bytes()).unwrap();
            line
        })
    }

    #[test]
    fn snapshot_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("observation.sock");
        let server = serve_once(
            UnixListener::bind(&sock).unwrap(),
            "{\"accessibility_granted\":true,\"screen_recording_granted\":false,\
             \"secure_field_focused\":false,\"focused_window\":null,\
             \"display_topology\":[],\"cursor_global_px\":null}\n",
        );
        let snap = snapshot_at(&sock).unwrap();
        assert!(snap.accessibility_granted);
        assert!(!snap.screen_recording_granted);
        assert_eq!(server.join().unwrap(), "{\"command\":\"snapshot\"}\n");
    }

    #[test]
    fn missing_socket_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(snapshot_at(&dir.path().join("nope.sock")).is_err());
    }

    #[test]
    fn garbage_reply_is_invalid_data() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("observation.sock");
        let server = serve_once(UnixListener::bind(&sock).unwrap(), "not json\n");
        let err = snapshot_at(&sock).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        server.join().unwrap();
    }
}
