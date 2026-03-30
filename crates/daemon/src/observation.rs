use agenthandover_common::event::{CursorPosition, DisplayInfo, WindowInfo};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ObservationSnapshot {
    pub accessibility_granted: bool,
    pub screen_recording_granted: bool,
    pub secure_field_focused: bool,
    pub focused_window: Option<WindowInfo>,
    pub display_topology: Vec<DisplayInfo>,
    pub cursor_global_px: Option<CursorPosition>,
}

#[cfg(unix)]
pub fn snapshot() -> Option<ObservationSnapshot> {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;
    use std::time::Duration;

    let home = std::env::var("HOME").ok()?;
    let socket_path = format!(
        "{}/Library/Application Support/agenthandover/observation.sock",
        home
    );

    let mut stream = UnixStream::connect(&socket_path).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .ok()?;
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .ok()?;

    stream
        .write_all(br#"{"command":"snapshot"}"#)
        .ok()?;
    stream.write_all(b"\n").ok()?;

    let mut response = String::new();
    let mut reader = BufReader::new(stream);
    reader.read_line(&mut response).ok()?;

    if response.trim().is_empty() {
        return None;
    }

    serde_json::from_str(response.trim()).ok()
}

#[cfg(not(unix))]
pub fn snapshot() -> Option<ObservationSnapshot> {
    None
}
