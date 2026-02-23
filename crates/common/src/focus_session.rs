//! Focus Recording Mode — signal file IPC between CLI/SwiftUI and daemon.
//!
//! The focus session signal is a JSON file (`focus-session.json`) in the
//! standard data directory that the CLI or SwiftUI app writes to start/stop
//! a focus recording session.  The daemon reads it each poll cycle (500ms)
//! to tag events with `focus_session_id` metadata.
//!
//! File location: `~/Library/Application Support/oc-apprentice/focus-session.json`

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{debug, warn};

/// Focus session signal filename.
pub const FOCUS_SESSION_FILE: &str = "focus-session.json";

/// Focus session status — only two valid states.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FocusSessionStatus {
    /// Actively recording user actions.
    Recording,
    /// Recording finished, waiting for worker to process.
    Stopped,
}

/// Signal written by CLI or SwiftUI to start/stop a focus recording session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FocusSessionSignal {
    /// Unique session identifier (UUID).
    pub session_id: String,
    /// User-provided label for the workflow being recorded.
    pub title: String,
    /// When the session was started (ISO 8601).
    pub started_at: String,
    /// Current status.
    pub status: FocusSessionStatus,
}

impl FocusSessionSignal {
    /// Returns true if the session is actively recording.
    pub fn is_recording(&self) -> bool {
        self.status == FocusSessionStatus::Recording
    }

    /// Returns true if the session has been stopped.
    pub fn is_stopped(&self) -> bool {
        self.status == FocusSessionStatus::Stopped
    }
}

/// Read the focus session signal file from the given state directory.
///
/// Returns `None` if the file doesn't exist or can't be parsed.
pub fn read_focus_signal(state_dir: &Path) -> Option<FocusSessionSignal> {
    let path = state_dir.join(FOCUS_SESSION_FILE);
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return None,
    };
    match serde_json::from_str(&content) {
        Ok(signal) => Some(signal),
        Err(e) => {
            warn!(
                error = %e,
                path = %path.display(),
                "Failed to parse focus-session.json"
            );
            None
        }
    }
}

/// Write a focus session signal file (atomic: tmp + fsync + rename).
pub fn write_focus_signal(state_dir: &Path, signal: &FocusSessionSignal) -> std::io::Result<()> {
    use std::io::Write;

    std::fs::create_dir_all(state_dir)?;
    let target = state_dir.join(FOCUS_SESSION_FILE);
    let tmp = state_dir.join(format!(".{}.tmp", FOCUS_SESSION_FILE));
    let json = serde_json::to_string_pretty(signal)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let mut file = std::fs::File::create(&tmp)?;
    file.write_all(json.as_bytes())?;
    file.sync_all()?;
    std::fs::rename(&tmp, &target)?;
    Ok(())
}

/// Remove the focus session signal file after the worker consumes it.
pub fn clear_focus_signal(state_dir: &Path) {
    let path = state_dir.join(FOCUS_SESSION_FILE);
    if path.exists() {
        if let Err(e) = std::fs::remove_file(&path) {
            warn!(
                error = %e,
                path = %path.display(),
                "Failed to remove focus-session.json"
            );
        } else {
            debug!(path = %path.display(), "Cleared focus-session.json");
        }
    }
}

/// Helper to get the focus signal file path from the standard data directory.
pub fn focus_signal_path() -> PathBuf {
    crate::status::data_dir().join(FOCUS_SESSION_FILE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_signal() -> FocusSessionSignal {
        FocusSessionSignal {
            session_id: "550e8400-e29b-41d4-a716-446655440000".to_string(),
            title: "Expense report filing".to_string(),
            started_at: "2026-02-23T10:00:00Z".to_string(),
            status: FocusSessionStatus::Recording,
        }
    }

    #[test]
    fn test_write_and_read_signal() {
        let tmp = TempDir::new().unwrap();
        let signal = make_signal();

        write_focus_signal(tmp.path(), &signal).unwrap();
        let read = read_focus_signal(tmp.path()).unwrap();

        assert_eq!(read, signal);
        assert_eq!(read.session_id, "550e8400-e29b-41d4-a716-446655440000");
        assert_eq!(read.title, "Expense report filing");
        assert_eq!(read.status, FocusSessionStatus::Recording);
    }

    #[test]
    fn test_read_nonexistent_returns_none() {
        let tmp = TempDir::new().unwrap();
        assert!(read_focus_signal(tmp.path()).is_none());
    }

    #[test]
    fn test_clear_signal() {
        let tmp = TempDir::new().unwrap();
        let signal = make_signal();

        write_focus_signal(tmp.path(), &signal).unwrap();
        assert!(tmp.path().join(FOCUS_SESSION_FILE).exists());

        clear_focus_signal(tmp.path());
        assert!(!tmp.path().join(FOCUS_SESSION_FILE).exists());
    }

    #[test]
    fn test_clear_nonexistent_no_panic() {
        let tmp = TempDir::new().unwrap();
        clear_focus_signal(tmp.path()); // Should not panic
    }

    #[test]
    fn test_is_recording() {
        let signal = make_signal();
        assert!(signal.is_recording());
        assert!(!signal.is_stopped());
    }

    #[test]
    fn test_is_stopped() {
        let mut signal = make_signal();
        signal.status = FocusSessionStatus::Stopped;
        assert!(!signal.is_recording());
        assert!(signal.is_stopped());
    }

    #[test]
    fn test_json_roundtrip() {
        let signal = make_signal();
        let json = serde_json::to_string_pretty(&signal).unwrap();
        let parsed: FocusSessionSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(signal, parsed);
    }

    #[test]
    fn test_write_overwrites_existing() {
        let tmp = TempDir::new().unwrap();
        let mut signal = make_signal();

        write_focus_signal(tmp.path(), &signal).unwrap();

        signal.status = FocusSessionStatus::Stopped;
        write_focus_signal(tmp.path(), &signal).unwrap();

        let read = read_focus_signal(tmp.path()).unwrap();
        assert_eq!(read.status, FocusSessionStatus::Stopped);
    }

    #[test]
    fn test_invalid_json_returns_none() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join(FOCUS_SESSION_FILE);
        std::fs::write(&path, "not valid json{{{").unwrap();

        assert!(read_focus_signal(tmp.path()).is_none());
    }

    /// Integration test: simulates the full CLI `focus start` → `focus stop`
    /// flow by writing a recording signal, verifying it, mutating to stopped,
    /// and verifying the worker would pick it up.
    #[test]
    fn test_cli_start_stop_roundtrip() {
        let tmp = TempDir::new().unwrap();

        // Step 1: No session exists
        assert!(read_focus_signal(tmp.path()).is_none());

        // Step 2: CLI "focus start" — write recording signal
        let signal = FocusSessionSignal {
            session_id: "cli-test-uuid-001".to_string(),
            title: "Expense report filing".to_string(),
            started_at: "2026-02-23T14:30:00Z".to_string(),
            status: FocusSessionStatus::Recording,
        };
        write_focus_signal(tmp.path(), &signal).unwrap();

        // Daemon would read this and start tagging events
        let active = read_focus_signal(tmp.path()).unwrap();
        assert!(active.is_recording());
        assert_eq!(active.session_id, "cli-test-uuid-001");

        // Step 3: CLI "focus stop" — read existing, set to stopped, rewrite
        let mut stopped = read_focus_signal(tmp.path()).unwrap();
        stopped.status = FocusSessionStatus::Stopped;
        write_focus_signal(tmp.path(), &stopped).unwrap();

        // Worker would read this and process the session
        let final_signal = read_focus_signal(tmp.path()).unwrap();
        assert!(final_signal.is_stopped());
        assert_eq!(final_signal.title, "Expense report filing");
        assert_eq!(final_signal.started_at, "2026-02-23T14:30:00Z");

        // Step 4: Worker clears after processing
        clear_focus_signal(tmp.path());
        assert!(read_focus_signal(tmp.path()).is_none());
    }

    /// Verify enum serializes to lowercase JSON strings for Python/Swift interop.
    #[test]
    fn test_status_enum_json_serialization() {
        let recording = FocusSessionStatus::Recording;
        let stopped = FocusSessionStatus::Stopped;

        // Must serialize to lowercase strings matching Python/Swift expectations
        let rec_json = serde_json::to_string(&recording).unwrap();
        assert_eq!(rec_json, "\"recording\"");

        let stop_json = serde_json::to_string(&stopped).unwrap();
        assert_eq!(stop_json, "\"stopped\"");

        // Must deserialize from lowercase strings (written by Python/Swift)
        let parsed: FocusSessionStatus = serde_json::from_str("\"recording\"").unwrap();
        assert_eq!(parsed, FocusSessionStatus::Recording);

        let parsed: FocusSessionStatus = serde_json::from_str("\"stopped\"").unwrap();
        assert_eq!(parsed, FocusSessionStatus::Stopped);

        // Invalid status should fail deserialization
        assert!(serde_json::from_str::<FocusSessionStatus>("\"paused\"").is_err());
    }
}
