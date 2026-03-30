use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Info about an active focus recording session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusSessionInfo {
    pub session_id: String,
    pub title: String,
    pub started_at: String,
}

/// Status written by the daemon every 60s to daemon-status.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub pid: u32,
    pub version: String,
    pub started_at: DateTime<Utc>,
    pub heartbeat: DateTime<Utc>,
    pub events_today: u64,
    pub permissions_ok: bool,
    pub accessibility_permitted: bool,
    pub screen_recording_permitted: bool,
    pub db_path: String,
    pub uptime_seconds: u64,
    /// Timestamp of the last message received from the Chrome extension
    /// via Native Messaging.  `None` if no message has been received yet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_extension_message: Option<DateTime<Utc>>,
    /// Active focus recording session, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focus_session: Option<FocusSessionInfo>,
}

/// Status written by the worker each poll cycle to worker-status.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStatus {
    pub pid: u32,
    pub version: String,
    pub started_at: DateTime<Utc>,
    pub heartbeat: DateTime<Utc>,
    pub events_processed_today: u64,
    pub sops_generated: u64,
    pub last_pipeline_duration_ms: Option<u64>,
    pub consecutive_errors: u32,
    pub vlm_available: bool,
    pub sop_inducer_available: bool,
    /// Number of VLM jobs currently pending in the queue.
    #[serde(default)]
    pub vlm_queue_pending: u64,
    /// Number of VLM jobs dispatched today (UTC).
    #[serde(default)]
    pub vlm_jobs_today: u64,
    /// Number of VLM jobs dropped today due to backpressure.
    #[serde(default)]
    pub vlm_dropped_today: u64,
    /// VLM operating mode: "local" or "remote".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vlm_mode: Option<String>,
    /// VLM provider when in remote mode (e.g. "openai", "anthropic", "google").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vlm_provider: Option<String>,
}

/// Standard data directory for AgentHandover.
///
/// Used by status files, PID files, logs, database, and artifacts.
/// - macOS: `~/Library/Application Support/agenthandover`
/// - Linux: `~/.local/share/agenthandover`
pub fn data_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    if cfg!(target_os = "macos") {
        PathBuf::from(home).join("Library/Application Support/agenthandover")
    } else {
        PathBuf::from(home).join(".local/share/agenthandover")
    }
}

/// Standard location for status files (delegates to `data_dir()`).
pub fn status_dir() -> PathBuf {
    data_dir()
}

/// Atomically write a status file (tmp + fsync + rename).
pub fn write_status_file(filename: &str, status: &impl Serialize) -> std::io::Result<()> {
    use std::io::Write;

    let dir = status_dir();
    std::fs::create_dir_all(&dir)?;
    let target = dir.join(filename);
    let tmp = dir.join(format!(".{}.tmp", filename));
    let json = serde_json::to_string_pretty(status)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let mut file = std::fs::File::create(&tmp)?;
    file.write_all(json.as_bytes())?;
    file.sync_all()?;
    std::fs::rename(&tmp, &target)?;
    Ok(())
}

/// Heartbeat written by the NM bridge process to extension-heartbeat.json.
///
/// The NM bridge is a separate OS process launched by Chrome (not the main
/// daemon).  It cannot update the daemon's in-memory `Arc<AtomicI64>`, so it
/// writes this file instead.  The daemon health watcher, CLI, and SwiftUI app
/// all read it to determine extension connection status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionHeartbeat {
    pub pid: u32,
    pub last_message: DateTime<Utc>,
    pub messages_this_session: u64,
    pub session_started: DateTime<Utc>,
}

/// Extension heartbeat filename.
pub const EXTENSION_HEARTBEAT_FILE: &str = "extension-heartbeat.json";

/// Read the extension heartbeat file and return the `last_message` timestamp
/// if the heartbeat is fresh (written within the last 2 minutes).
///
/// Returns `None` if the file doesn't exist, can't be parsed, or is stale.
pub fn read_extension_heartbeat() -> Option<DateTime<Utc>> {
    let heartbeat: ExtensionHeartbeat = read_status_file(EXTENSION_HEARTBEAT_FILE).ok()?;
    let age = Utc::now().signed_duration_since(heartbeat.last_message);
    if age.num_seconds() <= 120 {
        Some(heartbeat.last_message)
    } else {
        None
    }
}

/// Read and deserialize a status file.
pub fn read_status_file<T: serde::de::DeserializeOwned>(filename: &str) -> std::io::Result<T> {
    let path = status_dir().join(filename);
    let content = std::fs::read_to_string(&path)?;
    serde_json::from_str(&content)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}
