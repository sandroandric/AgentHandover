use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

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
    /// When the daily storage maintenance last completed, and what it
    /// removed. Absent until the first run on this machine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_maintenance: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub events_purged: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifacts_purged: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_freed: Option<u64>,
    /// What capture cost since the daemon started and in the last hour
    /// (`agenthandover status --compute`). Absent from older daemons.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute: Option<CaptureCompute>,
    /// `[privacy] read_app_state`: whether macOS lets AgentHandover read
    /// each app it tried (by app name), asked without prompting.  Absent
    /// until an app-state read was attempted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_state_consent: Option<BTreeMap<String, AutomationConsent>>,
}

/// macOS's Automation (Apple Events) consent for AgentHandover to query
/// one app, from `AEDeterminePermissionToAutomateTarget` without a prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutomationConsent {
    Granted,
    /// Turned off in System Settings > Privacy & Security > Automation.
    Denied,
    /// macOS has not asked the user yet.
    NotAsked,
    /// The app was not running when asked.
    NotRunning,
    /// Any other answer (an older macOS, an error).
    Unknown,
}

impl AutomationConsent {
    /// The consent an `AEDeterminePermissionToAutomateTarget` status means.
    pub fn from_status(status: i32) -> Self {
        match status {
            0 => Self::Granted,
            -1743 => Self::Denied,   // errAEEventNotPermitted
            -1744 => Self::NotAsked, // errAEEventWouldRequireUserConsent
            -600 => Self::NotRunning, // procNotFound
            _ => Self::Unknown,
        }
    }
}

/// Capture-side counters: frames taken from the app's capture socket,
/// frames kept, frames not taken or dropped per reason, and the
/// milliseconds spent capturing and in OCR.
///
/// Gate reasons: `privacy_zone` (a transition in a blocked or
/// metadata-only window), `rate_limited` (a transition over the minute's
/// frame budget), `capture_failed` (the capture socket gave no usable frame),
/// `privacy_changed` (a private window came up during the capture),
/// `unchanged` (same screen as the last kept frame) and `same_text` (a
/// heartbeat frame whose text did not change).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureCounts {
    pub frames_captured: u64,
    pub frames_kept: u64,
    #[serde(default)]
    pub frames_gated_by_reason: BTreeMap<String, u64>,
    pub capture_ms: u64,
    pub ocr_ms: u64,
}

impl CaptureCounts {
    /// Add `other`'s counters to these.
    pub fn add(&mut self, other: &CaptureCounts) {
        self.frames_captured += other.frames_captured;
        self.frames_kept += other.frames_kept;
        for (reason, n) in &other.frames_gated_by_reason {
            *self.frames_gated_by_reason.entry(reason.clone()).or_default() += n;
        }
        self.capture_ms += other.capture_ms;
        self.ocr_ms += other.ocr_ms;
    }
}

/// [`CaptureCounts`] since the daemon started (flattened) and for the last
/// hour, as written into `daemon-status.json` under `compute`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureCompute {
    #[serde(flatten)]
    pub since_start: CaptureCounts,
    pub last_hour: CaptureCounts,
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
    /// The worker's inference thread and its passive backlog.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference: Option<InferenceStatus>,
}

/// `worker-status.json`'s `inference` block: the job the worker's inference
/// thread runs, the jobs waiting, and the passive annotation backlog.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct InferenceStatus {
    /// The job running, with its priority (`"P0 focus"`).
    #[serde(default)]
    pub running: Option<String>,
    #[serde(default)]
    pub running_seconds: Option<f64>,
    /// Jobs waiting, in the order they will run.
    #[serde(default)]
    pub queued: Vec<String>,
    /// Jobs waiting plus the one running.
    #[serde(default)]
    pub depth: u32,
    /// Passive frames waiting for annotation.
    #[serde(default)]
    pub annotation_backlog: Option<u64>,
    /// The backlog needs hours of the worker's annotation budget to clear:
    /// the daemon keeps fewer passive frames meanwhile.
    #[serde(default)]
    pub lower_capture_cadence: bool,
}

/// A worker heartbeat older than this is from a worker that is not running:
/// its requests are ignored.
pub const WORKER_HEARTBEAT_FRESH_SECS: i64 = 120;

impl WorkerStatus {
    /// True when a running worker (heartbeat within
    /// [`WORKER_HEARTBEAT_FRESH_SECS`] of `now`) asks for fewer passive frames.
    pub fn asks_lower_capture_cadence(&self, now: DateTime<Utc>) -> bool {
        (now - self.heartbeat).num_seconds() <= WORKER_HEARTBEAT_FRESH_SECS
            && self
                .inference
                .as_ref()
                .is_some_and(|i| i.lower_capture_cadence)
    }
}

/// `worker-status.json` in `dir`, or None when it is missing or unreadable.
pub fn read_worker_status(dir: &Path) -> Option<WorkerStatus> {
    let content = std::fs::read_to_string(dir.join("worker-status.json")).ok()?;
    serde_json::from_str(&content).ok()
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

/// `<data dir>/sockets`: holds `capture.sock`, `observation.sock` (served by
/// the menu bar app) and `control.sock` (served by the daemon).  Mode 0700,
/// so no other account can reach the sockets.
pub fn socket_dir() -> PathBuf {
    data_dir().join("sockets")
}

/// Create `dir` mode 0700, or tighten an existing one to 0700.  Refuses a
/// symlink, a non-directory, or a directory owned by another user.
#[cfg(unix)]
pub fn ensure_private_dir(dir: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};

    if let Some(parent) = dir.parent() {
        std::fs::create_dir_all(parent)?;
    }
    match std::fs::DirBuilder::new().mode(0o700).create(dir) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e),
    }
    let meta = std::fs::symlink_metadata(dir)?;
    // SAFETY: getuid has no preconditions and cannot fail.
    let uid = unsafe { libc::getuid() };
    if !meta.file_type().is_dir() || meta.uid() != uid {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!("{} is not a directory owned by this user", dir.display()),
        ));
    }
    if meta.mode() & 0o777 != 0o700 {
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
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
