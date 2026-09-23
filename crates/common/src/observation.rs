//! Client for the menu bar app's observation socket.
//!
//! AgentHandover.app owns the Accessibility and Screen Recording grants and
//! serves what they allow over a Unix socket in the data dir: the daemon
//! asks it for window, display and permission state, and `doctor` asks it
//! whether the grants are in place. One client for both.
//!
//! Every request is bounded by [`SNAPSHOT_TIMEOUT`] so a wedged app cannot
//! stall the caller. The async entry point ([`snapshot_async`]) is what the
//! daemon's observer tick uses; it bounds the whole round trip and logs a
//! rate-limited warning on timeout. The blocking entry points
//! ([`snapshot`], [`snapshot_at`]) serve synchronous callers: the daemon's
//! health and control replies, and `agenthandover doctor`.

use crate::event::{CursorPosition, DisplayInfo, FocusedElement, WindowInfo};
use chrono::{DateTime, Utc};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Upper bound for one snapshot round trip (connect + request + response).
pub const SNAPSHOT_TIMEOUT: Duration = Duration::from_secs(2);

/// A wedged app is reported at most once per this interval.
const WARN_INTERVAL_SECS: u64 = 60;

const SNAPSHOT_REQUEST: &[u8] = b"{\"command\":\"snapshot\"}\n";

#[derive(Debug, Clone, Deserialize)]
pub struct ObservationSnapshot {
    pub accessibility_granted: bool,
    pub screen_recording_granted: bool,
    pub secure_field_focused: bool,
    pub focused_window: Option<WindowInfo>,
    pub display_topology: Vec<DisplayInfo>,
    pub cursor_global_px: Option<CursorPosition>,
    /// The most recent sleep the app saw (NSWorkspace `willSleep` then
    /// `didWake`), None until the Mac has slept while the app ran.
    #[serde(default)]
    pub last_sleep: Option<SleepRecord>,
}

/// One sleep of the Mac, as the app's NSWorkspace observer recorded it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SleepRecord {
    pub slept_at: DateTime<Utc>,
    pub woke_at: DateTime<Utc>,
}

/// Why a snapshot request failed. The message names the failing step so
/// `doctor` can report it.
#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    /// The app did not answer within the timeout (wedged or stalled).
    #[error("no reply within {} ms (the app may be wedged)", .0.as_millis())]
    Timeout(Duration),
    /// No app listening, or the connection failed.
    #[error("{0}")]
    Io(#[from] std::io::Error),
    /// The app answered with an empty or unparseable line.
    #[error("bad reply: {0}")]
    BadResponse(String),
}

/// Rate limiter for the "observation socket timed out" warning.
pub struct WarnLimiter {
    last_warn_secs: AtomicU64,
}

impl WarnLimiter {
    pub const fn new() -> Self {
        Self {
            last_warn_secs: AtomicU64::new(0),
        }
    }

    /// True when a warning should be emitted at `now_secs`.
    pub fn should_warn(&self, now_secs: u64) -> bool {
        let last = self.last_warn_secs.load(Ordering::Relaxed);
        if last != 0 && now_secs.saturating_sub(last) < WARN_INTERVAL_SECS {
            return false;
        }
        self.last_warn_secs
            .compare_exchange(last, now_secs.max(1), Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    }
}

impl Default for WarnLimiter {
    fn default() -> Self {
        Self::new()
    }
}

static TIMEOUT_WARNINGS: WarnLimiter = WarnLimiter::new();

/// `<data dir>/sockets/observation.sock`, served by ObservationServer.swift.
pub fn observation_socket_path() -> PathBuf {
    crate::status::socket_dir().join("observation.sock")
}

/// `<data dir>/sockets/capture.sock`, served by ScreenCaptureServer.swift.
pub fn capture_socket_path() -> PathBuf {
    crate::status::socket_dir().join("capture.sock")
}

fn parse_response<T: DeserializeOwned>(line: &str) -> Result<T, SnapshotError> {
    let line = line.trim();
    if line.is_empty() {
        return Err(SnapshotError::BadResponse("empty reply".to_string()));
    }
    serde_json::from_str(line).map_err(|e| SnapshotError::BadResponse(e.to_string()))
}

/// The observation server's reply to a `focused_element` request.
#[derive(Debug, Deserialize)]
struct FocusedElementReply {
    focused_element: Option<FocusedElement>,
}

/// The `focused_element` request line.  `capture_values` carries
/// `[privacy] capture_input_values`: without it the app reads no `AXValue`.
fn focused_element_request(capture_values: bool) -> Vec<u8> {
    let mut line = serde_json::json!({
        "command": "focused_element",
        "capture_values": capture_values,
    })
    .to_string()
    .into_bytes();
    line.push(b'\n');
    line
}

/// Async snapshot from the observation server at `path`.
///
/// The whole round trip is bounded by `timeout`, so a wedged app cannot
/// stall the observer tick. A timeout logs a warning, rate-limited by
/// `limiter`, so a stalled app is visible in the daemon log.
#[cfg(unix)]
pub async fn snapshot_at_async(
    path: &Path,
    timeout: Duration,
    limiter: &WarnLimiter,
) -> Result<ObservationSnapshot, SnapshotError> {
    request_at_async(path, SNAPSHOT_REQUEST, timeout, limiter).await
}

/// The focused UI element from the observation server at `path` (None when
/// nothing is focused, the element is a secure field, or Accessibility is
/// not granted).  Bounded like [`snapshot_at_async`].
#[cfg(unix)]
pub async fn focused_element_at_async(
    path: &Path,
    capture_values: bool,
    timeout: Duration,
    limiter: &WarnLimiter,
) -> Result<Option<FocusedElement>, SnapshotError> {
    let request = focused_element_request(capture_values);
    let reply: FocusedElementReply = request_at_async(path, &request, timeout, limiter).await?;
    Ok(reply.focused_element)
}

/// The focused UI element for the daemon's observer, `None` on any failure.
#[cfg(unix)]
pub async fn focused_element_async(capture_values: bool) -> Option<FocusedElement> {
    focused_element_at_async(
        &observation_socket_path(),
        capture_values,
        SNAPSHOT_TIMEOUT,
        &TIMEOUT_WARNINGS,
    )
    .await
    .ok()
    .flatten()
}

#[cfg(not(unix))]
pub async fn focused_element_async(_capture_values: bool) -> Option<FocusedElement> {
    None
}

/// One request line to the observation server and its one-line JSON reply.
#[cfg(unix)]
async fn request_at_async<T: DeserializeOwned>(
    path: &Path,
    request: &[u8],
    timeout: Duration,
    limiter: &WarnLimiter,
) -> Result<T, SnapshotError> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let round_trip = async {
        let mut stream = tokio::net::UnixStream::connect(path).await?;
        stream.write_all(request).await?;
        let mut response = String::new();
        BufReader::new(stream).read_line(&mut response).await?;
        parse_response(&response)
    };

    match tokio::time::timeout(timeout, round_trip).await {
        Ok(result) => result,
        Err(_) => {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            if limiter.should_warn(now) {
                tracing::warn!(
                    socket = %path.display(),
                    timeout_ms = timeout.as_millis() as u64,
                    "Observation socket did not answer in time; the app may be wedged \
                     (capture skipped this tick, warning repeats at most once a minute)"
                );
            }
            Err(SnapshotError::Timeout(timeout))
        }
    }
}

/// Snapshot of app-owned observation state for the daemon's async observer
/// loop, `None` on any failure. Bounded by [`SNAPSHOT_TIMEOUT`].
#[cfg(unix)]
pub async fn snapshot_async() -> Option<ObservationSnapshot> {
    snapshot_at_async(&observation_socket_path(), SNAPSHOT_TIMEOUT, &TIMEOUT_WARNINGS)
        .await
        .ok()
}

#[cfg(not(unix))]
pub async fn snapshot_async() -> Option<ObservationSnapshot> {
    None
}

/// Blocking snapshot from the observation server at `socket_path`.
///
/// Sends `{"command":"snapshot"}\n` and parses one JSON line back. Each
/// socket operation is bounded by [`SNAPSHOT_TIMEOUT`]; a read or write that
/// runs out of time is reported as [`SnapshotError::Timeout`].
#[cfg(unix)]
pub fn snapshot_at(socket_path: &Path) -> Result<ObservationSnapshot, SnapshotError> {
    use std::io::{BufRead, BufReader, ErrorKind, Write};
    use std::os::unix::net::UnixStream;

    let timed_out = |e: std::io::Error| match e.kind() {
        ErrorKind::WouldBlock | ErrorKind::TimedOut => SnapshotError::Timeout(SNAPSHOT_TIMEOUT),
        _ => SnapshotError::Io(e),
    };

    let mut stream = UnixStream::connect(socket_path)?;
    stream.set_read_timeout(Some(SNAPSHOT_TIMEOUT))?;
    stream.set_write_timeout(Some(SNAPSHOT_TIMEOUT))?;
    stream.write_all(SNAPSHOT_REQUEST).map_err(timed_out)?;

    let mut response = String::new();
    BufReader::new(stream)
        .read_line(&mut response)
        .map_err(timed_out)?;
    parse_response(&response)
}

#[cfg(not(unix))]
pub fn snapshot_at(_socket_path: &Path) -> Result<ObservationSnapshot, SnapshotError> {
    Err(SnapshotError::Io(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "observation socket requires a Unix platform",
    )))
}

/// Blocking snapshot from the default socket, `None` on any failure.
pub fn snapshot() -> Option<ObservationSnapshot> {
    snapshot_at(&observation_socket_path()).ok()
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixListener;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    const REPLY_OK: &str = "{\"accessibility_granted\":true,\"screen_recording_granted\":false,\
         \"secure_field_focused\":false,\"focused_window\":null,\
         \"display_topology\":[],\"cursor_global_px\":null}\n";

    fn serve_once(listener: UnixListener, reply: &'static str) -> std::thread::JoinHandle<String> {
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut line = String::new();
            BufReader::new(&stream).read_line(&mut line).unwrap();
            (&stream).write_all(reply.as_bytes()).unwrap();
            line
        })
    }

    // ---- blocking client (doctor, health, control) ----

    #[test]
    fn snapshot_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("observation.sock");
        let server = serve_once(UnixListener::bind(&sock).unwrap(), REPLY_OK);
        let snap = snapshot_at(&sock).unwrap();
        assert!(snap.accessibility_granted);
        assert!(!snap.screen_recording_granted);
        assert_eq!(server.join().unwrap(), "{\"command\":\"snapshot\"}\n");
    }

    #[test]
    fn missing_socket_is_an_io_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = snapshot_at(&dir.path().join("nope.sock")).unwrap_err();
        assert!(matches!(err, SnapshotError::Io(_)), "got {err:?}");
    }

    #[test]
    fn garbage_reply_is_bad_response() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("observation.sock");
        let server = serve_once(UnixListener::bind(&sock).unwrap(), "not json\n");
        let err = snapshot_at(&sock).unwrap_err();
        assert!(matches!(err, SnapshotError::BadResponse(_)), "got {err:?}");
        server.join().unwrap();
    }

    #[test]
    fn blocking_wedged_app_times_out_within_bound() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("observation.sock");
        let listener = UnixListener::bind(&sock).unwrap();
        // Accept and read the request, then hold the connection without answering.
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut line = String::new();
            BufReader::new(&stream).read_line(&mut line).unwrap();
            let _ = release_rx.recv_timeout(Duration::from_secs(30));
            drop(stream);
        });

        let started = Instant::now();
        let err = snapshot_at(&sock).unwrap_err();
        let elapsed = started.elapsed();
        release_tx.send(()).unwrap();
        server.join().unwrap();

        assert!(matches!(err, SnapshotError::Timeout(_)), "got {err:?}");
        assert!(elapsed < Duration::from_millis(2500), "blocked for {elapsed:?}");
    }

    // ---- async client (daemon observer tick) ----

    #[derive(Clone, Default)]
    struct Captured(Arc<Mutex<Vec<u8>>>);

    impl Write for Captured {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn wedged_app_times_out_within_bound_and_warns() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("observation.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        // Accept and hold the connection without ever answering.
        let _hold = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_secs(30)).await;
            drop(stream);
        });

        let logs = Captured::default();
        let writer = logs.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);

        let limiter = WarnLimiter::new();
        let started = Instant::now();
        let result = snapshot_at_async(&path, SNAPSHOT_TIMEOUT, &limiter).await;
        let elapsed = started.elapsed();

        assert!(matches!(result, Err(SnapshotError::Timeout(_))), "got {result:?}");
        assert!(elapsed < Duration::from_millis(2500), "tick blocked for {elapsed:?}");
        let text = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
        assert!(text.contains("WARN"), "no warning logged: {text}");
        assert!(text.contains("Observation socket did not answer in time"), "{text}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn answering_app_returns_snapshot() {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("observation.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            reader.read_line(&mut line).await.unwrap();
            assert_eq!(line, "{\"command\":\"snapshot\"}\n");
            let body = br#"{"accessibility_granted":true,"screen_recording_granted":true,"secure_field_focused":false,"focused_window":null,"display_topology":[],"cursor_global_px":null}"#;
            let stream = reader.get_mut();
            stream.write_all(body).await.unwrap();
            stream.write_all(b"\n").await.unwrap();
        });

        let snapshot = snapshot_at_async(&path, SNAPSHOT_TIMEOUT, &WarnLimiter::new())
            .await
            .expect("snapshot");
        assert!(snapshot.screen_recording_granted);
        assert!(!snapshot.secure_field_focused);
    }

    /// Serve one request with `reply`, returning the request line.
    fn serve_one_async(
        listener: tokio::net::UnixListener,
        reply: &'static str,
    ) -> tokio::task::JoinHandle<String> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            reader.read_line(&mut line).await.unwrap();
            let stream = reader.get_mut();
            stream.write_all(reply.as_bytes()).await.unwrap();
            stream.write_all(b"\n").await.unwrap();
            line
        })
    }

    #[tokio::test(flavor = "current_thread")]
    async fn focused_element_request_carries_the_value_flag_and_parses_the_reply() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("observation.sock");
        let server = serve_one_async(
            tokio::net::UnixListener::bind(&path).unwrap(),
            r#"{"focused_element":{"role":"AXCell","value":"42","column_header":"Qty","ancestors":[{"role":"AXRow","index":2},{"role":"AXTable","title":"Orders"}]}}"#,
        );

        let element = focused_element_at_async(&path, true, SNAPSHOT_TIMEOUT, &WarnLimiter::new())
            .await
            .expect("reply")
            .expect("an element");
        assert_eq!(element.role, "AXCell");
        assert_eq!(element.value.as_deref(), Some("42"));
        assert_eq!(element.column_header.as_deref(), Some("Qty"));
        assert_eq!(element.ancestors[0].index, Some(2));
        assert_eq!(element.ancestors[1].title.as_deref(), Some("Orders"));

        let request: serde_json::Value =
            serde_json::from_str(&server.await.unwrap()).unwrap();
        assert_eq!(request["command"], "focused_element");
        assert_eq!(request["capture_values"], true);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn no_focused_element_is_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("observation.sock");
        let server = serve_one_async(
            tokio::net::UnixListener::bind(&path).unwrap(),
            r#"{"focused_element":null}"#,
        );
        let element = focused_element_at_async(&path, false, SNAPSHOT_TIMEOUT, &WarnLimiter::new())
            .await
            .expect("reply");
        assert!(element.is_none());
        let request: serde_json::Value =
            serde_json::from_str(&server.await.unwrap()).unwrap();
        assert_eq!(request["capture_values"], false);
    }

    #[test]
    fn snapshot_parses_last_sleep_and_tolerates_its_absence() {
        let with: ObservationSnapshot = parse_response(
            r#"{"accessibility_granted":true,"screen_recording_granted":true,"secure_field_focused":false,"focused_window":null,"display_topology":[],"cursor_global_px":null,"last_sleep":{"slept_at":"2026-09-20T22:10:00Z","woke_at":"2026-09-21T07:02:30Z"}}"#,
        )
        .unwrap();
        let sleep = with.last_sleep.expect("last_sleep");
        assert_eq!(sleep.slept_at.to_rfc3339(), "2026-09-20T22:10:00+00:00");
        assert_eq!(sleep.woke_at.to_rfc3339(), "2026-09-21T07:02:30+00:00");

        let without: ObservationSnapshot = parse_response(REPLY_OK).unwrap();
        assert!(without.last_sleep.is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn missing_socket_is_io_error_not_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let result = snapshot_at_async(
            &dir.path().join("absent.sock"),
            SNAPSHOT_TIMEOUT,
            &WarnLimiter::new(),
        )
        .await;
        assert!(matches!(result, Err(SnapshotError::Io(_))));
    }

    #[test]
    fn warning_is_rate_limited_to_once_a_minute() {
        let limiter = WarnLimiter::new();
        assert!(limiter.should_warn(1_000));
        assert!(!limiter.should_warn(1_010));
        assert!(!limiter.should_warn(1_059));
        assert!(limiter.should_warn(1_060));
    }
}
