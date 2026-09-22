use agenthandover_common::event::{CursorPosition, DisplayInfo, WindowInfo};
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Deserialize)]
pub struct ObservationSnapshot {
    pub accessibility_granted: bool,
    pub screen_recording_granted: bool,
    pub secure_field_focused: bool,
    pub focused_window: Option<WindowInfo>,
    pub display_topology: Vec<DisplayInfo>,
    pub cursor_global_px: Option<CursorPosition>,
}

/// Upper bound for one snapshot round trip (connect + request + response).
pub const SNAPSHOT_TIMEOUT: Duration = Duration::from_secs(2);

/// A wedged app is reported at most once per this interval.
const WARN_INTERVAL_SECS: u64 = 60;

const SNAPSHOT_REQUEST: &[u8] = b"{\"command\":\"snapshot\"}\n";

#[derive(Debug)]
pub enum SnapshotError {
    /// The app did not answer within the timeout (wedged or stalled).
    Timeout,
    /// No app listening, or the connection failed.
    Io(std::io::Error),
    /// The app answered with an empty or unparseable line.
    BadResponse,
}

/// Rate limiter for the "observation socket timed out" warning.
pub struct WarnLimiter {
    last_warn_secs: AtomicU64,
}

impl WarnLimiter {
    pub const fn new() -> Self {
        Self { last_warn_secs: AtomicU64::new(0) }
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

fn socket_path() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join("Library/Application Support/agenthandover/observation.sock"))
}

fn parse_response(line: &str) -> Result<ObservationSnapshot, SnapshotError> {
    let line = line.trim();
    if line.is_empty() {
        return Err(SnapshotError::BadResponse);
    }
    serde_json::from_str(line).map_err(|_| SnapshotError::BadResponse)
}

/// Async snapshot for the observer loop: the whole round trip is bounded by
/// `timeout`, so a wedged app cannot stall the tick. A timeout logs a
/// rate-limited warning so a stalled app is visible in the daemon log.
#[cfg(unix)]
pub async fn snapshot_at(
    path: &std::path::Path,
    timeout: Duration,
    limiter: &WarnLimiter,
) -> Result<ObservationSnapshot, SnapshotError> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let round_trip = async {
        let mut stream = tokio::net::UnixStream::connect(path)
            .await
            .map_err(SnapshotError::Io)?;
        stream.write_all(SNAPSHOT_REQUEST).await.map_err(SnapshotError::Io)?;
        let mut response = String::new();
        BufReader::new(stream)
            .read_line(&mut response)
            .await
            .map_err(SnapshotError::Io)?;
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
            Err(SnapshotError::Timeout)
        }
    }
}

/// Snapshot of app-owned observation state for the async observer loop.
#[cfg(unix)]
pub async fn snapshot_async() -> Option<ObservationSnapshot> {
    let path = socket_path()?;
    snapshot_at(&path, SNAPSHOT_TIMEOUT, &TIMEOUT_WARNINGS).await.ok()
}

#[cfg(not(unix))]
pub async fn snapshot_async() -> Option<ObservationSnapshot> {
    None
}

/// Blocking snapshot for synchronous callers (health and control replies).
/// Each socket operation is bounded by `SNAPSHOT_TIMEOUT`.
#[cfg(unix)]
pub fn snapshot() -> Option<ObservationSnapshot> {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;

    let mut stream = UnixStream::connect(socket_path()?).ok()?;
    stream.set_read_timeout(Some(SNAPSHOT_TIMEOUT)).ok()?;
    stream.set_write_timeout(Some(SNAPSHOT_TIMEOUT)).ok()?;
    stream.write_all(SNAPSHOT_REQUEST).ok()?;

    let mut response = String::new();
    BufReader::new(stream).read_line(&mut response).ok()?;
    parse_response(&response).ok()
}

#[cfg(not(unix))]
pub fn snapshot() -> Option<ObservationSnapshot> {
    None
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

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
        let result = snapshot_at(&path, SNAPSHOT_TIMEOUT, &limiter).await;
        let elapsed = started.elapsed();

        assert!(matches!(result, Err(SnapshotError::Timeout)), "got {result:?}");
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

        let snapshot = snapshot_at(&path, SNAPSHOT_TIMEOUT, &WarnLimiter::new())
            .await
            .expect("snapshot");
        assert!(snapshot.screen_recording_granted);
        assert!(!snapshot.secure_field_focused);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn missing_socket_is_io_error_not_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let result =
            snapshot_at(&dir.path().join("absent.sock"), SNAPSHOT_TIMEOUT, &WarnLimiter::new()).await;
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
