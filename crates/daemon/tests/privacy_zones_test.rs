//! The observer loop evaluates privacy before the shutter: a failed
//! observation snapshot, a blocked app or title, a blocked page URL and a
//! metadata-only app never cause a capture request to reach the app.
//!
//! Each test runs the real `run_observer_loop` against a fake observation
//! socket (scripted window sequence) and a fake capture socket that counts
//! the capture requests it receives. Everything lives in a temp dir: no
//! real data dir, socket or app is touched.

use agenthandover_common::browser_transition::{
    write_browser_transition, BrowserTransition, BrowserTransitionKind,
};
use agenthandover_common::config::PrivacyZonesConfig;
use agenthandover_common::event::{Event, EventKind};
use agenthandover_daemon::observer::event_loop::{run_observer_loop, ObserverConfig, ObserverMessage};
use agenthandover_storage::artifact_store::ArtifactStore;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;
use tokio::sync::{mpsc, watch};

fn window_reply(app_id: &str, title: &str) -> String {
    serde_json::json!({
        "accessibility_granted": true,
        "screen_recording_granted": true,
        "secure_field_focused": false,
        "focused_window": {
            "window_id": "1",
            "app_id": app_id,
            "title": title,
            "bounds_global_px": [0, 0, 800, 600],
            "z_order": 0,
            "is_fullscreen": false
        },
        "display_topology": [],
        "cursor_global_px": null
    })
    .to_string()
}

/// Serve `replies` in order, one per connection; the last one repeats.
fn fake_observation_server(path: &Path, replies: Vec<String>) {
    let listener = UnixListener::bind(path).unwrap();
    tokio::spawn(async move {
        let mut i = 0usize;
        loop {
            let Ok((stream, _)) = listener.accept().await else { return };
            let reply = replies[i.min(replies.len() - 1)].clone();
            i += 1;
            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            let _ = reader.read_line(&mut line).await;
            let stream = reader.get_mut();
            let _ = stream.write_all(reply.as_bytes()).await;
            let _ = stream.write_all(b"\n").await;
        }
    });
}

/// Count capture requests; answer each by closing the connection (the
/// capture fails, which is fine: the count is what the tests check).
fn fake_capture_server(path: &Path) -> Arc<AtomicUsize> {
    let listener = UnixListener::bind(path).unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&count);
    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else { return };
            seen.fetch_add(1, Ordering::SeqCst);
            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            let _ = reader.read_line(&mut line).await;
        }
    });
    count
}

struct Harness {
    _dir: tempfile::TempDir,
    root: PathBuf,
    config: ObserverConfig,
    captures: Arc<AtomicUsize>,
}

fn harness(zones: PrivacyZonesConfig) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let state_dir = root.join("state");
    std::fs::create_dir_all(&state_dir).unwrap();
    let captures = fake_capture_server(&root.join("capture.sock"));
    let config = ObserverConfig {
        poll_interval: Duration::from_millis(40),
        screenshots_dir: root.join("screenshots"),
        db_path: root.join("events.db"),
        zones,
        observation_socket: root.join("observation.sock"),
        capture_socket: root.join("capture.sock"),
        state_dir,
        ..Default::default()
    };
    Harness { _dir: dir, root, config, captures }
}

/// Run the loop for `run_for`, then shut it down; return every event sent.
async fn run_loop(config: ObserverConfig, root: &Path, run_for: Duration) -> Vec<Event> {
    let (tx, mut rx) = mpsc::channel(1000);
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let store = Arc::new(ArtifactStore::new(root.join("artifacts"), [7u8; 32]));
    let handle = tokio::spawn(run_observer_loop(
        config,
        tx,
        shutdown_rx,
        Some(store),
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    ));
    tokio::time::sleep(run_for).await;
    shutdown_tx.send(true).unwrap();
    handle.await.unwrap().unwrap();
    let mut events = Vec::new();
    while let Some(ObserverMessage::Event { event }) = rx.recv().await {
        events.push(event);
    }
    events
}

fn app_switches(events: &[Event]) -> Vec<(String, String, String)> {
    events
        .iter()
        .filter_map(|e| match &e.kind {
            EventKind::AppSwitch { from_app, to_app } => Some((
                from_app.clone(),
                to_app.clone(),
                e.window.as_ref().map(|w| w.title.clone()).unwrap_or_default(),
            )),
            _ => None,
        })
        .collect()
}

fn title_changes(events: &[Event]) -> usize {
    events
        .iter()
        .filter(|e| matches!(e.kind, EventKind::WindowTitleChange))
        .count()
}

fn frames(events: &[Event]) -> usize {
    events
        .iter()
        .filter(|e| matches!(e.kind, EventKind::DwellSnapshot))
        .count()
}

#[derive(Clone, Default)]
struct Logs(Arc<Mutex<Vec<u8>>>);

impl Write for Logs {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// #132: with observation.sock absent the tick is unsafe (the secure-field
/// check cannot run): nothing is captured, no transition is fabricated, and
/// the failure is warned once (rate-limited), naming the failing step.
#[tokio::test(flavor = "current_thread")]
async fn missing_observation_socket_captures_nothing_and_warns_once() {
    let logs = Logs::default();
    let writer = logs.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(move || writer.clone())
        .with_ansi(false)
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);

    let h = harness(PrivacyZonesConfig::default());
    // A browser transition would fire the shutter if the tick went ahead.
    write_browser_transition(
        &h.config.state_dir,
        &BrowserTransition {
            kind: BrowserTransitionKind::Navigation,
            at: chrono::Utc::now() + chrono::Duration::milliseconds(100),
            url: Some("https://example.com/".into()),
        },
    )
    .unwrap();

    let events = run_loop(h.config, &h.root, Duration::from_millis(600)).await;

    assert_eq!(h.captures.load(Ordering::SeqCst), 0, "no capture request");
    assert_eq!(frames(&events), 0);
    assert_eq!(title_changes(&events), 0);
    assert!(events.is_empty(), "no events at all: {events:?}");
    let text = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
    assert_eq!(
        text.matches("Observation snapshot unavailable").count(),
        1,
        "one rate-limited warning over ~15 failed ticks: {text}"
    );
    assert!(text.contains("step: connect"), "{text}");
}

/// A window from a blocked app (the default password-manager list): the
/// switch into it is recorded without its title, and no capture request
/// is ever sent while it is focused.
#[tokio::test(flavor = "current_thread")]
async fn blocked_app_never_reaches_the_shutter() {
    let h = harness(PrivacyZonesConfig::default());
    fake_observation_server(
        &h.config.observation_socket,
        vec![
            window_reply("pid:10:TextEdit", "notes.txt"),
            window_reply("pid:11:1Password", "Vault — Bank logins"),
            window_reply("pid:11:1Password", "Vault — Bank logins — edit"),
        ],
    );
    let events = run_loop(h.config, &h.root, Duration::from_millis(500)).await;

    assert_eq!(h.captures.load(Ordering::SeqCst), 0, "no capture request for 1Password");
    assert_eq!(frames(&events), 0);
    assert_eq!(
        app_switches(&events),
        vec![("pid:10:TextEdit".into(), "pid:11:1Password".into(), String::new())],
        "the switch is metadata only, title withheld"
    );
    assert_eq!(title_changes(&events), 0, "a blocked window's titles are not recorded");
}

/// Control for the test above: the same switch into an unblocked app
/// fires the shutter (so the zero above is the zone, not the harness).
#[tokio::test(flavor = "current_thread")]
async fn unblocked_app_switch_requests_a_frame() {
    let h = harness(PrivacyZonesConfig::default());
    fake_observation_server(
        &h.config.observation_socket,
        vec![
            window_reply("pid:10:TextEdit", "notes.txt"),
            window_reply("pid:12:Preview", "invoice.pdf"),
        ],
    );
    let events = run_loop(h.config, &h.root, Duration::from_millis(500)).await;
    assert!(h.captures.load(Ordering::SeqCst) >= 1, "the app switch requested a frame");
    assert_eq!(app_switches(&events).len(), 1);
    assert_eq!(app_switches(&events)[0].2, "invoice.pdf");
}

/// `blocked` patterns from config.toml match the title too, case-folded.
#[tokio::test(flavor = "current_thread")]
async fn configured_title_pattern_blocks() {
    let h = harness(PrivacyZonesConfig {
        blocked: vec!["*payslip*".into()],
        ..Default::default()
    });
    fake_observation_server(
        &h.config.observation_socket,
        vec![
            window_reply("pid:10:TextEdit", "notes.txt"),
            window_reply("pid:12:Preview", "PAYSLIP-2026-08.pdf"),
        ],
    );
    let events = run_loop(h.config, &h.root, Duration::from_millis(500)).await;
    assert_eq!(h.captures.load(Ordering::SeqCst), 0);
    assert_eq!(frames(&events), 0);
}

/// A metadata-only app keeps its switch and title events but never gets a
/// frame.
#[tokio::test(flavor = "current_thread")]
async fn metadata_only_app_gets_events_but_no_frame() {
    let h = harness(PrivacyZonesConfig {
        metadata_only: vec!["Messages".into()],
        ..Default::default()
    });
    fake_observation_server(
        &h.config.observation_socket,
        vec![
            window_reply("pid:10:TextEdit", "notes.txt"),
            window_reply("pid:13:Messages", "Alice"),
            window_reply("pid:13:Messages", "Bob"),
        ],
    );
    let events = run_loop(h.config, &h.root, Duration::from_millis(500)).await;
    assert_eq!(h.captures.load(Ordering::SeqCst), 0);
    assert_eq!(frames(&events), 0);
    assert_eq!(app_switches(&events)[0].2, "Alice", "metadata keeps the title");
    let titles: Vec<String> = events
        .iter()
        .filter(|e| matches!(e.kind, EventKind::WindowTitleChange))
        .filter_map(|e| e.window.as_ref().map(|w| w.title.clone()))
        .collect();
    assert_eq!(titles, vec!["Alice".to_string(), "Bob".to_string()]);
}

/// The browser's page URL (handed over by the native-messaging bridge) is
/// checked against `blocked_urls` before the shutter.
#[tokio::test(flavor = "current_thread")]
async fn blocked_page_url_in_the_browser_is_not_captured() {
    for (url, expect_capture) in [
        ("https://vault.example.com/items", false),
        ("https://docs.example.com/guide", true),
    ] {
        let h = harness(PrivacyZonesConfig {
            blocked_urls: vec!["*vault.example.com*".into()],
            ..Default::default()
        });
        write_browser_transition(
            &h.config.state_dir,
            &BrowserTransition {
                kind: BrowserTransitionKind::Navigation,
                at: chrono::Utc::now() + chrono::Duration::milliseconds(50),
                url: Some(url.into()),
            },
        )
        .unwrap();
        fake_observation_server(
            &h.config.observation_socket,
            vec![window_reply("pid:20:Google Chrome", "Example")],
        );
        let _ = run_loop(h.config, &h.root, Duration::from_millis(500)).await;
        let captured = h.captures.load(Ordering::SeqCst) >= 1;
        assert_eq!(captured, expect_capture, "url {url}");
    }
}

/// #132: a failed tick between two identical snapshots leaves the last
/// app and title alone, so no transition (and no frame) is fabricated.
#[tokio::test(flavor = "current_thread")]
async fn a_failed_tick_does_not_fabricate_a_transition() {
    let h = harness(PrivacyZonesConfig::default());
    fake_observation_server(
        &h.config.observation_socket,
        vec![
            window_reply("pid:10:TextEdit", "notes.txt"),
            "not json".to_string(),
            "not json".to_string(),
            window_reply("pid:10:TextEdit", "notes.txt"),
        ],
    );
    let events = run_loop(h.config, &h.root, Duration::from_millis(500)).await;
    assert!(events.is_empty(), "no events: {events:?}");
    assert_eq!(h.captures.load(Ordering::SeqCst), 0);
}
