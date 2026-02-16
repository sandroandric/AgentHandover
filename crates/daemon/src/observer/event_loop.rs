use anyhow::Result;
use chrono::Utc;
use oc_apprentice_common::event::*;
use oc_apprentice_common::redaction::Redactor;
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time;
use tracing::{info, debug, error};
use uuid::Uuid;

/// Messages sent from the observer to the storage writer.
#[derive(Debug)]
pub enum ObserverMessage {
    Event(Event),
    Shutdown,
}

/// Configuration for the observer event loop.
pub struct ObserverConfig {
    pub t_dwell_seconds: u64,
    pub t_scroll_read_seconds: u64,
    pub capture_screenshots: bool,
    pub screenshot_max_per_minute: u32,
    pub poll_interval: Duration,
    pub db_path: PathBuf,
}

impl Default for ObserverConfig {
    fn default() -> Self {
        Self {
            t_dwell_seconds: 3,
            t_scroll_read_seconds: 8,
            capture_screenshots: true,
            screenshot_max_per_minute: 20,
            poll_interval: Duration::from_millis(500),
            db_path: PathBuf::from("openmimic.db"),
        }
    }
}

/// The observer event loop — runs as an async task.
/// Polls OS state at regular intervals and emits events.
pub async fn run_observer_loop(
    config: ObserverConfig,
    tx: mpsc::Sender<ObserverMessage>,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> Result<()> {
    info!("Observer event loop starting");

    let redactor = Redactor::new();
    let mut dwell = super::dwell::DwellTracker::new(
        Duration::from_secs(config.t_dwell_seconds),
        Duration::from_secs(config.t_scroll_read_seconds),
    );

    let mut last_window_title: Option<String> = None;
    let mut last_app_id: Option<String> = None;
    let mut screenshot_count_this_minute: u32 = 0;
    let mut last_minute_reset = Utc::now();

    let mut interval = time::interval(config.poll_interval);

    loop {
        tokio::select! {
            _ = interval.tick() => {
                // Check shutdown
                if *shutdown_rx.borrow() {
                    info!("Observer loop received shutdown signal");
                    let _ = tx.send(ObserverMessage::Shutdown).await;
                    break;
                }

                // Reset screenshot counter every minute
                let now = Utc::now();
                if (now - last_minute_reset).num_seconds() >= 60 {
                    screenshot_count_this_minute = 0;
                    last_minute_reset = now;
                }

                // 1. Check for secure field — if focused, skip all capture
                #[cfg(target_os = "macos")]
                {
                    if crate::platform::accessibility::is_secure_field_focused() {
                        debug!("Secure field focused — skipping capture");
                        continue;
                    }
                }

                // 2. Get display topology
                #[cfg(target_os = "macos")]
                let display_topology = crate::platform::window_capture::get_display_topology();
                #[cfg(not(target_os = "macos"))]
                let display_topology: Vec<DisplayInfo> = vec![];

                let primary_display_id = display_topology
                    .first()
                    .map(|d| d.display_id.clone())
                    .unwrap_or_else(|| "unknown".to_string());

                // 3. Get focused window
                #[cfg(target_os = "macos")]
                let window = crate::platform::window_capture::get_focused_window();
                #[cfg(not(target_os = "macos"))]
                let window: Option<WindowInfo> = None;

                // 4. Check for focus/title changes
                let current_title = window.as_ref().map(|w| w.title.clone());
                let current_app = window.as_ref().map(|w| w.app_id.clone());

                // App switch detection
                if current_app != last_app_id {
                    if let (Some(from), Some(to)) = (&last_app_id, &current_app) {
                        let event = make_event(
                            EventKind::AppSwitch {
                                from_app: redactor.redact(from),
                                to_app: redactor.redact(to),
                            },
                            &window,
                            &display_topology,
                            &primary_display_id,
                        );
                        let _ = tx.send(ObserverMessage::Event(event)).await;
                        dwell.on_manipulation_input();
                    }
                    last_app_id = current_app.clone();
                }

                // Title change detection
                if current_title != last_window_title {
                    if last_window_title.is_some() {
                        let event = make_event(
                            EventKind::WindowTitleChange,
                            &window,
                            &display_topology,
                            &primary_display_id,
                        );
                        let _ = tx.send(ObserverMessage::Event(event)).await;
                    }
                    last_window_title = current_title;
                }

                // 5. Dwell/scroll-read detection
                dwell.tick();

                if dwell.is_dwelling() {
                    let event = make_event(
                        EventKind::DwellSnapshot,
                        &window,
                        &display_topology,
                        &primary_display_id,
                    );
                    let _ = tx.send(ObserverMessage::Event(event)).await;
                }

                if dwell.is_scroll_reading() {
                    let event = make_event(
                        EventKind::ScrollReadSnapshot,
                        &window,
                        &display_topology,
                        &primary_display_id,
                    );
                    let _ = tx.send(ObserverMessage::Event(event)).await;
                }

                // Suppress unused variable warnings for screenshot rate limiter
                // (will be used when screenshot capture is wired in)
                let _ = screenshot_count_this_minute;
                let _ = config.screenshot_max_per_minute;
                let _ = config.capture_screenshots;
            }
            _ = shutdown_rx.changed() => {
                info!("Observer loop: shutdown watch triggered");
                let _ = tx.send(ObserverMessage::Shutdown).await;
                break;
            }
        }
    }

    info!("Observer event loop stopped");
    Ok(())
}

/// Storage writer task — receives events and writes them to SQLite.
pub async fn run_storage_writer(
    db_path: PathBuf,
    mut rx: mpsc::Receiver<ObserverMessage>,
) -> Result<()> {
    info!(path = %db_path.display(), "Storage writer starting");

    let store = oc_apprentice_storage::EventStore::open(&db_path)?;
    let mut event_count = 0u64;

    while let Some(msg) = rx.recv().await {
        match msg {
            ObserverMessage::Event(event) => {
                if let Err(e) = store.insert_event(&event) {
                    error!(error = %e, "Failed to insert event");
                } else {
                    event_count += 1;
                    if event_count % 100 == 0 {
                        debug!(event_count, "Events stored");
                    }
                }
            }
            ObserverMessage::Shutdown => {
                info!(event_count, "Storage writer shutting down");
                break;
            }
        }
    }

    Ok(())
}

fn make_event(
    kind: EventKind,
    window: &Option<WindowInfo>,
    display_topology: &[DisplayInfo],
    primary_display_id: &str,
) -> Event {
    Event {
        id: Uuid::new_v4(),
        timestamp: Utc::now(),
        kind,
        window: window.clone(),
        display_topology: display_topology.to_vec(),
        primary_display_id: primary_display_id.to_string(),
        cursor_global_px: None, // Will be populated with actual cursor tracking
        ui_scale: None,
        artifact_ids: vec![],
        metadata: serde_json::json!({}),
    }
}
