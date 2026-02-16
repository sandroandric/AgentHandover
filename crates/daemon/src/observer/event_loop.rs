use anyhow::Result;
use chrono::Utc;
use oc_apprentice_common::event::*;
use oc_apprentice_common::redaction::Redactor;
use oc_apprentice_storage::artifact_store::ArtifactStore;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time;
use tracing::{info, debug, warn, error};
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
    artifact_store: Option<Arc<ArtifactStore>>,
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
                        let mut event = make_event(
                            EventKind::AppSwitch {
                                from_app: from.clone(),
                                to_app: to.clone(),
                            },
                            &window,
                            &display_topology,
                            &primary_display_id,
                        );
                        redact_event(&mut event, &redactor);
                        let _ = tx.send(ObserverMessage::Event(event)).await;
                        dwell.on_manipulation_input();
                    }
                    last_app_id = current_app.clone();
                }

                // Title change detection
                if current_title != last_window_title {
                    if last_window_title.is_some() {
                        let mut event = make_event(
                            EventKind::WindowTitleChange,
                            &window,
                            &display_topology,
                            &primary_display_id,
                        );
                        redact_event(&mut event, &redactor);
                        let _ = tx.send(ObserverMessage::Event(event)).await;
                    }
                    last_window_title = current_title;
                }

                // 5. Dwell/scroll-read detection
                dwell.tick();

                if dwell.is_dwelling() {
                    let mut event = make_event(
                        EventKind::DwellSnapshot,
                        &window,
                        &display_topology,
                        &primary_display_id,
                    );

                    // Capture screenshot on dwell if enabled and under rate limit
                    if config.capture_screenshots
                        && screenshot_count_this_minute < config.screenshot_max_per_minute
                    {
                        if let Some(ref store) = artifact_store {
                            event.artifact_ids = capture_and_store_screenshot(store);
                            screenshot_count_this_minute += event.artifact_ids.len() as u32;
                        }
                    }

                    redact_event(&mut event, &redactor);
                    let _ = tx.send(ObserverMessage::Event(event)).await;
                }

                if dwell.is_scroll_reading() {
                    let mut event = make_event(
                        EventKind::ScrollReadSnapshot,
                        &window,
                        &display_topology,
                        &primary_display_id,
                    );

                    // Capture screenshot on scroll-read if enabled and under rate limit
                    if config.capture_screenshots
                        && screenshot_count_this_minute < config.screenshot_max_per_minute
                    {
                        if let Some(ref store) = artifact_store {
                            event.artifact_ids = capture_and_store_screenshot(store);
                            screenshot_count_this_minute += event.artifact_ids.len() as u32;
                        }
                    }

                    redact_event(&mut event, &redactor);
                    let _ = tx.send(ObserverMessage::Event(event)).await;
                }
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

/// Capture a screenshot and store it as an artifact.
/// Returns a vec with a UUID for the artifact on success, or empty vec on failure.
/// The UUID is used as the event's artifact_ids; the ArtifactStore's content-hash
/// based ID is logged for filesystem correlation.
#[cfg(target_os = "macos")]
fn capture_and_store_screenshot(store: &ArtifactStore) -> Vec<Uuid> {
    const MAX_RETRIES: u32 = 2;

    for attempt in 0..=MAX_RETRIES {
        match crate::capture::screenshot::capture_main_display() {
            Some((_width, _height, raw_pixels)) => {
                match store.store(&raw_pixels, "screenshot") {
                    Ok(artifact_id) => {
                        let uuid = Uuid::new_v4();
                        debug!(uuid = %uuid, store_id = %artifact_id, "Screenshot captured and stored");
                        return vec![uuid];
                    }
                    Err(e) => {
                        warn!(error = %e, "Failed to store screenshot artifact");
                        return vec![];
                    }
                }
            }
            None => {
                if attempt < MAX_RETRIES {
                    warn!(attempt = attempt + 1, "Screenshot capture returned None, retrying");
                    std::thread::sleep(std::time::Duration::from_millis(50));
                } else {
                    error!("Screenshot capture failed after {} retries (no permission?)", MAX_RETRIES + 1);
                }
            }
        }
    }
    vec![]
}

#[cfg(not(target_os = "macos"))]
fn capture_and_store_screenshot(_store: &ArtifactStore) -> Vec<Uuid> {
    vec![]
}

fn make_event(
    kind: EventKind,
    window: &Option<WindowInfo>,
    display_topology: &[DisplayInfo],
    primary_display_id: &str,
) -> Event {
    let display_ids_spanned = detect_spanning_displays(window, display_topology);

    // Get cursor position from CoreGraphics.
    #[cfg(target_os = "macos")]
    let cursor_global_px = crate::platform::window_capture::get_cursor_position();
    #[cfg(not(target_os = "macos"))]
    let cursor_global_px: Option<CursorPosition> = None;

    // Determine ui_scale based on which display the cursor is on.
    let ui_scale = cursor_global_px
        .as_ref()
        .and_then(|cursor| {
            #[cfg(target_os = "macos")]
            {
                crate::platform::window_capture::get_ui_scale_for_position(cursor, display_topology)
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = cursor;
                None
            }
        });

    Event {
        id: Uuid::new_v4(),
        timestamp: Utc::now(),
        kind,
        window: window.clone(),
        display_topology: display_topology.to_vec(),
        primary_display_id: primary_display_id.to_string(),
        cursor_global_px,
        ui_scale,
        artifact_ids: vec![],
        metadata: serde_json::json!({}),
        display_ids_spanned,
    }
}

/// Detect if a window spans multiple displays by checking if any of the
/// window's four corners fall within different displays' bounds.
/// Returns Some(vec) with the display IDs if spanning, None otherwise.
fn detect_spanning_displays(
    window: &Option<WindowInfo>,
    displays: &[DisplayInfo],
) -> Option<Vec<u32>> {
    let win = window.as_ref()?;
    if displays.len() < 2 {
        return None;
    }

    let [wx, wy, ww, wh] = win.bounds_global_px;

    // Guard against zero-size windows which cause underflow in corner calculation
    if ww == 0 || wh == 0 {
        return None;
    }

    // Window corners: top-left, top-right, bottom-left, bottom-right
    let corners = [
        (wx, wy),
        (wx + ww - 1, wy),
        (wx, wy + wh - 1),
        (wx + ww - 1, wy + wh - 1),
    ];

    let mut spanned_ids: Vec<u32> = Vec::new();

    for display in displays {
        let [dx, dy, dw, dh] = display.bounds_global_px;
        let contains_corner = corners.iter().any(|&(cx, cy)| {
            cx >= dx && cx < dx + dw && cy >= dy && cy < dy + dh
        });
        if contains_corner {
            if let Ok(id) = display.display_id.parse::<u32>() {
                if !spanned_ids.contains(&id) {
                    spanned_ids.push(id);
                }
            }
        }
    }

    if spanned_ids.len() > 1 {
        Some(spanned_ids)
    } else {
        None
    }
}

/// Apply the Redactor to all text fields in an Event that could contain secrets.
fn redact_event(event: &mut Event, redactor: &Redactor) {
    // Redact EventKind fields
    match &mut event.kind {
        EventKind::AppSwitch { from_app, to_app } => {
            *from_app = redactor.redact(from_app);
            *to_app = redactor.redact(to_app);
        }
        EventKind::ClickIntent { target_description } => {
            *target_description = redactor.redact(target_description);
        }
        EventKind::ClipboardChange { content_hash, .. } => {
            *content_hash = redactor.redact(content_hash);
        }
        EventKind::PasteDetected { matched_copy_hash } => {
            if let Some(hash) = matched_copy_hash {
                *hash = redactor.redact(hash);
            }
        }
        _ => {}
    }

    // Redact window title
    if let Some(ref mut win) = event.window {
        win.title = redactor.redact(&win.title);
    }

    // Redact metadata_json string values recursively
    event.metadata = redact_json_value(&event.metadata, redactor);
}

/// Recursively redact all string values in a serde_json::Value.
fn redact_json_value(value: &serde_json::Value, redactor: &Redactor) -> serde_json::Value {
    match value {
        serde_json::Value::String(s) => serde_json::Value::String(redactor.redact(s)),
        serde_json::Value::Object(map) => {
            let redacted: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), redact_json_value(v, redactor)))
                .collect();
            serde_json::Value::Object(redacted)
        }
        serde_json::Value::Array(arr) => {
            let redacted: Vec<serde_json::Value> = arr
                .iter()
                .map(|v| redact_json_value(v, redactor))
                .collect();
            serde_json::Value::Array(redacted)
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_event_with_kind(kind: EventKind) -> Event {
        Event {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            kind,
            window: Some(WindowInfo {
                window_id: "win1".into(),
                app_id: "com.app.Test".into(),
                title: "My Title".into(),
                bounds_global_px: [0, 0, 800, 600],
                z_order: 0,
                is_fullscreen: false,
            }),
            display_topology: vec![],
            primary_display_id: "d1".into(),
            cursor_global_px: None,
            ui_scale: None,
            artifact_ids: vec![],
            metadata: serde_json::json!({}),
            display_ids_spanned: None,
        }
    }

    #[test]
    fn test_redact_event_window_title_with_secret() {
        let redactor = Redactor::new();
        let mut event = make_test_event_with_kind(EventKind::FocusChange);
        event.window.as_mut().unwrap().title = "Edit: api_key=sk_live_ABC1234567890123456789".into();

        redact_event(&mut event, &redactor);

        let title = &event.window.as_ref().unwrap().title;
        assert!(!title.contains("sk_live_ABC1234567890123456789"), "Secret should be redacted from window title");
        assert!(title.contains("[REDACTED"), "Window title should contain redaction marker");
    }

    #[test]
    fn test_redact_event_app_switch_with_secret() {
        let redactor = Redactor::new();
        let mut event = make_test_event_with_kind(EventKind::AppSwitch {
            from_app: "com.app.Test ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl".into(),
            to_app: "com.app.Other".into(),
        });

        redact_event(&mut event, &redactor);

        if let EventKind::AppSwitch { from_app, .. } = &event.kind {
            assert!(!from_app.contains("ghp_"), "GitHub token should be redacted from from_app");
            assert!(from_app.contains("[REDACTED_GITHUB_TOKEN]"));
        } else {
            panic!("Expected AppSwitch kind");
        }
    }

    #[test]
    fn test_redact_event_click_intent_with_secret() {
        let redactor = Redactor::new();
        let mut event = make_test_event_with_kind(EventKind::ClickIntent {
            target_description: "Button with api_token=mySecretTokenValue1234".into(),
        });

        redact_event(&mut event, &redactor);

        if let EventKind::ClickIntent { target_description } = &event.kind {
            assert!(!target_description.contains("mySecretTokenValue1234"), "Secret should be redacted from click target");
        } else {
            panic!("Expected ClickIntent kind");
        }
    }

    #[test]
    fn test_redact_event_metadata_with_secret() {
        let redactor = Redactor::new();
        let mut event = make_test_event_with_kind(EventKind::FocusChange);
        event.metadata = serde_json::json!({
            "url": "https://example.com",
            "token": "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl",
            "nested": {
                "secret": "AKIAIOSFODNN7EXAMPLE1"
            }
        });

        redact_event(&mut event, &redactor);

        let meta = &event.metadata;
        assert!(!meta["token"].as_str().unwrap().contains("ghp_"), "GitHub token in metadata should be redacted");
        assert!(!meta["nested"]["secret"].as_str().unwrap().contains("AKIA"), "AWS key in nested metadata should be redacted");
        // Non-secret values should remain
        assert_eq!(meta["url"].as_str().unwrap(), "https://example.com");
    }

    #[test]
    fn test_redact_json_value_preserves_non_strings() {
        let redactor = Redactor::new();
        let input = serde_json::json!({
            "count": 42,
            "active": true,
            "tags": ["safe", "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl"]
        });

        let output = redact_json_value(&input, &redactor);

        assert_eq!(output["count"], 42);
        assert_eq!(output["active"], true);
        assert_eq!(output["tags"][0].as_str().unwrap(), "safe");
        assert!(output["tags"][1].as_str().unwrap().contains("[REDACTED_GITHUB_TOKEN]"));
    }

    #[test]
    fn test_redact_event_no_window_no_panic() {
        let redactor = Redactor::new();
        let mut event = Event {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            kind: EventKind::FocusChange,
            window: None,
            display_topology: vec![],
            primary_display_id: "d1".into(),
            cursor_global_px: None,
            ui_scale: None,
            artifact_ids: vec![],
            metadata: serde_json::json!({}),
            display_ids_spanned: None,
        };

        // Should not panic with no window
        redact_event(&mut event, &redactor);
        assert!(event.window.is_none());
    }

    #[test]
    fn test_detect_spanning_single_display() {
        let window = Some(WindowInfo {
            window_id: "w1".into(),
            app_id: "com.app.Test".into(),
            title: "Test".into(),
            bounds_global_px: [100, 100, 800, 600],
            z_order: 0,
            is_fullscreen: false,
        });
        let displays = vec![DisplayInfo {
            display_id: "1".into(),
            bounds_global_px: [0, 0, 2560, 1440],
            scale_factor: 2.0,
            orientation: 0,
        }];

        let result = detect_spanning_displays(&window, &displays);
        assert!(result.is_none(), "Single display should not report spanning");
    }

    #[test]
    fn test_detect_spanning_across_two_displays() {
        let window = Some(WindowInfo {
            window_id: "w1".into(),
            app_id: "com.app.Test".into(),
            title: "Test".into(),
            // Window straddles the boundary between two side-by-side displays
            bounds_global_px: [2400, 100, 400, 600],
            z_order: 0,
            is_fullscreen: false,
        });
        let displays = vec![
            DisplayInfo {
                display_id: "1".into(),
                bounds_global_px: [0, 0, 2560, 1440],
                scale_factor: 2.0,
                orientation: 0,
            },
            DisplayInfo {
                display_id: "2".into(),
                bounds_global_px: [2560, 0, 1920, 1080],
                scale_factor: 1.0,
                orientation: 0,
            },
        ];

        let result = detect_spanning_displays(&window, &displays);
        assert!(result.is_some(), "Window spanning two displays should be detected");
        let ids = result.unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    #[test]
    fn test_detect_spanning_window_within_single_display_multi_monitor() {
        let window = Some(WindowInfo {
            window_id: "w1".into(),
            app_id: "com.app.Test".into(),
            title: "Test".into(),
            bounds_global_px: [100, 100, 400, 300],
            z_order: 0,
            is_fullscreen: false,
        });
        let displays = vec![
            DisplayInfo {
                display_id: "1".into(),
                bounds_global_px: [0, 0, 2560, 1440],
                scale_factor: 2.0,
                orientation: 0,
            },
            DisplayInfo {
                display_id: "2".into(),
                bounds_global_px: [2560, 0, 1920, 1080],
                scale_factor: 1.0,
                orientation: 0,
            },
        ];

        let result = detect_spanning_displays(&window, &displays);
        assert!(result.is_none(), "Window within a single display should not report spanning");
    }

    #[test]
    fn test_detect_spanning_no_window() {
        let displays = vec![DisplayInfo {
            display_id: "1".into(),
            bounds_global_px: [0, 0, 2560, 1440],
            scale_factor: 2.0,
            orientation: 0,
        }];

        let result = detect_spanning_displays(&None, &displays);
        assert!(result.is_none(), "No window should return None");
    }
}
