use anyhow::Result;
use chrono::Utc;
use oc_apprentice_common::event::*;
use oc_apprentice_common::redaction::Redactor;
use oc_apprentice_storage::artifact_store::{ArtifactMeta, ArtifactStore};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time;
use tracing::{info, debug, warn, error};
use uuid::Uuid;

/// Messages sent from the observer to the storage writer.
///
/// The storage writer exits when all senders are dropped (channel close),
/// so there is no explicit Shutdown variant — proper teardown ordering in
/// `main.rs` guarantees all producers are stopped before we await the
/// writer task.
pub enum ObserverMessage {
    Event {
        event: Event,
        /// Metadata for artifacts associated with this event, so the storage
        /// writer can insert rows into the `artifacts` table.
        artifacts: Vec<ArtifactMeta>,
    },
}

/// Configuration for the observer event loop.
pub struct ObserverConfig {
    pub t_dwell_seconds: u64,
    pub t_scroll_read_seconds: u64,
    pub capture_screenshots: bool,
    pub screenshot_max_per_minute: u32,
    pub poll_interval: Duration,
    pub db_path: PathBuf,
    /// dHash perceptual hash threshold for screenshot change detection.
    /// Lower = stricter dedup (fewer screenshots stored).
    pub dhash_threshold: u32,
    /// Screenshot format: "jpeg" or "png".
    pub screenshot_format: String,
    /// JPEG quality 1-100 (default: 70). Only used when format = "jpeg".
    pub screenshot_quality: u8,
    /// Scale factor for VLM screenshots (default: 0.5 = half resolution).
    pub screenshot_scale: f64,
    /// Directory for plain JPEG screenshots (VLM annotation pipeline).
    pub screenshots_dir: PathBuf,
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
            dhash_threshold: 10,
            screenshot_format: "jpeg".to_string(),
            screenshot_quality: 70,
            screenshot_scale: 0.5,
            screenshots_dir: PathBuf::from("/tmp/openmimic-screenshots"),
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
    let mut last_dhash: u64 = 0;

    // Idle detector: uses CGEventSourceSecondsSinceLastEventType to detect
    // keyboard/mouse/trackpad activity. We use this to reset the dwell tracker
    // when the user is actively interacting, so DwellSnapshots fire after each
    // idle period (not just once after the first AppSwitch).
    #[cfg(target_os = "macos")]
    let idle_detector = crate::platform::IdleDetector::new();
    // Track whether we last saw the user as active, to avoid spamming on_manipulation_input
    #[cfg(target_os = "macos")]
    let mut user_was_active = true;

    // Focus recording: capture screenshots aggressively (every ~2s with dHash dedup)
    // instead of waiting for 3s idle dwell threshold.
    let focus_capture_interval = Duration::from_secs(2);
    let mut last_focus_capture = Instant::now();

    // Focus recording session tracking
    let state_dir = oc_apprentice_common::status::data_dir();
    let mut active_focus_session_id: Option<String> = None;

    // AppleScript throttle: at most once per 2 seconds per app
    #[cfg(target_os = "macos")]
    let mut applescript_last_query: HashMap<String, Instant> = HashMap::new();
    #[cfg(target_os = "macos")]
    let applescript_throttle = Duration::from_secs(2);

    let mut interval = time::interval(config.poll_interval);

    loop {
        tokio::select! {
            _ = interval.tick() => {
                // Check shutdown
                if *shutdown_rx.borrow() {
                    info!("Observer loop received shutdown signal");
                    break;
                }

                // Check for focus recording session signal (cheap stat() call)
                match oc_apprentice_common::focus_session::read_focus_signal(&state_dir) {
                    Some(signal) if signal.is_recording() => {
                        if active_focus_session_id.as_deref() != Some(&signal.session_id) {
                            info!(
                                session_id = %signal.session_id,
                                title = %signal.title,
                                "Focus recording session started"
                            );
                            active_focus_session_id = Some(signal.session_id);
                        }
                    }
                    Some(signal) if signal.is_stopped() => {
                        if active_focus_session_id.is_some() {
                            info!(
                                session_id = %signal.session_id,
                                "Focus recording session stopped"
                            );
                            active_focus_session_id = None;
                        }
                    }
                    _ => {
                        // No signal file or unknown status — clear active session
                        if active_focus_session_id.is_some() {
                            debug!("Focus session signal file removed, clearing active session");
                            active_focus_session_id = None;
                        }
                    }
                }

                // Reset screenshot counter every minute
                let now = Utc::now();
                if (now - last_minute_reset).num_seconds() >= 60 {
                    screenshot_count_this_minute = 0;
                    last_minute_reset = now;

                    // Evict stale AppleScript throttle entries to bound map size
                    #[cfg(target_os = "macos")]
                    {
                        let stale_cutoff = Duration::from_secs(30);
                        applescript_last_query.retain(|_, ts| ts.elapsed() < stale_cutoff);
                    }
                }

                // 1. Check for secure field — if focused, skip all capture.
                // Uses the async version with spawn_blocking + 100ms timeout
                // to prevent AX API deadlocks (Mach IPC hangs).
                #[cfg(target_os = "macos")]
                {
                    if crate::platform::accessibility::is_secure_field_focused_async().await {
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

                        // Query AppleScript state for known native apps
                        #[cfg(target_os = "macos")]
                        {
                            let should_query = if let Some(last) = applescript_last_query.get(to.as_str()) {
                                last.elapsed() >= applescript_throttle
                            } else {
                                true
                            };

                            if should_query {
                                // Try by bundle ID first, then by app name (async, off executor)
                                let app_name_to_query: Option<String> = if crate::platform::applescript::is_supported_bundle_id(to) {
                                    crate::platform::applescript::app_name_for_bundle_id(to)
                                        .map(|n| n.to_string())
                                } else {
                                    let short_name = to.rsplit('.').next().unwrap_or(to);
                                    if crate::platform::applescript::is_supported_app(short_name) {
                                        Some(short_name.to_string())
                                    } else {
                                        None
                                    }
                                };
                                let app_state = if let Some(name) = app_name_to_query {
                                    crate::platform::applescript::query_app_state_async(name).await
                                } else {
                                    None
                                };

                                if let Some(state) = app_state {
                                    if let Ok(state_json) = serde_json::to_value(&state) {
                                        event.metadata["app_state"] = state_json;
                                    }
                                }
                                applescript_last_query.insert(to.clone(), Instant::now());
                            }
                        }

                        tag_focus_session(&mut event, &active_focus_session_id);
                        redact_event(&mut event, &redactor);
                        let _ = tx.send(ObserverMessage::Event { event, artifacts: vec![] }).await;
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
                        tag_focus_session(&mut event, &active_focus_session_id);
                        redact_event(&mut event, &redactor);
                        let _ = tx.send(ObserverMessage::Event { event, artifacts: vec![] }).await;
                    }
                    last_window_title = current_title;
                }

                // 5. Detect user activity via HID idle time and reset dwell tracker
                // This ensures DwellSnapshots fire after EVERY idle period, not just
                // after an AppSwitch. Without this, dwell_fired stays true forever
                // when the user stays in one app.
                #[cfg(target_os = "macos")]
                {
                    let idle_secs = idle_detector.seconds_since_last_input();
                    let is_active = idle_secs < config.t_dwell_seconds as f64;
                    if is_active && !user_was_active {
                        // User just became active again after being idle
                        // Reset dwell tracker so next idle period can fire
                        dwell.on_manipulation_input();
                    }
                    user_was_active = is_active;
                }

                // 5b. Focus recording: aggressive capture every ~2s regardless of dwell state.
                // Must run BEFORE dwell detection because dwell's dHash-skip `continue`
                // would bypass this block entirely.
                // During focus, we use dHash threshold 0 to capture EVERY frame
                // (the user explicitly chose to record — every frame matters).
                if active_focus_session_id.is_some()
                    && last_focus_capture.elapsed() >= focus_capture_interval
                    && config.capture_screenshots
                    && screenshot_count_this_minute < config.screenshot_max_per_minute
                {
                    if let Some(ref store) = artifact_store {
                        // dHash threshold = 0 means always capture (bypass dedup)
                        let result = capture_and_store_screenshot(
                            store,
                            last_dhash,
                            0, // Focus mode: capture every frame
                            &config.screenshots_dir,
                            config.screenshot_scale,
                            config.screenshot_quality,
                        ).await;

                        last_focus_capture = Instant::now();
                        last_dhash = result.dhash;

                        if result.skipped_dhash {
                            // With threshold=0, this should never happen, but handle gracefully
                            debug!("Focus capture: screen unchanged (dHash=0), skipping");
                        } else {
                            last_dhash = result.dhash;

                            let mut event = make_event(
                                EventKind::DwellSnapshot,
                                &window,
                                &display_topology,
                                &primary_display_id,
                            );

                            let mut focus_artifacts: Vec<ArtifactMeta> = vec![];

                            screenshot_count_this_minute += result.artifact_ids.len() as u32;
                            event.artifact_ids = result.artifact_ids;
                            focus_artifacts = result.artifact_metas;

                            if let Some(ref path) = result.screenshot_path {
                                event.metadata["screenshot_path"] = serde_json::Value::String(
                                    path.to_string_lossy().to_string(),
                                );
                            }

                            // Run OCR on focus capture
                            #[cfg(target_os = "macos")]
                            if let Some((w, h, pixels)) = result.raw_pixels {
                                if let Some(ocr_result) = crate::platform::ocr::recognize_text_async(pixels, w, h).await {
                                    if let Ok(ocr_json) = serde_json::to_value(&ocr_result) {
                                        event.metadata["ocr"] = ocr_json;
                                    }
                                }
                            }

                            event.metadata["focus_capture"] = serde_json::Value::Bool(true);
                            tag_focus_session(&mut event, &active_focus_session_id);
                            redact_event(&mut event, &redactor);
                            debug!("Focus capture: DwellSnapshot with screenshot");
                            let _ = tx.send(ObserverMessage::Event { event, artifacts: focus_artifacts }).await;
                        }
                    }
                }

                // 6. Dwell/scroll-read detection (passive mode)
                dwell.tick();

                if dwell.is_dwelling() {
                    let mut captured_result: Option<ScreenshotResult> = None;
                    let mut should_skip_dhash = false;

                    // Capture screenshot on dwell if enabled and under rate limit
                    if config.capture_screenshots
                        && screenshot_count_this_minute < config.screenshot_max_per_minute
                    {
                        if let Some(ref store) = artifact_store {
                            let result = capture_and_store_screenshot(
                                store,
                                last_dhash,
                                config.dhash_threshold,
                                &config.screenshots_dir,
                                config.screenshot_scale,
                                config.screenshot_quality,
                            ).await;

                            if result.skipped_dhash {
                                // Screen hasn't changed enough — skip entire event
                                should_skip_dhash = true;
                                last_dhash = result.dhash;
                            } else {
                                last_dhash = result.dhash;
                                captured_result = Some(result);
                            }
                        }
                    }

                    if should_skip_dhash {
                        debug!("Screen unchanged (dHash), skipping DwellSnapshot");
                        continue;
                    }

                    let mut event = make_event(
                        EventKind::DwellSnapshot,
                        &window,
                        &display_topology,
                        &primary_display_id,
                    );

                    let mut dwell_artifacts: Vec<ArtifactMeta> = vec![];

                    if let Some(result) = captured_result {
                        screenshot_count_this_minute += result.artifact_ids.len() as u32;
                        event.artifact_ids = result.artifact_ids;
                        dwell_artifacts = result.artifact_metas;

                        // Store screenshot path for VLM annotation pipeline
                        if let Some(ref path) = result.screenshot_path {
                            event.metadata["screenshot_path"] = serde_json::Value::String(
                                path.to_string_lossy().to_string(),
                            );
                        }

                        // Run OCR on captured screenshot pixels (async, off executor)
                        #[cfg(target_os = "macos")]
                        if let Some((w, h, pixels)) = result.raw_pixels {
                            if let Some(ocr_result) = crate::platform::ocr::recognize_text_async(pixels, w, h).await {
                                if let Ok(ocr_json) = serde_json::to_value(&ocr_result) {
                                    event.metadata["ocr"] = ocr_json;
                                }
                            }
                        }
                    }

                    tag_focus_session(&mut event, &active_focus_session_id);
                    redact_event(&mut event, &redactor);
                    let _ = tx.send(ObserverMessage::Event { event, artifacts: dwell_artifacts }).await;
                }

                if dwell.is_scroll_reading() {
                    let mut captured_result: Option<ScreenshotResult> = None;
                    let mut should_skip_dhash = false;

                    // Capture screenshot on scroll-read if enabled and under rate limit
                    if config.capture_screenshots
                        && screenshot_count_this_minute < config.screenshot_max_per_minute
                    {
                        if let Some(ref store) = artifact_store {
                            let result = capture_and_store_screenshot(
                                store,
                                last_dhash,
                                config.dhash_threshold,
                                &config.screenshots_dir,
                                config.screenshot_scale,
                                config.screenshot_quality,
                            ).await;

                            if result.skipped_dhash {
                                should_skip_dhash = true;
                                last_dhash = result.dhash;
                            } else {
                                last_dhash = result.dhash;
                                captured_result = Some(result);
                            }
                        }
                    }

                    if should_skip_dhash {
                        debug!("Screen unchanged (dHash), skipping ScrollReadSnapshot");
                        continue;
                    }

                    let mut event = make_event(
                        EventKind::ScrollReadSnapshot,
                        &window,
                        &display_topology,
                        &primary_display_id,
                    );

                    let mut scroll_artifacts: Vec<ArtifactMeta> = vec![];

                    if let Some(result) = captured_result {
                        screenshot_count_this_minute += result.artifact_ids.len() as u32;
                        event.artifact_ids = result.artifact_ids;
                        scroll_artifacts = result.artifact_metas;

                        // Store screenshot path for VLM annotation pipeline
                        if let Some(ref path) = result.screenshot_path {
                            event.metadata["screenshot_path"] = serde_json::Value::String(
                                path.to_string_lossy().to_string(),
                            );
                        }

                        // Run OCR on captured screenshot pixels (async, off executor)
                        #[cfg(target_os = "macos")]
                        if let Some((w, h, pixels)) = result.raw_pixels {
                            if let Some(ocr_result) = crate::platform::ocr::recognize_text_async(pixels, w, h).await {
                                if let Ok(ocr_json) = serde_json::to_value(&ocr_result) {
                                    event.metadata["ocr"] = ocr_json;
                                }
                            }
                        }
                    }

                    tag_focus_session(&mut event, &active_focus_session_id);
                    redact_event(&mut event, &redactor);
                    let _ = tx.send(ObserverMessage::Event { event, artifacts: scroll_artifacts }).await;
                }

                // (Focus capture runs in step 5b above, before dwell detection)
            }
            _ = shutdown_rx.changed() => {
                info!("Observer loop: shutdown signal received, stopping");
                // Don't send a Shutdown message — just break.  Dropping `tx`
                // (along with aborting other producer tasks in main.rs) will
                // close the channel, causing the storage writer's `recv()` to
                // return None and exit cleanly.  This eliminates the race
                // where events queued after a try_recv() drain could be lost.
                break;
            }
        }
    }

    info!("Observer event loop stopped");
    Ok(())
}

/// Storage writer task — receives events and writes them to SQLite.
///
/// The optional `shared_counter` is incremented on each successful insert so
/// that the health watcher can report an accurate `events_today` value.
///
/// Exits when all channel senders are dropped (i.e. when the observer loop
/// and all background producers have been stopped).  This guarantees every
/// event that was successfully sent to the channel gets written — no race
/// between a one-shot drain and concurrent producers.
pub async fn run_storage_writer(
    db_path: PathBuf,
    mut rx: mpsc::Receiver<ObserverMessage>,
    shared_counter: Option<std::sync::Arc<std::sync::atomic::AtomicU64>>,
) -> Result<()> {
    info!(path = %db_path.display(), "Storage writer starting");

    let store = oc_apprentice_storage::EventStore::open(&db_path)?;
    let mut event_count = 0u64;

    // recv() returns None only when ALL senders have been dropped, so every
    // in-flight event is guaranteed to be processed before we exit.
    while let Some(msg) = rx.recv().await {
        let ObserverMessage::Event { event, artifacts } = msg;
        if let Err(e) = store.insert_event(&event) {
            error!(error = %e, "Failed to insert event");
        } else {
            // Insert artifact records so maintenance can find real files.
            let event_id_str = event.id.to_string();
            for meta in &artifacts {
                if let Err(e) = store.insert_artifact(
                    &meta.artifact_id,
                    &event_id_str,
                    &meta.artifact_type,
                    &meta.file_path.display().to_string(),
                    meta.original_size_bytes,
                    meta.stored_size_bytes,
                ) {
                    warn!(error = %e, artifact_id = %meta.artifact_id, "Failed to insert artifact record");
                }
            }

            event_count += 1;
            if let Some(ref counter) = shared_counter {
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if event_count % 100 == 0 {
                debug!(event_count, "Events stored");
            }
        }
    }

    info!(event_count, "Storage writer: channel closed, all events persisted");
    Ok(())
}

/// Result of screenshot capture: artifact metadata and optional raw pixel data for OCR.
struct ScreenshotResult {
    artifact_ids: Vec<String>,
    /// Full metadata for DB insertion.
    artifact_metas: Vec<ArtifactMeta>,
    /// Raw BGRA pixels + dimensions, available for OCR processing.
    raw_pixels: Option<(usize, usize, Vec<u8>)>,
    /// Path to the plain JPEG screenshot for VLM annotation pipeline.
    screenshot_path: Option<PathBuf>,
    /// Perceptual hash (dHash) of this screenshot.
    dhash: u64,
    /// True if this screenshot was skipped due to dHash similarity threshold.
    skipped_dhash: bool,
}

/// Capture a screenshot, check dHash for change detection, store as artifact,
/// and save a plain JPEG for the VLM annotation pipeline.
///
/// If the screenshot is too similar to the previous one (hamming distance < threshold),
/// returns a result with `skipped_dhash = true` — the caller should skip the event.
#[cfg(target_os = "macos")]
async fn capture_and_store_screenshot(
    store: &ArtifactStore,
    last_dhash: u64,
    dhash_threshold: u32,
    screenshots_dir: &Path,
    screenshot_scale: f64,
    screenshot_quality: u8,
) -> ScreenshotResult {
    const MAX_RETRIES: u32 = 2;

    for attempt in 0..=MAX_RETRIES {
        match crate::capture::screenshot::capture_main_display() {
            Some((width, height, raw_pixels)) => {
                // Compute perceptual hash for change detection
                let dhash = crate::capture::dhash::compute_dhash(&raw_pixels, width, height);
                let distance = crate::capture::dhash::hamming_distance(dhash, last_dhash);

                if distance < dhash_threshold {
                    debug!(
                        dhash = dhash,
                        distance = distance,
                        threshold = dhash_threshold,
                        "Screenshot similar to previous, skipping"
                    );
                    return ScreenshotResult {
                        artifact_ids: vec![],
                        artifact_metas: vec![],
                        raw_pixels: None,
                        screenshot_path: None,
                        dhash,
                        skipped_dhash: true,
                    };
                }

                // Store encrypted artifact (existing behavior for audit trail)
                match store.store(&raw_pixels, "screenshot") {
                    Ok(meta) => {
                        debug!(
                            artifact_id = %meta.artifact_id,
                            dhash_distance = distance,
                            "Screenshot captured and stored"
                        );
                        let id = meta.artifact_id.clone();

                        // Save plain JPEG for VLM annotation pipeline
                        let screenshot_path = save_vlm_jpeg(
                            &raw_pixels,
                            width,
                            height,
                            screenshots_dir,
                            screenshot_scale,
                            screenshot_quality,
                        );

                        return ScreenshotResult {
                            artifact_ids: vec![id],
                            artifact_metas: vec![meta],
                            raw_pixels: Some((width, height, raw_pixels)),
                            screenshot_path,
                            dhash,
                            skipped_dhash: false,
                        };
                    }
                    Err(e) => {
                        warn!(error = %e, "Failed to store screenshot artifact");
                        return ScreenshotResult {
                            artifact_ids: vec![],
                            artifact_metas: vec![],
                            raw_pixels: None,
                            screenshot_path: None,
                            dhash,
                            skipped_dhash: false,
                        };
                    }
                }
            }
            None => {
                if attempt < MAX_RETRIES {
                    warn!(attempt = attempt + 1, "Screenshot capture returned None, retrying");
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                } else {
                    error!("Screenshot capture failed after {} retries (no permission?)", MAX_RETRIES + 1);
                }
            }
        }
    }
    ScreenshotResult {
        artifact_ids: vec![],
        artifact_metas: vec![],
        raw_pixels: None,
        screenshot_path: None,
        dhash: 0,
        skipped_dhash: false,
    }
}

#[cfg(not(target_os = "macos"))]
async fn capture_and_store_screenshot(
    _store: &ArtifactStore,
    _last_dhash: u64,
    _dhash_threshold: u32,
    _screenshots_dir: &Path,
    _screenshot_scale: f64,
    _screenshot_quality: u8,
) -> ScreenshotResult {
    ScreenshotResult {
        artifact_ids: vec![],
        artifact_metas: vec![],
        raw_pixels: None,
        screenshot_path: None,
        dhash: 0,
        skipped_dhash: false,
    }
}

/// Save a plain (unencrypted) half-resolution JPEG for the VLM annotation pipeline.
///
/// The Python worker's scene annotator reads these directly via the `screenshot_path`
/// stored in event metadata. Screenshots are deleted after successful VLM annotation.
fn save_vlm_jpeg(
    raw_pixels: &[u8],
    width: usize,
    height: usize,
    screenshots_dir: &Path,
    scale: f64,
    quality: u8,
) -> Option<PathBuf> {
    if let Err(e) = std::fs::create_dir_all(screenshots_dir) {
        warn!(error = %e, dir = %screenshots_dir.display(), "Failed to create screenshots directory");
        return None;
    }

    let filename = format!("{}.jpg", Uuid::new_v4());
    let path = screenshots_dir.join(&filename);

    match crate::capture::jpeg_converter::save_screenshot_jpeg(
        raw_pixels,
        width as u32,
        height as u32,
        scale,
        quality,
        &path,
    ) {
        Ok(size) => {
            debug!(path = %path.display(), size_bytes = size, "VLM screenshot JPEG saved");
            Some(path)
        }
        Err(e) => {
            warn!(error = %e, "Failed to save VLM screenshot JPEG");
            None
        }
    }
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

/// Tag an event with the active focus session ID, if any.
fn tag_focus_session(event: &mut Event, focus_session_id: &Option<String>) {
    if let Some(ref session_id) = focus_session_id {
        event.metadata["focus_session_id"] = serde_json::Value::String(session_id.clone());
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
    fn test_tag_focus_session_active() {
        let mut event = make_test_event_with_kind(EventKind::DwellSnapshot);
        let session_id = Some("test-session-123".to_string());

        tag_focus_session(&mut event, &session_id);

        assert_eq!(
            event.metadata["focus_session_id"].as_str().unwrap(),
            "test-session-123"
        );
    }

    #[test]
    fn test_tag_focus_session_none() {
        let mut event = make_test_event_with_kind(EventKind::DwellSnapshot);
        let session_id: Option<String> = None;

        tag_focus_session(&mut event, &session_id);

        assert!(event.metadata.get("focus_session_id").is_none());
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
