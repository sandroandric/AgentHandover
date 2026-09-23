use anyhow::Result;
use chrono::{Local, Utc};
use agenthandover_common::browser_transition::{BrowserTransitionKind, BrowserTransitionWatch};
use agenthandover_common::config::PrivacyZonesConfig;
use agenthandover_common::event::*;
use agenthandover_common::redaction::Redactor;
use agenthandover_storage::artifact_store::ArtifactStore;
#[cfg(target_os = "macos")]
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time;
use tracing::{info, debug, warn, error};
use uuid::Uuid;

use super::capture_policy::{ocr_words, CapturePolicy, CaptureTrigger, FrameDecision};
use super::privacy_zones::{PrivacyZones, Zone};
use crate::observation::{
    focused_element_at_async, snapshot_at_async, ObservationSnapshot, SleepRecord, WarnLimiter,
    SNAPSHOT_TIMEOUT,
};
use crate::capture::dhash;

/// Messages sent from the observer to the storage writer.
///
/// The storage writer exits when all senders are dropped (channel close),
/// so there is no explicit Shutdown variant — proper teardown ordering in
/// `main.rs` guarantees all producers are stopped before we await the
/// writer task.
pub enum ObserverMessage {
    Event { event: Event },
}

/// Configuration for the observer event loop.
pub struct ObserverConfig {
    pub capture_screenshots: bool,
    /// Ceiling on frames kept per minute outside focus recordings.
    pub screenshot_max_per_minute: u32,
    /// Ceiling on frames kept per minute during a focus recording (its own
    /// counter, so passive capture never leaves a recording blind).
    pub focus_screenshot_max_per_minute: u32,
    pub poll_interval: Duration,
    pub db_path: PathBuf,
    /// Changed blocks (of the frame's block-mean signature) needed to keep
    /// a frame; see `capture::dhash`. Higher = fewer frames kept.
    pub dhash_threshold: u32,
    /// Screenshot format: "jpeg" or "png".
    pub screenshot_format: String,
    /// JPEG quality 1-100 (default: 70). Only used when format = "jpeg".
    pub screenshot_quality: u8,
    /// Scale factor for VLM screenshots (default: 0.5 = half resolution).
    pub screenshot_scale: f64,
    /// Directory for plain JPEG screenshots (VLM annotation pipeline).
    pub screenshots_dir: PathBuf,
    /// `[privacy.zones]`: evaluated every tick before any capture request.
    pub zones: PrivacyZonesConfig,
    /// The app's observation socket (window, permissions, secure field).
    pub observation_socket: PathBuf,
    /// The app's capture socket (frames).
    pub capture_socket: PathBuf,
    /// Data dir holding the focus-recording signal and the browser
    /// transition file.
    pub state_dir: PathBuf,
    /// `[privacy] capture_input_values`: whether the focused element's
    /// `AXValue` is read (never for a secure field).
    pub capture_input_values: bool,
}

impl Default for ObserverConfig {
    fn default() -> Self {
        Self {
            capture_screenshots: true,
            screenshot_max_per_minute: 20,
            focus_screenshot_max_per_minute: 40,
            poll_interval: Duration::from_millis(500),
            db_path: PathBuf::from("agenthandover.db"),
            dhash_threshold: 1,
            screenshot_format: "jpeg".to_string(),
            screenshot_quality: 70,
            screenshot_scale: 0.5,
            screenshots_dir: PathBuf::from("/tmp/agenthandover-screenshots"),
            zones: PrivacyZonesConfig::default(),
            observation_socket: agenthandover_common::observation::observation_socket_path(),
            capture_socket: agenthandover_common::observation::capture_socket_path(),
            state_dir: agenthandover_common::status::data_dir(),
            capture_input_values: false,
        }
    }
}

/// The observer event loop — runs as an async task.
/// Polls OS state at regular intervals and emits events.
///
/// Screenshots are event-driven (see `capture_policy`): the shutter fires on
/// an app switch, a window-title change, a browser navigation, form submit
/// or click (handed over by the native-messaging bridge through
/// `browser_transition`), a mouse click, a clipboard change
/// (`clipboard_changed`, set by the clipboard monitor), every 2 s during a
/// focus recording, and otherwise as a heartbeat once a minute.
///
/// Privacy comes first each tick: a tick whose observation snapshot failed
/// records nothing (the secure-field check cannot run), a focused secure
/// field records nothing, and `[privacy.zones]` is evaluated for the focused
/// window (app, title, and the browser's page URL) before the shutter: a
/// blocked window gets no frame, no OCR and no title, a metadata-only window
/// gets its app-switch and title events but no frame.
///
/// When `capture_paused` is `true` AND there is no active focus recording
/// session, the loop skips all event capture (AppSwitch, WindowTitleChange,
/// frames). Focus recordings always run regardless of the pause flag so the
/// user can record a session while normally paused.
pub async fn run_observer_loop(
    config: ObserverConfig,
    tx: mpsc::Sender<ObserverMessage>,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
    artifact_store: Option<Arc<ArtifactStore>>,
    capture_paused: Arc<AtomicBool>,
    clipboard_changed: Arc<AtomicBool>,
) -> Result<()> {
    info!("Observer event loop starting");

    let mut last_window_title: Option<String> = None;
    let mut last_app_id: Option<String> = None;
    let mut last_sleep: Option<SleepRecord> = None;
    let mut policy = CapturePolicy::new(
        config.screenshot_max_per_minute,
        config.focus_screenshot_max_per_minute,
        config.dhash_threshold,
        Instant::now(),
    );
    let mut last_tick = Instant::now();
    let zones = PrivacyZones::new(&config.zones);
    let snapshot_warnings = WarnLimiter::new();
    // The browser's current page (origin + path), from the extension's
    // navigations, tab switches, clicks and form submits.
    let mut browser_url: Option<String> = None;

    // Mouse clicks are read from the HID system's "seconds since last
    // mouse-down" (no event tap, nothing intercepted).
    #[cfg(target_os = "macos")]
    let idle_detector = crate::platform::IdleDetector::new();

    // Focus recording session tracking; browser transitions share the dir.
    let state_dir = config.state_dir.clone();
    let mut active_focus_session_id: Option<String> = None;
    let mut browser_watch = BrowserTransitionWatch::since(Utc::now());

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

                let tick_elapsed = last_tick.elapsed();
                last_tick = Instant::now();

                // Check for focus recording session signal (cheap stat() call)
                match agenthandover_common::focus_session::read_focus_signal(&state_dir) {
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

                // Transitions from other tasks and processes, consumed every
                // tick so one that happened while paused never fires later.
                let clipboard_transition = clipboard_changed.swap(false, Ordering::AcqRel);
                let browser_transition = browser_watch.poll(&state_dir);
                if let Some(url) = browser_transition.as_ref().and_then(|t| t.url.clone()) {
                    browser_url = Some(url);
                }
                let browser_transition = browser_transition.map(|t| t.kind);

                // If capture is paused and no active focus recording, skip
                // all event capture this tick. Focus recordings always run
                // regardless of pause state (the user explicitly pressed Record).
                if capture_paused.load(Ordering::Acquire)
                    && active_focus_session_id.is_none()
                {
                    continue;
                }

                // 1. Pull app-owned observation state. The app is the sole
                // TCC principal for Accessibility and Screen Recording.
                // Without a snapshot the secure-field check cannot run, so
                // the tick is unsafe: capture nothing, and leave the last
                // app/title as they were so no transition is fabricated.
                // (The failure is warned, rate-limited, by the client.)
                let Ok(observation) = snapshot_at_async(
                    &config.observation_socket,
                    SNAPSHOT_TIMEOUT,
                    &snapshot_warnings,
                )
                .await
                else {
                    continue;
                };

                // 2. Check for secure field — if focused, skip all capture.
                let Some(zone) = privacy_zone(&zones, &observation, browser_url.as_deref()) else {
                    debug!("Secure field focused — skipping capture");
                    continue;
                };

                // 3. Observation metadata comes from the app-owned socket.
                let display_topology = observation.display_topology.clone();

                let primary_display_id = display_topology
                    .first()
                    .map(|d| d.display_id.clone())
                    .unwrap_or_else(|| "unknown".to_string());

                let observed_window = if observation.accessibility_granted {
                    observation.focused_window.clone()
                } else {
                    None
                };
                let cursor_global_px = if observation.accessibility_granted {
                    observation.cursor_global_px.clone()
                } else {
                    None
                };

                // The privacy zone (above) is known before anything is
                // recorded. A blocked window's title is withheld from every
                // event; only the app switch into it is recorded.
                let window = observed_window.clone().map(|mut w| {
                    if zone == Zone::Blocked {
                        w.title.clear();
                    }
                    w
                });

                // 3b. A sleep the app saw since the last tick is a boundary:
                // no episode or task segment may span it.
                if let Some(sleep) = new_sleep(&last_sleep, &observation) {
                    let mut event = make_event(
                        EventKind::SystemSleep {
                            slept_at: sleep.slept_at,
                            woke_at: sleep.woke_at,
                        },
                        &None,
                        &display_topology,
                        &primary_display_id,
                        None,
                    );
                    // Ordered before anything captured after the wake.
                    event.timestamp = sleep.woke_at;
                    tag_focus_session(&mut event, &active_focus_session_id);
                    let _ = tx.send(ObserverMessage::Event { event }).await;
                    info!(slept_at = %sleep.slept_at, woke_at = %sleep.woke_at, "System sleep boundary");
                    last_sleep = Some(sleep);
                }

                // 4. Collect this tick's transitions, cheapest first.
                let mut transitions: Vec<CaptureTrigger> = Vec::new();
                let current_title = observed_window.as_ref().map(|w| w.title.clone());
                let current_app = observed_window.as_ref().map(|w| w.app_id.clone());

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
                            cursor_global_px.clone(),
                        );

                        // Query AppleScript state for known native apps
                        #[cfg(target_os = "macos")]
                        {
                            // App state (page URL, document, message) is
                            // content: never read for a zoned window.
                            let should_query = zone == Zone::Full
                                && applescript_last_query
                                    .get(to.as_str())
                                    .is_none_or(|last| last.elapsed() >= applescript_throttle);

                            if should_query {
                                let app_name_to_query =
                                    crate::platform::applescript::app_name_for_app_id(to)
                                        .map(str::to_string);
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
                                // Bound the map: forget apps not queried recently.
                                applescript_last_query
                                    .retain(|_, ts| ts.elapsed() < Duration::from_secs(30));
                                applescript_last_query.insert(to.clone(), Instant::now());
                            }
                        }

                        tag_focus_session(&mut event, &active_focus_session_id);
                        let _ = tx.send(ObserverMessage::Event { event }).await;
                        transitions.push(CaptureTrigger::AppSwitch);
                    }
                    last_app_id = current_app.clone();
                }

                // Title change detection (not recorded for a blocked window)
                if current_title != last_window_title {
                    if last_window_title.is_some() && zone != Zone::Blocked {
                        let mut event = make_event(
                            EventKind::WindowTitleChange,
                            &window,
                            &display_topology,
                            &primary_display_id,
                            cursor_global_px.clone(),
                        );
                        tag_focus_session(&mut event, &active_focus_session_id);
                        let _ = tx.send(ObserverMessage::Event { event }).await;
                        transitions.push(CaptureTrigger::TitleChange);
                    }
                    last_window_title = current_title;
                }

                // A mouse button went down since the previous tick.
                #[cfg(target_os = "macos")]
                if idle_detector.seconds_since_last_click() < tick_elapsed.as_secs_f64() {
                    transitions.push(CaptureTrigger::Click);
                }
                #[cfg(not(target_os = "macos"))]
                let _ = tick_elapsed;

                if clipboard_transition {
                    transitions.push(CaptureTrigger::Clipboard);
                }
                if let Some(kind) = browser_transition {
                    transitions.push(match kind {
                        BrowserTransitionKind::Navigation => CaptureTrigger::Navigation,
                        BrowserTransitionKind::FormSubmit => CaptureTrigger::FormSubmit,
                        BrowserTransitionKind::Click => CaptureTrigger::Click,
                    });
                }

                // 5. At most one frame per tick, only when something happened
                // (or the heartbeat / focus interval is due) and budget remains.
                // A blocked or metadata-only window never reaches the shutter.
                if zone != Zone::Full {
                    debug!(zone = zone.as_str(), "Privacy zone: no frame");
                    continue;
                }
                if !config.capture_screenshots || artifact_store.is_none() {
                    continue;
                }
                let focus_recording = active_focus_session_id.is_some();
                let Some(trigger) =
                    policy.shutter_trigger(&transitions, focus_recording, Instant::now())
                else {
                    continue;
                };
                let gate = PrivacyGate {
                    zones: &zones,
                    warnings: &snapshot_warnings,
                    browser_url: browser_url.as_deref(),
                };
                let Some(kept) =
                    capture_kept_frame(&mut policy, trigger, focus_recording, &config, &gate).await
                else {
                    continue;
                };

                let mut event = make_event(
                    EventKind::DwellSnapshot,
                    &window,
                    &display_topology,
                    &primary_display_id,
                    cursor_global_px.clone(),
                );
                // Every kept frame (a click, app switch, focus-recording
                // tick, ...) carries what had focus when it was taken.
                event.focused_element =
                    focused_element_for(&observation, &config, &snapshot_warnings).await;
                if let Some(path) = kept.screenshot_path {
                    event.metadata["screenshot_path"] =
                        serde_json::Value::String(path.to_string_lossy().to_string());
                }
                if let Some(ocr) = kept.ocr {
                    event.metadata["ocr"] = ocr;
                }
                // The page the frame shows, so the worker's zone check (the
                // second line) sees the URL too.
                if let (Some(url), Some(w)) = (&browser_url, &window) {
                    let app = super::privacy_zones::strip_pid_prefix(&w.app_id).to_lowercase();
                    if super::privacy_zones::is_browser(&app) {
                        event.metadata["url"] = serde_json::Value::String(url.clone());
                    }
                }
                event.metadata["capture_trigger"] = serde_json::Value::String(trigger.as_str().to_string());
                if focus_recording {
                    event.metadata["focus_capture"] = serde_json::Value::Bool(true);
                }
                tag_focus_session(&mut event, &active_focus_session_id);
                debug!(trigger = trigger.as_str(), "Frame kept");
                let _ = tx.send(ObserverMessage::Event { event }).await;
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

    let store = agenthandover_storage::EventStore::open(&db_path)?;
    // Redaction is a storage invariant: every producer (observer loop,
    // clipboard monitor, Chrome native messaging) funnels through this
    // writer, so redacting here and only here guarantees no event reaches
    // SQLite unredacted, whichever path produced it.
    let redactor = Redactor::new();
    let mut event_count = 0u64;

    // recv() returns None only when ALL senders have been dropped, so every
    // in-flight event is guaranteed to be processed before we exit.
    while let Some(msg) = rx.recv().await {
        let ObserverMessage::Event { mut event } = msg;
        redact_event(&mut event, &redactor);
        if let Err(e) = store.insert_event(&event) {
            error!(error = %e, "Failed to insert event");
        } else {
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

/// A frame that passed the gate: its JPEG on disk (for the VLM annotation
/// pipeline) and its accurate OCR.
struct KeptFrame {
    screenshot_path: Option<PathBuf>,
    ocr: Option<serde_json::Value>,
}

/// The zone of the snapshot's focused window, or `None` when a secure
/// (password) field is focused and nothing may be recorded.
fn privacy_zone(
    zones: &PrivacyZones,
    observation: &crate::observation::ObservationSnapshot,
    browser_url: Option<&str>,
) -> Option<Zone> {
    if !observation.accessibility_granted {
        return Some(Zone::Full);
    }
    if observation.secure_field_focused {
        return None;
    }
    Some(observation.focused_window.as_ref().map_or(Zone::Full, |w| {
        zones.zone(&w.app_id, &w.title, browser_url, Local::now().time())
    }))
}

/// What a captured frame is checked against once it is in hand.
struct PrivacyGate<'a> {
    zones: &'a PrivacyZones,
    warnings: &'a WarnLimiter,
    browser_url: Option<&'a str>,
}

impl PrivacyGate<'_> {
    /// Whether the window on screen *now* may still be recorded: the user
    /// can switch to a blocked window, or focus a password field, while
    /// the frame is being taken. A failed snapshot is a no.
    async fn still_allows(&self, config: &ObserverConfig) -> bool {
        let Ok(now) = snapshot_at_async(&config.observation_socket, SNAPSHOT_TIMEOUT, self.warnings).await
        else {
            return false;
        };
        privacy_zone(self.zones, &now, self.browser_url) == Some(Zone::Full)
    }
}

/// Capture one frame for `trigger` and run it through the policy, cheapest
/// check first: the privacy zone of the window on screen once the frame is
/// in hand (it may have changed during the capture), the block-mean
/// signature against the last kept frame, then
/// (heartbeats only) fast OCR against the last kept frame's words, then, for
/// a frame that is kept, accurate OCR and the JPEG write. Nothing reaches
/// disk for a dropped frame. `None` when the capture failed or the frame
/// was dropped.
async fn capture_kept_frame(
    policy: &mut CapturePolicy,
    trigger: CaptureTrigger,
    focus_recording: bool,
    config: &ObserverConfig,
    gate: &PrivacyGate<'_>,
) -> Option<KeptFrame> {
    let (width, height, bgra, jpeg) =
        grab_frame(&config.capture_socket, config.screenshot_scale, config.screenshot_quality).await?;
    if !gate.still_allows(config).await {
        debug!("The window changed to a private one during the capture, dropping the frame");
        return None;
    }
    let signature = dhash::compute_signature(&bgra, width, height)?;

    let mut pixels = bgra;
    match policy.frame_decision(trigger, focus_recording, &signature) {
        FrameDecision::Unchanged => {
            debug!(trigger = trigger.as_str(), "Screen unchanged since the last kept frame, dropping");
            return None;
        }
        FrameDecision::CheckText => {
            let (back, fast) = ocr_frame(pixels, width, height, OcrPass::Fast).await;
            pixels = back;
            // Without OCR there is no text to compare: the pixels changed, keep it.
            if let Some((text, _)) = fast {
                if !policy.text_changed(&ocr_words(&text)) {
                    debug!("Heartbeat frame shows the same text as the last kept frame, dropping");
                    return None;
                }
            }
        }
        FrameDecision::Keep => {}
    }

    let (_, accurate) = ocr_frame(pixels, width, height, OcrPass::Accurate).await;
    let screenshot_path = write_vlm_jpeg(&jpeg, &config.screenshots_dir);
    let (words, ocr) = match accurate {
        Some((text, json)) => (Some(ocr_words(&text)), Some(json)),
        None => (None, None),
    };
    policy.record_kept(signature, words, focus_recording);
    Some(KeptFrame { screenshot_path, ocr })
}

/// Capture one frame from the app's capture socket: `(width, height, BGRA
/// pixels, JPEG bytes)`. Transient socket errors are retried twice.
#[cfg(target_os = "macos")]
async fn grab_frame(socket: &Path, scale: f64, quality: u8) -> Option<(usize, usize, Vec<u8>, Vec<u8>)> {
    use crate::capture::screenshot::{capture_frame_at, CAPTURE_TIMEOUT};
    const MAX_RETRIES: u32 = 2;

    for attempt in 0..=MAX_RETRIES {
        match capture_frame_at(socket.to_path_buf(), scale, quality, CAPTURE_TIMEOUT).await {
            Ok(frame) => {
                debug!(
                    width = frame.width,
                    height = frame.height,
                    jpeg_bytes = frame.jpeg.len(),
                    "Screenshot captured"
                );
                return Some((frame.width, frame.height, frame.bgra, frame.jpeg));
            }
            Err(e) if e.is_retryable() && attempt < MAX_RETRIES => {
                warn!(attempt = attempt + 1, error = %e, "Screenshot capture failed, retrying");
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
            Err(e) => {
                error!(error = %e, attempts = attempt + 1, "Screenshot capture failed");
                break;
            }
        }
    }
    None
}

#[cfg(not(target_os = "macos"))]
async fn grab_frame(_socket: &Path, _scale: f64, _quality: u8) -> Option<(usize, usize, Vec<u8>, Vec<u8>)> {
    None
}

/// Which OCR pass to run on a frame.
#[derive(Clone, Copy)]
enum OcrPass {
    /// Vision's fast level, to decide whether a heartbeat frame's text changed.
    Fast,
    /// Vision's accurate level, for a frame that is kept.
    Accurate,
}

/// Run OCR on a frame. Returns the pixels back plus `(full text, result as
/// JSON)` on success.
#[cfg(target_os = "macos")]
async fn ocr_frame(
    pixels: Vec<u8>,
    width: usize,
    height: usize,
    pass: OcrPass,
) -> (Vec<u8>, Option<(String, serde_json::Value)>) {
    use crate::platform::ocr::{recognize_text_async, OcrLevel};
    let level = match pass {
        OcrPass::Fast => OcrLevel::Fast,
        OcrPass::Accurate => OcrLevel::Accurate,
    };
    let (pixels, result) = recognize_text_async(pixels, width, height, level).await;
    let result = result.and_then(|r| {
        let json = serde_json::to_value(&r).ok()?;
        Some((r.full_text, json))
    });
    (pixels, result)
}

#[cfg(not(target_os = "macos"))]
async fn ocr_frame(
    pixels: Vec<u8>,
    _width: usize,
    _height: usize,
    _pass: OcrPass,
) -> (Vec<u8>, Option<(String, serde_json::Value)>) {
    (pixels, None)
}

/// Write the app-encoded JPEG for the VLM annotation pipeline (no re-encode).
///
/// The Python worker's scene annotator reads these directly via the `screenshot_path`
/// stored in event metadata. The worker deletes each file once it has been
/// annotated; daily maintenance deletes any left older than `vlm.job_ttl_days`.
fn write_vlm_jpeg(jpeg: &[u8], screenshots_dir: &Path) -> Option<PathBuf> {
    if let Err(e) = std::fs::create_dir_all(screenshots_dir) {
        warn!(error = %e, dir = %screenshots_dir.display(), "Failed to create screenshots directory");
        return None;
    }

    let path = screenshots_dir.join(format!("{}.jpg", Uuid::new_v4()));
    match std::fs::write(&path, jpeg) {
        Ok(()) => {
            debug!(path = %path.display(), size_bytes = jpeg.len(), "VLM screenshot JPEG saved");
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
    cursor_global_px: Option<CursorPosition>,
) -> Event {
    let display_ids_spanned = detect_spanning_displays(window, display_topology);

    // Determine ui_scale based on which display the cursor is on.
    let ui_scale = cursor_global_px
        .as_ref()
        .and_then(|cursor| ui_scale_for_position(cursor, display_topology));

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
        focused_element: None,
    }
}

fn ui_scale_for_position(
    cursor: &CursorPosition,
    displays: &[DisplayInfo],
) -> Option<f64> {
    for display in displays {
        let [dx, dy, dw, dh] = display.bounds_global_px;
        if cursor.x >= dx && cursor.x < dx + dw && cursor.y >= dy && cursor.y < dy + dh {
            return Some(display.scale_factor);
        }
    }
    displays.first().map(|display| display.scale_factor)
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

/// The sleep in `observation` when it is not the one `last` already
/// reported.
fn new_sleep(last: &Option<SleepRecord>, observation: &ObservationSnapshot) -> Option<SleepRecord> {
    let sleep = observation.last_sleep.clone()?;
    (last.as_ref() != Some(&sleep)).then_some(sleep)
}

/// The focused element for a kept frame, read by the app over the
/// configured observation socket; None when the app has no Accessibility
/// grant or did not answer (warned through `warnings`).
async fn focused_element_for(
    observation: &ObservationSnapshot,
    config: &ObserverConfig,
    warnings: &WarnLimiter,
) -> Option<FocusedElement> {
    if !observation.accessibility_granted {
        return None;
    }
    focused_element_at_async(
        &config.observation_socket,
        config.capture_input_values,
        SNAPSHOT_TIMEOUT,
        warnings,
    )
    .await
    .ok()
    .flatten()
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
        EventKind::ClipboardChange { content_hash, content_preview, .. } => {
            *content_hash = redactor.redact(content_hash);
            if let Some(preview) = content_preview {
                *preview = redactor.redact(preview);
            }
        }
        EventKind::PasteDetected { matched_copy_hash } => {
            if let Some(hash) = matched_copy_hash {
                *hash = redactor.redact(hash);
            }
        }
        EventKind::ShellCommand { cwd, command } => {
            *cwd = redactor.redact(cwd);
            *command = redactor.redact(command);
        }
        EventKind::NavigationChange { url, title } => {
            *url = redactor.redact(url);
            *title = redactor.redact(title);
        }
        EventKind::FormSubmit { url, form_anchor } => {
            *url = redactor.redact(url);
            *form_anchor = redact_json_value(form_anchor, redactor);
        }
        EventKind::FocusChange
        | EventKind::WindowTitleChange
        | EventKind::DwellSnapshot
        | EventKind::ScrollReadSnapshot
        | EventKind::SecureFieldFocus
        | EventKind::SystemSleep { .. } => {}
    }

    // Redact window title
    if let Some(ref mut win) = event.window {
        win.title = redactor.redact(&win.title);
    }

    // Redact every text the focused element carries
    if let Some(ref mut element) = event.focused_element {
        let redact = |text: &mut Option<String>| {
            if let Some(t) = text {
                *t = redactor.redact(t);
            }
        };
        redact(&mut element.title);
        redact(&mut element.description);
        redact(&mut element.value);
        redact(&mut element.column_header);
        redact(&mut element.document);
        redact(&mut element.url);
        for ancestor in &mut element.ancestors {
            redact(&mut ancestor.title);
            redact(&mut ancestor.description);
        }
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
            focused_element: None,
        }
    }

    #[test]
    fn test_redact_event_focused_element_texts() {
        let redactor = Redactor::new();
        let mut event = make_test_event_with_kind(EventKind::DwellSnapshot);
        event.focused_element = Some(FocusedElement {
            role: "AXTextField".into(),
            title: Some("Token".into()),
            value: Some("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl".into()),
            ancestors: vec![AxAncestor {
                role: "AXGroup".into(),
                description: Some("password: hunter2".into()),
                ..Default::default()
            }],
            ..Default::default()
        });

        redact_event(&mut event, &redactor);

        let element = event.focused_element.unwrap();
        assert_eq!(element.title.as_deref(), Some("Token"));
        assert_eq!(element.value.as_deref(), Some("[REDACTED_GITHUB_TOKEN]"));
        assert!(!element.ancestors[0].description.as_ref().unwrap().contains("hunter2"));
    }

    #[test]
    fn test_redact_event_shell_command() {
        let redactor = Redactor::new();
        let mut event = make_test_event_with_kind(EventKind::ShellCommand {
            cwd: "/Users/me/app".into(),
            command: "curl -u me:ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl https://api.example.com/x".into(),
        });
        redact_event(&mut event, &redactor);
        match event.kind {
            EventKind::ShellCommand { cwd, command } => {
                assert_eq!(cwd, "/Users/me/app");
                assert!(!command.contains("ghp_ABCDEFGHIJ"), "{command}");
                assert!(command.starts_with("curl -u me:"), "the rest of the command stays: {command}");
            }
            other => panic!("kind changed: {other:?}"),
        }
    }

    fn snapshot_with_sleep(sleep: Option<SleepRecord>) -> ObservationSnapshot {
        let mut snapshot: ObservationSnapshot = serde_json::from_value(serde_json::json!({
            "accessibility_granted": true,
            "screen_recording_granted": true,
            "secure_field_focused": false,
            "focused_window": null,
            "display_topology": [],
            "cursor_global_px": null,
        }))
        .unwrap();
        snapshot.last_sleep = sleep;
        snapshot
    }

    #[test]
    fn test_new_sleep_reports_each_sleep_once() {
        let at = |h: u32| chrono::TimeZone::with_ymd_and_hms(&Utc, 2026, 9, 21, h, 0, 0).unwrap();
        let first = SleepRecord { slept_at: at(1), woke_at: at(7) };
        let second = SleepRecord { slept_at: at(12), woke_at: at(13) };

        assert_eq!(new_sleep(&None, &snapshot_with_sleep(None)), None, "never slept");
        assert_eq!(new_sleep(&None, &snapshot_with_sleep(Some(first.clone()))), Some(first.clone()));
        assert_eq!(
            new_sleep(&Some(first.clone()), &snapshot_with_sleep(Some(first.clone()))),
            None,
            "already reported"
        );
        assert_eq!(
            new_sleep(&Some(first), &snapshot_with_sleep(Some(second.clone()))),
            Some(second)
        );
    }

    #[tokio::test]
    async fn test_no_focused_element_without_accessibility() {
        let mut snapshot = snapshot_with_sleep(None);
        snapshot.accessibility_granted = false;
        // The socket is never consulted without the grant.
        let config = ObserverConfig {
            observation_socket: PathBuf::from("/nonexistent/observation.sock"),
            capture_input_values: true,
            ..ObserverConfig::default()
        };
        let warnings = WarnLimiter::new();
        assert!(focused_element_for(&snapshot, &config, &warnings).await.is_none());
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
    fn test_redact_event_navigation_and_form_submit() {
        let redactor = Redactor::new();
        let secret = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl";

        let mut nav = make_test_event_with_kind(EventKind::NavigationChange {
            url: format!("https://example.com/{secret}"),
            title: format!("Token {secret}"),
        });
        redact_event(&mut nav, &redactor);
        let EventKind::NavigationChange { url, title } = &nav.kind else { panic!() };
        assert!(!url.contains("ghp_") && !title.contains("ghp_"));

        let mut submit = make_test_event_with_kind(EventKind::FormSubmit {
            url: "https://example.com/login".into(),
            form_anchor: serde_json::json!({"ariaLabel": format!("key {secret}"), "tag": "form"}),
        });
        redact_event(&mut submit, &redactor);
        let EventKind::FormSubmit { form_anchor, .. } = &submit.kind else { panic!() };
        assert!(!form_anchor["ariaLabel"].as_str().unwrap().contains("ghp_"));
        assert_eq!(form_anchor["tag"], "form");
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
            focused_element: None,
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
