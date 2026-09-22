use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use tokio::sync::{mpsc, watch};
use tracing::{info, warn, error, debug};
use tracing_subscriber::EnvFilter;
use tracing_appender::rolling;
use sha2::{Digest, Sha256};

use agenthandover_daemon::control::ControlContext;
use agenthandover_daemon::ipc::native_messaging;
use agenthandover_daemon::observer::event_loop::{
    ObserverConfig, ObserverMessage, run_observer_loop, run_storage_writer,
};
use agenthandover_daemon::observer::health::HealthWatcher;

/// Check if this process was launched by Chrome Native Messaging.
///
/// Chrome NM launches the host binary with stdin connected to a pipe (not a
/// terminal and not /dev/null).  The launchd-managed daemon has stdin pointing
/// to /dev/null.  We detect this by checking if stdin is *not* a tty and if
/// the `--native-messaging` flag was passed (Chrome doesn't pass it, but we
/// add it to the NM manifest to be explicit), OR if stdin is a pipe.
fn is_native_messaging_mode() -> bool {
    // Explicit flag check first
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--native-messaging") {
        return true;
    }
    // Auto-detect: check if stdin is a pipe (Chrome NM mode)
    // When launched by launchd, stdin is /dev/null (not a pipe)
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        unsafe {
            let fd = std::io::stdin().as_raw_fd();
            let mut stat: libc::stat = std::mem::zeroed();
            if libc::fstat(fd, &mut stat) == 0 {
                // S_IFIFO = pipe/FIFO
                return (stat.st_mode & libc::S_IFMT) == libc::S_IFIFO;
            }
        }
    }
    false
}

#[tokio::main]
async fn main() -> Result<()> {
    // If launched by Chrome NM, run in lightweight NM-only mode
    if is_native_messaging_mode() {
        return run_native_messaging_bridge().await;
    }

    // Detach from the parent shell's session and process group so SIGHUP
    // from the launching shell closing doesn't propagate to us.
    //
    // Without this, hikoae's v0.2.9 repro was: run `agenthandover restart`
    // from a shell, then let that shell exit. The daemon's PPID would
    // correctly reparent to 1 (init/launchd), but its PGID stayed tied
    // to the launching shell's process group. When the shell closed its
    // controlling TTY, SIGHUP was sent to every process in that group —
    // including the orphaned daemon. Default SIGHUP action is terminate
    // (no signal handler invoked, no stderr output, no unified log entry,
    // no crash report). The daemon just silently disappeared within
    // seconds of the shell exiting.
    //
    // `setsid()` creates a new session with the daemon as leader, detaching
    // it from the launching shell's session and process group. It fails
    // with EPERM if the caller is already a session leader (the desired
    // end state) — we ignore the error because both success and EPERM
    // mean "daemon is now detached."
    #[cfg(unix)]
    unsafe {
        let _ = libc::setsid();
    }

    let log_dir = {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        if cfg!(target_os = "macos") {
            PathBuf::from(&home).join("Library/Application Support/agenthandover/logs")
        } else {
            PathBuf::from(&home).join(".local/share/agenthandover/logs")
        }
    };
    std::fs::create_dir_all(&log_dir).ok();

    // One file per day, `daemon.YYYY-MM-DD.log`, the newest 7 kept.
    // `agenthandover logs daemon` resolves the newest of these.
    let file_appender = rolling::Builder::new()
        .rotation(rolling::Rotation::DAILY)
        .filename_prefix("daemon")
        .filename_suffix("log")
        .max_log_files(7)
        .build(&log_dir)?;
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("info".parse()?),
        )
        .with_writer(non_blocking)
        .with_ansi(false)
        .init();

    // Install a panic hook so any panic in a Tokio task (or elsewhere) gets
    // logged to daemon.log with a location + payload BEFORE the process
    // exits. The default Rust panic printer writes to stderr, which the
    // daemon spawn paths (Swift Process() + CLI Command::spawn) redirect to
    // /dev/null — meaning panics vanished silently on hikoae's v0.2.7
    // machine.  Logging via `error!` ensures the panic hits the tracing
    // subscriber's file writer and is visible for diagnostics.
    //
    // Note: this does NOT prevent process termination — a panic still
    // unwinds and (for `panic = "abort"` builds) aborts the process.  It
    // just makes sure we see WHY before that happens.
    std::panic::set_hook(Box::new(|info| {
        let payload = info.payload();
        let msg = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(|s| s.as_str()))
            .unwrap_or("<non-string panic payload>");
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown>".to_string());
        tracing::error!(location = %location, "PANIC: {}", msg);
    }));

    info!("agenthandover-daemon starting");

    // Write PID file
    let pid_path = agenthandover_common::pid::write_pid_file("daemon")
        .expect("Failed to write daemon PID file");
    info!(path = %pid_path.display(), "PID file written");

    // Do NOT prompt for permissions on startup. AgentHandover.app owns both
    // Accessibility and Screen Recording and serves observation/capture data
    // to the daemon over local sockets.

    let start_time = chrono::Utc::now();

    // Shared event counter — incremented by storage writer, read by health watcher
    let event_counter = Arc::new(AtomicU64::new(0));

    // Timestamp (epoch millis) of the last Chrome extension NM message.
    // 0 = no message received yet.  Updated by the NM forwarder task.
    let last_nm_message_epoch_ms = Arc::new(AtomicI64::new(0));

    let app_config = load_app_config();

    // Convert AppConfig -> ObserverConfig
    let config = ObserverConfig {
        capture_screenshots: app_config.observer.capture_screenshots,
        screenshot_max_per_minute: app_config.observer.screenshot_max_per_minute,
        focus_screenshot_max_per_minute: app_config.observer.focus_screenshot_max_per_minute,
        poll_interval: std::time::Duration::from_millis(500),
        db_path: {
            // Use the same standard path the worker expects so both
            // processes find the database without manual --db-path flags.
            let data_dir = if cfg!(target_os = "macos") {
                dirs_or_home("Library/Application Support/agenthandover")
            } else {
                dirs_or_home(".local/share/agenthandover")
            };
            std::fs::create_dir_all(&data_dir).ok();
            data_dir.join("events.db")
        },
        dhash_threshold: app_config.observer.dhash_threshold,
        screenshot_format: app_config.observer.screenshot_format.clone(),
        screenshot_quality: app_config.observer.screenshot_quality,
        screenshot_scale: app_config.observer.screenshot_scale,
        screenshots_dir: {
            // Screenshots dir lives alongside the database
            let data_dir = if cfg!(target_os = "macos") {
                dirs_or_home("Library/Application Support/agenthandover")
            } else {
                dirs_or_home(".local/share/agenthandover")
            };
            let dir = data_dir.join("screenshots");
            std::fs::create_dir_all(&dir).ok();
            dir
        },
    };

    // Channel for observer -> storage communication
    let (tx, rx) = mpsc::channel(1000);

    // Shutdown signal
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // Handle Ctrl+C
    let shutdown_tx_clone = shutdown_tx.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        info!("Received Ctrl+C, shutting down...");
        let _ = shutdown_tx_clone.send(true);
    });

    // Handle SIGTERM (sent by launchctl bootout)
    #[cfg(unix)]
    {
        let shutdown_tx_term = shutdown_tx.clone();
        tokio::spawn(async move {
            let mut sigterm = tokio::signal::unix::signal(
                tokio::signal::unix::SignalKind::terminate(),
            )
            .expect("failed to register SIGTERM handler");
            sigterm.recv().await;
            info!("Received SIGTERM, shutting down gracefully...");
            let _ = shutdown_tx_term.send(true);
        });
    }

    // Handle SIGHUP as a clean shutdown signal.  With setsid() at startup
    // the daemon is normally detached from the launching shell's session,
    // so SIGHUP shouldn't arrive via that path.  This handler is defense
    // in depth: if setsid() fails for any reason, or if something else
    // sends SIGHUP (logrotate conventions, explicit `kill -HUP`), we
    // exit cleanly with full shutdown logs + cleanup instead of dying
    // silently on the default handler.
    #[cfg(unix)]
    {
        let shutdown_tx_hup = shutdown_tx.clone();
        tokio::spawn(async move {
            let mut sighup = tokio::signal::unix::signal(
                tokio::signal::unix::SignalKind::hangup(),
            )
            .expect("failed to register SIGHUP handler");
            sighup.recv().await;
            info!("Received SIGHUP, shutting down gracefully...");
            let _ = shutdown_tx_hup.send(true);
        });
    }

    // Atomic capture-paused flag shared between control API and observer loop.
    // The control API sets it; the observer reads it each tick.
    let capture_paused = Arc::new(AtomicBool::new(false));

    // Spawn Unix domain socket control server
    let control_status_dir = agenthandover_common::status::status_dir();
    let control_handle = tokio::spawn(agenthandover_daemon::control::serve(ControlContext {
        status_dir: control_status_dir,
        capture_paused: Arc::clone(&capture_paused),
        shutdown_rx: shutdown_tx.subscribe(),
    }));

    let db_path = config.db_path.clone();

    // Spawn storage writer
    let storage_handle = tokio::spawn({
        let db = db_path.clone();
        let counter = Arc::clone(&event_counter);
        async move { run_storage_writer(db, rx, Some(counter)).await }
    });

    // Spawn native messaging server (Chrome extension bridge)
    let native_tx = tx.clone();
    let nm_ts_clone = Arc::clone(&last_nm_message_epoch_ms);
    let native_handle = tokio::spawn(async move {
        // Create a channel to receive events from the native messaging server
        let (nm_event_tx, mut nm_event_rx) = mpsc::channel(256);

        // Spawn the forwarder that bridges Event -> ObserverMessage
        let forwarder_tx = native_tx;
        let nm_ts = nm_ts_clone;
        let forwarder_handle = tokio::spawn(async move {
            while let Some(event) = nm_event_rx.recv().await {
                // Track last NM message timestamp for extension-connected detection
                nm_ts.store(chrono::Utc::now().timestamp_millis(), Ordering::Relaxed);
                note_browser_transition(&event);
                match forwarder_tx.try_send(ObserverMessage::Event { event }) {
                    Ok(()) => {}
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        warn!("Native messaging forwarder: main channel full (backpressure), dropping event");
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        info!("Native messaging forwarder: main channel closed");
                        break;
                    }
                }
            }
        });

        // Run the native messaging server on stdio
        // Full daemon mode doesn't send outbound commands to Chrome — use a dummy channel
        let server = native_messaging::stdio_server();
        let (_dummy_cmd_tx, dummy_cmd_rx) =
            mpsc::channel::<native_messaging::DaemonCommand>(1);
        if let Err(e) = server.run(nm_event_tx, dummy_cmd_rx).await {
            warn!("Native messaging server exited: {}", e);
        }

        forwarder_handle.abort();
    });

    // Spawn health watcher (periodic background health checks + status file writing)
    let health_shutdown_rx = shutdown_tx.subscribe();
    let health_db_path = db_path.clone();
    let health_start_time = start_time;
    let health_event_counter = Arc::clone(&event_counter);
    let health_nm_ts = Arc::clone(&last_nm_message_epoch_ms);
    let health_handle = tokio::spawn(async move {
        let artifact_dir = health_db_path.parent()
            .map(|p| p.join("artifacts"))
            .unwrap_or_else(|| std::env::temp_dir().join("agenthandover-artifacts"));
        let watcher = HealthWatcher::new(5, 512)
            .with_artifact_path(artifact_dir);
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
        let mut shutdown_rx = health_shutdown_rx;

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    let status = watcher.check();
                    if !status.is_healthy() {
                        warn!(
                            accessibility = status.accessibility_permitted,
                            screen_recording = status.screen_recording_permitted,
                            disk_ok = status.disk_space_ok,
                            free_gb = status.free_disk_gb,
                            memory_mb = status.daemon_memory_mb,
                            "Health check: unhealthy"
                        );
                    }

                    // Write daemon status file for external consumers (CLI, menu bar app)
                    let now = chrono::Utc::now();
                    // Convert last NM message epoch_ms to DateTime, if any.
                    // If the in-memory atomic is 0 (no NM message in THIS daemon
                    // process), fall back to the extension-heartbeat.json file
                    // written by the separate NM bridge process.
                    let last_nm_ms = health_nm_ts.load(Ordering::Relaxed);
                    let last_ext_msg = if last_nm_ms > 0 {
                        chrono::DateTime::from_timestamp_millis(last_nm_ms)
                    } else {
                        // Fall back to extension heartbeat file (written by NM bridge)
                        agenthandover_common::status::read_extension_heartbeat()
                    };

                    // Read focus session signal for status reporting
                    let focus_session_info = {
                        let state_dir = agenthandover_common::status::data_dir();
                        agenthandover_common::focus_session::read_focus_signal(&state_dir)
                            .filter(|s| s.is_recording())
                            .map(|s| agenthandover_common::status::FocusSessionInfo {
                                session_id: s.session_id,
                                title: s.title,
                                started_at: s.started_at,
                            })
                    };

                    let last_maintenance = agenthandover_daemon::maintenance::read_record(
                        &agenthandover_common::status::data_dir(),
                    );

                    let daemon_status = agenthandover_common::status::DaemonStatus {
                        pid: std::process::id(),
                        version: env!("CARGO_PKG_VERSION").to_string(),
                        started_at: health_start_time,
                        heartbeat: now,
                        events_today: health_event_counter.load(Ordering::Relaxed),
                        permissions_ok: status.is_healthy(),
                        accessibility_permitted: status.accessibility_permitted,
                        screen_recording_permitted: status.screen_recording_permitted,
                        db_path: health_db_path.display().to_string(),
                        uptime_seconds: now.signed_duration_since(health_start_time)
                            .num_seconds()
                            .unsigned_abs(),
                        last_extension_message: last_ext_msg,
                        focus_session: focus_session_info,
                        last_maintenance: last_maintenance.as_ref().map(|r| r.finished_at),
                        events_purged: last_maintenance.as_ref().map(|r| r.events_purged),
                        artifacts_purged: last_maintenance.as_ref().map(|r| r.artifacts_purged),
                        bytes_freed: last_maintenance.as_ref().map(|r| r.bytes_freed),
                    };
                    if let Err(e) = agenthandover_common::status::write_status_file(
                        "daemon-status.json",
                        &daemon_status,
                    ) {
                        warn!("Failed to write daemon status file: {}", e);
                    }
                }
                _ = shutdown_rx.changed() => {
                    info!("Health watcher shutting down");
                    break;
                }
            }
        }
    });

    // Spawn the daily maintenance scheduler. It checks at startup and then
    // hourly; a run happens whenever the persisted date marker is older than
    // today, so a machine asleep overnight still gets its daily run.
    let maint_shutdown_rx = shutdown_tx.subscribe();
    let maint_db_path = db_path.clone();
    let maint_storage_config = app_config.storage.clone();
    let maint_screenshots_dir = config.screenshots_dir.clone();
    let maint_screenshot_ttl_days = app_config.vlm.job_ttl_days;
    let maint_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(3600));
        let mut shutdown_rx = maint_shutdown_rx;

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    let db_path = maint_db_path.clone();
                    let storage_config = maint_storage_config.clone();
                    let screenshots_dir = maint_screenshots_dir.clone();
                    let outcome = tokio::task::spawn_blocking(move || {
                        let state_dir = agenthandover_common::status::data_dir();
                        let today = chrono::Local::now().date_naive();
                        agenthandover_daemon::maintenance::run_if_due(&state_dir, today, || {
                            info!("Daily maintenance due — running full maintenance");
                            run_maintenance(
                                &db_path,
                                &storage_config,
                                &screenshots_dir,
                                maint_screenshot_ttl_days,
                            )
                        })
                    })
                    .await;
                    match outcome {
                        Ok(Ok(Some(record))) => info!(
                            events_purged = record.events_purged,
                            artifacts_purged = record.artifacts_purged,
                            bytes_freed = record.bytes_freed,
                            "Daily maintenance completed"
                        ),
                        Ok(Ok(None)) => debug!("Daily maintenance already ran today"),
                        Ok(Err(e)) => error!("Daily maintenance failed: {}", e),
                        Err(e) => error!("Daily maintenance task panicked: {}", e),
                    }
                }
                _ = shutdown_rx.changed() => {
                    info!("Maintenance timer shutting down");
                    break;
                }
            }
        }
    });

    // Set by the clipboard forwarder, consumed by the observer: a clipboard
    // change is a capture transition.
    let clipboard_changed = Arc::new(AtomicBool::new(false));

    // Spawn clipboard monitor (macOS only)
    #[cfg(target_os = "macos")]
    let clipboard_handle = {
        use agenthandover_daemon::platform::clipboard_monitor;
        let clip_tx = tx.clone();
        let clip_changed = Arc::clone(&clipboard_changed);
        let clip_shutdown_rx = shutdown_tx.subscribe();
        // Clipboard text is stored only when the user opted in.
        let clip_preview_max_chars = app_config
            .privacy
            .enable_clipboard_preview
            .then_some(app_config.privacy.clipboard_preview_max_chars);
        tokio::spawn(async move {
            let (clip_event_tx, mut clip_event_rx) = mpsc::channel(256);

            // Spawn the forwarder that converts ClipboardMessage -> ObserverMessage
            let fwd_tx = clip_tx;
            let forwarder = tokio::spawn(async move {
                let mut hash_tracker = clipboard_monitor::ClipboardHashTracker::new();
                while let Some(msg) = clip_event_rx.recv().await {
                    match msg {
                        clipboard_monitor::ClipboardMessage::Change(change) => {
                            hash_tracker.record(change.content_hash.clone());
                            clip_changed.store(true, Ordering::Release);
                            let event = agenthandover_common::event::Event {
                                id: uuid::Uuid::new_v4(),
                                timestamp: change.timestamp,
                                kind: agenthandover_common::event::EventKind::ClipboardChange {
                                    content_types: change.content_types,
                                    byte_size: change.byte_size,
                                    high_entropy: change.high_entropy,
                                    content_hash: change.content_hash,
                                    content_preview: change.content_preview,
                                },
                                window: None,
                                display_topology: vec![],
                                primary_display_id: "unknown".to_string(),
                                cursor_global_px: None,
                                ui_scale: None,
                                artifact_ids: vec![],
                                metadata: serde_json::json!({}),
                                display_ids_spanned: None,
                            };
                            match fwd_tx.try_send(ObserverMessage::Event { event }) {
                                Ok(()) => {}
                                Err(mpsc::error::TrySendError::Full(_)) => {
                                    warn!("Clipboard forwarder: main channel full (backpressure), dropping event");
                                }
                                Err(mpsc::error::TrySendError::Closed(_)) => {
                                    info!("Clipboard forwarder: main channel closed");
                                    break;
                                }
                            }
                        }
                        clipboard_monitor::ClipboardMessage::Shutdown => break,
                    }
                }
            });

            clipboard_monitor::run_clipboard_monitor(clip_event_tx, clip_shutdown_rx, clip_preview_max_chars).await;
            forwarder.abort();
        })
    };

    // Create artifact store for screenshot capture.
    // Key is derived from machine-specific data so each installation gets
    // a unique encryption key, while remaining deterministic on the same machine.
    let artifact_store = {
        use agenthandover_storage::artifact_store::ArtifactStore;

        let artifact_dir = match db_path.parent() {
            Some(parent) => parent.join("artifacts"),
            None => {
                let fallback = std::env::temp_dir().join("agenthandover-artifacts");
                error!("db_path has no parent directory, using temp dir: {}", fallback.display());
                fallback
            }
        };
        let key = derive_machine_key();

        Some(std::sync::Arc::new(
            ArtifactStore::new(artifact_dir, key),
        ))
    };

    // Run observer loop (blocks until shutdown)
    let observer_result = run_observer_loop(
        config,
        tx,
        shutdown_rx,
        artifact_store,
        capture_paused,
        clipboard_changed,
    )
    .await;

    // Abort background tasks
    native_handle.abort();
    health_handle.abort();
    maint_handle.abort();
    control_handle.abort();
    #[cfg(target_os = "macos")]
    clipboard_handle.abort();

    // Wait for storage writer to finish
    storage_handle.await??;

    // Remove PID file on clean shutdown
    agenthandover_common::pid::remove_pid_file("daemon");

    // Remove daemon-status.json on clean shutdown so CLI/app status tooling
    // reports a clean "Stopped" state instead of reading a stale heartbeat
    // with a dead PID (which renders as "not responding" — the confusing
    // v0.2.6 symptom hikoae reported after toggling "Observe Me" off).
    // Matches the worker's _remove_worker_status() behavior on clean exit.
    let status_path = agenthandover_common::status::status_dir().join("daemon-status.json");
    let _ = std::fs::remove_file(status_path);

    info!("agenthandover-daemon stopped");
    observer_result
}

/// Lightweight mode: only run the Native Messaging bridge.
///
/// Chrome launches a new process for each NM session.  This mode skips the
/// full observer loop, health watcher, clipboard monitor, and maintenance
/// timer — it only relays browser events from Chrome to the shared SQLite DB
/// (which is managed by the launchd-managed daemon instance).
async fn run_native_messaging_bridge() -> Result<()> {
    // Minimal logging — write to a separate log so NM output stays clean
    let log_dir = {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        if cfg!(target_os = "macos") {
            PathBuf::from(&home).join("Library/Application Support/agenthandover/logs")
        } else {
            PathBuf::from(&home).join(".local/share/agenthandover/logs")
        }
    };
    std::fs::create_dir_all(&log_dir).ok();

    let file_appender = rolling::daily(&log_dir, "nm-bridge.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("info".parse()?),
        )
        .with_writer(non_blocking)
        .with_ansi(false)
        .init();

    info!("agenthandover-daemon starting in native-messaging bridge mode");

    // Open the shared database (created by the full daemon)
    let db_path = {
        let data_dir = if cfg!(target_os = "macos") {
            dirs_or_home("Library/Application Support/agenthandover")
        } else {
            dirs_or_home(".local/share/agenthandover")
        };
        data_dir.join("events.db")
    };

    if !db_path.exists() {
        error!("Database not found at {}. Is the daemon running?", db_path.display());
        // Still run — the NM server needs to respond to Chrome even if DB is missing
    }

    // Open the storage writer directly
    let (tx, rx) = mpsc::channel(1000);
    let storage_handle = tokio::spawn({
        let db = db_path.clone();
        async move { run_storage_writer(db, rx, None).await }
    });

    // Write initial extension heartbeat immediately so CLI/SwiftUI can detect
    // the bridge right away (before any Chrome messages arrive).
    let session_started = chrono::Utc::now();
    let initial_heartbeat = agenthandover_common::status::ExtensionHeartbeat {
        pid: std::process::id(),
        last_message: session_started,
        messages_this_session: 0,
        session_started,
    };
    if let Err(e) = agenthandover_common::status::write_status_file(
        agenthandover_common::status::EXTENSION_HEARTBEAT_FILE,
        &initial_heartbeat,
    ) {
        warn!("Failed to write initial extension heartbeat: {}", e);
    }

    // Run native messaging — this blocks until Chrome closes the pipe
    let server = native_messaging::stdio_server();
    let (nm_event_tx, mut nm_event_rx) = mpsc::channel::<agenthandover_common::event::Event>(256);

    // Track the last active Chrome tab ID for outbound DOM snapshot requests.
    // Updated by the forwarder task from incoming extension messages.
    let last_active_tab_id = Arc::new(AtomicI64::new(-1));

    // Command channel: DOM request task -> NM server -> Chrome extension
    let (cmd_tx, cmd_rx) = mpsc::channel::<native_messaging::DaemonCommand>(64);

    let forwarder_tx = tx;
    let forwarder_tab_id = Arc::clone(&last_active_tab_id);
    let forwarder_handle = tokio::spawn(async move {
        let mut message_count: u64 = 0;
        let mut last_heartbeat_write = std::time::Instant::now();

        while let Some(event) = nm_event_rx.recv().await {
            message_count += 1;

            // Track last active tab ID for outbound DOM snapshot requests
            if let Some(tab_id) = event.metadata.get("tabId").and_then(|v| v.as_i64()) {
                forwarder_tab_id.store(tab_id, Ordering::Relaxed);
            }

            // Tell the observer to capture the page the user just acted on.
            note_browser_transition(&event);

            // Write extension heartbeat every 5 seconds (throttled)
            if last_heartbeat_write.elapsed() >= std::time::Duration::from_secs(5) {
                let heartbeat = agenthandover_common::status::ExtensionHeartbeat {
                    pid: std::process::id(),
                    last_message: chrono::Utc::now(),
                    messages_this_session: message_count,
                    session_started,
                };
                if let Err(e) = agenthandover_common::status::write_status_file(
                    agenthandover_common::status::EXTENSION_HEARTBEAT_FILE,
                    &heartbeat,
                ) {
                    warn!("Failed to write extension heartbeat: {}", e);
                }
                last_heartbeat_write = std::time::Instant::now();
            }

            match forwarder_tx.try_send(ObserverMessage::Event { event }) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    warn!("NM bridge: channel full, dropping event");
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    info!("NM bridge: channel closed");
                    break;
                }
            }
        }

        // Write final heartbeat on clean exit so stale detection is accurate
        let final_heartbeat = agenthandover_common::status::ExtensionHeartbeat {
            pid: std::process::id(),
            last_message: chrono::Utc::now(),
            messages_this_session: message_count,
            session_started,
        };
        if let Err(e) = agenthandover_common::status::write_status_file(
            agenthandover_common::status::EXTENSION_HEARTBEAT_FILE,
            &final_heartbeat,
        ) {
            warn!("Failed to write final extension heartbeat: {}", e);
        }
    });

    // Spawn DOM request task: polls focus signal file every 2s and sends
    // `request_snapshot` to Chrome when focus recording is active.  Each
    // request carries `[privacy] capture_input_values` so the content
    // script knows whether to include form-field values in the snapshot.
    let capture_input_values = load_app_config().privacy.capture_input_values;
    let dom_tab_id = Arc::clone(&last_active_tab_id);
    let dom_cmd_tx = cmd_tx.clone();
    let dom_request_handle = tokio::spawn(async move {
        let state_dir = agenthandover_common::status::data_dir();
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(2));

        loop {
            interval.tick().await;

            // Only request DOM snapshots during active focus recording
            let focus_signal = agenthandover_common::focus_session::read_focus_signal(&state_dir);
            let is_focus = focus_signal
                .as_ref()
                .map(|s| s.is_recording())
                .unwrap_or(false);

            if !is_focus {
                continue;
            }

            let tab_id = dom_tab_id.load(Ordering::Relaxed);
            if tab_id < 0 {
                // No tab seen yet — Chrome hasn't sent any messages
                debug!("DOM request: focus active but no tab ID yet");
                continue;
            }

            info!("DOM request: sending request_snapshot to tab {}", tab_id);

            let payload = serde_json::json!({
                "tabId": tab_id,
                "reason": "focus_recording",
                "captureInputValues": capture_input_values,
            });

            if dom_cmd_tx
                .send(native_messaging::DaemonCommand::SendMessage {
                    msg_type: native_messaging::DaemonMessageType::RequestSnapshot,
                    payload,
                })
                .await
                .is_err()
            {
                // Channel closed — server is shutting down
                break;
            }
        }
    });

    // Drop our clone of cmd_tx so the channel can close when dom_request_handle
    // is aborted (its clone is the only remaining sender).
    drop(cmd_tx);

    if let Err(e) = server.run(nm_event_tx, cmd_rx).await {
        // Normal: Chrome closed the NM connection
        info!("Native messaging session ended: {}", e);
    }
    // nm_event_tx is now dropped (it was moved into server.run()).
    // This causes nm_event_rx.recv() in the forwarder to return None,
    // so the forwarder loop exits cleanly after draining any buffered items.

    // Abort DOM request task — no more commands needed
    dom_request_handle.abort();

    // Await the forwarder — do NOT abort, so it can flush remaining events
    // from nm_event_rx into the storage writer channel (tx/forwarder_tx).
    if let Err(e) = forwarder_handle.await {
        warn!("NM bridge: forwarder task error: {}", e);
    }
    // forwarder_tx is now dropped (the forwarder task owned it and has exited),
    // so the storage writer's rx.recv() will return None after draining.

    // Await the storage writer to flush any remaining events to SQLite.
    match storage_handle.await {
        Ok(Ok(())) => info!("NM bridge: storage writer drained cleanly"),
        Ok(Err(e)) => warn!("NM bridge: storage writer error: {}", e),
        Err(e) => warn!("NM bridge: storage writer join error: {}", e),
    }

    info!("NM bridge exiting");
    Ok(())
}

/// Hand a browser click, navigation or form submit to the observer through
/// the transition file (the bridge is Chrome's process, not the observer's).
fn note_browser_transition(event: &agenthandover_common::event::Event) {
    use agenthandover_common::browser_transition::{write_browser_transition, BrowserTransition};

    let Some(kind) = native_messaging::browser_transition_kind(&event.kind) else {
        return;
    };
    let transition = BrowserTransition { kind, at: chrono::Utc::now() };
    let dir = agenthandover_common::status::data_dir();
    if let Err(e) = write_browser_transition(&dir, &transition) {
        warn!("Failed to write browser transition: {}", e);
    }
}

/// Load AppConfig from the standard config file location, falling back to
/// defaults when the file is missing or unparseable.
fn load_app_config() -> agenthandover_common::config::AppConfig {
    use agenthandover_common::config::AppConfig;

    let config_path = if cfg!(target_os = "macos") {
        std::env::var("HOME").ok().map(|home| {
            std::path::PathBuf::from(home)
                .join("Library/Application Support/agenthandover/config.toml")
        })
    } else {
        std::env::var("HOME").ok().map(|home| {
            std::path::PathBuf::from(home)
                .join(".config/agenthandover/config.toml")
        })
    };

    match config_path {
        Some(ref path) if path.is_file() => {
            match AppConfig::from_file(path) {
                Ok(cfg) => {
                    info!(path = %path.display(), "Loaded configuration from file");
                    cfg
                }
                Err(e) => {
                    error!(path = %path.display(), error = %e, "Failed to parse config, using defaults");
                    AppConfig::default()
                }
            }
        }
        _ => {
            info!("No config file found, using defaults");
            AppConfig::default()
        }
    }
}

/// Derive an encryption key unique to this machine.
///
/// On macOS, uses `sysctl -n kern.uuid` (IOPlatformUUID) as seed material.
/// Falls back to hostname + username if that fails.
/// Hashes "agenthandover-" + machine_id + "-artifact-key-v1" with SHA-256.
fn derive_machine_key() -> [u8; 32] {
    let machine_id = get_machine_id();
    let mut hasher = Sha256::new();
    hasher.update(format!("agenthandover-{}-artifact-key-v1", machine_id).as_bytes());
    hasher.finalize().into()
}

fn get_machine_id() -> String {
    // Try sysctl kern.uuid (macOS IOPlatformUUID)
    if let Ok(output) = std::process::Command::new("sysctl")
        .args(["-n", "kern.uuid"])
        .output()
    {
        let uuid = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !uuid.is_empty() {
            return uuid;
        }
    }

    // Fallback: hostname + username
    let hostname = std::process::Command::new("hostname")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown-host".to_string());
    let username = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "unknown-user".to_string());
    format!("{}-{}", hostname, username)
}

/// Resolve a subpath under $HOME, e.g. `".local/share/agenthandover"`.
fn dirs_or_home(subpath: &str) -> PathBuf {
    std::env::var("HOME")
        .map(|h| PathBuf::from(h).join(subpath))
        .unwrap_or_else(|_| PathBuf::from(subpath))
}

/// Run full database maintenance cycle using values from the loaded config.
fn run_maintenance(
    db_path: &std::path::Path,
    storage_config: &agenthandover_common::config::StorageConfig,
    screenshots_dir: &std::path::Path,
    screenshot_ttl_days: u32,
) -> Result<agenthandover_storage::maintenance::MaintenanceReport> {
    use agenthandover_storage::maintenance::{MaintenanceOptions, MaintenanceRunner};

    // Default artifact max: 10 GB
    const ARTIFACT_MAX_BYTES: u64 = 10 * 1024 * 1024 * 1024;

    let artifact_dir = db_path.parent().map(|p| p.join("artifacts"));
    let conn = rusqlite::Connection::open(db_path)?;
    // Wait for the worker's writes instead of failing with SQLITE_BUSY.
    conn.pragma_update(None, "busy_timeout", 30000)?;
    let runner = MaintenanceRunner::new(&conn);
    runner.run_full_maintenance(&MaintenanceOptions {
        db_path,
        retention_days_raw: storage_config.retention_days_raw,
        retention_days_episodes: storage_config.retention_days_episodes,
        min_free_gb: storage_config.vacuum_min_free_gb,
        vacuum_safety_multiplier: storage_config.vacuum_safety_multiplier,
        artifact_dir: artifact_dir.as_deref(),
        artifact_max_bytes: Some(ARTIFACT_MAX_BYTES),
        screenshots_dir: Some(screenshots_dir),
        screenshot_ttl_days,
    })
}
