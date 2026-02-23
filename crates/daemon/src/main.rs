use anyhow::Result;
use chrono::Timelike;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use tokio::sync::{mpsc, watch};
use tracing::{info, warn, error};
use tracing_subscriber::EnvFilter;
use tracing_appender::rolling;
use sha2::{Digest, Sha256};

use oc_apprentice_daemon::ipc::native_messaging;
use oc_apprentice_daemon::observer::event_loop::{
    ObserverConfig, ObserverMessage, run_observer_loop, run_storage_writer,
};
use oc_apprentice_daemon::observer::health::HealthWatcher;

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

    let log_dir = {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        if cfg!(target_os = "macos") {
            PathBuf::from(&home).join("Library/Application Support/oc-apprentice/logs")
        } else {
            PathBuf::from(&home).join(".local/share/oc-apprentice/logs")
        }
    };
    std::fs::create_dir_all(&log_dir).ok();

    let file_appender = rolling::daily(&log_dir, "daemon.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("info".parse()?),
        )
        .with_writer(non_blocking)
        .with_ansi(false)
        .init();

    info!("oc-apprentice-daemon starting");

    // Write PID file
    let pid_path = oc_apprentice_common::pid::write_pid_file("daemon")
        .expect("Failed to write daemon PID file");
    info!(path = %pid_path.display(), "PID file written");

    let start_time = chrono::Utc::now();

    // Shared event counter — incremented by storage writer, read by health watcher
    let event_counter = Arc::new(AtomicU64::new(0));

    // Timestamp (epoch millis) of the last Chrome extension NM message.
    // 0 = no message received yet.  Updated by the NM forwarder task.
    let last_nm_message_epoch_ms = Arc::new(AtomicI64::new(0));

    // Load AppConfig from standard config file location, fall back to defaults
    let app_config = {
        use oc_apprentice_common::config::AppConfig;

        let config_path = if cfg!(target_os = "macos") {
            std::env::var("HOME").ok().map(|home| {
                std::path::PathBuf::from(home)
                    .join("Library/Application Support/oc-apprentice/config.toml")
            })
        } else {
            std::env::var("HOME").ok().map(|home| {
                std::path::PathBuf::from(home)
                    .join(".config/oc-apprentice/config.toml")
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
    };

    // Convert AppConfig -> ObserverConfig
    let config = ObserverConfig {
        t_dwell_seconds: app_config.observer.t_dwell_seconds,
        t_scroll_read_seconds: app_config.observer.t_scroll_read_seconds,
        capture_screenshots: app_config.observer.capture_screenshots,
        screenshot_max_per_minute: app_config.observer.screenshot_max_per_minute,
        poll_interval: std::time::Duration::from_millis(500),
        db_path: {
            // Use the same standard path the worker expects so both
            // processes find the database without manual --db-path flags.
            let data_dir = if cfg!(target_os = "macos") {
                dirs_or_home("Library/Application Support/oc-apprentice")
            } else {
                dirs_or_home(".local/share/oc-apprentice")
            };
            std::fs::create_dir_all(&data_dir).ok();
            data_dir.join("events.db")
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
                match forwarder_tx.try_send(ObserverMessage::Event(event)) {
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
        let mut server = native_messaging::stdio_server();
        if let Err(e) = server.run(nm_event_tx).await {
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
            .unwrap_or_else(|| std::env::temp_dir().join("openmimic-artifacts"));
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
                        oc_apprentice_common::status::read_extension_heartbeat()
                    };

                    // Read focus session signal for status reporting
                    let focus_session_info = {
                        let state_dir = oc_apprentice_common::status::data_dir();
                        oc_apprentice_common::focus_session::read_focus_signal(&state_dir)
                            .filter(|s| s.is_recording())
                            .map(|s| oc_apprentice_common::status::FocusSessionInfo {
                                session_id: s.session_id,
                                title: s.title,
                                started_at: s.started_at,
                            })
                    };

                    let daemon_status = oc_apprentice_common::status::DaemonStatus {
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
                    };
                    if let Err(e) = oc_apprentice_common::status::write_status_file(
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

    // Spawn nightly maintenance trigger
    let maint_shutdown_rx = shutdown_tx.subscribe();
    let maint_db_path = db_path.clone();
    let maint_storage_config = app_config.storage.clone();
    let maint_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(3600));
        let mut shutdown_rx = maint_shutdown_rx;

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    let hour = chrono::Local::now().hour();
                    if hour >= 1 && hour < 5 {
                        info!("Nightly maintenance window — running full maintenance");
                        let artifact_dir = maint_db_path.parent()
                            .map(|p| p.join("artifacts"));
                        match run_maintenance(
                            &maint_db_path,
                            &maint_storage_config,
                            artifact_dir.as_deref(),
                        ) {
                            Ok(report) => {
                                // Actually delete artifact files from disk
                                let mut files_deleted = 0usize;
                                let mut files_failed = 0usize;
                                for path_str in &report.artifact_paths_to_delete {
                                    let path = std::path::Path::new(path_str);
                                    if path.exists() {
                                        match std::fs::remove_file(path) {
                                            Ok(()) => files_deleted += 1,
                                            Err(e) => {
                                                warn!(
                                                    path = %path.display(),
                                                    error = %e,
                                                    "Failed to delete artifact file"
                                                );
                                                files_failed += 1;
                                            }
                                        }
                                    }
                                }
                                info!(
                                    events_purged = report.events_purged,
                                    episodes_purged = report.episodes_purged,
                                    vlm_purged = report.vlm_jobs_purged,
                                    artifact_rows = report.artifact_paths_to_delete.len(),
                                    artifact_size_evicted = report.artifact_size_evicted,
                                    artifact_files_deleted = files_deleted,
                                    artifact_files_failed = files_failed,
                                    vacuumed = report.vacuumed,
                                    "Nightly maintenance completed"
                                );
                            }
                            Err(e) => {
                                error!("Nightly maintenance failed: {}", e);
                            }
                        }
                    }
                }
                _ = shutdown_rx.changed() => {
                    info!("Maintenance timer shutting down");
                    break;
                }
            }
        }
    });

    // Spawn clipboard monitor (macOS only)
    #[cfg(target_os = "macos")]
    let clipboard_handle = {
        use oc_apprentice_daemon::platform::clipboard_monitor;
        let clip_tx = tx.clone();
        let clip_shutdown_rx = shutdown_tx.subscribe();
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
                            let event = oc_apprentice_common::event::Event {
                                id: uuid::Uuid::new_v4(),
                                timestamp: change.timestamp,
                                kind: oc_apprentice_common::event::EventKind::ClipboardChange {
                                    content_types: change.content_types,
                                    byte_size: change.byte_size,
                                    high_entropy: change.high_entropy,
                                    content_hash: change.content_hash,
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
                            match fwd_tx.try_send(ObserverMessage::Event(event)) {
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

            clipboard_monitor::run_clipboard_monitor(clip_event_tx, clip_shutdown_rx).await;
            forwarder.abort();
        })
    };

    // Create artifact store for screenshot capture.
    // Key is derived from machine-specific data so each installation gets
    // a unique encryption key, while remaining deterministic on the same machine.
    let artifact_store = {
        use oc_apprentice_storage::artifact_store::ArtifactStore;

        let artifact_dir = match db_path.parent() {
            Some(parent) => parent.join("artifacts"),
            None => {
                let fallback = std::env::temp_dir().join("openmimic-artifacts");
                error!("db_path has no parent directory, using temp dir: {}", fallback.display());
                fallback
            }
        };
        let key = derive_machine_key();

        Some(std::sync::Arc::new(ArtifactStore::new(artifact_dir, key)))
    };

    // Run observer loop (blocks until shutdown)
    let observer_result = run_observer_loop(config, tx, shutdown_rx, artifact_store).await;

    // Abort background tasks
    native_handle.abort();
    health_handle.abort();
    maint_handle.abort();
    #[cfg(target_os = "macos")]
    clipboard_handle.abort();

    // Wait for storage writer to finish
    storage_handle.await??;

    // Remove PID file on clean shutdown
    oc_apprentice_common::pid::remove_pid_file("daemon");

    info!("oc-apprentice-daemon stopped");
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
            PathBuf::from(&home).join("Library/Application Support/oc-apprentice/logs")
        } else {
            PathBuf::from(&home).join(".local/share/oc-apprentice/logs")
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

    info!("oc-apprentice-daemon starting in native-messaging bridge mode");

    // Open the shared database (created by the full daemon)
    let db_path = {
        let data_dir = if cfg!(target_os = "macos") {
            dirs_or_home("Library/Application Support/oc-apprentice")
        } else {
            dirs_or_home(".local/share/oc-apprentice")
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
    let initial_heartbeat = oc_apprentice_common::status::ExtensionHeartbeat {
        pid: std::process::id(),
        last_message: session_started,
        messages_this_session: 0,
        session_started,
    };
    if let Err(e) = oc_apprentice_common::status::write_status_file(
        oc_apprentice_common::status::EXTENSION_HEARTBEAT_FILE,
        &initial_heartbeat,
    ) {
        warn!("Failed to write initial extension heartbeat: {}", e);
    }

    // Run native messaging — this blocks until Chrome closes the pipe
    let mut server = native_messaging::stdio_server();
    let (nm_event_tx, mut nm_event_rx) = mpsc::channel(256);

    let forwarder_tx = tx;
    let forwarder_handle = tokio::spawn(async move {
        let mut message_count: u64 = 0;
        let mut last_heartbeat_write = std::time::Instant::now();

        while let Some(event) = nm_event_rx.recv().await {
            message_count += 1;

            // Write extension heartbeat every 5 seconds (throttled)
            if last_heartbeat_write.elapsed() >= std::time::Duration::from_secs(5) {
                let heartbeat = oc_apprentice_common::status::ExtensionHeartbeat {
                    pid: std::process::id(),
                    last_message: chrono::Utc::now(),
                    messages_this_session: message_count,
                    session_started,
                };
                if let Err(e) = oc_apprentice_common::status::write_status_file(
                    oc_apprentice_common::status::EXTENSION_HEARTBEAT_FILE,
                    &heartbeat,
                ) {
                    warn!("Failed to write extension heartbeat: {}", e);
                }
                last_heartbeat_write = std::time::Instant::now();
            }

            match forwarder_tx.try_send(ObserverMessage::Event(event)) {
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
        let final_heartbeat = oc_apprentice_common::status::ExtensionHeartbeat {
            pid: std::process::id(),
            last_message: chrono::Utc::now(),
            messages_this_session: message_count,
            session_started,
        };
        if let Err(e) = oc_apprentice_common::status::write_status_file(
            oc_apprentice_common::status::EXTENSION_HEARTBEAT_FILE,
            &final_heartbeat,
        ) {
            warn!("Failed to write final extension heartbeat: {}", e);
        }
    });

    if let Err(e) = server.run(nm_event_tx).await {
        // Normal: Chrome closed the NM connection
        info!("Native messaging session ended: {}", e);
    }
    // nm_event_tx is now dropped (it was moved into server.run()).
    // This causes nm_event_rx.recv() in the forwarder to return None,
    // so the forwarder loop exits cleanly after draining any buffered items.

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

/// Derive an encryption key unique to this machine.
///
/// On macOS, uses `sysctl -n kern.uuid` (IOPlatformUUID) as seed material.
/// Falls back to hostname + username if that fails.
/// Hashes "openmimic-" + machine_id + "-artifact-key-v1" with SHA-256.
fn derive_machine_key() -> [u8; 32] {
    let machine_id = get_machine_id();
    let mut hasher = Sha256::new();
    hasher.update(format!("openmimic-{}-artifact-key-v1", machine_id).as_bytes());
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

/// Resolve a subpath under $HOME, e.g. `".local/share/oc-apprentice"`.
fn dirs_or_home(subpath: &str) -> PathBuf {
    std::env::var("HOME")
        .map(|h| PathBuf::from(h).join(subpath))
        .unwrap_or_else(|_| PathBuf::from(subpath))
}

/// Run full database maintenance cycle using values from the loaded config.
fn run_maintenance(
    db_path: &std::path::Path,
    storage_config: &oc_apprentice_common::config::StorageConfig,
    artifact_dir: Option<&std::path::Path>,
) -> Result<oc_apprentice_storage::maintenance::MaintenanceReport> {
    use oc_apprentice_storage::maintenance::MaintenanceRunner;

    // Default artifact max: 10 GB
    const ARTIFACT_MAX_BYTES: u64 = 10 * 1024 * 1024 * 1024;

    let conn = rusqlite::Connection::open(db_path)?;
    let runner = MaintenanceRunner::new(&conn);
    runner.run_full_maintenance(
        db_path,
        storage_config.retention_days_raw,
        storage_config.retention_days_episodes,
        storage_config.vacuum_min_free_gb,
        storage_config.vacuum_safety_multiplier,
        artifact_dir,
        Some(ARTIFACT_MAX_BYTES),
    )
}
