use anyhow::Result;
use chrono::Timelike;
use std::path::PathBuf;
use tokio::sync::{mpsc, watch};
use tracing::{info, warn, error};
use tracing_subscriber::EnvFilter;
use sha2::{Digest, Sha256};

use oc_apprentice_daemon::ipc::native_messaging;
use oc_apprentice_daemon::observer::event_loop::{
    ObserverConfig, ObserverMessage, run_observer_loop, run_storage_writer,
};
use oc_apprentice_daemon::observer::health::HealthWatcher;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("info".parse()?),
        )
        .init();

    info!("oc-apprentice-daemon starting");

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
                    .join(".config/openclaw-apprentice/config.toml")
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
    let storage_handle = tokio::spawn(run_storage_writer(db_path.clone(), rx));

    // Spawn native messaging server (Chrome extension bridge)
    let native_tx = tx.clone();
    let native_handle = tokio::spawn(async move {
        // Create a channel to receive events from the native messaging server
        let (nm_event_tx, mut nm_event_rx) = mpsc::channel(256);

        // Spawn the forwarder that bridges Event -> ObserverMessage
        let forwarder_tx = native_tx;
        let forwarder_handle = tokio::spawn(async move {
            while let Some(event) = nm_event_rx.recv().await {
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

    // Spawn health watcher (periodic background health checks)
    let health_shutdown_rx = shutdown_tx.subscribe();
    let health_db_path = db_path.clone();
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
                        match run_maintenance(&maint_db_path, &maint_storage_config) {
                            Ok(report) => {
                                info!(
                                    events_purged = report.events_purged,
                                    episodes_purged = report.episodes_purged,
                                    vlm_purged = report.vlm_jobs_purged,
                                    artifacts = report.artifact_paths_to_delete.len(),
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

    info!("oc-apprentice-daemon stopped");
    observer_result
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
) -> Result<oc_apprentice_storage::maintenance::MaintenanceReport> {
    use oc_apprentice_storage::maintenance::MaintenanceRunner;

    let conn = rusqlite::Connection::open(db_path)?;
    let runner = MaintenanceRunner::new(&conn);
    runner.run_full_maintenance(
        db_path,
        storage_config.retention_days_raw,
        storage_config.retention_days_episodes,
        storage_config.vacuum_min_free_gb,
        storage_config.vacuum_safety_multiplier,
    )
}
