use anyhow::Result;
use tokio::sync::{mpsc, watch};
use tracing::info;
use tracing_subscriber::EnvFilter;

use oc_apprentice_daemon::observer::event_loop::{
    ObserverConfig, run_observer_loop, run_storage_writer,
};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("info".parse()?),
        )
        .init();

    info!("oc-apprentice-daemon starting");

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

    let config = ObserverConfig::default();
    let db_path = config.db_path.clone();

    // Spawn storage writer
    let storage_handle = tokio::spawn(run_storage_writer(db_path, rx));

    // Run observer loop (blocks until shutdown)
    let observer_result = run_observer_loop(config, tx, shutdown_rx).await;

    // Wait for storage writer to finish
    storage_handle.await??;

    info!("oc-apprentice-daemon stopped");
    observer_result
}
