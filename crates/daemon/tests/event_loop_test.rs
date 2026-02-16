use oc_apprentice_daemon::observer::event_loop::{ObserverConfig, ObserverMessage, run_observer_loop};
use tokio::sync::{mpsc, watch};
use std::time::Duration;

#[tokio::test]
async fn test_observer_loop_starts_and_stops() {
    let (tx, mut rx) = mpsc::channel(100);
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let config = ObserverConfig {
        poll_interval: Duration::from_millis(50),
        ..Default::default()
    };

    // Run the loop for a short time then signal shutdown
    let handle = tokio::spawn(async move {
        run_observer_loop(config, tx, shutdown_rx).await
    });

    // Let it run briefly
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Signal shutdown
    shutdown_tx.send(true).unwrap();

    // Should complete without error
    let result = handle.await.unwrap();
    assert!(result.is_ok());

    // Should have received a shutdown message
    let mut got_shutdown = false;
    while let Ok(msg) = rx.try_recv() {
        if matches!(msg, ObserverMessage::Shutdown) {
            got_shutdown = true;
        }
    }
    assert!(got_shutdown, "Should have received shutdown message");
}

#[tokio::test]
async fn test_observer_config_defaults() {
    let config = ObserverConfig::default();
    assert_eq!(config.t_dwell_seconds, 3);
    assert_eq!(config.t_scroll_read_seconds, 8);
    assert!(config.capture_screenshots);
    assert_eq!(config.screenshot_max_per_minute, 20);
}
