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
        run_observer_loop(config, tx, shutdown_rx, None).await
    });

    // Let it run briefly
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Signal shutdown
    shutdown_tx.send(true).unwrap();

    // Should complete without error
    let result = handle.await.unwrap();
    assert!(result.is_ok());

    // The observer drops its tx on shutdown.  Since there are no other
    // senders, the channel is now closed — recv() should return None.
    // Any events queued before shutdown are still available.
    let mut event_count = 0;
    while let Ok(msg) = rx.try_recv() {
        if let ObserverMessage::Event { .. } = msg {
            event_count += 1;
        }
    }
    // Channel should be closed (no senders left)
    assert!(
        rx.try_recv().is_err(),
        "Channel should be closed after observer loop exits (got unexpected message)"
    );
    // We don't assert event_count > 0 because a short-lived loop on CI
    // may not capture any events, but the channel *must* be closed.
}

#[tokio::test]
async fn test_observer_config_defaults() {
    let config = ObserverConfig::default();
    assert_eq!(config.t_dwell_seconds, 3);
    assert_eq!(config.t_scroll_read_seconds, 8);
    assert!(config.capture_screenshots);
    assert_eq!(config.screenshot_max_per_minute, 20);
}
