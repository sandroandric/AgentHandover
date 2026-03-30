use agenthandover_daemon::observer::health::HealthWatcher;

#[test]
fn test_health_check_does_not_panic() {
    let watcher = HealthWatcher::new(1, 512);
    let status = watcher.check();
    // Basic sanity — should not panic
    let _ = status.is_healthy();
}

#[test]
fn test_health_status_disk_space() {
    let watcher = HealthWatcher::new(0, 999999);
    let status = watcher.check();
    // With 0 GB minimum, disk should be ok
    assert!(status.disk_space_ok);
}

#[test]
fn test_health_status_with_impossible_disk_requirement() {
    let watcher = HealthWatcher::new(999999, 512);
    let status = watcher.check();
    assert!(!status.disk_space_ok);
}
