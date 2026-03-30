use agenthandover_daemon::observer::dwell::DwellTracker;
use std::time::Duration;

#[test]
fn test_dwell_starts_inactive() {
    let mut tracker = DwellTracker::new(Duration::from_secs(3), Duration::from_secs(8));
    assert!(!tracker.is_dwelling());
    assert!(!tracker.is_scroll_reading());
}

#[test]
fn test_manipulation_resets_dwell() {
    let mut tracker = DwellTracker::new(Duration::from_secs(3), Duration::from_secs(8));
    tracker.on_navigation_input();
    tracker.on_manipulation_input();
    assert!(!tracker.is_dwelling());
}

#[test]
fn test_dwell_triggers_after_threshold() {
    let mut tracker = DwellTracker::new(Duration::from_millis(50), Duration::from_millis(200));
    std::thread::sleep(Duration::from_millis(60));
    tracker.tick();
    assert!(tracker.is_dwelling());
}

#[test]
fn test_scroll_reading_triggers_after_threshold() {
    let mut tracker = DwellTracker::new(Duration::from_millis(50), Duration::from_millis(100));
    for _ in 0..5 {
        tracker.on_navigation_input();
        std::thread::sleep(Duration::from_millis(25));
        tracker.tick();
    }
    assert!(tracker.is_scroll_reading());
}
