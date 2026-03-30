#[cfg(target_os = "macos")]
mod macos_tests {
    use agenthandover_daemon::platform::IdleDetector;

    #[test]
    fn test_idle_seconds_returns_non_negative() {
        let detector = IdleDetector::new();
        let idle = detector.seconds_since_last_input();
        assert!(
            idle >= 0.0,
            "Idle time should be non-negative, got: {}",
            idle
        );
    }

    #[test]
    fn test_is_user_idle_with_threshold() {
        let detector = IdleDetector::new();
        // With a very high threshold, user should not be "idle"
        assert!(!detector.is_idle(999_999.0));
    }

    #[test]
    fn test_default_constructor() {
        let detector = IdleDetector::default();
        let idle = detector.seconds_since_last_input();
        assert!(idle >= 0.0);
    }
}
