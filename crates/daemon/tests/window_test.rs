#[cfg(target_os = "macos")]
mod window_tests {
    use agenthandover_daemon::platform::window_capture::{get_display_topology, get_focused_window};

    #[test]
    fn test_display_topology_returns_at_least_one() {
        let displays = get_display_topology();
        assert!(!displays.is_empty(), "Should detect at least one display");
        for d in &displays {
            assert!(d.bounds_global_px[2] > 0, "Display width should be positive");
            assert!(d.bounds_global_px[3] > 0, "Display height should be positive");
            assert!(d.scale_factor >= 1.0, "Scale factor should be >= 1.0");
        }
    }

    #[test]
    fn test_focused_window_does_not_panic() {
        let _window = get_focused_window();
    }
}
