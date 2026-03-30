#[cfg(target_os = "macos")]
mod screenshot_tests {
    use agenthandover_daemon::capture::screenshot::{
        capture_main_display, has_screen_recording_permission,
    };

    #[test]
    fn test_capture_returns_data_if_permitted() {
        // May return None without screen recording permission
        if has_screen_recording_permission() {
            let result = capture_main_display();
            assert!(result.is_some());
            let (width, height, bytes) = result.unwrap();
            assert!(width > 0);
            assert!(height > 0);
            assert!(!bytes.is_empty());
        }
    }

    #[test]
    fn test_permission_check_does_not_panic() {
        // This should never panic regardless of permission state
        let _has_perm = has_screen_recording_permission();
    }
}
