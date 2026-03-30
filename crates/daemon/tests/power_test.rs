#[cfg(target_os = "macos")]
mod power_tests {
    use agenthandover_daemon::platform::power::{check_power_gates, get_power_state};

    #[test]
    fn test_power_state_returns_valid_data() {
        let state = get_power_state().unwrap();
        // Battery percent should be 0-100 if present
        if let Some(pct) = state.battery_percent {
            assert!(pct <= 100, "Battery should be 0-100, got {}", pct);
        }
    }

    #[test]
    fn test_check_power_gates_with_relaxed_thresholds() {
        // Very relaxed thresholds should pass
        let result = check_power_gates(false, 0, 200, 100).unwrap();
        assert!(
            result.passed,
            "Relaxed gates should pass: {:?}",
            result.rejection_reasons
        );
    }

    #[test]
    fn test_check_power_gates_with_strict_thresholds() {
        // Impossibly strict thresholds should fail
        let result = check_power_gates(true, 100, 0, 0).unwrap();
        assert!(!result.passed, "Impossibly strict gates should fail");
        assert!(!result.rejection_reasons.is_empty());
    }
}
