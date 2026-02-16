//! Integration tests for the CDP (Chrome DevTools Protocol) bridge.
//!
//! Tests cover:
//!   - CdpConfig defaults (disabled, empty allowlist)
//!   - is_allowed: disabled, not in allowlist, enabled + allowlisted
//!   - validate_localhost: valid and invalid hosts
//!   - connect: Disabled, NotAllowlisted, NotLocalhost results
//!   - CdpTarget serialization roundtrip
//!   - build_cdp_eval_script: valid JS with escaped selector

use oc_apprentice_daemon::ipc::cdp_bridge::{
    build_cdp_eval_script, CdpBridge, CdpConfig, CdpConnectionResult, CdpTarget,
};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// CdpConfig default tests
// ---------------------------------------------------------------------------

#[test]
fn cdp_config_default_is_disabled() {
    let config = CdpConfig::default();
    assert!(!config.enabled);
}

#[test]
fn cdp_config_default_has_empty_allowlist() {
    let config = CdpConfig::default();
    assert!(config.allowlist.is_empty());
}

#[test]
fn cdp_config_default_timeout() {
    let config = CdpConfig::default();
    assert_eq!(config.connect_timeout_ms, 5000);
}

#[test]
fn cdp_config_default_retries() {
    let config = CdpConfig::default();
    assert_eq!(config.max_retries, 3);
}

// ---------------------------------------------------------------------------
// is_allowed tests
// ---------------------------------------------------------------------------

#[test]
fn cdp_is_allowed_returns_false_when_disabled() {
    let config = CdpConfig {
        enabled: false,
        allowlist: HashSet::from(["com.test.app".to_string()]),
        ..CdpConfig::default()
    };
    let bridge = CdpBridge::new(config);
    assert!(!bridge.is_allowed("com.test.app"));
}

#[test]
fn cdp_is_allowed_returns_false_when_not_in_allowlist() {
    let config = CdpConfig {
        enabled: true,
        allowlist: HashSet::from(["com.test.allowed".to_string()]),
        ..CdpConfig::default()
    };
    let bridge = CdpBridge::new(config);
    assert!(!bridge.is_allowed("com.test.other"));
}

#[test]
fn cdp_is_allowed_returns_true_when_enabled_and_allowlisted() {
    let config = CdpConfig {
        enabled: true,
        allowlist: HashSet::from(["com.test.app".to_string()]),
        ..CdpConfig::default()
    };
    let bridge = CdpBridge::new(config);
    assert!(bridge.is_allowed("com.test.app"));
}

#[test]
fn cdp_is_allowed_returns_false_when_enabled_but_empty_allowlist() {
    let config = CdpConfig {
        enabled: true,
        allowlist: HashSet::new(),
        ..CdpConfig::default()
    };
    let bridge = CdpBridge::new(config);
    assert!(!bridge.is_allowed("com.test.app"));
}

// ---------------------------------------------------------------------------
// validate_localhost tests
// ---------------------------------------------------------------------------

#[test]
fn cdp_validate_localhost_accepts_127_0_0_1() {
    assert!(CdpBridge::validate_localhost("127.0.0.1"));
}

#[test]
fn cdp_validate_localhost_accepts_localhost() {
    assert!(CdpBridge::validate_localhost("localhost"));
}

#[test]
fn cdp_validate_localhost_accepts_ipv6_loopback() {
    assert!(CdpBridge::validate_localhost("::1"));
}

#[test]
fn cdp_validate_localhost_accepts_ipv6_bracketed() {
    assert!(CdpBridge::validate_localhost("[::1]"));
}

#[test]
fn cdp_validate_localhost_rejects_private_192() {
    assert!(!CdpBridge::validate_localhost("192.168.1.1"));
}

#[test]
fn cdp_validate_localhost_rejects_private_10() {
    assert!(!CdpBridge::validate_localhost("10.0.0.1"));
}

#[test]
fn cdp_validate_localhost_rejects_hostname() {
    assert!(!CdpBridge::validate_localhost("evil.com"));
}

#[test]
fn cdp_validate_localhost_rejects_0_0_0_0() {
    assert!(!CdpBridge::validate_localhost("0.0.0.0"));
}

#[test]
fn cdp_validate_localhost_rejects_empty() {
    assert!(!CdpBridge::validate_localhost(""));
}

// ---------------------------------------------------------------------------
// connect tests
// ---------------------------------------------------------------------------

#[test]
fn cdp_connect_returns_disabled_when_not_enabled() {
    let config = CdpConfig::default(); // enabled: false
    let bridge = CdpBridge::new(config);

    let result = bridge.connect("com.test.app", "127.0.0.1", 9222);
    assert_eq!(result, CdpConnectionResult::Disabled);
}

#[test]
fn cdp_connect_returns_not_allowlisted() {
    let config = CdpConfig {
        enabled: true,
        allowlist: HashSet::from(["com.test.other".to_string()]),
        ..CdpConfig::default()
    };
    let bridge = CdpBridge::new(config);

    let result = bridge.connect("com.test.unlisted", "127.0.0.1", 9222);
    match result {
        CdpConnectionResult::NotAllowlisted { bundle_id } => {
            assert_eq!(bundle_id, "com.test.unlisted");
        }
        other => panic!("Expected NotAllowlisted, got {:?}", other),
    }
}

#[test]
fn cdp_connect_returns_not_localhost_for_remote_host() {
    let config = CdpConfig {
        enabled: true,
        allowlist: HashSet::from(["com.test.app".to_string()]),
        ..CdpConfig::default()
    };
    let bridge = CdpBridge::new(config);

    let result = bridge.connect("com.test.app", "192.168.1.1", 9222);
    assert_eq!(result, CdpConnectionResult::NotLocalhost);
}

#[test]
fn cdp_connect_returns_not_localhost_for_evil_hostname() {
    let config = CdpConfig {
        enabled: true,
        allowlist: HashSet::from(["com.test.app".to_string()]),
        ..CdpConfig::default()
    };
    let bridge = CdpBridge::new(config);

    let result = bridge.connect("com.test.app", "evil.com", 9222);
    assert_eq!(result, CdpConnectionResult::NotLocalhost);
}

#[test]
fn cdp_connect_to_closed_port_returns_connection_failed() {
    let config = CdpConfig {
        enabled: true,
        allowlist: HashSet::from(["com.test.app".to_string()]),
        // Short timeout so the test doesn't take too long
        connect_timeout_ms: 100,
        max_retries: 1,
    };
    let bridge = CdpBridge::new(config);

    // Use an unlikely-to-be-open port
    let result = bridge.connect("com.test.app", "127.0.0.1", 19999);
    match result {
        CdpConnectionResult::ConnectionFailed { reason } => {
            assert!(
                reason.contains("Failed to connect"),
                "Expected failure message, got: {}",
                reason
            );
        }
        other => panic!("Expected ConnectionFailed, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// CdpTarget serialization tests
// ---------------------------------------------------------------------------

#[test]
fn cdp_target_serde_roundtrip() {
    let target = CdpTarget {
        id: "target-123".to_string(),
        title: "Main Page".to_string(),
        url: "https://app.example.com".to_string(),
        target_type: "page".to_string(),
    };

    let json = serde_json::to_string(&target).unwrap();
    let deserialized: CdpTarget = serde_json::from_str(&json).unwrap();
    assert_eq!(target, deserialized);
}

#[test]
fn cdp_target_serde_service_worker() {
    let target = CdpTarget {
        id: "sw-456".to_string(),
        title: "Service Worker".to_string(),
        url: "chrome-extension://abc123/background.js".to_string(),
        target_type: "service_worker".to_string(),
    };

    let json = serde_json::to_string(&target).unwrap();
    let deserialized: CdpTarget = serde_json::from_str(&json).unwrap();
    assert_eq!(target.id, deserialized.id);
    assert_eq!(target.target_type, deserialized.target_type);
}

// ---------------------------------------------------------------------------
// CdpConfig serialization tests
// ---------------------------------------------------------------------------

#[test]
fn cdp_config_serde_roundtrip() {
    let config = CdpConfig {
        enabled: true,
        allowlist: HashSet::from([
            "com.app.one".to_string(),
            "com.app.two".to_string(),
        ]),
        connect_timeout_ms: 3000,
        max_retries: 5,
    };

    let json = serde_json::to_string(&config).unwrap();
    let deserialized: CdpConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.enabled, config.enabled);
    assert_eq!(deserialized.allowlist, config.allowlist);
    assert_eq!(deserialized.connect_timeout_ms, config.connect_timeout_ms);
    assert_eq!(deserialized.max_retries, config.max_retries);
}

// ---------------------------------------------------------------------------
// build_cdp_eval_script tests
// ---------------------------------------------------------------------------

#[test]
fn cdp_eval_script_contains_selector() {
    let script = build_cdp_eval_script("#my-button").unwrap();
    let expected = r##""#my-button""##;
    assert!(
        script.contains(expected),
        "Script should contain JSON-encoded selector, got: {}",
        script
    );
}

#[test]
fn cdp_eval_script_is_iife() {
    let script = build_cdp_eval_script("div.test").unwrap();
    assert!(script.starts_with("(function()"));
    assert!(script.ends_with("})()"));
}

#[test]
fn cdp_eval_script_queries_dom() {
    let script = build_cdp_eval_script(".some-class").unwrap();
    assert!(script.contains("document.querySelector("));
}

#[test]
fn cdp_eval_script_returns_semantic_properties() {
    let script = build_cdp_eval_script("button").unwrap();
    assert!(script.contains("tagName"));
    assert!(script.contains("role"));
    assert!(script.contains("ariaLabel"));
    assert!(script.contains("textContent"));
    assert!(script.contains("getBoundingClientRect"));
}

#[test]
fn cdp_eval_script_escapes_special_chars() {
    // A selector with special characters that need JSON escaping
    let script = build_cdp_eval_script(r#"[data-test="hello \"world\""]"#).unwrap();
    // The selector should be JSON-encoded, escaping the inner quotes
    assert!(
        script.contains("document.querySelector("),
        "Script should still contain querySelector: {}",
        script
    );
    // Verify it's valid JS by checking the escaped quotes are present
    assert!(
        script.contains(r#"\""#),
        "Script should contain escaped quotes: {}",
        script
    );
}

#[test]
fn cdp_eval_script_truncates_text_content() {
    let script = build_cdp_eval_script("p").unwrap();
    assert!(
        script.contains(".substring(0, 500)"),
        "Script should truncate textContent to 500 chars"
    );
}

#[test]
fn cdp_eval_script_null_check() {
    let script = build_cdp_eval_script("div").unwrap();
    assert!(
        script.contains("if (!el) return null"),
        "Script should return null when element not found"
    );
}
