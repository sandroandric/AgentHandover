use agenthandover_common::config::AppConfig;

#[test]
fn test_default_config_is_valid() {
    let config = AppConfig::default();
    assert_eq!(config.observer.t_dwell_seconds, 3);
    assert_eq!(config.observer.t_scroll_read_seconds, 8);
    assert!(config.observer.capture_screenshots);
    assert_eq!(config.observer.screenshot_max_per_minute, 20);
    assert!(config.privacy.enable_inline_secret_redaction);
    assert!(!config.privacy.enable_clipboard_preview);
    assert!(config.privacy.secure_field_drop);
    assert_eq!(config.storage.retention_days_raw, 14);
    assert!(config.storage.sqlite_wal_mode);
    assert_eq!(config.storage.vacuum_min_free_gb, 5);
    assert!(config.idle_jobs.require_ac_power);
    assert_eq!(config.idle_jobs.min_battery_percent, 50);
    assert_eq!(config.idle_jobs.max_cpu_percent, 30);
    assert_eq!(config.vlm.max_jobs_per_day, 50);
    assert!(config.openclaw.atomic_writes);
}

#[test]
fn test_config_from_toml_string() {
    let toml_str = r#"
[observer]
t_dwell_seconds = 5
screenshot_max_per_minute = 10

[privacy]
enable_clipboard_preview = true
clipboard_preview_max_chars = 100

[storage]
retention_days_raw = 7
"#;
    let config = AppConfig::from_toml_str(toml_str).unwrap();
    assert_eq!(config.observer.t_dwell_seconds, 5);
    assert_eq!(config.observer.screenshot_max_per_minute, 10);
    assert!(config.privacy.enable_clipboard_preview);
    assert_eq!(config.privacy.clipboard_preview_max_chars, 100);
    assert_eq!(config.storage.retention_days_raw, 7);
    // defaults still apply for unset fields
    assert!(config.observer.capture_screenshots);
    assert!(config.privacy.secure_field_drop);
}
