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
    assert!(!config.privacy.capture_input_values);
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
capture_input_values = true

[storage]
retention_days_raw = 7
"#;
    let config = AppConfig::from_toml_str(toml_str).unwrap();
    assert_eq!(config.observer.t_dwell_seconds, 5);
    assert_eq!(config.observer.screenshot_max_per_minute, 10);
    assert!(config.privacy.enable_clipboard_preview);
    assert_eq!(config.privacy.clipboard_preview_max_chars, 100);
    assert!(config.privacy.capture_input_values);
    assert_eq!(config.storage.retention_days_raw, 7);
    // defaults still apply for unset fields
    assert!(config.observer.capture_screenshots);
    assert!(config.privacy.secure_field_drop);
}

#[test]
fn test_legacy_multi_monitor_mode_key_is_ignored() {
    // Installed configs written before the key was removed still load.
    let toml_str = r#"
[observer]
t_dwell_seconds = 4
multi_monitor_mode = "focused_window"
"#;
    let config = AppConfig::from_toml_str(toml_str).unwrap();
    assert_eq!(config.observer.t_dwell_seconds, 4);
}

#[test]
fn test_config_example_parses() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config.example.toml");
    let config = AppConfig::from_file(&path).unwrap();
    assert_eq!(config.observer.t_dwell_seconds, 3);
    assert!(!config.privacy.read_app_state, "AppleScript app state is opt-in");
}

/// Reading app state through AppleScript makes macOS ask the user to let
/// AgentHandover "control" each app; it must never be on unless chosen.
#[test]
fn test_read_app_state_defaults_off() {
    assert!(!AppConfig::default().privacy.read_app_state);
    let config = AppConfig::from_toml_str("[privacy]\nenable_inline_secret_redaction = true\nenable_clipboard_preview = false\nclipboard_preview_max_chars = 200\nsecure_field_drop = true\n").unwrap();
    assert!(!config.privacy.read_app_state);
}

#[test]
fn test_memory_config_defaults_match_the_shared_contract() {
    let memory = AppConfig::default().memory;
    assert!(!memory.enabled);
    assert_eq!(memory.model, "embeddinggemma-2:440m");
    assert_eq!(memory.dimensions, 512);
    assert_eq!(memory.retention_days, 365);
    assert_eq!(memory.thumbnail_days, 7);
    assert!(!memory.agents_access);
    assert_eq!(memory.max_frames_per_hour, 600);
    assert!(!memory.writes_thumbnails(), "off by default");
    // A config without [memory] (every 0.5 install) loads the defaults.
    assert_eq!(AppConfig::from_toml_str("").unwrap().memory, memory);
}

#[test]
fn test_memory_config_partial_table_keeps_other_defaults() {
    let config = AppConfig::from_toml_str("[memory]\nenabled = true\nthumbnail_days = 3\n").unwrap();
    assert!(config.memory.enabled);
    assert_eq!(config.memory.thumbnail_days, 3);
    assert_eq!(config.memory.dimensions, 512);
    assert!(config.memory.writes_thumbnails());
    let none_kept =
        AppConfig::from_toml_str("[memory]\nenabled = true\nthumbnail_days = 0\n").unwrap();
    assert!(!none_kept.memory.writes_thumbnails(), "thumbnail_days = 0 keeps none");
}
