use chrono::Utc;
use oc_apprentice_common::status::{DaemonStatus, ExtensionHeartbeat, WorkerStatus};
use std::sync::Mutex;

/// Mutex to serialize tests that modify the HOME env var (process-global state).
static HOME_LOCK: Mutex<()> = Mutex::new(());

fn sample_daemon_status() -> DaemonStatus {
    let now = Utc::now();
    DaemonStatus {
        pid: 12345,
        version: "0.1.0".to_string(),
        started_at: now,
        heartbeat: now,
        events_today: 42,
        permissions_ok: true,
        accessibility_permitted: true,
        screen_recording_permitted: true,
        db_path: "/tmp/test.db".to_string(),
        uptime_seconds: 3600,
        last_extension_message: None,
        focus_session: None,
    }
}

fn sample_worker_status() -> WorkerStatus {
    let now = Utc::now();
    WorkerStatus {
        pid: 54321,
        version: "0.1.0".to_string(),
        started_at: now,
        heartbeat: now,
        events_processed_today: 100,
        sops_generated: 3,
        last_pipeline_duration_ms: Some(250),
        consecutive_errors: 0,
        vlm_available: true,
        sop_inducer_available: false,
        vlm_queue_pending: 0,
        vlm_jobs_today: 0,
        vlm_dropped_today: 0,
        vlm_mode: None,
        vlm_provider: None,
    }
}

#[test]
fn daemon_status_serialization_roundtrip() {
    let status = sample_daemon_status();
    let json = serde_json::to_string_pretty(&status).expect("serialize");
    let deserialized: DaemonStatus = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(status.pid, deserialized.pid);
    assert_eq!(status.version, deserialized.version);
    assert_eq!(status.started_at, deserialized.started_at);
    assert_eq!(status.heartbeat, deserialized.heartbeat);
    assert_eq!(status.events_today, deserialized.events_today);
    assert_eq!(status.permissions_ok, deserialized.permissions_ok);
    assert_eq!(status.accessibility_permitted, deserialized.accessibility_permitted);
    assert_eq!(status.screen_recording_permitted, deserialized.screen_recording_permitted);
    assert_eq!(status.db_path, deserialized.db_path);
    assert_eq!(status.uptime_seconds, deserialized.uptime_seconds);
}

#[test]
fn worker_status_serialization_roundtrip() {
    let status = sample_worker_status();
    let json = serde_json::to_string_pretty(&status).expect("serialize");
    let deserialized: WorkerStatus = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(status.pid, deserialized.pid);
    assert_eq!(status.version, deserialized.version);
    assert_eq!(status.started_at, deserialized.started_at);
    assert_eq!(status.heartbeat, deserialized.heartbeat);
    assert_eq!(status.events_processed_today, deserialized.events_processed_today);
    assert_eq!(status.sops_generated, deserialized.sops_generated);
    assert_eq!(status.last_pipeline_duration_ms, deserialized.last_pipeline_duration_ms);
    assert_eq!(status.consecutive_errors, deserialized.consecutive_errors);
    assert_eq!(status.vlm_available, deserialized.vlm_available);
    assert_eq!(status.sop_inducer_available, deserialized.sop_inducer_available);
}

#[test]
fn worker_status_optional_fields() {
    let mut status = sample_worker_status();
    status.last_pipeline_duration_ms = None;
    let json = serde_json::to_string(&status).expect("serialize");
    let deserialized: WorkerStatus = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.last_pipeline_duration_ms, None);
}

#[test]
fn write_and_read_daemon_status_file() {
    let _lock = HOME_LOCK.lock().unwrap();
    let tmp = tempfile::tempdir().expect("create tempdir");
    let original_home = std::env::var("HOME").ok();
    std::env::set_var("HOME", tmp.path());

    let status = sample_daemon_status();
    oc_apprentice_common::status::write_status_file("daemon-status.json", &status)
        .expect("write status file");

    let read_back: DaemonStatus =
        oc_apprentice_common::status::read_status_file("daemon-status.json")
            .expect("read status file");

    // Restore HOME before assertions (so panics don't leave it wrong)
    if let Some(home) = original_home {
        std::env::set_var("HOME", home);
    }

    assert_eq!(status.pid, read_back.pid);
    assert_eq!(status.version, read_back.version);
    assert_eq!(status.events_today, read_back.events_today);
    assert_eq!(status.db_path, read_back.db_path);
}

#[test]
fn write_and_read_worker_status_file() {
    let _lock = HOME_LOCK.lock().unwrap();
    let tmp = tempfile::tempdir().expect("create tempdir");
    let original_home = std::env::var("HOME").ok();
    std::env::set_var("HOME", tmp.path());

    let status = sample_worker_status();
    oc_apprentice_common::status::write_status_file("worker-status.json", &status)
        .expect("write status file");

    let read_back: WorkerStatus =
        oc_apprentice_common::status::read_status_file("worker-status.json")
            .expect("read status file");

    if let Some(home) = original_home {
        std::env::set_var("HOME", home);
    }

    assert_eq!(status.pid, read_back.pid);
    assert_eq!(status.events_processed_today, read_back.events_processed_today);
    assert_eq!(status.sops_generated, read_back.sops_generated);
}

#[test]
fn read_nonexistent_status_file_returns_error() {
    let _lock = HOME_LOCK.lock().unwrap();
    let tmp = tempfile::tempdir().expect("create tempdir");
    let original_home = std::env::var("HOME").ok();
    std::env::set_var("HOME", tmp.path());

    let result = oc_apprentice_common::status::read_status_file::<DaemonStatus>("nonexistent.json");

    if let Some(home) = original_home {
        std::env::set_var("HOME", home);
    }

    assert!(result.is_err());
}

#[test]
fn write_status_creates_directory_if_missing() {
    let _lock = HOME_LOCK.lock().unwrap();
    let tmp = tempfile::tempdir().expect("create tempdir");
    let original_home = std::env::var("HOME").ok();
    std::env::set_var("HOME", tmp.path());

    let expected_dir = if cfg!(target_os = "macos") {
        tmp.path().join("Library/Application Support/oc-apprentice")
    } else {
        tmp.path().join(".local/share/oc-apprentice")
    };
    assert!(!expected_dir.exists());

    let status = sample_daemon_status();
    oc_apprentice_common::status::write_status_file("daemon-status.json", &status)
        .expect("write status file");

    let dir_exists = expected_dir.exists();

    if let Some(home) = original_home {
        std::env::set_var("HOME", home);
    }

    assert!(dir_exists);
}

#[test]
fn atomic_write_overwrites_previous_status() {
    let _lock = HOME_LOCK.lock().unwrap();
    let tmp = tempfile::tempdir().expect("create tempdir");
    let original_home = std::env::var("HOME").ok();
    std::env::set_var("HOME", tmp.path());

    let mut status = sample_daemon_status();
    status.events_today = 10;
    oc_apprentice_common::status::write_status_file("daemon-status.json", &status)
        .expect("write first");

    status.events_today = 99;
    status.heartbeat = Utc::now();
    oc_apprentice_common::status::write_status_file("daemon-status.json", &status)
        .expect("write second");

    let read_back: DaemonStatus =
        oc_apprentice_common::status::read_status_file("daemon-status.json")
            .expect("read back");

    if let Some(home) = original_home {
        std::env::set_var("HOME", home);
    }

    assert_eq!(read_back.events_today, 99);
}

#[test]
fn daemon_status_with_extension_message_roundtrip() {
    let now = Utc::now();
    let status = DaemonStatus {
        pid: 12345,
        version: "0.1.0".to_string(),
        started_at: now,
        heartbeat: now,
        events_today: 10,
        permissions_ok: true,
        accessibility_permitted: true,
        screen_recording_permitted: true,
        db_path: "/tmp/test.db".to_string(),
        uptime_seconds: 600,
        last_extension_message: Some(now),
        focus_session: None,
    };

    let json = serde_json::to_string_pretty(&status).expect("serialize");
    // Verify the field is actually present in the JSON
    assert!(json.contains("last_extension_message"));

    let deserialized: DaemonStatus = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(status.last_extension_message, deserialized.last_extension_message);
    assert!(deserialized.last_extension_message.is_some());
}

#[test]
fn daemon_status_without_extension_message_omits_field() {
    let status = sample_daemon_status();
    assert!(status.last_extension_message.is_none());

    let json = serde_json::to_string(&status).expect("serialize");
    // Field should be omitted from JSON when None
    assert!(!json.contains("last_extension_message"));

    // Should still deserialize back correctly
    let deserialized: DaemonStatus = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.last_extension_message, None);
}

// =============================================================================
// ExtensionHeartbeat tests
// =============================================================================

fn sample_extension_heartbeat() -> ExtensionHeartbeat {
    let now = Utc::now();
    ExtensionHeartbeat {
        pid: 99999,
        last_message: now,
        messages_this_session: 42,
        session_started: now - chrono::Duration::minutes(5),
    }
}

#[test]
fn extension_heartbeat_serialization_roundtrip() {
    let heartbeat = sample_extension_heartbeat();
    let json = serde_json::to_string_pretty(&heartbeat).expect("serialize");
    let deserialized: ExtensionHeartbeat = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(heartbeat.pid, deserialized.pid);
    assert_eq!(heartbeat.last_message, deserialized.last_message);
    assert_eq!(heartbeat.messages_this_session, deserialized.messages_this_session);
    assert_eq!(heartbeat.session_started, deserialized.session_started);
}

#[test]
fn extension_heartbeat_json_contains_all_fields() {
    let heartbeat = sample_extension_heartbeat();
    let json = serde_json::to_string_pretty(&heartbeat).expect("serialize");

    assert!(json.contains("pid"));
    assert!(json.contains("last_message"));
    assert!(json.contains("messages_this_session"));
    assert!(json.contains("session_started"));
}

#[test]
fn write_and_read_extension_heartbeat_file() {
    let _lock = HOME_LOCK.lock().unwrap();
    let tmp = tempfile::tempdir().expect("create tempdir");
    let original_home = std::env::var("HOME").ok();
    std::env::set_var("HOME", tmp.path());

    let heartbeat = sample_extension_heartbeat();
    oc_apprentice_common::status::write_status_file(
        oc_apprentice_common::status::EXTENSION_HEARTBEAT_FILE,
        &heartbeat,
    )
    .expect("write heartbeat file");

    let read_back: ExtensionHeartbeat = oc_apprentice_common::status::read_status_file(
        oc_apprentice_common::status::EXTENSION_HEARTBEAT_FILE,
    )
    .expect("read heartbeat file");

    if let Some(home) = original_home {
        std::env::set_var("HOME", home);
    }

    assert_eq!(heartbeat.pid, read_back.pid);
    assert_eq!(heartbeat.messages_this_session, read_back.messages_this_session);
}

#[test]
fn read_extension_heartbeat_fresh_returns_some() {
    let _lock = HOME_LOCK.lock().unwrap();
    let tmp = tempfile::tempdir().expect("create tempdir");
    let original_home = std::env::var("HOME").ok();
    std::env::set_var("HOME", tmp.path());

    // Write a heartbeat with a recent timestamp
    let heartbeat = sample_extension_heartbeat();
    oc_apprentice_common::status::write_status_file(
        oc_apprentice_common::status::EXTENSION_HEARTBEAT_FILE,
        &heartbeat,
    )
    .expect("write");

    let result = oc_apprentice_common::status::read_extension_heartbeat();

    if let Some(home) = original_home {
        std::env::set_var("HOME", home);
    }

    assert!(result.is_some(), "fresh heartbeat should return Some");
}

#[test]
fn read_extension_heartbeat_stale_returns_none() {
    let _lock = HOME_LOCK.lock().unwrap();
    let tmp = tempfile::tempdir().expect("create tempdir");
    let original_home = std::env::var("HOME").ok();
    std::env::set_var("HOME", tmp.path());

    // Write a heartbeat with a timestamp from 5 minutes ago (stale)
    let stale_heartbeat = ExtensionHeartbeat {
        pid: 99999,
        last_message: Utc::now() - chrono::Duration::minutes(5),
        messages_this_session: 10,
        session_started: Utc::now() - chrono::Duration::minutes(10),
    };
    oc_apprentice_common::status::write_status_file(
        oc_apprentice_common::status::EXTENSION_HEARTBEAT_FILE,
        &stale_heartbeat,
    )
    .expect("write");

    let result = oc_apprentice_common::status::read_extension_heartbeat();

    if let Some(home) = original_home {
        std::env::set_var("HOME", home);
    }

    assert!(result.is_none(), "stale heartbeat (5 min old) should return None");
}

#[test]
fn read_extension_heartbeat_missing_file_returns_none() {
    let _lock = HOME_LOCK.lock().unwrap();
    let tmp = tempfile::tempdir().expect("create tempdir");
    let original_home = std::env::var("HOME").ok();
    std::env::set_var("HOME", tmp.path());

    // Don't write any heartbeat file
    let result = oc_apprentice_common::status::read_extension_heartbeat();

    if let Some(home) = original_home {
        std::env::set_var("HOME", home);
    }

    assert!(result.is_none(), "missing file should return None");
}

// =============================================================================
// WorkerStatus vlm_mode / vlm_provider tests
// =============================================================================

#[test]
fn worker_status_with_vlm_mode_roundtrip() {
    let mut status = sample_worker_status();
    status.vlm_mode = Some("remote".to_string());
    status.vlm_provider = Some("openai".to_string());

    let json = serde_json::to_string_pretty(&status).expect("serialize");
    assert!(json.contains("vlm_mode"));
    assert!(json.contains("vlm_provider"));

    let deserialized: WorkerStatus = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.vlm_mode, Some("remote".to_string()));
    assert_eq!(deserialized.vlm_provider, Some("openai".to_string()));
}

#[test]
fn worker_status_without_vlm_mode_omits_field() {
    let status = sample_worker_status();
    assert!(status.vlm_mode.is_none());
    assert!(status.vlm_provider.is_none());

    let json = serde_json::to_string(&status).expect("serialize");
    // Fields should be omitted when None (skip_serializing_if)
    assert!(!json.contains("vlm_mode"));
    assert!(!json.contains("vlm_provider"));

    // Should still deserialize back
    let deserialized: WorkerStatus = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.vlm_mode, None);
    assert_eq!(deserialized.vlm_provider, None);
}

#[test]
fn worker_status_deserialize_with_extra_vlm_fields() {
    // Simulate a worker-status.json with vlm_mode/vlm_provider
    let json = r#"{
        "pid": 12345,
        "version": "0.1.0",
        "started_at": "2025-01-01T00:00:00Z",
        "heartbeat": "2025-01-01T00:01:00Z",
        "events_processed_today": 50,
        "sops_generated": 2,
        "last_pipeline_duration_ms": 100,
        "consecutive_errors": 0,
        "vlm_available": true,
        "sop_inducer_available": true,
        "vlm_queue_pending": 5,
        "vlm_jobs_today": 10,
        "vlm_dropped_today": 1,
        "vlm_mode": "remote",
        "vlm_provider": "anthropic"
    }"#;

    let status: WorkerStatus = serde_json::from_str(json).expect("deserialize");
    assert_eq!(status.vlm_mode, Some("remote".to_string()));
    assert_eq!(status.vlm_provider, Some("anthropic".to_string()));
    assert!(status.vlm_available);
    assert_eq!(status.vlm_queue_pending, 5);
}

// =============================================================================
// Config parsing tests (VlmConfig remote fields)
// =============================================================================

#[test]
fn vlm_config_defaults() {
    let config = oc_apprentice_common::config::AppConfig::default();
    assert_eq!(config.vlm.mode, "local");
    assert!(config.vlm.provider.is_none());
    assert!(config.vlm.model.is_none());
    assert!(config.vlm.api_key_env.is_none());
    assert!(config.vlm.enabled);
}

#[test]
fn vlm_config_remote_mode_parsing() {
    let toml_str = r#"
[vlm]
enabled = true
mode = "remote"
provider = "openai"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
max_jobs_per_day = 50
max_queue_size = 500
job_ttl_days = 7
max_compute_minutes_per_day = 20
"#;
    let config = oc_apprentice_common::config::AppConfig::from_toml_str(toml_str)
        .expect("parse config");
    assert_eq!(config.vlm.mode, "remote");
    assert_eq!(config.vlm.provider, Some("openai".to_string()));
    assert_eq!(config.vlm.model, Some("gpt-4o-mini".to_string()));
    assert_eq!(config.vlm.api_key_env, Some("OPENAI_API_KEY".to_string()));
}

#[test]
fn llm_config_defaults() {
    let config = oc_apprentice_common::config::AppConfig::default();
    assert!(config.llm.enhance_sops);
    assert_eq!(config.llm.max_enhancements_per_day, 20);
    assert_eq!(config.llm.model, "");
    assert_eq!(config.llm.timeout_seconds, 60);
    assert!((config.llm.temperature - 0.3).abs() < 0.001);
    assert_eq!(config.llm.max_tokens, 800);
}

#[test]
fn llm_config_custom_parsing() {
    let toml_str = r#"
[llm]
enhance_sops = false
max_enhancements_per_day = 5
model = "llama3.2:3b"
timeout_seconds = 30
temperature = 0.5
max_tokens = 400
"#;
    let config = oc_apprentice_common::config::AppConfig::from_toml_str(toml_str)
        .expect("parse config");
    assert!(!config.llm.enhance_sops);
    assert_eq!(config.llm.max_enhancements_per_day, 5);
    assert_eq!(config.llm.model, "llama3.2:3b");
    assert_eq!(config.llm.timeout_seconds, 30);
    assert_eq!(config.llm.max_tokens, 400);
}
