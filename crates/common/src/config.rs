use figment::{Figment, providers::{Format, Toml, Serialized}};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub observer: ObserverConfig,
    #[serde(default)]
    pub privacy: PrivacyConfig,
    #[serde(default)]
    pub browser: BrowserConfig,
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub idle_jobs: IdleJobsConfig,
    #[serde(default)]
    pub vlm: VlmConfig,
    #[serde(default)]
    pub llm: LlmConfig,
    #[serde(default)]
    pub openclaw: OpenClawConfig,
    #[serde(default)]
    pub export: ExportConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObserverConfig {
    pub t_dwell_seconds: u64,
    pub t_scroll_read_seconds: u64,
    pub capture_screenshots: bool,
    pub screenshot_max_per_minute: u32,
    pub multi_monitor_mode: String,
    /// Screenshot format: "jpeg" (default, half-res) or "png" (full-res).
    #[serde(default = "default_screenshot_format")]
    pub screenshot_format: String,
    /// JPEG quality 1-100 (default: 70). Only used when format = "jpeg".
    #[serde(default = "default_screenshot_quality")]
    pub screenshot_quality: u8,
    /// Screenshot scale factor (default: 0.5 = half resolution).
    #[serde(default = "default_screenshot_scale")]
    pub screenshot_scale: f64,
    /// dHash perceptual hash threshold (hamming distance). Lower = stricter dedup.
    #[serde(default = "default_dhash_threshold")]
    pub dhash_threshold: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    pub enable_inline_secret_redaction: bool,
    pub enable_clipboard_preview: bool,
    pub clipboard_preview_max_chars: usize,
    pub secure_field_drop: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserConfig {
    pub extension_id: String,
    pub native_host_name: String,
    pub deny_network_egress: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub retention_days_raw: u32,
    pub retention_days_episodes: u32,
    pub sqlite_wal_mode: bool,
    pub vacuum_min_free_gb: u64,
    pub vacuum_safety_multiplier: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdleJobsConfig {
    pub require_ac_power: bool,
    pub min_battery_percent: u32,
    pub max_cpu_percent: u32,
    pub max_temp_c: u32,
    pub run_window_local_time: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VlmConfig {
    pub enabled: bool,
    pub max_jobs_per_day: u32,
    pub max_queue_size: u32,
    pub job_ttl_days: u32,
    pub max_compute_minutes_per_day: u32,
    /// VLM mode: "local" (default) or "remote" (cloud API).
    #[serde(default = "default_vlm_mode")]
    pub mode: String,
    /// Remote provider: "openai" | "anthropic" | "google".
    #[serde(default)]
    pub provider: Option<String>,
    /// Provider-specific model name override.
    #[serde(default)]
    pub model: Option<String>,
    /// Environment variable name holding the API key (NEVER store keys directly!).
    #[serde(default)]
    pub api_key_env: Option<String>,

    // --- v2 scene annotation pipeline fields ---

    /// Model for per-frame annotation (default: "qwen3.5:2b").
    #[serde(default = "default_annotation_model")]
    pub annotation_model: String,
    /// Model for SOP generation (default: "qwen3.5:4b").
    #[serde(default = "default_sop_model")]
    pub sop_model: String,
    /// Enable continuous scene annotation pipeline.
    #[serde(default = "default_annotation_enabled")]
    pub annotation_enabled: bool,
    /// Skip annotation after N consecutive non-workflow same-app frames.
    #[serde(default = "default_stale_skip_count")]
    pub stale_skip_count: u32,
    /// Max age (seconds) for sliding window context frames (default: 600 = 10 min).
    #[serde(default = "default_sliding_window_max_age_sec")]
    pub sliding_window_max_age_sec: u64,
}

fn default_screenshot_format() -> String { "jpeg".to_string() }
fn default_screenshot_quality() -> u8 { 70 }
fn default_screenshot_scale() -> f64 { 0.5 }
fn default_dhash_threshold() -> u32 { 10 }

fn default_vlm_mode() -> String { "local".to_string() }
fn default_annotation_model() -> String { "qwen3.5:2b".to_string() }
fn default_sop_model() -> String { "qwen3.5:4b".to_string() }
fn default_annotation_enabled() -> bool { true }
fn default_stale_skip_count() -> u32 { 3 }
fn default_sliding_window_max_age_sec() -> u64 { 600 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Enable LLM-enhanced SOP descriptions.
    #[serde(default = "default_llm_enhance_sops")]
    pub enhance_sops: bool,
    /// Maximum SOP enhancements per day.
    #[serde(default = "default_llm_max_enhancements")]
    pub max_enhancements_per_day: u32,
    /// Model override (empty = inherit from VLM config).
    #[serde(default)]
    pub model: String,
    /// Timeout for LLM inference in seconds.
    #[serde(default = "default_llm_timeout")]
    pub timeout_seconds: u32,
    /// Temperature for LLM inference.
    #[serde(default = "default_llm_temperature")]
    pub temperature: f64,
    /// Max tokens for LLM response.
    #[serde(default = "default_llm_max_tokens")]
    pub max_tokens: u32,
}

fn default_llm_enhance_sops() -> bool { true }
fn default_llm_max_enhancements() -> u32 { 20 }
fn default_llm_timeout() -> u32 { 60 }
fn default_llm_temperature() -> f64 { 0.3 }
fn default_llm_max_tokens() -> u32 { 800 }

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            enhance_sops: default_llm_enhance_sops(),
            max_enhancements_per_day: default_llm_max_enhancements(),
            model: String::new(),
            timeout_seconds: default_llm_timeout(),
            temperature: default_llm_temperature(),
            max_tokens: default_llm_max_tokens(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenClawConfig {
    pub workspace_path: String,
    pub sop_output_dir: String,
    pub index_path: String,
    pub atomic_writes: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Which adapter to use: "openclaw" or "generic"
    pub adapter: String,
    /// Enable JSON export alongside Markdown
    pub json_export: bool,
    /// Output directory for the generic adapter
    pub generic_output_dir: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            observer: ObserverConfig::default(),
            privacy: PrivacyConfig::default(),
            browser: BrowserConfig::default(),
            storage: StorageConfig::default(),
            idle_jobs: IdleJobsConfig::default(),
            vlm: VlmConfig::default(),
            llm: LlmConfig::default(),
            openclaw: OpenClawConfig::default(),
            export: ExportConfig::default(),
        }
    }
}

impl Default for ObserverConfig {
    fn default() -> Self {
        Self {
            t_dwell_seconds: 3,
            t_scroll_read_seconds: 8,
            capture_screenshots: true,
            screenshot_max_per_minute: 20,
            multi_monitor_mode: "focused_window".into(),
            screenshot_format: default_screenshot_format(),
            screenshot_quality: default_screenshot_quality(),
            screenshot_scale: default_screenshot_scale(),
            dhash_threshold: default_dhash_threshold(),
        }
    }
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            enable_inline_secret_redaction: true,
            enable_clipboard_preview: false,
            clipboard_preview_max_chars: 200,
            secure_field_drop: true,
        }
    }
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            extension_id: "knldjmfmopnpolahpmmgbagdohdnhkik".into(),
            native_host_name: "com.openclaw.apprentice".into(),
            deny_network_egress: true,
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            retention_days_raw: 14,
            retention_days_episodes: 90,
            sqlite_wal_mode: true,
            vacuum_min_free_gb: 5,
            vacuum_safety_multiplier: 2.5,
        }
    }
}

impl Default for IdleJobsConfig {
    fn default() -> Self {
        Self {
            require_ac_power: true,
            min_battery_percent: 50,
            max_cpu_percent: 30,
            max_temp_c: 80,
            run_window_local_time: "01:00-05:00".into(),
        }
    }
}

impl Default for VlmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_jobs_per_day: 50,
            max_queue_size: 500,
            job_ttl_days: 7,
            max_compute_minutes_per_day: 20,
            mode: default_vlm_mode(),
            provider: None,
            model: None,
            api_key_env: None,
            annotation_model: default_annotation_model(),
            sop_model: default_sop_model(),
            annotation_enabled: default_annotation_enabled(),
            stale_skip_count: default_stale_skip_count(),
            sliding_window_max_age_sec: default_sliding_window_max_age_sec(),
        }
    }
}

impl Default for OpenClawConfig {
    fn default() -> Self {
        Self {
            workspace_path: "~/.openclaw/workspace".into(),
            sop_output_dir: "memory/apprentice/sops".into(),
            index_path: "memory/apprentice/index.md".into(),
            atomic_writes: true,
        }
    }
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            adapter: "openclaw".into(),
            json_export: false,
            generic_output_dir: "sops".into(),
        }
    }
}

impl AppConfig {
    pub fn from_toml_str(toml_str: &str) -> Result<Self, figment::Error> {
        Figment::from(Serialized::defaults(AppConfig::default()))
            .merge(Toml::string(toml_str))
            .extract()
    }

    pub fn from_file(path: &std::path::Path) -> Result<Self, figment::Error> {
        Figment::from(Serialized::defaults(AppConfig::default()))
            .merge(Toml::file(path))
            .extract()
    }
}
