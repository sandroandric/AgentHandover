use anyhow::Result;
use chrono::Utc;
use colored::Colorize;

use crate::paths;

/// Detect which install channel produced this AgentHandover installation.
fn detect_install_channel() -> &'static str {
    // Check for pkg install
    if std::path::Path::new("/usr/local/bin/agenthandover-daemon").exists()
        && std::path::Path::new("/usr/local/lib/agenthandover").exists()
    {
        return "pkg";
    }
    // Check for Homebrew
    if let Ok(output) = std::process::Command::new("brew")
        .args(["--prefix", "agenthandover"])
        .output()
    {
        if output.status.success() {
            return "homebrew";
        }
    }
    // Check for source build (look for Cargo.toml in ancestor dirs)
    let mut dir = std::env::current_dir().unwrap_or_default();
    for _ in 0..5 {
        if dir.join("Cargo.toml").exists() && dir.join("worker").exists() {
            return "source";
        }
        if !dir.pop() {
            break;
        }
    }
    "unknown"
}

/// Print a channel-aware install/repair hint.
fn print_install_hint(channel: &str) {
    match channel {
        "pkg" => eprintln!("  Fix: Re-run the .pkg installer to repair."),
        "homebrew" => eprintln!("  Fix: brew reinstall agenthandover"),
        "source" => eprintln!("  Fix: just build-all"),
        _ => eprintln!("  Fix: Re-install AgentHandover (.pkg recommended)."),
    }
}

/// Counters for the three check result states.
struct CheckCounts {
    passes: u32,
    skips: u32,
    failures: u32,
}

impl CheckCounts {
    fn new() -> Self {
        Self {
            passes: 0,
            skips: 0,
            failures: 0,
        }
    }

    fn record_pass(&mut self) {
        self.passes += 1;
    }

    fn record_skip(&mut self) {
        self.skips += 1;
    }

    fn record_fail(&mut self) {
        self.failures += 1;
    }
}

pub fn run() -> Result<()> {
    println!("{}", "AgentHandover Doctor".bold());
    println!("{}", "=".repeat(50));
    println!();

    let channel = detect_install_channel();
    let mut counts = CheckCounts::new();

    // Check 1: Daemon binary exists
    //
    // v0.2.1 renamed the installed daemon from `agenthandover-daemon` to
    // `ah-observer` and moved it from `/usr/local/bin/` to
    // `/usr/local/lib/agenthandover/`. The Swift menu bar app hardcodes
    // that new path as the place to spawn the daemon from. We use
    // paths::find_daemon_binary() so pkg/cask installs (ah-observer),
    // legacy installs, homebrew libexec, and cargo source builds all
    // resolve through a single helper.
    let daemon_path = paths::find_daemon_binary();
    if let Some(ref p) = daemon_path {
        check_pass(&mut counts, &format!("Daemon binary ({})", p.display()));
    } else {
        check_fail(&mut counts, "Daemon binary");
        print_install_hint(channel);
    }

    // Check 2: CLI binary (we're running it, so it exists)
    check_pass(&mut counts, "CLI binary");

    // Check 3: Data directory exists
    let data_dir = agenthandover_common::status::status_dir();
    if data_dir.exists() {
        check_pass(&mut counts, "Data directory");
    } else {
        check_fail(&mut counts, "Data directory");
    }

    // Check 4: Config file exists (optional — defaults are fine)
    let config_path = data_dir.join("config.toml");
    if config_path.exists() {
        check_pass(&mut counts, "Config file");
    } else {
        check_skip(&mut counts, "Config file", "Using defaults");
    }

    // Check 5 + 6: TCC permissions (Accessibility + Screen Recording)
    //
    // macOS TCC permissions are granted PER APP BUNDLE, not system-wide.
    // The daemon runs under the AgentHandover.app TCC principal (because
    // the menu bar app spawns it via Process() — see ServiceController
    // .daemonExecutableURL), so what we actually need to check is whether
    // AgentHandover.app has the permissions, NOT whether this CLI
    // process has them.
    //
    // Calling AXIsProcessTrusted() / CGDisplayCreateImage() from the
    // CLI checks THIS process, which is the wrong thing — the CLI will
    // always read "no accessibility" because you never granted it, but
    // the app can have full permissions. The previous version of doctor
    // had this bug and falsely reported Accessibility as failed on
    // every pkg install.
    //
    // There is no clean API to query TCC for a different bundle id
    // without reading the TCC sqlite database, which itself requires
    // Full Disk Access (ironic). Since we can't definitively check,
    // print an advisory info line instead and let the user verify in
    // System Settings directly.
    #[cfg(target_os = "macos")]
    {
        println!(
            "  {} Accessibility permission ({})",
            "info".dimmed(),
            "verify AgentHandover in System Settings > Privacy & Security > Accessibility"
                .dimmed()
        );
        println!(
            "  {} Screen Recording permission ({})",
            "info".dimmed(),
            "verify AgentHandover in System Settings > Privacy & Security > Screen Recording"
                .dimmed()
        );
    }

    // Check 7: Database exists and is writable
    let db_path = data_dir.join("events.db");
    let daemon_pid_exists = data_dir.join("daemon.pid").exists();
    let db_ok = db_path.exists()
        && std::fs::OpenOptions::new()
            .write(true)
            .open(&db_path)
            .is_ok();
    if daemon_pid_exists {
        // Daemon has started at least once — DB should exist
        if db_ok {
            check_pass(&mut counts, "Database");
        } else {
            check_fail(&mut counts, "Database");
        }
    } else {
        // Fresh install — daemon never ran, DB expected to not exist
        if db_ok {
            check_pass(&mut counts, "Database");
        } else {
            check_skip(
                &mut counts,
                "Database",
                "Run 'agenthandover start' to create database",
            );
        }
    }

    // Check 8: Native messaging host manifest (content validation)
    if check_native_messaging_manifest() {
        check_pass(&mut counts, "Native messaging host");
    } else {
        check_fail(&mut counts, "Native messaging host");
    }

    // Check 9: Worker launchd plist
    //
    // Only the worker uses launchd in v0.2.1+. The daemon is spawned
    // directly by the menu bar app via Process() (see ServiceController
    // .startDaemon()) so it has no launchd plist anymore. The old
    // com.agenthandover.daemon.plist was deleted in v0.2.1 — do NOT
    // add a check for it here, it will always fail and mislead users.
    let launch_agents = launch_agents_dir();
    if launch_agents.join("com.agenthandover.worker.plist").exists() {
        check_pass(&mut counts, "Worker launchd plist");
    } else {
        check_fail(&mut counts, "Worker launchd plist");
        print_install_hint(channel);
    }

    // Check 10: Heartbeat freshness (advisory — does not affect counts)
    check_heartbeat_freshness("Daemon", "daemon-status.json");
    check_heartbeat_freshness("Worker", "worker-status.json");

    // Check 11: Disk space
    if free_disk_gb() > 1 {
        check_pass(&mut counts, "Disk space (>1GB free)");
    } else {
        check_fail(&mut counts, "Disk space (>1GB free)");
    }

    // Check 12: Python virtual environment (optional — not present in source builds)
    if paths::find_venv_python().is_some() {
        check_pass(&mut counts, "Python virtual environment");
    } else {
        let hint = match channel {
            "pkg" => "Re-run .pkg installer to repair",
            "homebrew" => "brew reinstall agenthandover",
            "source" => "venv not found — expected for source builds",
            _ => "Re-install AgentHandover (.pkg recommended)",
        };
        check_skip(&mut counts, "Python virtual environment", hint);
    }

    // Check 13: Chrome extension (optional — not all users need it)
    if paths::find_any_extension_path().is_some() {
        check_pass(&mut counts, "Browser extension");
    } else {
        let hint = match channel {
            "pkg" => "Re-run .pkg installer to repair",
            "homebrew" => "brew reinstall agenthandover",
            "source" => "cd extension && npm run build",
            _ => "Re-install AgentHandover (.pkg recommended)",
        };
        check_skip(&mut counts, "Browser extension", hint);
    }

    // Check 14: Worker process alive (optional — may not be started yet)
    if agenthandover_common::pid::check_pid_file("worker").is_some() {
        check_pass(&mut counts, "Worker process");
    } else {
        check_skip(
            &mut counts,
            "Worker process",
            "Start with: agenthandover start",
        );
    }

    // Check 15: Ollama reachable
    let ollama_ok = std::process::Command::new("curl")
        .args(["-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:11434/"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "200")
        .unwrap_or(false);
    if ollama_ok {
        check_pass(&mut counts, "Ollama service");
    } else {
        check_fail(&mut counts, "Ollama service");
        eprintln!("  Fix: Install Ollama from https://ollama.com and run it");
    }

    // Check 16: Required models pulled
    //
    // v0.2.0 introduced RAM-based model tiers (Qwen 3.5 for 8GB, Gemma 4
    // for 16GB+). The previous version of this check hardcoded
    // `qwen3.5:2b` + `qwen3.5:4b` which is wrong for anyone on a 16GB+
    // Mac — they'd have Gemma 4 installed and doctor would report both
    // qwen models as missing. False failures.
    //
    // Correct approach: read the user's config.toml to find their
    // ACTUALLY configured annotation_model + sop_model + embedding
    // model, then check Ollama for those specific names. Gracefully
    // fall back if config.toml is missing or fields aren't set.
    //
    // Also: if config has `vlm.mode = "remote"`, skip the local model
    // check entirely — they're using a cloud provider.
    if ollama_ok {
        let config_str = std::fs::read_to_string(&config_path).unwrap_or_default();

        let vlm_mode = read_config_string_field(&config_str, "vlm", "mode")
            .unwrap_or_else(|| "local".to_string());

        if vlm_mode == "remote" {
            check_skip(
                &mut counts,
                "Local models",
                "vlm.mode = remote (using cloud API)",
            );
        } else {
            let annotation_model = read_config_string_field(&config_str, "vlm", "annotation_model")
                .unwrap_or_else(|| "qwen3.5:2b".to_string());
            let sop_model = read_config_string_field(&config_str, "vlm", "sop_model")
                .unwrap_or_else(|| "qwen3.5:4b".to_string());
            let embedding_model = read_config_string_field(&config_str, "embedding", "model")
                .unwrap_or_else(|| "nomic-embed-text".to_string());

            let models_output = std::process::Command::new("ollama")
                .arg("list")
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                .unwrap_or_default();

            // Gemma 4 is a single-model tier where annotation_model ==
            // sop_model; deduplicate so we don't report the same model
            // twice with different purposes.
            let mut models_to_check: Vec<(String, &str)> = Vec::new();
            if !annotation_model.is_empty() {
                models_to_check.push((annotation_model.clone(), "scene annotation"));
            }
            if !sop_model.is_empty() && sop_model != annotation_model {
                models_to_check.push((sop_model.clone(), "SOP generation"));
            }
            if !embedding_model.is_empty() {
                models_to_check.push((embedding_model, "semantic search"));
            }

            for (model, purpose) in &models_to_check {
                if models_output.contains(model.as_str()) {
                    check_pass(&mut counts, &format!("Model {} ({})", model, purpose));
                } else {
                    check_fail(&mut counts, &format!("Model {} ({})", model, purpose));
                    eprintln!("  Fix: ollama pull {}", model);
                }
            }
        }
    }

    // Summary
    println!();
    println!(
        "Results: {} passed, {} skipped, {} failed",
        counts.passes, counts.skips, counts.failures
    );
    if counts.failures > 0 {
        println!(
            "{}",
            "Some checks failed. See above for details."
                .yellow()
                .bold()
        );
    } else if counts.skips > 0 {
        println!(
            "{}",
            format!(
                "All required checks passed ({} optional check(s) skipped)",
                counts.skips
            )
            .green()
            .bold()
        );
    } else {
        println!("{}", "All checks passed!".green().bold());
    }

    Ok(())
}

fn check_pass(counts: &mut CheckCounts, name: &str) {
    counts.record_pass();
    println!("  {} {}", "pass".green(), name);
}

fn check_fail(counts: &mut CheckCounts, name: &str) {
    counts.record_fail();
    println!("  {} {}", "FAIL".red(), name);
}

fn check_skip(counts: &mut CheckCounts, name: &str, reason: &str) {
    counts.record_skip();
    println!("  {} {} ({})", "SKIP".yellow(), name, reason);
}

/// Advisory heartbeat freshness check for a service status file.
///
/// Reads `~/Library/Application Support/agenthandover/<filename>`, parses the
/// `heartbeat` ISO-8601 timestamp, and prints a WARNING if it is older than 60
/// seconds.  If the file doesn't exist (first install, service never started),
/// prints an INFO note.  This check never affects the overall doctor result.
fn check_heartbeat_freshness(service_name: &str, status_filename: &str) {
    let label = format!("{} heartbeat", service_name);

    let status_dir = agenthandover_common::status::status_dir();
    let path = status_dir.join(status_filename);

    if !path.exists() {
        println!(
            "  {} {} ({})",
            "info".dimmed(),
            label,
            "No status file yet (service may not have started)"
        );
        return;
    }

    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => {
            println!("  {} {} ({})", "info".dimmed(), label, "Cannot read status file");
            return;
        }
    };

    let parsed: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(_) => {
            println!("  {} {} ({})", "info".dimmed(), label, "Invalid JSON in status file");
            return;
        }
    };

    let heartbeat_str = match parsed.get("heartbeat").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => {
            println!(
                "  {} {} ({})",
                "info".dimmed(),
                label,
                "No heartbeat field in status file"
            );
            return;
        }
    };

    let heartbeat = match chrono::DateTime::parse_from_rfc3339(&heartbeat_str) {
        Ok(dt) => dt.with_timezone(&Utc),
        Err(_) => {
            println!(
                "  {} {} ({})",
                "info".dimmed(),
                label,
                "Cannot parse heartbeat timestamp"
            );
            return;
        }
    };

    let age_secs = Utc::now()
        .signed_duration_since(heartbeat)
        .num_seconds();

    if age_secs <= 60 {
        println!("  {} {} ({}s ago)", "pass".green(), label, age_secs);
    } else {
        println!(
            "  {} {} (last heartbeat {}s ago — service may be hung)",
            "WARN".yellow(),
            label,
            age_secs
        );
    }
}

// Path resolution functions (find_homebrew_libexec, find_venv_python,
// find_extension_dir, find_local_extension_dist, find_daemon_binary)
// live in crate::paths. Don't duplicate them here.
//
// The dead helpers `which`, `source_build_daemon_exists`,
// `accessibility_sys_check`, and `screen_recording_check` were removed
// in v0.2.3. The first two are subsumed by paths::find_daemon_binary;
// the last two checked TCC permissions for the CLI process instead of
// the app bundle (wrong semantics) and were replaced with advisory
// info output — see Check 5/6 above.

/// Extract a string field from a TOML config file without pulling in
/// a TOML parser dependency. Handles:
///
/// - Section headers (`[section]`)
/// - `key = "value"` (quoted)
/// - `key = value` (unquoted, e.g. numbers or booleans)
/// - `#` comments (whole line or trailing)
/// - Empty lines
///
/// Returns `None` if the section or key isn't found. Does NOT handle
/// nested tables, arrays, or multi-line strings — only flat scalar
/// fields, which is all doctor actually needs to read. If you need
/// anything more, add the `toml` crate as a dependency instead.
fn read_config_string_field(config: &str, section: &str, key: &str) -> Option<String> {
    let target_header = format!("[{}]", section);
    let mut in_section = false;
    for line in config.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // Section header
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            in_section = trimmed == target_header;
            continue;
        }
        if !in_section {
            continue;
        }
        // key = value
        if let Some((k, v)) = trimmed.split_once('=') {
            if k.trim() == key {
                let v = v.trim();
                // Strip trailing comment (after the value)
                let v = v.split('#').next().unwrap_or(v).trim();
                // Strip surrounding quotes
                let v = v.trim_matches(|c| c == '"' || c == '\'');
                return Some(v.to_string());
            }
        }
    }
    None
}

fn native_messaging_manifest_path() -> std::path::PathBuf {
    // Check all supported Chromium browsers, return the first found manifest
    let manifest_name = "com.agenthandover.host.json";
    for dir in crate::paths::native_messaging_hosts_dirs() {
        let path = dir.join(manifest_name);
        if path.exists() {
            return path;
        }
    }
    // Fallback to Chrome default for error reporting
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    std::path::PathBuf::from(home).join(
        "Library/Application Support/Google/Chrome/NativeMessagingHosts/com.agenthandover.host.json",
    )
}

fn launch_agents_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    std::path::PathBuf::from(home).join("Library/LaunchAgents")
}

fn free_disk_gb() -> u64 {
    #[cfg(unix)]
    {
        use std::ffi::CString;
        let c_path = CString::new("/").unwrap();
        unsafe {
            let mut stat: libc::statvfs = std::mem::zeroed();
            if libc::statvfs(c_path.as_ptr(), &mut stat) == 0 {
                stat.f_bavail as u64 * stat.f_frsize as u64 / (1024 * 1024 * 1024)
            } else {
                0
            }
        }
    }
    #[cfg(not(unix))]
    {
        0
    }
}

fn check_native_messaging_manifest() -> bool {
    let manifest_path = native_messaging_manifest_path();
    if !manifest_path.exists() {
        println!(
            "    {} Manifest not found at: {}",
            "→".dimmed(),
            manifest_path.display()
        );
        println!(
            "    Run {} to install it.",
            "agenthandover setup --extension".cyan()
        );
        return false;
    }

    // Read and parse
    let content = match std::fs::read_to_string(&manifest_path) {
        Ok(c) => c,
        Err(e) => {
            println!("    {} Cannot read manifest: {}", "→".dimmed(), e);
            return false;
        }
    };

    let parsed: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            println!("    {} Invalid JSON in manifest: {}", "→".dimmed(), e);
            return false;
        }
    };

    // Verify name
    let name = parsed.get("name").and_then(|v| v.as_str()).unwrap_or("");
    if name != "com.agenthandover.host" {
        println!(
            "    {} Wrong host name: expected 'com.agenthandover.host', got '{}'",
            "→".dimmed(),
            name
        );
        return false;
    }

    // Verify path points to existing executable
    let daemon_path_str = parsed.get("path").and_then(|v| v.as_str()).unwrap_or("");
    if daemon_path_str.is_empty() {
        println!("    {} Manifest 'path' is empty", "→".dimmed());
        return false;
    }
    let daemon_path = std::path::Path::new(daemon_path_str);
    if !daemon_path.exists() {
        println!(
            "    {} Daemon binary not found at manifest path: {}",
            "→".dimmed(),
            daemon_path_str
        );
        println!(
            "    Re-run {} to update.",
            "agenthandover setup --extension".cyan()
        );
        return false;
    }

    // Verify allowed_origins contains expected extension ID
    let origins = parsed.get("allowed_origins").and_then(|v| v.as_array());
    let expected_origin = "chrome-extension://jpemkdcihaijkolbkankcldmiimmmnfo/";
    let has_origin = origins
        .map(|arr| arr.iter().any(|v| v.as_str() == Some(expected_origin)))
        .unwrap_or(false);
    if !has_origin {
        println!(
            "    {} allowed_origins missing expected extension ID",
            "→".dimmed()
        );
        return false;
    }

    true
}
