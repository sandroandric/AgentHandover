use anyhow::Result;
use chrono::Utc;
use colored::Colorize;

use agenthandover_common::observation::{
    capture_socket_path, observation_socket_path, snapshot_at,
};

use agenthandover_common::tiers;
use crate::{machine, ollama, paths};

/// Detect which install channel produced this AgentHandover installation.
fn detect_install_channel() -> &'static str {
    // Check for pkg install (the daemon is installed as <lib dir>/ah-observer)
    if paths::lib_dir().join("ah-observer").exists() {
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

/// Counters for the check result states.  A warning does not fail doctor
/// but is counted, so a summary never reads "all passed" over one.
struct CheckCounts {
    passes: u32,
    skips: u32,
    warnings: u32,
    failures: u32,
}

impl CheckCounts {
    fn new() -> Self {
        Self {
            passes: 0,
            skips: 0,
            warnings: 0,
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

    // Check 5-8: the app's capture servers and its TCC grants.
    //
    // Accessibility and Screen Recording are granted to AgentHandover.app,
    // not to this CLI (the app spawns the daemon, so the daemon runs under
    // the app's grants). Asking TCC about this process would always say
    // "no", so we ask the app: its observation server reports both grants.
    // If it does not answer, the app is not running or its server failed
    // to bind, and the daemon captures nothing.
    check_capture_servers(&mut counts);

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

    // Check 12: Python virtual environment. Absent is fine for a source
    // build; when the lib dir exists (a pkg install) a missing or broken
    // venv means the worker cannot run at all, so it is a failure.
    check_venv(&mut counts, channel);

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

    // Check 15: the Ollama service (the configured host), its version
    // against the minimum, the installs on this Mac and which one serves
    // the port, and memory headroom for the configured tier.
    let config_str = std::fs::read_to_string(&config_path).unwrap_or_default();
    let host_url = read_config_string_field(&config_str, "vlm", "ollama_host").unwrap_or_default();
    let report = machine::gather(&host_url, &configured_models(&config_str));
    print_findings(&mut counts, &report.findings);
    let ollama_ok = report.version.is_some();

    // Check 16: Required models pulled
    //
    // The models depend on the RAM tier (tiers.json), so a hard-coded
    // list reports false failures on every other tier.
    //
    // Correct approach: read the user's config.toml to find their
    // ACTUALLY configured annotation_model + sop_model + embedding
    // model, then check Ollama for those specific names. Gracefully
    // fall back if config.toml is missing or fields aren't set.
    //
    // Also: if config has `vlm.mode = "remote"`, skip the local model
    // check entirely — they're using a cloud provider.
    if ollama_ok {
        let vlm_mode = read_config_string_field(&config_str, "vlm", "mode")
            .unwrap_or_else(|| "local".to_string());

        if vlm_mode == "remote" {
            check_skip(
                &mut counts,
                "Local models",
                "vlm.mode = remote (using cloud API)",
            );
        } else {
            // Missing model keys come from the config's tier, else the
            // RAM tier, exactly as the worker resolves them.
            let machine::ConfiguredModels {
                annotation: annotation_model,
                sop: sop_model,
                ..
            } = configured_models(&config_str);
            let embedding_model = read_config_string_field(&config_str, "embedding", "model")
                .unwrap_or_else(|| tiers::table().embedding_model.clone());

            // Over HTTP, as `setup` asks (see crate::ollama for why).
            let installed = ollama::installed_models(&report.host.url).unwrap_or_default();

            // The standard tier runs one model for both jobs; deduplicate
            // so we don't report the same model twice.
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
                if ollama::model_installed(&installed, model) {
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
        "Results: {} passed, {} skipped, {} warning(s), {} failed",
        counts.passes, counts.skips, counts.warnings, counts.failures
    );
    if counts.failures > 0 {
        println!(
            "{}",
            "Some checks failed. See above for details."
                .yellow()
                .bold()
        );
    } else if counts.warnings > 0 {
        println!(
            "{}",
            format!(
                "Required checks passed, with {} warning(s): see WARN above",
                counts.warnings
            )
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

    if counts.failures > 0 {
        anyhow::bail!("{} doctor check(s) failed", counts.failures);
    }
    Ok(())
}

/// `agenthandover doctor --ollama-json`: the Ollama and memory report the
/// app's onboarding shows (the same findings `doctor` prints).
pub fn print_ollama_json() -> Result<()> {
    let config_path = agenthandover_common::status::status_dir().join("config.toml");
    let config_str = std::fs::read_to_string(config_path).unwrap_or_default();
    let host_url = read_config_string_field(&config_str, "vlm", "ollama_host").unwrap_or_default();
    let report = machine::gather(&host_url, &configured_models(&config_str));
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

/// The config's tier and models, as the worker resolves them.
fn configured_models(config_str: &str) -> machine::ConfiguredModels {
    let (tier, annotation, sop) = tiers::resolve_models(
        read_config_string_field(config_str, "vlm", "tier").as_deref(),
        read_config_string_field(config_str, "vlm", "annotation_model").as_deref(),
        read_config_string_field(config_str, "vlm", "sop_model").as_deref(),
    );
    machine::ConfiguredModels { tier_label: tier.label.clone(), annotation, sop }
}

fn print_findings(counts: &mut CheckCounts, findings: &[machine::Finding]) {
    for f in findings {
        match f.level {
            machine::Level::Pass => check_pass(counts, &f.title),
            machine::Level::Warn => check_warn(counts, &f.title),
            machine::Level::Fail => check_fail(counts, &f.title),
        }
        for line in &f.lines {
            eprintln!("  {line}");
        }
    }
}

fn check_capture_servers(counts: &mut CheckCounts) {
    let obs_sock = observation_socket_path();
    match snapshot_at(&obs_sock) {
        Ok(snap) => {
            check_pass(
                counts,
                &format!("Observation server ({})", obs_sock.display()),
            );
            if snap.accessibility_granted {
                check_pass(counts, "Accessibility permission (AgentHandover.app)");
            } else {
                check_fail(counts, "Accessibility permission (AgentHandover.app)");
                eprintln!(
                    "  Fix: System Settings > Privacy & Security > Accessibility: enable AgentHandover"
                );
            }
            if snap.screen_recording_granted {
                check_pass(counts, "Screen Recording permission (AgentHandover.app)");
            } else {
                check_fail(counts, "Screen Recording permission (AgentHandover.app)");
                eprintln!(
                    "  Fix: System Settings > Privacy & Security > Screen Recording: enable AgentHandover"
                );
            }
        }
        Err(e) => {
            check_fail(
                counts,
                &format!("Observation server ({})", obs_sock.display()),
            );
            eprintln!(
                "  AgentHandover.app is not running or its capture server failed to bind \
                 — the daemon captures nothing without it ({})",
                e
            );
            eprintln!("  Fix: open AgentHandover.app (it lives in /Applications)");
        }
    }

    let cap_sock = capture_socket_path();
    if is_socket(&cap_sock) {
        check_pass(counts, &format!("Capture socket ({})", cap_sock.display()));
    } else {
        check_fail(counts, &format!("Capture socket ({})", cap_sock.display()));
        eprintln!("  AgentHandover.app serves screenshots on this socket; open the app.");
    }
}

#[cfg(unix)]
fn is_socket(path: &std::path::Path) -> bool {
    use std::os::unix::fs::FileTypeExt;
    std::fs::metadata(path)
        .map(|m| m.file_type().is_socket())
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_socket(_path: &std::path::Path) -> bool {
    false
}

/// Python run in the venv: the worker and its MCP dependency must import,
/// and it prints the installed worker version.
const VENV_PROBE: &str = "import importlib.metadata as m, agenthandover_worker, mcp; \
                          print(m.version('agenthandover-worker'))";

fn check_venv(counts: &mut CheckCounts, channel: &str) {
    const NAME: &str = "Python virtual environment";
    let lib_dir = paths::lib_dir();
    let Some(python) = paths::find_venv_python() else {
        if lib_dir.exists() {
            check_fail(counts, NAME);
            eprintln!("  {} has no working venv/bin/python", lib_dir.display());
            print_install_hint(channel);
        } else {
            let hint = match channel {
                "homebrew" => "brew reinstall agenthandover",
                "source" => "venv not found — expected for source builds",
                _ => "Re-install AgentHandover (.pkg recommended)",
            };
            check_skip(counts, NAME, hint);
        }
        return;
    };

    let probe = std::process::Command::new(&python)
        .args(["-c", VENV_PROBE])
        .output();
    let outcome = match probe {
        Ok(out) if out.status.success() => {
            venv_version_verdict(&String::from_utf8_lossy(&out.stdout), env!("CARGO_PKG_VERSION"))
        }
        Ok(out) => Err(format!(
            "worker does not import: {}",
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .last()
                .unwrap_or("unknown error")
        )),
        Err(e) => Err(format!("cannot run {}: {}", python.display(), e)),
    };
    match outcome {
        Ok(version) => check_pass(counts, &format!("{} (worker {})", NAME, version)),
        Err(reason) => {
            check_fail(counts, &format!("{} ({})", NAME, reason));
            print_install_hint(channel);
        }
    }
}

/// Compare the venv's reported worker version with the CLI's own.
fn venv_version_verdict(probe_stdout: &str, cli_version: &str) -> Result<String, String> {
    let worker_version = probe_stdout.trim();
    if worker_version == cli_version {
        Ok(worker_version.to_string())
    } else {
        Err(format!(
            "worker {} does not match CLI {}",
            if worker_version.is_empty() { "?" } else { worker_version },
            cli_version
        ))
    }
}

fn check_pass(counts: &mut CheckCounts, name: &str) {
    counts.record_pass();
    println!("  {} {}", "pass".green(), name);
}

fn check_fail(counts: &mut CheckCounts, name: &str) {
    counts.record_fail();
    println!("  {} {}", "FAIL".red(), name);
}

fn check_warn(counts: &mut CheckCounts, name: &str) {
    counts.warnings += 1;
    println!("  {} {}", "WARN".yellow(), name);
}

fn check_skip(counts: &mut CheckCounts, name: &str, reason: &str) {
    counts.record_skip();
    println!("  {} {} ({})", "SKIP".yellow(), name, reason);
}

/// Advisory heartbeat freshness check for a service status file
/// (`~/Library/Application Support/agenthandover/<filename>`).  Never
/// affects the overall doctor result.
fn check_heartbeat_freshness(service_name: &str, status_filename: &str) {
    let label = format!("{} heartbeat", service_name);
    let path = agenthandover_common::status::status_dir().join(status_filename);
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => Some(c),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(_) => {
            println!("  {} {} (Cannot read status file)", "info".dimmed(), label);
            return;
        }
    };
    let (level, detail) = heartbeat_verdict(
        content.as_deref(),
        Utc::now(),
        agenthandover_common::pid::is_process_running,
    );
    let tag = match level {
        HeartbeatLevel::Pass => "pass".green(),
        HeartbeatLevel::Info => "info".dimmed(),
        HeartbeatLevel::Warn => "WARN".yellow(),
    };
    println!("  {} {} ({})", tag, label, detail);
}

#[derive(Debug, PartialEq)]
enum HeartbeatLevel {
    Pass,
    Info,
    Warn,
}

/// Classifies a status file's heartbeat.  `content` is None when the file
/// does not exist.  A status file whose `pid` is not a live process is left
/// over from an earlier run (an upgrade, a crash, a stop): the service is
/// not running, and the age of that file's heartbeat means nothing.
fn heartbeat_verdict(
    content: Option<&str>,
    now: chrono::DateTime<Utc>,
    pid_alive: impl Fn(u32) -> bool,
) -> (HeartbeatLevel, String) {
    let info = |m: &str| (HeartbeatLevel::Info, m.to_string());
    let Some(content) = content else {
        return info("No status file yet (service may not have started)");
    };
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(content) else {
        return info("Invalid JSON in status file");
    };
    let pid = parsed
        .get("pid")
        .and_then(|v| v.as_u64())
        .and_then(|p| u32::try_from(p).ok());
    if !pid.map(&pid_alive).unwrap_or(false) {
        return info("not running");
    }
    let Some(heartbeat_str) = parsed.get("heartbeat").and_then(|v| v.as_str()) else {
        return info("No heartbeat field in status file");
    };
    let Ok(heartbeat) = chrono::DateTime::parse_from_rfc3339(heartbeat_str) else {
        return info("Cannot parse heartbeat timestamp");
    };
    let age_secs = now
        .signed_duration_since(heartbeat.with_timezone(&Utc))
        .num_seconds();
    if age_secs <= 60 {
        (HeartbeatLevel::Pass, format!("{}s ago", age_secs))
    } else {
        (
            HeartbeatLevel::Warn,
            format!("last heartbeat {}s ago — service may be hung", age_secs),
        )
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

/// `AH_LAUNCH_AGENTS` (a temporary-prefix install, as in postinstall) or
/// `~/Library/LaunchAgents`.
fn launch_agents_dir() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("AH_LAUNCH_AGENTS").filter(|d| !d.is_empty()) {
        return std::path::PathBuf::from(dir);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn venv_version_matches_cli() {
        assert_eq!(venv_version_verdict("0.4.0\n", "0.4.0"), Ok("0.4.0".to_string()));
    }

    #[test]
    fn venv_version_mismatch_fails() {
        let err = venv_version_verdict("0.3.2\n", "0.4.0").unwrap_err();
        assert!(err.contains("0.3.2") && err.contains("0.4.0"), "{err}");
    }

    #[test]
    fn venv_version_empty_output_fails() {
        assert!(venv_version_verdict("", "0.4.0").is_err());
    }

    fn now() -> chrono::DateTime<Utc> {
        chrono::DateTime::parse_from_rfc3339("2026-09-23T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    #[test]
    fn stale_status_file_of_a_dead_process_reads_not_running() {
        // A 177-day-old status file left by the previous install.
        let old = r#"{"pid": 4242, "heartbeat": "2026-03-30T12:00:00Z"}"#;
        assert_eq!(
            heartbeat_verdict(Some(old), now(), |_| false),
            (HeartbeatLevel::Info, "not running".to_string())
        );
    }

    #[test]
    fn live_process_heartbeat_age_is_reported() {
        let fresh = r#"{"pid": 4242, "heartbeat": "2026-09-23T11:59:50Z"}"#;
        assert_eq!(
            heartbeat_verdict(Some(fresh), now(), |p| p == 4242),
            (HeartbeatLevel::Pass, "10s ago".to_string())
        );
        let hung = r#"{"pid": 4242, "heartbeat": "2026-09-23T11:50:00Z"}"#;
        let (level, detail) = heartbeat_verdict(Some(hung), now(), |_| true);
        assert_eq!(level, HeartbeatLevel::Warn);
        assert!(detail.contains("600s ago"), "{detail}");
    }

    #[test]
    fn missing_file_and_missing_pid() {
        assert_eq!(heartbeat_verdict(None, now(), |_| true).0, HeartbeatLevel::Info);
        assert_eq!(
            heartbeat_verdict(Some(r#"{"heartbeat": "2026-09-23T11:59:50Z"}"#), now(), |_| true),
            (HeartbeatLevel::Info, "not running".to_string())
        );
    }
}
