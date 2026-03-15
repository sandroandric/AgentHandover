use anyhow::Result;
use chrono::Utc;
use colored::Colorize;

use crate::paths;

/// Detect which install channel produced this OpenMimic installation.
fn detect_install_channel() -> &'static str {
    // Check for pkg install
    if std::path::Path::new("/usr/local/bin/oc-apprentice-daemon").exists()
        && std::path::Path::new("/usr/local/lib/openmimic").exists()
    {
        return "pkg";
    }
    // Check for Homebrew
    if let Ok(output) = std::process::Command::new("brew")
        .args(["--prefix", "openmimic"])
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
        "homebrew" => eprintln!("  Fix: brew reinstall openmimic"),
        "source" => eprintln!("  Fix: just build-all"),
        _ => eprintln!("  Fix: Re-install OpenMimic (.pkg recommended)."),
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
    println!("{}", "OpenMimic Doctor".bold());
    println!("{}", "=".repeat(50));
    println!();

    let channel = detect_install_channel();
    let mut counts = CheckCounts::new();

    // Check 1: Daemon binary exists
    // Search pkg path, PATH, and source build directories.
    let daemon_found = std::path::Path::new("/usr/local/bin/oc-apprentice-daemon").exists()
        || which("oc-apprentice-daemon")
        || source_build_daemon_exists();
    if daemon_found {
        check_pass(&mut counts, "Daemon binary");
    } else {
        check_fail(&mut counts, "Daemon binary");
        print_install_hint(channel);
    }

    // Check 2: CLI binary (we're running it, so it exists)
    check_pass(&mut counts, "CLI binary");

    // Check 3: Data directory exists
    let data_dir = oc_apprentice_common::status::status_dir();
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

    // Check 5: Accessibility permission
    #[cfg(target_os = "macos")]
    {
        if accessibility_sys_check() {
            check_pass(&mut counts, "Accessibility permission");
        } else {
            check_fail(&mut counts, "Accessibility permission");
        }
    }

    // Check 6: Screen Recording permission (optional)
    #[cfg(target_os = "macos")]
    {
        if screen_recording_check() {
            check_pass(&mut counts, "Screen Recording permission");
        } else {
            check_skip(
                &mut counts,
                "Screen Recording permission",
                "Screenshots disabled",
            );
        }
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
                "Run 'openmimic start' to create database",
            );
        }
    }

    // Check 8: Native messaging host manifest (content validation)
    if check_native_messaging_manifest() {
        check_pass(&mut counts, "Native messaging host");
    } else {
        check_fail(&mut counts, "Native messaging host");
    }

    // Check 9: launchd plists installed
    let launch_agents = launch_agents_dir();
    if launch_agents.join("com.openmimic.daemon.plist").exists() {
        check_pass(&mut counts, "Daemon launchd plist");
    } else {
        check_fail(&mut counts, "Daemon launchd plist");
        print_install_hint(channel);
    }
    if launch_agents.join("com.openmimic.worker.plist").exists() {
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
            "homebrew" => "brew reinstall openmimic",
            "source" => "venv not found — expected for source builds",
            _ => "Re-install OpenMimic (.pkg recommended)",
        };
        check_skip(&mut counts, "Python virtual environment", hint);
    }

    // Check 13: Chrome extension (optional — not all users need it)
    if paths::find_any_extension_path().is_some() {
        check_pass(&mut counts, "Chrome extension");
    } else {
        let hint = match channel {
            "pkg" => "Re-run .pkg installer to repair",
            "homebrew" => "brew reinstall openmimic",
            "source" => "cd extension && npm run build",
            _ => "Re-install OpenMimic (.pkg recommended)",
        };
        check_skip(&mut counts, "Chrome extension", hint);
    }

    // Check 14: Worker process alive (optional — may not be started yet)
    if oc_apprentice_common::pid::check_pid_file("worker").is_some() {
        check_pass(&mut counts, "Worker process");
    } else {
        check_skip(
            &mut counts,
            "Worker process",
            "Start with: openmimic start",
        );
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
/// Reads `~/Library/Application Support/oc-apprentice/<filename>`, parses the
/// `heartbeat` ISO-8601 timestamp, and prints a WARNING if it is older than 60
/// seconds.  If the file doesn't exist (first install, service never started),
/// prints an INFO note.  This check never affects the overall doctor result.
fn check_heartbeat_freshness(service_name: &str, status_filename: &str) {
    let label = format!("{} heartbeat", service_name);

    let status_dir = oc_apprentice_common::status::status_dir();
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

fn which(binary: &str) -> bool {
    std::process::Command::new("which")
        .arg(binary)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check common source build output directories for the daemon binary.
fn source_build_daemon_exists() -> bool {
    let source_paths = [
        "target/release/oc-apprentice-daemon",
        "target/debug/oc-apprentice-daemon",
        "target/universal-release/oc-apprentice-daemon",
    ];
    for sp in &source_paths {
        if std::path::Path::new(sp).exists() {
            return true;
        }
    }
    false
}

// Path resolution functions (find_homebrew_libexec, find_venv_python,
// find_extension_dir, find_local_extension_dist) are now in crate::paths.

#[cfg(target_os = "macos")]
fn accessibility_sys_check() -> bool {
    // Use the same AXIsProcessTrusted() API the daemon uses.
    // accessibility-sys wraps this but we call it via the C ABI directly
    // to avoid pulling the full crate into the CLI.
    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        fn AXIsProcessTrusted() -> bool;
    }
    unsafe { AXIsProcessTrusted() }
}

#[cfg(target_os = "macos")]
fn screen_recording_check() -> bool {
    // Independently probe screen recording by attempting a display capture.
    // CGDisplayCreateImage returns NULL without Screen Recording permission.
    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGMainDisplayID() -> u32;
        fn CGDisplayCreateImage(display_id: u32) -> *const std::ffi::c_void;
    }
    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFRelease(cf: *const std::ffi::c_void);
    }
    unsafe {
        let image = CGDisplayCreateImage(CGMainDisplayID());
        if image.is_null() {
            false
        } else {
            CFRelease(image);
            true
        }
    }
}

fn native_messaging_manifest_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    // Chrome's NativeMessagingHosts location on macOS
    std::path::PathBuf::from(home).join(
        "Library/Application Support/Google/Chrome/NativeMessagingHosts/com.openclaw.apprentice.json",
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
            "openmimic setup --extension".cyan()
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
    if name != "com.openclaw.apprentice" {
        println!(
            "    {} Wrong host name: expected 'com.openclaw.apprentice', got '{}'",
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
            "openmimic setup --extension".cyan()
        );
        return false;
    }

    // Verify allowed_origins contains expected extension ID
    let origins = parsed.get("allowed_origins").and_then(|v| v.as_array());
    let expected_origin = "chrome-extension://knldjmfmopnpolahpmmgbagdohdnhkik/";
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
