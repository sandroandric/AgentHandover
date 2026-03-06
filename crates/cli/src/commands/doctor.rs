use anyhow::Result;
use colored::Colorize;

use crate::paths;

pub fn run() -> Result<()> {
    println!("{}", "OpenMimic Doctor".bold());
    println!("{}", "=".repeat(50));
    println!();

    let mut all_ok = true;

    // Check 1: Daemon binary exists
    all_ok &= check("Daemon binary", || {
        std::path::Path::new("/usr/local/bin/oc-apprentice-daemon").exists()
            || which("oc-apprentice-daemon")
    });

    // Check 2: CLI binary (we're running it, so it exists)
    check("CLI binary", || true);

    // Check 3: Data directory exists
    let data_dir = oc_apprentice_common::status::status_dir();
    all_ok &= check("Data directory", || data_dir.exists());

    // Check 4: Config file exists
    let config_path = data_dir.join("config.toml");
    check_optional("Config file", || config_path.exists(), "Using defaults");

    // Check 5: Accessibility permission
    #[cfg(target_os = "macos")]
    {
        all_ok &= check("Accessibility permission", accessibility_sys_check);
    }

    // Check 6: Screen Recording permission
    #[cfg(target_os = "macos")]
    {
        check_optional(
            "Screen Recording permission",
            screen_recording_check,
            "Screenshots disabled",
        );
    }

    // Check 7: Database exists and is writable
    // On fresh installs the daemon hasn't created the DB yet — advisory, not fatal.
    let db_path = data_dir.join("events.db");
    let daemon_pid_exists = data_dir.join("daemon.pid").exists();
    if daemon_pid_exists {
        // Daemon has started at least once — DB should exist
        all_ok &= check("Database", || {
            db_path.exists()
                && std::fs::OpenOptions::new()
                    .write(true)
                    .open(&db_path)
                    .is_ok()
        });
    } else {
        // Fresh install — daemon never ran, DB expected to not exist
        check_optional(
            "Database",
            || {
                db_path.exists()
                    && std::fs::OpenOptions::new()
                        .write(true)
                        .open(&db_path)
                        .is_ok()
            },
            "Run 'openmimic start' to create database",
        );
    }

    // Check 8: Native messaging host manifest (content validation)
    all_ok &= check("Native messaging host", check_native_messaging_manifest);

    // Check 9: launchd plists installed
    let launch_agents = launch_agents_dir();
    all_ok &= check("Daemon launchd plist", || {
        launch_agents.join("com.openmimic.daemon.plist").exists()
    });
    all_ok &= check("Worker launchd plist", || {
        launch_agents.join("com.openmimic.worker.plist").exists()
    });

    // Check 10: Disk space
    all_ok &= check("Disk space (>1GB free)", || free_disk_gb() > 1);

    // Check 11: Python virtual environment
    // Check pkg path, Homebrew libexec (resolved from binary), and known opt paths.
    check_optional(
        "Python virtual environment",
        || paths::find_venv_python().is_some(),
        "Run installer or: brew install --HEAD openmimic",
    );

    // Check 12: Chrome extension
    // Homebrew installs dist contents flat into libexec/extension/ (no dist subdir).
    // Pkg installer uses /usr/local/lib/openmimic/extension/dist/.
    check_optional(
        "Chrome extension",
        || paths::find_any_extension_path().is_some(),
        "Build with: cd extension && npm run build",
    );

    // Check 13: Worker process alive
    check_optional(
        "Worker process",
        || oc_apprentice_common::pid::check_pid_file("worker").is_some(),
        "Start with: openmimic start",
    );

    println!();
    if all_ok {
        println!("{}", "All checks passed!".green().bold());
    } else {
        println!(
            "{}",
            "Some checks failed. See above for details."
                .yellow()
                .bold()
        );
    }

    Ok(())
}

fn check(name: &str, test: impl FnOnce() -> bool) -> bool {
    let result = test();
    if result {
        println!("  {} {}", "pass".green(), name);
    } else {
        println!("  {} {}", "FAIL".red(), name);
    }
    result
}

fn check_optional(name: &str, test: impl FnOnce() -> bool, fallback_msg: &str) {
    let result = test();
    if result {
        println!("  {} {}", "pass".green(), name);
    } else {
        println!("  {} {} ({})", "skip".yellow(), name, fallback_msg);
    }
}

fn which(binary: &str) -> bool {
    std::process::Command::new("which")
        .arg(binary)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
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
