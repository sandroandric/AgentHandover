use anyhow::Result;
use colored::Colorize;

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
    let db_path = data_dir.join("events.db");
    all_ok &= check("Database", || {
        db_path.exists()
            && std::fs::OpenOptions::new()
                .write(true)
                .open(&db_path)
                .is_ok()
    });

    // Check 8: Native messaging host manifest
    let nm_manifest = native_messaging_manifest_path();
    all_ok &= check("Native messaging host", || nm_manifest.exists());

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
    check_optional(
        "Python virtual environment",
        || std::path::Path::new("/usr/local/lib/openmimic/venv/bin/python").exists(),
        "Run installer or: python3 -m venv /usr/local/lib/openmimic/venv",
    );

    // Check 12: Chrome extension dist
    check_optional(
        "Chrome extension dist",
        || {
            std::path::Path::new("/usr/local/lib/openmimic/extension/dist").exists()
                || find_local_extension_dist()
        },
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

/// Check if extension/dist exists relative to common repo locations.
fn find_local_extension_dist() -> bool {
    // Check relative to the binary's location
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            // e.g. target/debug/ -> repo root is ../../
            for ancestor in parent.ancestors().take(5) {
                if ancestor.join("extension/dist").exists() {
                    return true;
                }
            }
        }
    }
    // Check current working directory
    if let Ok(cwd) = std::env::current_dir() {
        if cwd.join("extension/dist").exists() {
            return true;
        }
    }
    false
}

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
