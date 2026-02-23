//! Interactive setup wizard for OpenMimic.
//!
//! Walks through permissions, services, Chrome extension, and VLM setup.
//! Supports `--check` (dry-run), `--extension` (extension only), and
//! `--vlm` (VLM only) modes.

use anyhow::Result;
use colored::Colorize;
use oc_apprentice_common::{config::AppConfig, pid, status};

use crate::paths;

/// Load config.toml from the standard OS location; falls back to defaults.
fn load_config() -> Result<AppConfig> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let path = if cfg!(target_os = "macos") {
        std::path::PathBuf::from(&home)
            .join("Library/Application Support/oc-apprentice/config.toml")
    } else {
        std::path::PathBuf::from(&home).join(".config/oc-apprentice/config.toml")
    };
    if path.exists() {
        Ok(AppConfig::from_file(&path)?)
    } else {
        Ok(AppConfig::default())
    }
}

/// Run the setup wizard.
///
/// `check_only`: just report status, don't modify anything.
/// `extension_only` / `vlm_only`: run only that step.
pub fn run(check_only: bool, extension_only: bool, vlm_only: bool) -> Result<()> {
    println!("{}", "OpenMimic Setup".bold());
    println!("{}", "=".repeat(50));
    println!();

    if extension_only {
        step_chrome_extension(check_only)?;
        return Ok(());
    }
    if vlm_only {
        step_vlm(check_only)?;
        return Ok(());
    }

    // Full wizard: run all steps sequentially
    step_permissions(check_only)?;
    println!();
    step_services(check_only)?;
    println!();
    step_chrome_extension(check_only)?;
    println!();
    step_vlm(check_only)?;
    println!();

    println!("{}", "Setup complete!".green().bold());
    println!(
        "Run {} to verify everything is working.",
        "openmimic status".cyan()
    );

    Ok(())
}

// =============================================================================
// Step 1: Permissions
// =============================================================================

fn step_permissions(check_only: bool) -> Result<()> {
    println!("{}", "Step 1: Permissions".bold());

    // Accessibility
    let ax_granted = check_accessibility();
    if ax_granted {
        println!("  {} Accessibility permission granted", "✓".green());
    } else {
        println!("  {} Accessibility permission {}", "✗".red(), "NOT granted".red());
        if !check_only {
            println!(
                "    Opening System Settings → Privacy & Security → Accessibility..."
            );
            open_accessibility_settings();
            println!(
                "    {}",
                "Add OpenMimic (or Terminal) and toggle ON.".dimmed()
            );
        }
    }

    // Screen Recording
    let sr_granted = check_screen_recording();
    if sr_granted {
        println!("  {} Screen Recording permission granted", "✓".green());
    } else {
        println!(
            "  {} Screen Recording permission {} (optional — needed for screenshots)",
            "~".yellow(),
            "not granted".yellow()
        );
        if !check_only {
            println!(
                "    Opening System Settings → Privacy & Security → Screen Recording..."
            );
            open_screen_recording_settings();
        }
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn check_accessibility() -> bool {
    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        fn AXIsProcessTrusted() -> bool;
    }
    unsafe { AXIsProcessTrusted() }
}

#[cfg(not(target_os = "macos"))]
fn check_accessibility() -> bool {
    true // Not applicable on non-macOS
}

#[cfg(target_os = "macos")]
fn check_screen_recording() -> bool {
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

#[cfg(not(target_os = "macos"))]
fn check_screen_recording() -> bool {
    true
}

fn open_accessibility_settings() {
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("/usr/bin/open")
            .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")
            .spawn();
    }
}

fn open_screen_recording_settings() {
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("/usr/bin/open")
            .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")
            .spawn();
    }
}

// =============================================================================
// Step 2: Services
// =============================================================================

fn step_services(check_only: bool) -> Result<()> {
    println!("{}", "Step 2: Services".bold());

    let daemon_running = pid::check_pid_file("daemon").is_some();
    let worker_running = pid::check_pid_file("worker").is_some();

    if daemon_running {
        println!("  {} Daemon running", "✓".green());
    } else {
        println!("  {} Daemon {}", "✗".red(), "not running".red());
        if !check_only {
            println!("    Starting daemon...");
            start_service("com.openmimic.daemon");
        }
    }

    if worker_running {
        println!("  {} Worker running", "✓".green());
    } else {
        println!("  {} Worker {}", "✗".red(), "not running".red());
        if !check_only {
            println!("    Starting worker...");
            start_service("com.openmimic.worker");
        }
    }

    Ok(())
}

fn start_service(label: &str) {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let plist_path = format!(
        "{}/Library/LaunchAgents/{}.plist",
        home, label
    );
    let plist = std::path::Path::new(&plist_path);
    if !plist.exists() {
        println!(
            "    {} launchd plist not found at {}",
            "⚠".yellow(),
            plist_path
        );
        println!(
            "    Run {} to install plists.",
            "brew install --HEAD openmimic".cyan()
        );
        return;
    }

    let output = std::process::Command::new("launchctl")
        .args(["load", "-w", &plist_path])
        .output();
    let stderr = match &output {
        Ok(o) => String::from_utf8_lossy(&o.stderr).trim().to_string(),
        Err(e) => format!("{}", e),
    };

    // Give launchd a moment to spawn, then verify the job is actually listed.
    std::thread::sleep(std::time::Duration::from_millis(500));
    let running = std::process::Command::new("launchctl")
        .args(["list", label])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if running {
        println!("    {} {} loaded and running", "✓".green(), label);
    } else {
        println!(
            "    {} {} may not have started{}",
            "⚠".yellow(),
            label,
            if stderr.is_empty() {
                String::from(". Verify with: openmimic status")
            } else {
                format!(": {}", stderr)
            }
        );
    }
}

// =============================================================================
// Step 3: Chrome Extension
// =============================================================================

fn step_chrome_extension(check_only: bool) -> Result<()> {
    println!("{}", "Step 3: Chrome Extension".bold());

    // Check if extension is already connected via heartbeat
    let heartbeat_fresh = status::read_extension_heartbeat().is_some();
    if heartbeat_fresh {
        println!(
            "  {} Chrome extension {}",
            "✓".green(),
            "connected".green()
        );
        return Ok(());
    }

    // Find extension directory
    let ext_path = paths::find_any_extension_path();
    match ext_path {
        Some(ref path) => {
            println!("  {} Extension found at: {}", "✓".green(), path.display());

            if check_only {
                println!(
                    "  {} Extension {}",
                    "~".yellow(),
                    "not connected (run without --check to set up)".yellow()
                );
                return Ok(());
            }

            // Copy path to clipboard
            let path_str = path.display().to_string();
            if copy_to_clipboard(&path_str) {
                println!("    📋 Extension path copied to clipboard!");
            } else {
                println!("    Path: {}", path_str.cyan());
                println!("    (Copy the path above manually)");
            }

            // Open Chrome extensions page
            println!("    Opening chrome://extensions...");
            let _ = std::process::Command::new("/usr/bin/open")
                .args(["-a", "Google Chrome", "chrome://extensions"])
                .spawn();

            println!();
            println!("    Follow these steps in Chrome:");
            println!(
                "      {} Enable {} (top-right toggle)",
                "1.".bold(),
                "Developer Mode".bold()
            );
            println!(
                "      {} Click {}",
                "2.".bold(),
                "Load Unpacked".bold()
            );
            println!(
                "      {} Paste the path ({}) and click Select",
                "3.".bold(),
                "Cmd+V".bold()
            );
            println!();

            // Poll for heartbeat (up to 45s)
            println!(
                "    {}",
                "Waiting for extension connection... (Ctrl+C to skip)".dimmed()
            );
            let start = std::time::Instant::now();
            let timeout = std::time::Duration::from_secs(45);
            while start.elapsed() < timeout {
                std::thread::sleep(std::time::Duration::from_secs(2));
                if status::read_extension_heartbeat().is_some() {
                    println!(
                        "\n    {} Extension {}!",
                        "✓".green(),
                        "connected".green().bold()
                    );
                    return Ok(());
                }
                let remaining = timeout.saturating_sub(start.elapsed()).as_secs();
                print!("\r    Waiting... {}s remaining ", remaining);
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
            println!();
            println!(
                "    {} Timed out waiting for connection.",
                "⚠".yellow()
            );
            println!(
                "    The extension may still work — check {} later.",
                "openmimic status".cyan()
            );
        }
        None => {
            println!(
                "  {} Chrome extension files {}",
                "✗".red(),
                "not found".red()
            );
            println!("    Install via: brew install --HEAD openmimic");
            println!(
                "    Or build from source: cd extension && npm install && npm run build"
            );
        }
    }

    Ok(())
}

fn copy_to_clipboard(text: &str) -> bool {
    match std::process::Command::new("pbcopy")
        .stdin(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            if let Some(ref mut stdin) = child.stdin {
                stdin.write_all(text.as_bytes())?;
            }
            child.wait()
        })
    {
        Ok(status) if status.success() => true,
        Ok(_) => {
            println!("    {} pbcopy exited with error", "⚠".yellow());
            false
        }
        Err(e) => {
            println!("    {} Failed to copy to clipboard: {}", "⚠".yellow(), e);
            false
        }
    }
}

// =============================================================================
// Step 4: VLM Setup (Optional)
// =============================================================================

fn step_vlm(check_only: bool) -> Result<()> {
    println!("{}", "Step 4: VLM Setup (Optional)".bold());

    // Check if VLM is already available from worker status
    if let Ok(worker) = status::read_status_file::<status::WorkerStatus>("worker-status.json") {
        if worker.vlm_available {
            // Show mode details if available
            let mode_label = worker
                .vlm_mode
                .as_deref()
                .unwrap_or("local");
            let provider_label = worker
                .vlm_provider
                .as_deref()
                .unwrap_or("auto-detect");
            println!(
                "  {} VLM {} ({}/{})",
                "✓".green(),
                "available".green(),
                mode_label,
                provider_label,
            );
            return Ok(());
        }
    }

    // Check config for remote mode
    if let Ok(config) = load_config() {
        if config.vlm.mode == "remote" {
            let provider = config.vlm.provider.as_deref().unwrap_or("unknown");
            let model = config.vlm.model.as_deref().unwrap_or("(default)");
            println!(
                "  {} VLM configured as {} ({}/{})",
                "✓".green(),
                "remote".cyan(),
                provider,
                model,
            );
            // Check if the API key env var is set
            if let Some(ref env_var) = config.vlm.api_key_env {
                if std::env::var(env_var).is_ok() {
                    println!(
                        "  {} API key ({}) {}",
                        "✓".green(),
                        env_var,
                        "is set".green()
                    );
                } else {
                    println!(
                        "  {} API key ({}) {}",
                        "✗".red(),
                        env_var,
                        "NOT set — export it in your shell profile".red()
                    );
                }
            }
            return Ok(());
        }
    }

    // Check if Ollama is installed (local mode)
    let ollama_path = find_ollama();
    match ollama_path {
        Some(ref path) => {
            println!("  {} Ollama installed at: {}", "✓".green(), path.display());

            // Check if a model is already pulled
            if has_ollama_model(path) {
                println!(
                    "  {} Ollama model available",
                    "✓".green()
                );
                return Ok(());
            }

            println!(
                "  {} No VLM model pulled yet",
                "~".yellow()
            );

            if check_only {
                println!("    Run {} to pull a model.", "openmimic setup --vlm".cyan());
                return Ok(());
            }

            println!(
                "    Pulling recommended model ({})...",
                "llava:7b".bold()
            );
            println!(
                "    {}",
                "This may take a few minutes on first download.".dimmed()
            );

            let pull_result = std::process::Command::new(path)
                .args(["pull", "llava:7b"])
                .stdout(std::process::Stdio::inherit())
                .stderr(std::process::Stdio::inherit())
                .status();

            match pull_result {
                Ok(exit) if exit.success() => {
                    println!(
                        "\n  {} Model {} pulled successfully!",
                        "✓".green(),
                        "llava:7b".bold()
                    );
                    println!(
                        "    Restart the worker to activate: {}",
                        "openmimic restart worker".cyan()
                    );
                }
                Ok(exit) => {
                    println!(
                        "\n  {} Model pull failed (exit code: {:?})",
                        "✗".red(),
                        exit.code()
                    );
                    println!("    Make sure Ollama is running: ollama serve");
                }
                Err(e) => {
                    println!("\n  {} Failed to run ollama: {}", "✗".red(), e);
                }
            }
        }
        None => {
            println!(
                "  {} Ollama {}",
                "~".yellow(),
                "not installed (VLM features disabled)".yellow()
            );
            if !check_only {
                println!("    To enable VLM features, install Ollama:");
                println!(
                    "      • Download: {}",
                    "https://ollama.com/download/mac".cyan()
                );
                println!(
                    "      • Or: {}",
                    "brew install ollama".cyan()
                );
                println!(
                    "    Then run {} again.",
                    "openmimic setup --vlm".cyan()
                );
            }
        }
    }

    Ok(())
}

fn find_ollama() -> Option<std::path::PathBuf> {
    // Check well-known paths
    for path in &[
        "/usr/local/bin/ollama",
        "/opt/homebrew/bin/ollama",
    ] {
        let p = std::path::PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    // Check PATH via `which`
    if let Ok(output) = std::process::Command::new("which")
        .arg("ollama")
        .output()
    {
        if output.status.success() {
            let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path_str.is_empty() {
                return Some(std::path::PathBuf::from(path_str));
            }
        }
    }
    None
}

fn has_ollama_model(ollama_path: &std::path::Path) -> bool {
    // Run `ollama list` and check if any model is available
    if let Ok(output) = std::process::Command::new(ollama_path)
        .arg("list")
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Output has a header line, then model lines
            // If there are more than 1 lines (header + at least one model), we have models
            let lines: Vec<&str> = stdout.lines().collect();
            return lines.len() > 1;
        }
    }
    false
}
