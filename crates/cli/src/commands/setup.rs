//! Interactive setup wizard for AgentHandover.
//!
//! Walks through permissions, services, Chrome extension, and VLM setup.
//! Supports `--check` (dry-run), `--extension` (extension only), and
//! `--vlm` (VLM only) modes.

use anyhow::Result;
use colored::Colorize;
use agenthandover_common::{config::AppConfig, pid, status, tiers};

use crate::paths;

/// Chrome Native Messaging host name -- must match `crates/daemon/src/ipc/native_messaging.rs:26`
const NM_HOST_NAME: &str = "com.agenthandover.host";
/// Chrome extension ID derived from the RSA key in `extension/manifest.json`.
const EXTENSION_ID: &str = "jpemkdcihaijkolbkankcldmiimmmnfo";

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

/// Load config.toml from the standard OS location; falls back to defaults.
fn load_config() -> Result<AppConfig> {
    let path = config_file();
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
pub fn run(
    check_only: bool,
    extension_only: bool,
    vlm_only: bool,
    tier: Option<&str>,
) -> Result<()> {
    println!("{}", "AgentHandover Setup".bold());
    println!("{}", "=".repeat(50));
    println!();

    if extension_only {
        step_chrome_extension(check_only)?;
        return Ok(());
    }
    if vlm_only {
        step_vlm(check_only, tier)?;
        return Ok(());
    }

    // Full wizard: run all steps sequentially, collect warnings.
    let mut warnings: Vec<String> = Vec::new();

    step_permissions(check_only)?;
    println!();

    if let Err(e) = step_services(check_only) {
        warnings.push(format!("Services: {}", e));
    }
    println!();

    step_chrome_extension(check_only)?;
    println!();
    step_vlm(check_only, tier)?;
    println!();

    if warnings.is_empty() {
        println!("{}", "Setup complete!".green().bold());
    } else {
        println!("{}", "Setup finished with warnings:".yellow().bold());
        for w in &warnings {
            println!("  {} {}", "⚠".yellow(), w);
        }
    }
    println!(
        "Run {} to verify everything is working.",
        "agenthandover status".cyan()
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
                "Add AgentHandover (or Terminal) and toggle ON.".dimmed()
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
    // Read-only TCC check: no capture (CGDisplayCreateImage is obsoleted in
    // macOS 15 and would itself trigger the permission prompt).
    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGPreflightScreenCaptureAccess() -> bool;
    }
    unsafe { CGPreflightScreenCaptureAccess() }
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

    let mut failures: Vec<String> = Vec::new();

    if daemon_running {
        println!("  {} Daemon running", "✓".green());
    } else {
        println!("  {} Daemon {}", "✗".red(), "not running".red());
        if !check_only {
            println!("    Starting daemon...");
            if !start_service("com.agenthandover.daemon") {
                failures.push("daemon failed to start".to_string());
            }
        }
    }

    if worker_running {
        println!("  {} Worker running", "✓".green());
    } else {
        println!("  {} Worker {}", "✗".red(), "not running".red());
        if !check_only {
            println!("    Starting worker...");
            if !start_service("com.agenthandover.worker") {
                failures.push("worker failed to start".to_string());
            }
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        anyhow::bail!("{}", failures.join(", "))
    }
}

/// Attempt to start a launchd service. Returns `true` if the service is
/// confirmed running after load, `false` otherwise.
fn start_service(label: &str) -> bool {
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
        let channel = detect_install_channel();
        match channel {
            "pkg" => println!("    Re-run the .pkg installer to repair."),
            "homebrew" => println!("    Run {} to install plists.", "brew reinstall agenthandover".cyan()),
            "source" => println!(
                "    Run: {} and copy plists to ~/Library/LaunchAgents/",
                "just build-all".cyan()
            ),
            _ => println!("    Re-install AgentHandover ({} recommended).", ".pkg".cyan()),
        }
        return false;
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
        true
    } else {
        println!(
            "    {} {} may not have started{}",
            "⚠".yellow(),
            label,
            if stderr.is_empty() {
                String::from(". Verify with: agenthandover status")
            } else {
                format!(": {}", stderr)
            }
        );
        false
    }
}

// =============================================================================
// Native Messaging Host Manifest
// =============================================================================

/// Install the Native Messaging host manifest for all detected Chromium browsers.
///
/// Writes `com.agenthandover.host.json` containing the daemon path and
/// allowed_origins so the extension can connect.
fn install_native_messaging_manifest() -> Result<bool> {
    let daemon_path = match paths::find_daemon_binary() {
        Some(p) => p,
        None => {
            println!(
                "  {} Could not find {} binary",
                "⚠".yellow(),
                "agenthandover-daemon".bold()
            );
            println!(
                "    Build with: {} or {}",
                "cargo build --release -p agenthandover-daemon".cyan(),
                "just build-all".cyan()
            );
            return Ok(false);
        }
    };

    let nm_dirs = paths::native_messaging_hosts_dirs();
    if nm_dirs.is_empty() {
        println!("  {} No supported browser directories found", "⚠".yellow());
        return Ok(false);
    }

    let manifest = serde_json::json!({
        "name": NM_HOST_NAME,
        "description": "AgentHandover Observer Bridge",
        "path": daemon_path.display().to_string(),
        "type": "stdio",
        "allowed_origins": [format!("chrome-extension://{}/", EXTENSION_ID)]
    });
    let manifest_str = serde_json::to_string_pretty(&manifest)?;
    let manifest_filename = format!("{}.json", NM_HOST_NAME);

    let mut installed = 0;
    for dir in &nm_dirs {
        // Only install into dirs that exist or whose parent exists
        // (Chrome creates NativeMessagingHosts/ on first use, but the
        // parent app support dir must exist to indicate the browser is installed)
        let parent_exists = dir.parent().map_or(false, |p| p.exists());
        if !dir.exists() && !parent_exists {
            continue;
        }

        if let Err(e) = std::fs::create_dir_all(dir) {
            println!(
                "    {} Failed to create {}: {}",
                "⚠".yellow(),
                dir.display(),
                e
            );
            continue;
        }

        let manifest_path = dir.join(&manifest_filename);
        if let Err(e) = std::fs::write(&manifest_path, &manifest_str) {
            println!(
                "    {} Failed to write {}: {}",
                "⚠".yellow(),
                manifest_path.display(),
                e
            );
            continue;
        }

        installed += 1;
        println!(
            "  {} NM manifest installed: {}",
            "✓".green(),
            manifest_path.display()
        );
    }

    if installed == 0 {
        println!(
            "  {} No browser NM directories found (is Chrome/Chromium/Brave installed?)",
            "⚠".yellow()
        );
        return Ok(false);
    }

    println!(
        "    Daemon path: {}",
        daemon_path.display().to_string().dimmed()
    );

    Ok(true)
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

            // Install NM manifest before loading extension
            println!();
            println!("  Installing Native Messaging host manifest...");
            install_native_messaging_manifest()?;
            println!();

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
                "agenthandover status".cyan()
            );
        }
        None => {
            println!(
                "  {} Chrome extension files {}",
                "✗".red(),
                "not found".red()
            );
            let channel = detect_install_channel();
            match channel {
                "pkg" => println!("    Re-run the .pkg installer to repair."),
                "homebrew" => println!("    Run: {}", "brew reinstall agenthandover".cyan()),
                "source" => println!(
                    "    Build from source: {}",
                    "cd extension && npm install && npm run build".cyan()
                ),
                _ => println!("    Re-install AgentHandover ({} recommended).", ".pkg".cyan()),
            }
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

fn step_vlm(check_only: bool, tier_arg: Option<&str>) -> Result<()> {
    println!("{}", "Step 4: VLM Setup (Optional)".bold());

    // Report what the running worker says, if it is up.
    if let Ok(worker) = status::read_status_file::<status::WorkerStatus>("worker-status.json") {
        if worker.vlm_available {
            println!(
                "  {} VLM {} ({}/{})",
                "✓".green(),
                "available".green(),
                worker.vlm_mode.as_deref().unwrap_or("local"),
                worker.vlm_provider.as_deref().unwrap_or("auto-detect"),
            );
        }
    }

    let config = load_config()?;
    if config.vlm.mode == "remote" && tier_arg.is_none() {
        let provider = config.vlm.provider.as_deref().unwrap_or("unknown");
        let model = config.vlm.model.as_deref().unwrap_or("(default)");
        println!(
            "  {} VLM configured as {} ({}/{})",
            "✓".green(),
            "remote".cyan(),
            provider,
            model,
        );
        if let Some(ref env_var) = config.vlm.api_key_env {
            if std::env::var(env_var).is_ok() {
                println!("  {} API key ({}) {}", "✓".green(), env_var, "is set".green());
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

    // Local mode: the models of the chosen tier (the one tier table,
    // tiers.json), pulled with Ollama; the choice is written to [vlm] in
    // the same bytes onboarding writes.
    let table = tiers::table();
    let tier = match tier_arg {
        Some(id) => table.get(id).ok_or_else(|| {
            let ids: Vec<&str> = table.tiers.iter().map(|t| t.id.as_str()).collect();
            anyhow::anyhow!("unknown tier '{}'; choose one of: {}", id, ids.join(", "))
        })?,
        None => tiers::resolve_models(config.vlm.tier.as_deref(), None, None).0,
    };
    let models = table.models(tier);
    println!(
        "  {} Tier: {} ({}; {} GB RAM detected)",
        "•".cyan(),
        tier.label.bold(),
        tier.subtitle,
        tiers::machine_ram_gb(),
    );

    let Some(path) = find_ollama() else {
        println!(
            "  {} Ollama {}",
            "~".yellow(),
            "not installed (VLM features disabled)".yellow()
        );
        if !check_only {
            println!("    To enable VLM features, install Ollama:");
            println!("      • Download: {}", "https://ollama.com/download/mac".cyan());
            println!(
                "    Then run {} again.",
                "agenthandover setup --vlm".cyan()
            );
        }
        return Ok(());
    };
    println!("  {} Ollama installed at: {}", "✓".green(), path.display());

    let Some(installed) = ollama_models(&path) else {
        println!(
            "  {} Ollama is not running: open the Ollama app, then run {} again.",
            "✗".red(),
            "agenthandover setup --vlm".cyan()
        );
        return Ok(());
    };
    let missing: Vec<&str> = models
        .iter()
        .copied()
        .filter(|m| !installed.iter().any(|i| i == m))
        .collect();

    if missing.is_empty() {
        println!("  {} Models present: {}", "✓".green(), models.join(", "));
        if !check_only {
            persist_tier(tier)?;
        }
        return Ok(());
    }
    println!("  {} Models to pull: {}", "~".yellow(), missing.join(", "));
    if check_only {
        println!("    Run {} to pull them.", "agenthandover setup --vlm".cyan());
        return Ok(());
    }

    let needs_new_ollama = missing.iter().any(|m| m.starts_with("gemma4"));
    let minimum = &table.minimum_ollama_version_for_gemma4;
    if needs_new_ollama {
        match ollama_version(&path) {
            Some(v) if !version_at_least(&v, minimum) => {
                println!(
                    "  {} Gemma 4 QAT models need Ollama {}+; you have {}. Update the app at {}.",
                    "✗".red(),
                    minimum,
                    v,
                    "https://ollama.com/download".cyan()
                );
                return Ok(());
            }
            _ => {}
        }
    }

    // Persist the choice before pulling, as onboarding does, so an
    // interrupted pull leaves the config naming the models being pulled.
    persist_tier(tier)?;
    for model in &missing {
        println!("    Pulling {}...", model.bold());
        match std::process::Command::new(&path)
            .args(["pull", model])
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .status()
        {
            Ok(exit) if exit.success() => {
                println!("  {} Model {} pulled", "✓".green(), model.bold());
            }
            Ok(exit) => {
                println!(
                    "\n  {} Pull of {} failed (exit code: {:?}); make sure the Ollama app is open.",
                    "✗".red(),
                    model,
                    exit.code()
                );
                return Ok(());
            }
            Err(e) => {
                println!("\n  {} Failed to run ollama: {}", "✗".red(), e);
                return Ok(());
            }
        }
    }
    println!(
        "    Restart the worker to use them: {}",
        "agenthandover restart worker".cyan()
    );
    Ok(())
}

/// Where `config.toml` lives (it may not exist yet).
fn config_file() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    if cfg!(target_os = "macos") {
        std::path::PathBuf::from(&home)
            .join("Library/Application Support/agenthandover/config.toml")
    } else {
        std::path::PathBuf::from(&home).join(".config/agenthandover/config.toml")
    }
}

/// Write `tier`'s managed `[vlm]` keys into config.toml (atomically).
fn persist_tier(tier: &tiers::Tier) -> Result<()> {
    let path = config_file();
    let current = std::fs::read_to_string(&path).unwrap_or_default();
    let updated = tiers::apply_vlm_block(&current, tier);
    if updated != current {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("toml.tmp");
        std::fs::write(&tmp, updated)?;
        std::fs::rename(&tmp, &path)?;
    }
    println!("  {} Saved tier {} to {}", "✓".green(), tier.id, path.display());
    Ok(())
}

/// `ollama list`'s model names (`:latest` dropped); None if it failed.
fn ollama_models(ollama_path: &std::path::Path) -> Option<Vec<String>> {
    let output = std::process::Command::new(ollama_path).arg("list").output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(parse_ollama_list(&String::from_utf8_lossy(&output.stdout)))
}

fn parse_ollama_list(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .skip(1) // header
        .filter_map(|line| line.split_whitespace().next())
        .map(|name| name.strip_suffix(":latest").unwrap_or(name).to_string())
        .collect()
}

/// `ollama --version` → "0.34.2".
fn ollama_version(ollama_path: &std::path::Path) -> Option<String> {
    let output = std::process::Command::new(ollama_path).arg("--version").output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    text.split_whitespace()
        .rev()
        .find(|w| w.chars().next().is_some_and(|c| c.is_ascii_digit()))
        .map(str::to_string)
}

/// `version` ("0.34.2-rc1") >= `minimum` ("0.34.0").
fn version_at_least(version: &str, minimum: &str) -> bool {
    let parse = |s: &str| -> Option<Vec<u64>> {
        s.split('-').next()?.split('.').map(|p| p.parse().ok()).collect()
    };
    match (parse(version), parse(minimum)) {
        (Some(have), Some(need)) => have >= need,
        _ => false,
    }
}

#[cfg(test)]
mod vlm_tests {
    use super::*;

    #[test]
    fn parses_ollama_list() {
        let out = "NAME                 ID      SIZE   MODIFIED\n\
                   qwen3.5:4b           2a65    3.4 GB 1 day ago\n\
                   nomic-embed-text:latest 0a10 274 MB 2 days ago\n";
        assert_eq!(parse_ollama_list(out), ["qwen3.5:4b", "nomic-embed-text"]);
    }

    #[test]
    fn compares_versions() {
        assert!(version_at_least("0.34.2", "0.34.0"));
        assert!(version_at_least("0.34.0-rc1", "0.34.0"));
        assert!(!version_at_least("0.30.6", "0.34.0"));
        assert!(!version_at_least("dev", "0.34.0"));
    }
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
