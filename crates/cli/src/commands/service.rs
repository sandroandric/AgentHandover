use anyhow::{bail, Result};
use std::process::Command;

const DAEMON_LABEL: &str = "com.openmimic.daemon";
const WORKER_LABEL: &str = "com.openmimic.worker";

fn launch_agents_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    std::path::PathBuf::from(home).join("Library/LaunchAgents")
}

fn plist_path(label: &str) -> String {
    launch_agents_dir()
        .join(format!("{}.plist", label))
        .display()
        .to_string()
}

/// Outcome of a launchctl invocation — carries stderr for diagnostic messages.
struct LaunchctlResult {
    stderr: String,
}

/// Run a launchctl command and capture stderr.
///
/// Note: `launchctl load` can return exit 0 yet emit errors on stderr
/// indicating the job was NOT actually loaded (e.g. "Could not find specified
/// service", domain errors).  Callers should use [`is_job_running`] to verify.
fn launchctl(args: &[&str]) -> Result<LaunchctlResult> {
    let output = Command::new("launchctl").args(args).output()?;
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !stderr.is_empty() {
        eprintln!("  launchctl: {}", stderr);
    }
    Ok(LaunchctlResult { stderr })
}

/// Check whether a launchd job is actually running by querying `launchctl list`.
fn is_job_running(label: &str) -> bool {
    Command::new("launchctl")
        .args(["list", label])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Start a single service, verifying it is actually running afterward.
fn start_one(label: &str, display_name: &str) -> Result<bool> {
    println!("Starting {}...", display_name);
    let result = launchctl(&["load", "-w", &plist_path(label)])?;

    // Give launchd a moment to spawn the process, then verify.
    std::thread::sleep(std::time::Duration::from_millis(500));
    let running = is_job_running(label);

    if running {
        println!("  {} started.", display_name);
        Ok(true)
    } else {
        eprintln!(
            "  ⚠ {} may not have started.{}",
            display_name,
            if !result.stderr.is_empty() {
                format!(" launchctl said: {}", result.stderr)
            } else {
                " Verify with: openmimic status".to_string()
            }
        );
        Ok(false)
    }
}

pub fn start(service: &str) -> Result<()> {
    match service {
        "daemon" => {
            start_one(DAEMON_LABEL, "daemon")?;
        }
        "worker" => {
            start_one(WORKER_LABEL, "worker")?;
        }
        "all" => {
            let d = start_one(DAEMON_LABEL, "daemon")?;
            let w = start_one(WORKER_LABEL, "worker")?;
            if d && w {
                println!("  All services started.");
            } else {
                eprintln!(
                    "  ⚠ One or more services may not have started. Run {} to check.",
                    "openmimic status"
                );
            }
        }
        _ => bail!(
            "Unknown service: {}. Use 'daemon', 'worker', or 'all'.",
            service
        ),
    }
    Ok(())
}

pub fn stop(service: &str) -> Result<()> {
    match service {
        "daemon" => {
            println!("Stopping daemon...");
            launchctl(&["unload", &plist_path(DAEMON_LABEL)])?;
            println!("  Daemon stopped.");
        }
        "worker" => {
            println!("Stopping worker...");
            launchctl(&["unload", &plist_path(WORKER_LABEL)])?;
            println!("  Worker stopped.");
        }
        "all" => {
            println!("Stopping all services...");
            launchctl(&["unload", &plist_path(DAEMON_LABEL)])?;
            launchctl(&["unload", &plist_path(WORKER_LABEL)])?;
            println!("  All services stopped.");
        }
        _ => bail!(
            "Unknown service: {}. Use 'daemon', 'worker', or 'all'.",
            service
        ),
    }
    Ok(())
}

pub fn restart(service: &str) -> Result<()> {
    stop(service)?;
    std::thread::sleep(std::time::Duration::from_secs(1));
    start(service)?;
    Ok(())
}
