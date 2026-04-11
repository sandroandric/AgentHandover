use anyhow::{bail, Result};
use std::process::{Command, Stdio};

// Worker is still managed via launchd — it has no TCC-sensitive permissions.
const WORKER_LABEL: &str = "com.agenthandover.worker";

// Daemon is launched DIRECTLY by the menu bar app (and here by the CLI)
// as a plain helper process.  It deliberately does NOT use launchd —
// having it as a launchd service meant it appeared as a second TCC
// principal (separate from the main app), which broke Screen Recording
// and Accessibility on macOS Tahoe.  The daemon binary lives outside the
// app bundle so codesign --deep doesn't re-register it either.  See
// issue #1 for the stale-plist confusion this architecture caused.
const DAEMON_BINARY_PATH: &str = "/usr/local/lib/agenthandover/ah-observer";

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

fn daemon_pid_file() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    std::path::PathBuf::from(home)
        .join("Library/Application Support/agenthandover/daemon.pid")
}

/// Check whether the daemon is currently running by looking up its pid file
/// and verifying the pid is actually ``ah-observer`` (not a recycled pid).
fn is_daemon_running() -> bool {
    let pid_file = daemon_pid_file();
    let pid_str = match std::fs::read_to_string(&pid_file) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let pid: i32 = match pid_str.trim().parse() {
        Ok(p) => p,
        Err(_) => return false,
    };
    // Verify the pid is actually ah-observer via `ps -p <pid> -o comm=`
    let output = Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "comm="])
        .output();
    match output {
        Ok(o) if o.status.success() => {
            let comm = String::from_utf8_lossy(&o.stdout);
            comm.contains("ah-observer")
        }
        _ => false,
    }
}

/// Start the daemon by spawning ``/usr/local/lib/agenthandover/ah-observer``
/// directly and detaching it so it survives after the CLI exits.
fn start_daemon_direct() -> Result<()> {
    println!("Starting daemon...");

    if is_daemon_running() {
        println!("  Daemon already running.");
        return Ok(());
    }

    let binary = std::path::Path::new(DAEMON_BINARY_PATH);
    if !binary.exists() {
        bail!(
            "daemon binary not found at {}.  Is AgentHandover installed?",
            DAEMON_BINARY_PATH
        );
    }

    // Spawn detached: redirect std streams to /dev/null so the child
    // doesn't hold onto the CLI's TTY, and drop the Child without
    // waiting so the process keeps running after the CLI exits.
    let child = Command::new(DAEMON_BINARY_PATH)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();

    match child {
        Ok(c) => {
            drop(c);
            std::thread::sleep(std::time::Duration::from_millis(500));
            if is_daemon_running() {
                println!("  Daemon started.");
                Ok(())
            } else {
                bail!(
                    "daemon failed to start (binary spawned but no pid file appeared). \
                     Check logs at ~/Library/Application Support/agenthandover/logs/daemon.log*"
                );
            }
        }
        Err(e) => bail!("failed to spawn daemon binary: {}", e),
    }
}

/// Stop the daemon by sending SIGTERM to its running process.
fn stop_daemon_direct() -> Result<()> {
    println!("Stopping daemon...");

    let pid_file = daemon_pid_file();
    let pid_str = match std::fs::read_to_string(&pid_file) {
        Ok(s) => s,
        Err(_) => {
            println!("  Daemon not running (no pid file).");
            return Ok(());
        }
    };
    let pid: i32 = match pid_str.trim().parse() {
        Ok(p) => p,
        Err(_) => {
            println!("  Daemon not running (invalid pid file).");
            return Ok(());
        }
    };

    // Graceful SIGTERM via kill(2).  Ignore errors if the process is already gone.
    let _ = Command::new("kill").args(["-TERM", &pid.to_string()]).status();
    println!("  Daemon stopped.");
    Ok(())
}

/// Outcome of a launchctl invocation — carries stderr for diagnostic messages.
struct LaunchctlResult {
    stderr: String,
}

fn launchctl(args: &[&str]) -> Result<LaunchctlResult> {
    let output = Command::new("launchctl").args(args).output()?;
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !stderr.is_empty() {
        eprintln!("  launchctl: {}", stderr);
    }
    Ok(LaunchctlResult { stderr })
}

fn is_job_running(label: &str) -> bool {
    Command::new("launchctl")
        .args(["list", label])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Start the launchd-managed worker job and verify it comes up.
fn start_worker_launchd() -> Result<()> {
    println!("Starting worker...");
    let result = launchctl(&["load", "-w", &plist_path(WORKER_LABEL)])?;
    std::thread::sleep(std::time::Duration::from_millis(500));
    if is_job_running(WORKER_LABEL) {
        println!("  Worker started.");
        Ok(())
    } else {
        let detail = if !result.stderr.is_empty() {
            format!("launchctl said: {}", result.stderr)
        } else {
            "job not listed after load".to_string()
        };
        bail!("worker failed to start: {}", detail);
    }
}

fn stop_worker_launchd() -> Result<()> {
    println!("Stopping worker...");
    launchctl(&["unload", &plist_path(WORKER_LABEL)])?;
    println!("  Worker stopped.");
    Ok(())
}

pub fn start(service: &str) -> Result<()> {
    match service {
        "daemon" => start_daemon_direct()?,
        "worker" => start_worker_launchd()?,
        "all" => {
            // Attempt both; collect failures so we report all of them.
            let d = start_daemon_direct();
            let w = start_worker_launchd();
            match (d, w) {
                (Ok(()), Ok(())) => println!("  All services started."),
                (Err(e1), Err(e2)) => bail!("{}\n{}", e1, e2),
                (Err(e), Ok(())) | (Ok(()), Err(e)) => return Err(e),
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
        "daemon" => stop_daemon_direct()?,
        "worker" => stop_worker_launchd()?,
        "all" => {
            println!("Stopping all services...");
            stop_daemon_direct()?;
            stop_worker_launchd()?;
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
