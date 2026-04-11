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

/// Outcome of a launchctl invocation — carries stderr for diagnostic
/// messages. The caller decides whether to surface stderr to the user
/// (unlike the v0.2.3 wrapper which auto-printed it, causing confusing
/// "launchctl: Load failed: 5" noise before the verification check
/// on already-running workers — see v0.2.4 fix for Issue B).
struct LaunchctlResult {
    stderr: String,
    success: bool,
}

/// Run ``launchctl`` with the given args and return stderr + success.
/// DOES NOT print stderr automatically — let the caller decide whether
/// a failure is real or just noise from an idempotent call (e.g.
/// bootstrap on an already-loaded job).
fn launchctl_silent(args: &[&str]) -> Result<LaunchctlResult> {
    let output = Command::new("launchctl").args(args).output()?;
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    Ok(LaunchctlResult {
        stderr,
        success: output.status.success(),
    })
}

fn is_job_running(label: &str) -> bool {
    Command::new("launchctl")
        .args(["list", label])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Return the GUI launchd domain for the current user: ``gui/<uid>``.
/// This is the same domain the menu bar app's ``ServiceController``
/// uses, so CLI-initiated and app-initiated bootstraps share state.
fn gui_domain() -> String {
    // Safe: getuid() is infallible on Unix.
    let uid = unsafe { libc::getuid() };
    format!("gui/{}", uid)
}

/// Start the launchd-managed worker job and verify it comes up.
///
/// Uses the modern ``bootstrap`` + ``kickstart`` API to match the
/// SwiftUI menu bar app's ``ServiceController.startWorker()`` —
/// previously used the deprecated ``launchctl load -w`` which
/// (a) is a macOS 13+ soft-deprecated path and (b) returns
/// ``Load failed: 5: Input/output error`` on stderr when the job
/// is already loaded, even though the worker is running fine.
/// That stderr was then auto-printed by the old ``launchctl()``
/// wrapper BEFORE the ``is_job_running`` success check fired,
/// producing output like:
///
///     Starting worker...
///       launchctl: Load failed: 5: Input/output error
///       Worker started.
///
/// Confusing UX reported by hikoae on v0.2.3 issue #1.
///
/// New flow: check if the worker is already running first (clean
/// idempotent no-op), then try bootstrap (loads the plist into the
/// user's GUI domain if it isn't already), then kickstart (forces
/// launchd to run the job now rather than waiting for a trigger).
/// Only surface launchctl stderr if the final verification fails.
fn start_worker_launchd() -> Result<()> {
    println!("Starting worker...");

    // Idempotent success: already running → no-op, no launchctl call,
    // no scary error output to confuse the user.
    if is_job_running(WORKER_LABEL) {
        println!("  Worker already running.");
        return Ok(());
    }

    let plist = plist_path(WORKER_LABEL);
    let domain = gui_domain();
    let label_in_domain = format!("{}/{}", domain, WORKER_LABEL);

    // Fail fast with a clear diagnostic if the plist isn't installed
    // in the user's LaunchAgents dir. ``launchctl bootstrap`` gives the
    // famously unhelpful "Bootstrap failed: 5: Input/output error" on
    // every failure, so we pre-check the most common cause ourselves.
    //
    // This is the exact failure mode that hikoae hit on v0.2.3: the
    // postinstall's ``/dev/console`` console-user detection returned
    // ``root``, so the plist was installed to
    // ``/var/root/Library/LaunchAgents`` instead of his user's
    // LaunchAgents. From his perspective, ``agenthandover start`` failed
    // with an opaque "Input/output error". v0.2.4 also fixes the
    // postinstall detection (see ``resources/pkg/scripts/postinstall``),
    // but if it still happens this error points directly at the fix.
    if !std::path::Path::new(&plist).exists() {
        bail!(
            "Worker plist not found at:\n  {}\n\
             \n\
             This usually means the postinstall script couldn't detect \
             the correct user at install time and installed the plist \
             to the wrong home directory. To fix:\n\
             \n\
             1. Re-run the pkg installer:\n\
                sudo installer -pkg AgentHandover-*.pkg -target /\n\
             \n\
             2. Or manually copy the plist from the pkg-shipped template:\n\
                cp /usr/local/lib/agenthandover/launchd/com.agenthandover.worker.plist \\\n\
                  ~/Library/LaunchAgents/\n\
             \n\
             3. Then: agenthandover start worker",
            plist
        );
    }

    // Modern API: bootstrap loads the plist into the user's GUI domain.
    // If the plist is already loaded this fails with "service already
    // loaded" — we don't care, we'll try kickstart next regardless.
    let bootstrap = launchctl_silent(&["bootstrap", &domain, &plist])?;

    // kickstart forces the job to actually run. -k means SIGKILL any
    // existing instance first (so we get a fresh start).
    let kickstart = launchctl_silent(&["kickstart", "-k", &label_in_domain])?;

    std::thread::sleep(std::time::Duration::from_millis(500));

    if is_job_running(WORKER_LABEL) {
        println!("  Worker started.");
        Ok(())
    } else {
        // Worker genuinely didn't come up — NOW surface whatever
        // launchctl had to say about it. Bootstrap stderr takes
        // priority since it fires first (and is usually the more
        // informative error — e.g. "file not found" for a missing
        // plist, or "nothing found to load" for the reverse).
        let detail = if !bootstrap.stderr.is_empty() && !bootstrap.success {
            format!("launchctl bootstrap: {}", bootstrap.stderr)
        } else if !kickstart.stderr.is_empty() && !kickstart.success {
            format!("launchctl kickstart: {}", kickstart.stderr)
        } else {
            "worker didn't appear in launchctl list after bootstrap + kickstart".to_string()
        };
        bail!(
            "worker failed to start: {}\n  \
             Plist: {}\n  \
             Check: ~/Library/Application Support/agenthandover/logs/worker.stderr.log",
            detail,
            plist
        );
    }
}

/// Stop the launchd-managed worker job.
///
/// Uses the modern ``bootout`` API (matching the Swift app) instead
/// of the deprecated ``launchctl unload``. Bootout is quieter on the
/// happy path and has consistent semantics with bootstrap above.
fn stop_worker_launchd() -> Result<()> {
    println!("Stopping worker...");

    // Idempotent: not running → no-op, no error.
    if !is_job_running(WORKER_LABEL) {
        println!("  Worker not running.");
        return Ok(());
    }

    let plist = plist_path(WORKER_LABEL);
    let result = launchctl_silent(&["bootout", &gui_domain(), &plist])?;

    if !result.success && !result.stderr.is_empty() {
        // bootout failed — report it, but don't fail the whole command
        // because the worker process itself may still be exiting
        // cleanly via SIGTERM propagation.
        eprintln!("  launchctl bootout: {}", result.stderr);
    }

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
