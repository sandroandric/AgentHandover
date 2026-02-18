use std::io;
use std::path::PathBuf;

/// Standard location for PID files (delegates to shared `data_dir()`).
pub fn pid_dir() -> PathBuf {
    crate::status::data_dir()
}

/// Write a PID file. Returns the path written.
pub fn write_pid_file(name: &str) -> io::Result<PathBuf> {
    let dir = pid_dir();
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.pid", name));
    std::fs::write(&path, std::process::id().to_string())?;
    Ok(path)
}

/// Remove a PID file on clean shutdown.
pub fn remove_pid_file(name: &str) {
    let path = pid_dir().join(format!("{}.pid", name));
    let _ = std::fs::remove_file(&path);
}

/// Check if a PID file exists and the process is still running.
/// Returns `Some(pid)` if process is alive, `None` if stale or missing.
pub fn check_pid_file(name: &str) -> Option<u32> {
    let path = pid_dir().join(format!("{}.pid", name));
    let content = std::fs::read_to_string(&path).ok()?;
    let pid: u32 = content.trim().parse().ok()?;

    // Check if process is still running
    if is_process_running(pid) {
        Some(pid)
    } else {
        // Stale PID file -- clean it up
        let _ = std::fs::remove_file(&path);
        None
    }
}

/// Check if a process with the given PID is running.
fn is_process_running(pid: u32) -> bool {
    // kill(pid, 0) checks if process exists without sending a signal
    unsafe { libc::kill(pid as i32, 0) == 0 }
}
