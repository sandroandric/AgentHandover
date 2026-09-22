//! `agenthandover logs daemon` prints the file the daemon actually writes.
//!
//! Runs the built CLI with HOME pointed at a temporary directory, so it
//! never reads the real data dir.

use std::process::Command;

fn logs_dir(home: &std::path::Path) -> std::path::PathBuf {
    if cfg!(target_os = "macos") {
        home.join("Library/Application Support/agenthandover/logs")
    } else {
        home.join(".local/share/agenthandover/logs")
    }
}

fn run_logs(home: &std::path::Path, service: &str) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_agenthandover"))
        .args(["logs", service, "--lines", "5"])
        .env("HOME", home)
        .output()
        .expect("run agenthandover logs");
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn prints_the_rolling_daemon_log() {
    let home = tempfile::tempdir().unwrap();
    let dir = logs_dir(home.path());
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("daemon.2026-09-13.log"), "yesterday\n").unwrap();
    std::fs::write(dir.join("daemon.2026-09-14.log"), "capture loop started\n").unwrap();

    let stdout = run_logs(home.path(), "daemon");
    assert!(stdout.contains("capture loop started"), "stdout: {stdout}");
    assert!(!stdout.contains("yesterday"), "stdout: {stdout}");
}

#[test]
fn reports_when_there_is_no_daemon_log() {
    let home = tempfile::tempdir().unwrap();
    let stdout = run_logs(home.path(), "daemon");
    assert!(stdout.contains("No daemon log file found"), "stdout: {stdout}");
}
