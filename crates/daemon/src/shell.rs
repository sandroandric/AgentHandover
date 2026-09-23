//! Ingest of the opt-in shell hook's log (`agenthandover setup --shell`).
//!
//! The zsh hook appends `{"ts": <epoch seconds>, "cwd": ..., "cmd": ...}`
//! lines to `<data dir>/shell.jsonl`.  Each [`take_commands`] pass moves
//! the log aside to `shell.jsonl.ingest` and returns (and deletes) what the
//! previous pass moved aside, so an append that was in flight during the
//! rename has a whole pass to land.  Commands become `ShellCommand` events;
//! the storage writer's redactor runs on them like on every event.

use agenthandover_common::event::{Event, EventKind};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tracing::warn;
use uuid::Uuid;

/// `<data dir>/shell.jsonl`, written by the zsh hook.
pub fn shell_log_path(data_dir: &Path) -> PathBuf {
    data_dir.join("shell.jsonl")
}

fn ingest_path(log: &Path) -> PathBuf {
    let mut name = log.as_os_str().to_owned();
    name.push(".ingest");
    PathBuf::from(name)
}

/// One command the hook recorded.
#[derive(Debug, Clone, PartialEq)]
pub struct ShellEntry {
    pub timestamp: DateTime<Utc>,
    pub cwd: String,
    pub command: String,
}

#[derive(Deserialize)]
struct RawEntry {
    ts: f64,
    cwd: String,
    cmd: String,
}

/// Parse one log line; None for a malformed line.
pub fn parse_line(line: &str) -> Option<ShellEntry> {
    let raw: RawEntry = serde_json::from_str(line.trim()).ok()?;
    if !raw.ts.is_finite() || raw.ts <= 0.0 || raw.cmd.trim().is_empty() {
        return None;
    }
    let secs = raw.ts.trunc() as i64;
    let nanos = ((raw.ts - raw.ts.trunc()) * 1e9).round().min(999_999_999.0) as u32;
    Some(ShellEntry {
        timestamp: DateTime::from_timestamp(secs, nanos)?,
        cwd: raw.cwd,
        command: raw.cmd,
    })
}

/// Return the commands moved aside by the previous pass (deleting that
/// file), then move the current log aside for the next pass.
pub fn take_commands(log: &Path) -> Vec<ShellEntry> {
    let staged = ingest_path(log);
    let mut entries = Vec::new();
    match std::fs::read_to_string(&staged) {
        Ok(text) => {
            let mut malformed = 0usize;
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                match parse_line(line) {
                    Some(entry) => entries.push(entry),
                    None => malformed += 1,
                }
            }
            if malformed > 0 {
                warn!(malformed, "Shell log: skipped malformed lines");
            }
            if let Err(e) = std::fs::remove_file(&staged) {
                warn!(error = %e, "Shell log: could not remove the ingested file");
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => warn!(error = %e, "Shell log: could not read the staged file"),
    }
    if log.exists() {
        if let Err(e) = std::fs::rename(log, &staged) {
            warn!(error = %e, "Shell log: could not stage the log");
        }
    }
    entries
}

/// The `ShellCommand` event of `entry`.
pub fn to_event(entry: ShellEntry) -> Event {
    Event {
        id: Uuid::new_v4(),
        timestamp: entry.timestamp,
        kind: EventKind::ShellCommand {
            cwd: entry.cwd,
            command: entry.command,
        },
        window: None,
        display_topology: vec![],
        primary_display_id: String::new(),
        cursor_global_px: None,
        ui_scale: None,
        artifact_ids: vec![],
        metadata: serde_json::json!({"source": "shell_hook"}),
        display_ids_spanned: None,
        focused_element: None,
    }
}

/// How often the log is staged and ingested.
const INGEST_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);

/// Ingest the shell log until shutdown.  While capture is paused and no
/// focus recording runs, ingested commands are dropped, as the observer
/// drops everything else it would capture.
pub async fn run_shell_ingest(
    data_dir: PathBuf,
    tx: tokio::sync::mpsc::Sender<crate::observer::event_loop::ObserverMessage>,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
    capture_paused: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    use crate::observer::event_loop::ObserverMessage;
    let log = shell_log_path(&data_dir);
    let mut interval = tokio::time::interval(INGEST_INTERVAL);
    loop {
        tokio::select! {
            _ = interval.tick() => {
                let path = log.clone();
                let Ok(entries) = tokio::task::spawn_blocking(move || take_commands(&path)).await
                else { continue };
                if entries.is_empty() {
                    continue;
                }
                let recording = agenthandover_common::focus_session::read_focus_signal(&data_dir)
                    .is_some_and(|s| s.is_recording());
                if capture_paused.load(std::sync::atomic::Ordering::Acquire) && !recording {
                    tracing::debug!(count = entries.len(), "Shell log: capture paused, commands dropped");
                    continue;
                }
                for entry in entries {
                    if tx.send(ObserverMessage::Event { event: to_event(entry) }).await.is_err() {
                        return;
                    }
                }
            }
            _ = shutdown_rx.changed() => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_hooks_line_format() {
        let entry = parse_line(
            r#"{"ts":1790000000.250000,"cwd":"/Users/me/my \"repo\"","cmd":"git commit -m \"x\"\necho y"}"#,
        )
        .unwrap();
        assert_eq!(entry.timestamp.to_rfc3339(), "2026-09-21T14:13:20.250+00:00");
        assert_eq!(entry.cwd, r#"/Users/me/my "repo""#);
        assert_eq!(entry.command, "git commit -m \"x\"\necho y");
    }

    #[test]
    fn rejects_malformed_lines() {
        assert!(parse_line("not json").is_none());
        assert!(parse_line(r#"{"ts":0,"cwd":"/","cmd":"ls"}"#).is_none());
        assert!(parse_line(r#"{"ts":1790000000,"cwd":"/","cmd":"  "}"#).is_none());
        assert!(parse_line(r#"{"cwd":"/","cmd":"ls"}"#).is_none());
    }

    #[test]
    fn two_passes_stage_then_ingest_and_nothing_is_read_twice() {
        let dir = tempfile::tempdir().unwrap();
        let log = shell_log_path(dir.path());
        std::fs::write(
            &log,
            "{\"ts\":1790000000,\"cwd\":\"/a\",\"cmd\":\"make\"}\ngarbage\n",
        )
        .unwrap();

        assert!(take_commands(&log).is_empty(), "the first pass only stages");
        assert!(!log.exists());

        // An append after the rename starts a new log.
        std::fs::write(&log, "{\"ts\":1790000005,\"cwd\":\"/a\",\"cmd\":\"make test\"}\n").unwrap();

        let second = take_commands(&log);
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].command, "make");

        let third = take_commands(&log);
        assert_eq!(third.len(), 1);
        assert_eq!(third[0].command, "make test");
        assert!(take_commands(&log).is_empty());
        assert!(!ingest_path(&log).exists());
    }

    #[test]
    fn event_carries_the_command_at_its_own_time() {
        let entry = parse_line(r#"{"ts":1790000000,"cwd":"/a","cmd":"ls -la"}"#).unwrap();
        let event = to_event(entry.clone());
        assert_eq!(event.timestamp, entry.timestamp);
        assert_eq!(
            event.kind,
            EventKind::ShellCommand { cwd: "/a".into(), command: "ls -la".into() }
        );
        let kind_json = serde_json::to_value(&event.kind).unwrap();
        assert_eq!(
            kind_json,
            serde_json::json!({"type": "ShellCommand", "cwd": "/a", "command": "ls -la"})
        );
    }
}
