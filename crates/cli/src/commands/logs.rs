use anyhow::{bail, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

fn logs_dir() -> PathBuf {
    agenthandover_common::status::data_dir().join("logs")
}

/// The daemon's current log file in `dir`.
///
/// The daemon writes one file per day, `daemon.YYYY-MM-DD.log`
/// (tracing-appender's rolling builder, 7 kept). Older daemons wrote
/// `daemon.log.YYYY-MM-DD`, and very old ones a plain `daemon.log`. We take
/// the file with the newest date across all three namings, and fall back
/// to `daemon.stderr.log` (where the app and CLI send the daemon's stderr,
/// which holds panics before logging starts) when there is none.
fn newest_daemon_log(dir: &Path) -> Option<PathBuf> {
    let newest = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let date = daemon_log_date(&name)?.to_string();
            Some((date, entry.path()))
        })
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, path)| path);
    newest.or_else(|| {
        let stderr_log = dir.join("daemon.stderr.log");
        stderr_log.is_file().then_some(stderr_log)
    })
}

/// The sortable date part of a daemon log file name ("" for the undated
/// legacy `daemon.log`), or `None` if `name` is not a daemon log.
fn daemon_log_date(name: &str) -> Option<&str> {
    if name == "daemon.log" {
        return Some("");
    }
    if let Some(date) = name.strip_prefix("daemon.log.") {
        return Some(date);
    }
    let date = name.strip_prefix("daemon.")?.strip_suffix(".log")?;
    // Exclude daemon.stderr.log and anything else that is not a date.
    (date.len() == 10 && date.bytes().all(|b| b.is_ascii_digit() || b == b'-')).then_some(date)
}

fn resolve_log_path(service: &str, dir: &Path) -> Result<Option<PathBuf>> {
    match service {
        "daemon" => Ok(newest_daemon_log(dir)),
        "worker" => {
            let path = dir.join("worker.log");
            Ok(path.exists().then_some(path))
        }
        _ => bail!("Unknown service: {}. Use 'daemon' or 'worker'.", service),
    }
}

pub fn run(service: &str, follow: bool, lines: usize) -> Result<()> {
    let dir = logs_dir();
    let Some(log_path) = resolve_log_path(service, &dir)? else {
        println!("No {} log file found in: {}", service, dir.display());
        println!("The service may not have started yet.");
        return Ok(());
    };

    if follow {
        let status = Command::new("tail")
            .args([
                "-f",
                "-n",
                &lines.to_string(),
                &log_path.display().to_string(),
            ])
            .status()?;
        if !status.success() {
            bail!("tail command failed");
        }
    } else {
        let output = Command::new("tail")
            .args(["-n", &lines.to_string(), &log_path.display().to_string()])
            .output()?;
        print!("{}", String::from_utf8_lossy(&output.stdout));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(dir: &Path, name: &str, body: &str) {
        std::fs::write(dir.join(name), body).unwrap();
    }

    #[test]
    fn picks_the_rolling_daemon_log() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "daemon.2026-09-14.log", "today\n");
        let path = resolve_log_path("daemon", dir.path()).unwrap().unwrap();
        assert_eq!(path, dir.path().join("daemon.2026-09-14.log"));
    }

    #[test]
    fn newest_date_wins_across_namings() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "daemon.log", "");
        touch(dir.path(), "daemon.log.2026-09-12", "");
        touch(dir.path(), "daemon.2026-09-13.log", "");
        touch(dir.path(), "daemon.2026-09-14.log", "");
        touch(dir.path(), "daemon.stderr.log", "");
        touch(dir.path(), "worker.log", "");
        let path = newest_daemon_log(dir.path()).unwrap();
        assert_eq!(path, dir.path().join("daemon.2026-09-14.log"));
    }

    #[test]
    fn legacy_suffix_naming_is_found() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "daemon.log.2026-09-12", "");
        let path = newest_daemon_log(dir.path()).unwrap();
        assert_eq!(path, dir.path().join("daemon.log.2026-09-12"));
    }

    #[test]
    fn falls_back_to_stderr_log() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "daemon.stderr.log", "panic\n");
        let path = newest_daemon_log(dir.path()).unwrap();
        assert_eq!(path, dir.path().join("daemon.stderr.log"));
    }

    #[test]
    fn nothing_found() {
        let dir = tempfile::tempdir().unwrap();
        assert!(resolve_log_path("daemon", dir.path()).unwrap().is_none());
        assert!(resolve_log_path("worker", dir.path()).unwrap().is_none());
        assert!(resolve_log_path("nope", dir.path()).is_err());
    }
}
