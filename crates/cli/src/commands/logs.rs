use anyhow::{bail, Result};
use std::process::Command;

fn logs_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    if cfg!(target_os = "macos") {
        std::path::PathBuf::from(home).join("Library/Application Support/agenthandover/logs")
    } else {
        std::path::PathBuf::from(home).join(".local/share/agenthandover/logs")
    }
}

pub fn run(service: &str, follow: bool, lines: usize) -> Result<()> {
    let log_file = match service {
        "daemon" => "daemon.log",
        "worker" => "worker.log",
        _ => bail!("Unknown service: {}. Use 'daemon' or 'worker'.", service),
    };

    let log_path = logs_dir().join(log_file);
    if !log_path.exists() {
        println!("No log file found at: {}", log_path.display());
        println!("The service may not have started yet.");
        return Ok(());
    }

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
