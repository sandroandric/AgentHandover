use anyhow::Result;
use chrono::{DateTime, Utc};
use colored::Colorize;
use oc_apprentice_common::pid;
use oc_apprentice_common::status;

pub fn run() -> Result<()> {
    println!("{}", "OpenMimic Status".bold());
    println!("{}", "=".repeat(50));
    println!();

    // Daemon status
    print_service_status("Daemon", "daemon-status.json", "daemon");
    println!();

    // Worker status
    print_service_status("Worker", "worker-status.json", "worker");

    Ok(())
}

/// Format a relative time string like "2s ago", "5m ago", "2h ago".
fn format_relative_time(timestamp: &str) -> String {
    match timestamp.parse::<DateTime<Utc>>() {
        Ok(dt) => {
            let now = Utc::now();
            let diff = now.signed_duration_since(dt);
            let secs = diff.num_seconds();
            if secs < 0 {
                "just now".to_string()
            } else if secs < 60 {
                format!("{}s ago", secs)
            } else if secs < 3600 {
                format!("{}m ago", secs / 60)
            } else if secs < 86400 {
                format!("{}h ago", secs / 3600)
            } else {
                format!("{}d ago", secs / 86400)
            }
        }
        Err(_) => timestamp.to_string(),
    }
}

/// Check if a heartbeat is stale (older than 2 minutes).
fn is_heartbeat_stale(timestamp: &str) -> bool {
    match timestamp.parse::<DateTime<Utc>>() {
        Ok(dt) => {
            let now = Utc::now();
            now.signed_duration_since(dt).num_seconds() > 120
        }
        Err(_) => false,
    }
}

fn print_service_status(name: &str, status_file: &str, pid_name: &str) {
    let pid_alive = pid::check_pid_file(pid_name);

    match status::read_status_file::<serde_json::Value>(status_file) {
        Ok(value) => {
            let status_icon = if pid_alive.is_some() {
                "●".green()
            } else {
                "●".red()
            };

            println!(
                "  {} {} {}",
                status_icon,
                name.bold(),
                if pid_alive.is_some() {
                    "(running)".green()
                } else {
                    "(not responding)".red()
                }
            );

            if let Some(pid) = value.get("pid").and_then(|v| v.as_u64()) {
                println!("    PID:        {}", pid);
            }
            if let Some(version) = value.get("version").and_then(|v| v.as_str()) {
                println!("    Version:    {}", version);
            }

            // Heartbeat with relative time and staleness check
            if let Some(heartbeat) = value.get("heartbeat").and_then(|v| v.as_str()) {
                let relative = format_relative_time(heartbeat);
                if is_heartbeat_stale(heartbeat) {
                    println!(
                        "    Heartbeat:  {}",
                        format!("{} (stale)", relative).yellow()
                    );
                } else {
                    println!("    Heartbeat:  {}", relative);
                }
            }

            // Daemon-specific fields
            if let Some(events) = value.get("events_today").and_then(|v| v.as_u64()) {
                println!("    Events:     {} captured today", format_number(events));
            }
            if let Some(perms) = value.get("permissions_ok").and_then(|v| v.as_bool()) {
                let perms_str = if perms {
                    "OK".green()
                } else {
                    "MISSING".red()
                };
                println!("    Perms:      {}", perms_str);
            }

            // Worker-specific fields
            if let Some(events) = value
                .get("events_processed_today")
                .and_then(|v| v.as_u64())
            {
                println!("    Events:     {} processed today", format_number(events));
            }
            if let Some(sops) = value.get("sops_generated").and_then(|v| v.as_u64()) {
                println!("    SOPs:       {} generated", sops);
            }
            if let Some(duration) = value
                .get("last_pipeline_duration_ms")
                .and_then(|v| v.as_u64())
            {
                println!("    Pipeline:   last run {}ms", duration);
            }
            if let Some(vlm) = value.get("vlm_available").and_then(|v| v.as_bool()) {
                let vlm_str = if vlm {
                    "available".green()
                } else {
                    "not installed".dimmed()
                };
                println!("    VLM:        {}", vlm_str);
            }
            if let Some(inducer) = value
                .get("sop_inducer_available")
                .and_then(|v| v.as_bool())
            {
                let inducer_str = if inducer {
                    "ready".green()
                } else {
                    "not installed".dimmed()
                };
                println!("    Inducer:    {}", inducer_str);
            }

            if let Some(errors) = value.get("consecutive_errors").and_then(|v| v.as_u64()) {
                if errors > 0 {
                    println!("    Errors:     {}", format!("{}", errors).red());
                }
            }
        }
        Err(_) => {
            let status_icon = "○".dimmed();
            println!(
                "  {} {} {}",
                status_icon,
                name.bold(),
                "(not running)".dimmed()
            );
            if let Some(pid) = pid_alive {
                println!("    PID {} is alive but no status file found", pid);
            }
        }
    }
}

/// Format a number with comma separators (e.g. 1247 -> "1,247").
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}
