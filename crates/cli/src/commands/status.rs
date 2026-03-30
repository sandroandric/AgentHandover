use anyhow::Result;
use colored::Colorize;
use agenthandover_common::focus_session::read_focus_signal;
use agenthandover_common::pid;
use agenthandover_common::status;

use crate::display::{format_number, format_relative_time, is_heartbeat_stale};

pub fn run() -> Result<()> {
    println!("{}", "AgentHandover Status".bold());
    println!("{}", "=".repeat(50));
    println!();

    // Daemon status
    print_service_status("Daemon", "daemon-status.json", "daemon");
    println!();

    // Worker status
    print_service_status("Worker", "worker-status.json", "worker");
    println!();

    // Focus session status
    print_focus_status();

    Ok(())
}

fn print_focus_status() {
    let state_dir = status::data_dir();
    match read_focus_signal(&state_dir) {
        Some(signal) if signal.is_recording() => {
            // Calculate elapsed time
            let elapsed = if let Ok(started) = chrono::DateTime::parse_from_rfc3339(&signal.started_at) {
                let duration = chrono::Utc::now().signed_duration_since(started);
                let mins = duration.num_minutes();
                let secs = duration.num_seconds() % 60;
                format!("{}m {}s", mins, secs)
            } else {
                "unknown".to_string()
            };

            println!(
                "  {} Focus: {} \"{}\" ({})",
                "●".red(),
                "Recording".red().bold(),
                signal.title,
                elapsed,
            );
        }
        Some(signal) if signal.is_stopped() => {
            println!(
                "  {} Focus: {} \"{}\" — awaiting worker processing",
                "◉".yellow(),
                "Stopped".yellow(),
                signal.title,
            );
        }
        _ => {
            println!("  {} Focus: {}", "○".dimmed(), "idle".dimmed());
        }
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

            // Chrome extension connection status
            // Check daemon-status.json first (populated when NM runs in-process),
            // then fall back to extension-heartbeat.json (written by separate NM bridge).
            let ext_ts_from_daemon = value.get("last_extension_message").and_then(|v| v.as_str());
            let ext_heartbeat = status::read_extension_heartbeat();

            if let Some(ext_ts) = ext_ts_from_daemon {
                let relative = format_relative_time(ext_ts);
                if is_heartbeat_stale(ext_ts) {
                    // Daemon's in-memory value is stale, but NM bridge file might be fresh
                    if ext_heartbeat.is_some() {
                        println!("    Extension:  {}", "connected (via NM bridge)".green());
                    } else {
                        println!(
                            "    Extension:  {}",
                            format!("last seen {} (stale)", relative).yellow()
                        );
                    }
                } else {
                    println!("    Extension:  {}", format!("connected ({})", relative).green());
                }
            } else if ext_heartbeat.is_some() {
                // No in-memory NM data, but heartbeat file is fresh
                println!("    Extension:  {}", "connected (via NM bridge)".green());
            } else {
                // No NM message from any source
                let uptime = value.get("uptime_seconds").and_then(|v| v.as_u64()).unwrap_or(0);
                if uptime > 600 {
                    println!(
                        "    Extension:  {}",
                        "not connected — load the Chrome extension for full capture".yellow()
                    );
                } else {
                    println!("    Extension:  {}", "waiting for connection...".dimmed());
                }
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
                let vlm_mode = value
                    .get("vlm_mode")
                    .and_then(|v| v.as_str())
                    .unwrap_or("local");
                let vlm_provider = value
                    .get("vlm_provider")
                    .and_then(|v| v.as_str());
                let vlm_str = if vlm {
                    // Show mode/provider info
                    let mode_info = if vlm_mode == "remote" {
                        let provider = vlm_provider.unwrap_or("unknown");
                        format!("remote ({})", provider)
                    } else {
                        "local".to_string()
                    };

                    // Show queue depth alongside availability
                    let pending = value
                        .get("vlm_queue_pending")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let jobs_today = value
                        .get("vlm_jobs_today")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let dropped = value
                        .get("vlm_dropped_today")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    if pending > 0 || jobs_today > 0 {
                        let mut parts = vec![format!("{} pending", pending)];
                        if jobs_today > 0 {
                            parts.push(format!("{} today", jobs_today));
                        }
                        if dropped > 0 {
                            parts.push(format!("{} dropped", dropped));
                        }
                        format!("{} ({})", mode_info, parts.join(", ")).into()
                    } else {
                        format!("{}", mode_info).green()
                    }
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
                println!("    SOP mining: {}", inducer_str);
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

