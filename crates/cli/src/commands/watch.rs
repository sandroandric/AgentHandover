use anyhow::Result;
use colored::Colorize;
use agenthandover_common::status;
use std::io::Write;

use crate::display::{format_number, format_uptime, is_heartbeat_stale};

/// Live-updating status display (like `watch` or `tail -f`).
///
/// Polls daemon-status.json and worker-status.json every 2 seconds
/// and prints a compact one-screen summary.  Press Ctrl-C to exit.
pub fn run() -> Result<()> {
    println!("{}", "AgentHandover Watch — press Ctrl-C to stop".bold());
    println!();

    loop {
        // Clear screen and move cursor to top-left
        print!("\x1B[2J\x1B[H");

        println!("{}", "AgentHandover Watch".bold());
        println!("{}", "═".repeat(50));

        // Read daemon status
        match status::read_status_file::<serde_json::Value>("daemon-status.json") {
            Ok(d) => {
                let events = d.get("events_today").and_then(|v| v.as_u64()).unwrap_or(0);
                let perms = d.get("permissions_ok").and_then(|v| v.as_bool()).unwrap_or(false);
                let heartbeat = d.get("heartbeat").and_then(|v| v.as_str()).unwrap_or("");
                let uptime = d.get("uptime_seconds").and_then(|v| v.as_u64()).unwrap_or(0);

                let hb_fresh = !heartbeat.is_empty() && !is_heartbeat_stale(heartbeat);
                let status_str = if hb_fresh {
                    "● running".green()
                } else {
                    "● stale".yellow()
                };
                println!("\n  {} {}", "Daemon".bold(), status_str);
                println!("    Events captured:  {}", format_number(events).bold());
                println!("    Uptime:           {}", format_uptime(uptime));
                println!(
                    "    Permissions:      {}",
                    if perms { "OK".green() } else { "MISSING".red() }
                );

                // Extension status
                if let Some(ext_ts) = d.get("last_extension_message").and_then(|v| v.as_str()) {
                    if is_heartbeat_stale(ext_ts) {
                        println!("    Extension:        {}", "stale".yellow());
                    } else {
                        println!("    Extension:        {}", "connected".green());
                    }
                } else if uptime > 600 {
                    println!(
                        "    Extension:        {}",
                        "NOT CONNECTED".red()
                    );
                } else {
                    println!("    Extension:        {}", "waiting...".dimmed());
                }
            }
            Err(_) => {
                println!("\n  {} {}", "Daemon".bold(), "○ not running".dimmed());
            }
        }

        // Read worker status
        match status::read_status_file::<serde_json::Value>("worker-status.json") {
            Ok(w) => {
                let events_proc = w
                    .get("events_processed_today")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let sops = w.get("sops_generated").and_then(|v| v.as_u64()).unwrap_or(0);
                let heartbeat = w.get("heartbeat").and_then(|v| v.as_str()).unwrap_or("");
                let pipeline_ms = w
                    .get("last_pipeline_duration_ms")
                    .and_then(|v| v.as_u64());
                let inducer = w
                    .get("sop_inducer_available")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let errors = w.get("consecutive_errors").and_then(|v| v.as_u64()).unwrap_or(0);

                let hb_fresh = !heartbeat.is_empty() && !is_heartbeat_stale(heartbeat);
                let status_str = if hb_fresh {
                    "● running".green()
                } else {
                    "● stale".yellow()
                };
                println!("\n  {} {}", "Worker".bold(), status_str);
                println!("    Events processed: {}", format_number(events_proc).bold());
                println!(
                    "    SOPs generated:   {}",
                    if sops > 0 {
                        format!("{}", sops).green().bold()
                    } else {
                        "0 (repeat workflows 2+ times)".dimmed().bold()
                    }
                );
                if let Some(ms) = pipeline_ms {
                    println!("    Last pipeline:    {}ms", ms);
                }
                println!(
                    "    SOP mining:       {}",
                    if inducer { "ready".green() } else { "disabled".red() }
                );
                if errors > 0 {
                    println!("    Errors:           {}", format!("{}", errors).red());
                }
            }
            Err(_) => {
                println!("\n  {} {}", "Worker".bold(), "○ not running".dimmed());
            }
        }

        println!("\n{}", "─".repeat(50).dimmed());
        println!("{}", "  Refreshing every 2s · Ctrl-C to exit".dimmed());

        std::io::stdout().flush().ok();
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
}

