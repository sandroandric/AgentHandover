use anyhow::Result;
use colored::Colorize;
use agenthandover_common::focus_session::read_focus_signal;
use agenthandover_common::pid;
use agenthandover_common::status;

use crate::display::{format_number, format_relative_time, is_heartbeat_stale};

/// `agenthandover status --compute`: what capture and the models cost in
/// the last hour, from the `compute` sections of daemon-status.json and
/// worker-status.json.
pub fn run_compute() -> Result<()> {
    let daemon = status::read_status_file::<serde_json::Value>("daemon-status.json").ok();
    let worker = status::read_status_file::<serde_json::Value>("worker-status.json").ok();
    println!("{}", "AgentHandover compute, last hour".bold());
    println!("{}", "=".repeat(50));
    print!("{}", format_compute(daemon.as_ref(), worker.as_ref()));
    Ok(())
}

fn u64_at(value: &serde_json::Value, key: &str) -> u64 {
    value.get(key).and_then(|v| v.as_u64()).unwrap_or(0)
}

fn f64_at(value: &serde_json::Value, key: &str) -> f64 {
    value.get(key).and_then(|v| v.as_f64()).unwrap_or(0.0)
}

fn seconds(ms: u64) -> String {
    format!("{:.1} s", ms as f64 / 1000.0)
}

/// `"a 3, b 1"` from a JSON object of counts, largest first.
fn counts(map: Option<&serde_json::Value>) -> String {
    let mut items: Vec<(String, u64)> = map
        .and_then(|m| m.as_object())
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0))).collect())
        .unwrap_or_default();
    if items.is_empty() {
        return "none".to_string();
    }
    items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    items
        .iter()
        .map(|(k, n)| format!("{k} {n}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// One line on "Upgrade everything" (the worker's re-embed of its text
/// vectors with EmbeddingGemma 2), from worker-status.json's
/// `embedding_upgrade`; None when the upgrade was never asked for.
pub fn embedding_upgrade_line(worker: &serde_json::Value) -> Option<String> {
    let upgrade = worker.get("embedding_upgrade")?;
    let state = upgrade.get("state").and_then(|v| v.as_str())?;
    let done = u64_at(upgrade, "done");
    let total = u64_at(upgrade, "total");
    let dropped = u64_at(upgrade, "dropped");
    let error = upgrade.get("error").and_then(|v| v.as_str());
    let progress = format!("{done} / {total} vectors");
    Some(match (state, error) {
        ("done", _) => format!(
            "done: text search uses EmbeddingGemma 2 ({total} vectors{})",
            if dropped > 0 { format!(", {dropped} dropped") } else { String::new() }
        ),
        ("failed", error) => format!(
            "failed: {} (text search stays on its current model)",
            error.unwrap_or("unknown error")
        ),
        ("running", None) => format!("re-embedding with EmbeddingGemma 2: {progress}"),
        ("running", Some(error)) => {
            format!("re-embedding with EmbeddingGemma 2: {progress}, waiting: {error}")
        }
        (_, None) => "waiting to start".to_string(),
        (_, Some(error)) => format!("waiting to start: {error}"),
    })
}

/// The last-hour report for `status --compute` (plain text, no colour).
fn format_compute(daemon: Option<&serde_json::Value>, worker: Option<&serde_json::Value>) -> String {
    let mut out = String::new();
    out.push_str("\n  Capture (daemon)\n");
    match daemon.map(|d| d.get("compute").and_then(|c| c.get("last_hour"))) {
        None => out.push_str("    not running (no daemon-status.json)\n"),
        Some(None) => out.push_str("    no compute counters (the daemon predates them)\n"),
        Some(Some(hour)) => {
            out.push_str(&format!(
                "    Frames:     {} captured, {} kept\n",
                u64_at(hour, "frames_captured"),
                u64_at(hour, "frames_kept"),
            ));
            out.push_str(&format!(
                "    Gated:      {}\n",
                counts(hour.get("frames_gated_by_reason"))
            ));
            out.push_str(&format!("    Capture:    {}\n", seconds(u64_at(hour, "capture_ms"))));
            out.push_str(&format!("    OCR:        {}\n", seconds(u64_at(hour, "ocr_ms"))));
        }
    }

    out.push_str("\n  Models (worker)\n");
    match worker.map(|w| (w, w.get("compute").and_then(|c| c.get("last_hour")))) {
        None => out.push_str("    not running (no worker-status.json)\n"),
        Some((_, None)) => out.push_str("    no compute counters (the worker predates them)\n"),
        Some((worker, Some(hour))) => {
            let by_type = hour.get("model_seconds_by_call_type").and_then(|v| v.as_object());
            let calls = hour.get("model_calls_by_call_type");
            let total_calls: u64 = calls
                .and_then(|c| c.as_object())
                .map(|c| c.values().filter_map(|v| v.as_u64()).sum())
                .unwrap_or(0);
            out.push_str(&format!(
                "    Model time: {:.1} s over {} calls\n",
                f64_at(hour, "model_seconds"),
                total_calls,
            ));
            let mut types: Vec<(&String, f64)> = by_type
                .map(|m| m.iter().map(|(k, v)| (k, v.as_f64().unwrap_or(0.0))).collect())
                .unwrap_or_default();
            types.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
            for (call_type, secs) in types {
                let n = calls.map(|c| u64_at(c, call_type)).unwrap_or(0);
                out.push_str(&format!(
                    "      {call_type:<12} {secs:>8.1} s  ({n} call{})\n",
                    if n == 1 { "" } else { "s" }
                ));
            }
            out.push_str(&format!(
                "    Annotation: {} retries, {} deferred\n",
                u64_at(hour, "annotation_retries"),
                u64_at(hour, "annotation_deferred"),
            ));
            out.push_str(&format!(
                "    Frame wall: {}\n",
                counts(hour.get("annotation_wall_ms_histogram"))
            ));
            if let Some(budget) = worker.get("annotation_budget") {
                out.push_str(&format!(
                    "    Budget:     {} / {} frames, {:.1} / {:.0} min today\n",
                    u64_at(budget, "frames"),
                    u64_at(budget, "max_frames"),
                    f64_at(budget, "compute_minutes"),
                    f64_at(budget, "max_compute_minutes"),
                ));
            }
            if let Some(reason) = worker.get("annotation_paused_reason").and_then(|v| v.as_str()) {
                out.push_str(&format!("    Paused:     {reason}\n"));
            }
            if let Some(budget) = worker.get("reasoning_budget") {
                let paused = worker
                    .get("reasoning_paused_reason")
                    .and_then(|v| v.as_str())
                    .map(|reason| format!(" ({reason})"))
                    .unwrap_or_default();
                out.push_str(&format!(
                    "    Reasoning:  {} / {} calls, {:.1} / {:.0} min today, {} skipped{paused}\n",
                    u64_at(budget, "calls"),
                    u64_at(budget, "max_calls"),
                    f64_at(budget, "compute_minutes"),
                    f64_at(budget, "max_compute_minutes"),
                    u64_at(budget, "skipped"),
                ));
            }
        }
    }
    out
}

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
            if value.get("skills_enabled").and_then(|v| v.as_bool()) == Some(false) {
                println!(
                    "    Mode:       {} (no model but Memory's; agenthandover skills learn on)",
                    "Memory only".cyan()
                );
            } else if let Some(vlm) = value.get("vlm_available").and_then(|v| v.as_bool()) {
                let vlm_str = if vlm {
                    "local".green()
                } else {
                    "not available".dimmed()
                };
                println!("    VLM:        {}", vlm_str);
            }
            if let Some(line) = embedding_upgrade_line(&value) {
                println!("    Upgrade:    {line}");
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

#[cfg(test)]
mod tests {
    use super::{embedding_upgrade_line, format_compute};
    use serde_json::json;

    #[test]
    fn the_embedding_upgrade_is_one_line_per_state() {
        let line = |upgrade: serde_json::Value| {
            embedding_upgrade_line(&json!({"pid": 1, "embedding_upgrade": upgrade}))
        };
        assert_eq!(embedding_upgrade_line(&json!({"pid": 1})), None);
        assert_eq!(
            line(json!({"state": "pending", "done": 0, "total": 9, "dropped": 0, "error": null})),
            Some("waiting to start".to_string())
        );
        assert_eq!(
            line(json!({"state": "running", "done": 120, "total": 400, "dropped": 0, "error": null})),
            Some("re-embedding with EmbeddingGemma 2: 120 / 400 vectors".to_string())
        );
        assert_eq!(
            line(json!({"state": "running", "done": 3, "total": 8, "dropped": 0,
                        "error": "Ollama failed: not reachable"})),
            Some(
                "re-embedding with EmbeddingGemma 2: 3 / 8 vectors, waiting: Ollama failed: \
                 not reachable"
                    .to_string()
            )
        );
        assert_eq!(
            line(json!({"state": "failed", "done": 0, "total": 8, "dropped": 0,
                        "error": "embeddinggemma-2:440m is not downloaded: download it \
                                  (ollama pull embeddinggemma-2:440m)"})),
            Some(
                "failed: embeddinggemma-2:440m is not downloaded: download it (ollama pull \
                 embeddinggemma-2:440m) (text search stays on its current model)"
                    .to_string()
            )
        );
        assert_eq!(
            line(json!({"state": "done", "done": 398, "total": 398, "dropped": 2, "error": null})),
            Some("done: text search uses EmbeddingGemma 2 (398 vectors, 2 dropped)".to_string())
        );
    }

    #[test]
    fn compute_report_reads_the_last_hour_of_both_files() {
        let daemon = json!({"compute": {
            "frames_captured": 900, "frames_kept": 300,
            "frames_gated_by_reason": {"unchanged": 500},
            "capture_ms": 1, "ocr_ms": 1,
            "last_hour": {
                "frames_captured": 120, "frames_kept": 40,
                "frames_gated_by_reason": {"unchanged": 60, "same_text": 10, "rate_limited": 10},
                "capture_ms": 3200, "ocr_ms": 12400
            }
        }});
        let worker = json!({
            "annotation_budget": {"frames": 12, "max_frames": 1500,
                                  "compute_minutes": 4.07, "max_compute_minutes": 60.0},
            "annotation_paused_reason": "annotation_budget_exhausted",
            "reasoning_budget": {"calls": 300, "max_calls": 300, "compute_minutes": 41.26,
                                 "max_compute_minutes": 90.0, "skipped": 7},
            "reasoning_paused_reason": "reasoning_budget_exhausted",
            "compute": {"model_seconds": 9999.0, "last_hour": {
                "model_seconds": 412.3,
                "model_seconds_by_call_type": {"sop": 112.2, "annotation": 300.1},
                "model_calls_by_call_type": {"sop": 1, "annotation": 38},
                "annotation_retries": 3, "annotation_deferred": 0,
                "annotation_wall_ms_histogram": {"le_1000": 20, "le_500": 10}
            }}
        });
        let text = format_compute(Some(&daemon), Some(&worker));
        assert!(text.contains("Frames:     120 captured, 40 kept"), "{text}");
        assert!(text.contains("Gated:      unchanged 60, rate_limited 10, same_text 10"), "{text}");
        assert!(text.contains("Capture:    3.2 s"), "{text}");
        assert!(text.contains("OCR:        12.4 s"), "{text}");
        assert!(text.contains("Model time: 412.3 s over 39 calls"), "{text}");
        let annotation = text.find("annotation      300.1 s  (38 calls)").expect(&text);
        let sop = text.find("sop             112.2 s  (1 call)").expect(&text);
        assert!(annotation < sop, "largest first: {text}");
        assert!(text.contains("Annotation: 3 retries, 0 deferred"), "{text}");
        assert!(text.contains("Frame wall: le_1000 20, le_500 10"), "{text}");
        assert!(text.contains("Budget:     12 / 1500 frames, 4.1 / 60 min today"), "{text}");
        assert!(text.contains("Paused:     annotation_budget_exhausted"), "{text}");
        assert!(
            text.contains(
                "Reasoning:  300 / 300 calls, 41.3 / 90 min today, 7 skipped \
                 (reasoning_budget_exhausted)"
            ),
            "{text}"
        );
        // Since-start totals are not the last hour.
        assert!(!text.contains("9999"), "{text}");
    }

    #[test]
    fn compute_report_names_what_is_missing() {
        let text = format_compute(None, Some(&json!({"pid": 1})));
        assert!(text.contains("not running (no daemon-status.json)"), "{text}");
        assert!(text.contains("no compute counters (the worker predates them)"), "{text}");
    }
}

