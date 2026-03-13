//! CLI commands for searching activity and recalling sessions.
//!
//! Uses trigger-file IPC: writes a query trigger, polls for a result file
//! from the worker, then displays results.

use anyhow::{bail, Result};
use colored::Colorize;
use oc_apprentice_common::status::data_dir;
use std::io::Write;
use std::path::PathBuf;

const SEARCH_TRIGGER_FILE: &str = "search-query-trigger.json";
const SEARCH_RESULT_FILE: &str = "search-query-result.json";
const RECALL_TRIGGER_FILE: &str = "recall-query-trigger.json";
const RECALL_RESULT_FILE: &str = "recall-query-result.json";

/// Write a JSON trigger file atomically (tmp + fsync + rename).
fn write_trigger(filename: &str, payload: &serde_json::Value) -> Result<PathBuf> {
    let state_dir = data_dir();
    std::fs::create_dir_all(&state_dir)?;
    let target = state_dir.join(filename);
    let tmp = state_dir.join(format!(".{}.tmp", filename));

    let json = serde_json::to_string_pretty(payload)?;
    let mut file = std::fs::File::create(&tmp)?;
    file.write_all(json.as_bytes())?;
    file.sync_all()?;
    std::fs::rename(&tmp, &target)?;

    Ok(target)
}

/// Poll for a result file with a timeout, then parse and clean up.
fn poll_result(
    trigger_file: &str,
    result_file: &str,
    timeout_secs: u64,
) -> Result<serde_json::Value> {
    let state_dir = data_dir();
    let result_path = state_dir.join(result_file);

    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(timeout_secs);

    while start.elapsed() < timeout {
        if result_path.exists() {
            let content = std::fs::read_to_string(&result_path)?;
            let parsed: serde_json::Value = serde_json::from_str(&content)?;

            // Clean up trigger and result files
            let _ = std::fs::remove_file(state_dir.join(trigger_file));
            let _ = std::fs::remove_file(&result_path);

            return Ok(parsed);
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }

    // Timeout — clean up trigger
    let _ = std::fs::remove_file(state_dir.join(trigger_file));
    bail!(
        "Timed out waiting for worker response. Is the worker running?\n\
         Check with: openmimic status"
    );
}

/// `openmimic search "query"` — full-text search over activity annotations.
pub fn search(
    query: &str,
    date: Option<&str>,
    app: Option<&str>,
    limit: usize,
) -> Result<()> {
    let mut trigger = serde_json::json!({
        "query": query,
        "limit": limit,
        "requested_at": chrono::Utc::now().to_rfc3339(),
    });

    if let Some(d) = date {
        trigger["date"] = serde_json::json!(d);
    }
    if let Some(a) = app {
        trigger["app"] = serde_json::json!(a);
    }

    write_trigger(SEARCH_TRIGGER_FILE, &trigger)?;
    println!("Searching for: {}", query.bold());

    let parsed = poll_result(SEARCH_TRIGGER_FILE, SEARCH_RESULT_FILE, 15)?;

    if let Some(results) = parsed.get("results").and_then(|r| r.as_array()) {
        if results.is_empty() {
            println!("{} No results found.", "·".dimmed());
            return Ok(());
        }

        println!(
            "\n{:<20}  {:<15}  {}",
            "Time".bold(),
            "App".bold(),
            "Activity".bold(),
        );
        println!("{}", "─".repeat(70));

        for result in results {
            let timestamp = result
                .get("timestamp")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let app_name = result
                .get("app")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let what_doing = result
                .get("what_doing")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let score = result
                .get("relevance_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);

            // Truncate timestamp to time only if it has a T
            let time_part = if let Some(pos) = timestamp.find('T') {
                &timestamp[pos + 1..std::cmp::min(pos + 9, timestamp.len())]
            } else {
                timestamp
            };

            let score_display = format!("{:.0}%", score * 100.0);
            println!(
                "  {:<20}  {:<15}  {} {}",
                time_part,
                app_name,
                what_doing,
                score_display.dimmed(),
            );
        }

        println!(
            "\n{} {} result(s)",
            "✓".green(),
            results.len()
        );
    } else if let Some(error) = parsed.get("error").and_then(|e| e.as_str()) {
        println!("{} {}", "✗".red(), error);
    } else {
        println!("{}", serde_json::to_string_pretty(&parsed)?);
    }

    Ok(())
}

/// `openmimic recall` — reconstruct what you were doing at a given time.
pub fn recall(
    date: Option<&str>,
    app: Option<&str>,
    start_time: Option<&str>,
    end_time: Option<&str>,
) -> Result<()> {
    let mut trigger = serde_json::json!({
        "requested_at": chrono::Utc::now().to_rfc3339(),
    });

    if let Some(d) = date {
        trigger["date"] = serde_json::json!(d);
    }
    if let Some(a) = app {
        trigger["app"] = serde_json::json!(a);
    }
    if let Some(s) = start_time {
        trigger["start_time"] = serde_json::json!(s);
    }
    if let Some(e) = end_time {
        trigger["end_time"] = serde_json::json!(e);
    }

    write_trigger(RECALL_TRIGGER_FILE, &trigger)?;

    let date_display = date.unwrap_or("today");
    println!("Recalling activity for: {}", date_display.bold());

    let parsed = poll_result(RECALL_TRIGGER_FILE, RECALL_RESULT_FILE, 15)?;

    // Display timeline
    if let Some(entries) = parsed.get("entries").and_then(|e| e.as_array()) {
        let total_minutes = parsed
            .get("total_active_minutes")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let apps = parsed
            .get("apps_used")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();

        if entries.is_empty() {
            println!("{} No activity recorded.", "·".dimmed());
            return Ok(());
        }

        println!(
            "\n{}  Active: {} min  |  Apps: {}",
            "Timeline".bold(),
            total_minutes,
            apps,
        );
        println!("{}", "─".repeat(70));

        for entry in entries {
            let timestamp = entry
                .get("timestamp")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let app_name = entry
                .get("app")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let what_doing = entry
                .get("what_doing")
                .and_then(|v| v.as_str())
                .unwrap_or("?");

            let time_part = if let Some(pos) = timestamp.find('T') {
                &timestamp[pos + 1..std::cmp::min(pos + 9, timestamp.len())]
            } else {
                timestamp
            };

            println!(
                "  {} │ {:<15} │ {}",
                time_part.cyan(),
                app_name,
                what_doing,
            );
        }

        println!(
            "\n{} {} entries, {} active minutes",
            "✓".green(),
            entries.len(),
            total_minutes,
        );
    } else if let Some(error) = parsed.get("error").and_then(|e| e.as_str()) {
        println!("{} {}", "✗".red(), error);
    } else {
        println!("{}", serde_json::to_string_pretty(&parsed)?);
    }

    Ok(())
}
