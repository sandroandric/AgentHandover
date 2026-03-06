//! CLI commands for listing, viewing, and managing SOPs.
//!
//! `approve`/`reject`/`retry` use trigger files that the worker picks up.
//! `failed` writes a query trigger and polls for a result file.

use anyhow::{bail, Result};
use colored::Colorize;
use oc_apprentice_common::status::data_dir;
use std::io::Write;
use std::path::PathBuf;

const APPROVE_TRIGGER_FILE: &str = "approve-trigger.json";
const FAILED_TRIGGER_FILE: &str = "failed-query-trigger.json";
const FAILED_RESULT_FILE: &str = "failed-query-result.json";
const RETRY_TRIGGER_FILE: &str = "retry-trigger.json";

fn sops_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    // Default OpenClaw workspace path
    PathBuf::from(home).join(".openclaw/workspace/memory/apprentice/sops")
}

/// Write a JSON trigger file atomically (tmp + fsync + rename), matching the
/// pattern used by `export.rs`.
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

pub fn list() -> Result<()> {
    let dir = sops_dir();
    if !dir.exists() {
        println!("No SOPs directory found at: {}", dir.display());
        println!("SOPs will be generated once enough workflow patterns are detected.");
        return Ok(());
    }

    let mut sop_files: Vec<_> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension().map_or(false, |ext| ext == "md")
                && e.file_name().to_string_lossy().starts_with("sop.")
        })
        .collect();

    if sop_files.is_empty() {
        println!("No SOPs generated yet.");
        println!("SOPs appear once the system detects repeated workflow patterns.");
        return Ok(());
    }

    sop_files.sort_by_key(|e| e.file_name());

    println!("Generated SOPs ({}):", sop_files.len());
    println!("{}", "-".repeat(60));
    for entry in &sop_files {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        // Extract slug from "sop.<slug>.md"
        let slug = name_str
            .strip_prefix("sop.")
            .and_then(|s| s.strip_suffix(".md"))
            .unwrap_or(&name_str);

        // Try to read first heading
        let title = std::fs::read_to_string(entry.path())
            .ok()
            .and_then(|content| {
                content
                    .lines()
                    .find(|l| l.starts_with("# "))
                    .map(|l| l[2..].trim().to_string())
            })
            .unwrap_or_else(|| slug.replace('-', " "));

        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        println!("  {} -- {} ({} bytes)", slug, title, size);
    }

    Ok(())
}

pub fn show(slug: &str) -> Result<()> {
    let file_path = sops_dir().join(format!("sop.{}.md", slug));
    if !file_path.exists() {
        bail!("SOP '{}' not found at: {}", slug, file_path.display());
    }
    let content = std::fs::read_to_string(&file_path)?;
    println!("{}", content);
    Ok(())
}

pub fn dir() -> Result<()> {
    println!("{}", sops_dir().display());
    Ok(())
}

pub fn approve(slug_or_id: &str) -> Result<()> {
    let trigger = serde_json::json!({
        "sop_id": slug_or_id,
        "action": "approve",
        "requested_at": chrono::Utc::now().to_rfc3339(),
    });

    write_trigger(APPROVE_TRIGGER_FILE, &trigger)?;

    println!("{} Approval queued for '{}'", "✓".green(), slug_or_id.bold());
    println!("The worker will export on its next cycle.");

    Ok(())
}

pub fn reject(slug_or_id: &str) -> Result<()> {
    let trigger = serde_json::json!({
        "sop_id": slug_or_id,
        "action": "reject",
        "requested_at": chrono::Utc::now().to_rfc3339(),
    });

    write_trigger(APPROVE_TRIGGER_FILE, &trigger)?;

    println!("{} Rejection queued for '{}'", "✓".green(), slug_or_id.bold());

    Ok(())
}

pub fn failed() -> Result<()> {
    let trigger = serde_json::json!({
        "query": "failed",
        "requested_at": chrono::Utc::now().to_rfc3339(),
    });

    write_trigger(FAILED_TRIGGER_FILE, &trigger)?;

    let state_dir = data_dir();
    let result_path = state_dir.join(FAILED_RESULT_FILE);

    // Poll for the result file with a 10-second timeout
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(10);

    while start.elapsed() < timeout {
        if result_path.exists() {
            // Check that the file was written after our trigger
            let content = std::fs::read_to_string(&result_path)?;
            let parsed: serde_json::Value = serde_json::from_str(&content)?;

            // Clean up trigger and result files
            let _ = std::fs::remove_file(state_dir.join(FAILED_TRIGGER_FILE));
            let _ = std::fs::remove_file(&result_path);

            // Display results
            if let Some(failures) = parsed.get("failures").and_then(|f| f.as_array()) {
                if failures.is_empty() {
                    println!("{} No failed generations.", "✓".green());
                    return Ok(());
                }

                println!("Failed generations ({}):", failures.len());
                println!(
                    "{:<36}  {:<30}  {}",
                    "ID".bold(),
                    "SOP".bold(),
                    "Error".bold()
                );
                println!("{}", "-".repeat(80));

                for failure in failures {
                    let id = failure
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?");
                    let sop = failure
                        .get("sop_slug")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?");
                    let error = failure
                        .get("error")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    println!("  {:<36}  {:<30}  {}", id, sop, error.red());
                }
            } else {
                println!("{}", content);
            }

            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }

    // Timeout — clean up trigger file
    let _ = std::fs::remove_file(state_dir.join(FAILED_TRIGGER_FILE));
    bail!(
        "Timed out waiting for worker response. Is the worker running?\n\
         Check with: openmimic status"
    );
}

pub fn retry(failure_id: &str) -> Result<()> {
    let trigger = serde_json::json!({
        "failure_id": failure_id,
        "requested_at": chrono::Utc::now().to_rfc3339(),
    });

    write_trigger(RETRY_TRIGGER_FILE, &trigger)?;

    println!(
        "{} Retry queued for failure '{}'",
        "✓".green(),
        failure_id.bold()
    );
    println!("The worker will re-attempt on its next cycle.");

    Ok(())
}
