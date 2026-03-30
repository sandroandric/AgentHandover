//! CLI commands for listing, viewing, and managing SOPs.
//!
//! `approve`/`reject`/`retry` use trigger files that the worker picks up.
//! `failed` writes a query trigger and polls for a result file.

use anyhow::{bail, Result};
use colored::Colorize;
use agenthandover_common::status::data_dir;
use std::io::Write;
use std::path::PathBuf;

const APPROVE_TRIGGER_FILE: &str = "approve-trigger.json";
const FAILED_TRIGGER_FILE: &str = "failed-query-trigger.json";
const FAILED_RESULT_FILE: &str = "failed-query-result.json";
const RETRY_TRIGGER_FILE: &str = "retry-trigger.json";
const PROMOTE_TRIGGER_FILE: &str = "lifecycle-promote-trigger.json";

fn sops_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    // Default OpenClaw workspace path
    PathBuf::from(home).join(".openclaw/workspace/memory/apprentice/sops")
}

fn sops_index_path() -> PathBuf {
    data_dir().join("sops-index.json")
}

/// Read the worker's sops-index.json and return the parsed JSON value.
fn read_sops_index() -> Option<serde_json::Value> {
    let path = sops_index_path();
    let content = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Return SOP entries from sops-index.json that have the given status.
fn entries_with_status(status: &str) -> Vec<serde_json::Value> {
    let index = match read_sops_index() {
        Some(idx) => idx,
        None => return Vec::new(),
    };
    let sops = match index.get("sops").and_then(|s| s.as_array()) {
        Some(arr) => arr,
        None => return Vec::new(),
    };
    sops.iter()
        .filter(|s| s.get("status").and_then(|v| v.as_str()) == Some(status))
        .cloned()
        .collect()
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

    // Build a set of approved/agent_ready slugs from sops-index.json.
    // Only SOPs tracked in the index are shown — legacy passive files on disk
    // that predate the index are hidden.
    let index = read_sops_index();
    let index_entries: Vec<serde_json::Value> = index
        .as_ref()
        .and_then(|idx| idx.get("sops"))
        .and_then(|s| s.as_array())
        .cloned()
        .unwrap_or_default();

    // Approved/agent_ready SOPs (shown as full skills)
    let approved: Vec<&serde_json::Value> = index_entries
        .iter()
        .filter(|s| {
            let status = s.get("status").and_then(|v| v.as_str()).unwrap_or("");
            status == "approved" || status == "agent_ready"
        })
        .collect();

    // Draft SOPs (shown separately)
    let drafts: Vec<&serde_json::Value> = index_entries
        .iter()
        .filter(|s| s.get("status").and_then(|v| v.as_str()) == Some("draft"))
        .collect();

    if approved.is_empty() && drafts.is_empty() {
        println!("No skills yet.");
        println!("Record a Focus Session to teach AgentHandover a workflow.");
        return Ok(());
    }

    // Show approved skills
    if !approved.is_empty() {
        println!("Skills ({}):", approved.len());
        println!("{}", "-".repeat(60));
        for entry in &approved {
            let slug = entry.get("slug").and_then(|v| v.as_str()).unwrap_or("?");
            let title = entry
                .get("short_title")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .or_else(|| entry.get("title").and_then(|v| v.as_str()))
                .unwrap_or("Untitled");
            let confidence = entry
                .get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let lifecycle = entry
                .get("lifecycle_state")
                .and_then(|v| v.as_str())
                .unwrap_or("approved");
            let source = entry
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let source_label = if source == "focus" { "Focus" } else { "Auto" };

            // Check if exported to disk
            let on_disk = dir.join(format!("sop.{}.md", slug)).exists();
            let disk_badge = if on_disk { "" } else { " [pending export]" };

            println!(
                "  {} -- {} ({}, {:.0}%, {}){}",
                slug.bold(),
                title,
                source_label,
                confidence * 100.0,
                lifecycle.green(),
                disk_badge,
            );
        }
    }

    // Show drafts awaiting review
    if !drafts.is_empty() {
        if !approved.is_empty() {
            println!();
        }
        println!("Drafts awaiting review ({}):", drafts.len());
        println!("{}", "-".repeat(60));
        for entry in &drafts {
            let slug = entry.get("slug").and_then(|v| v.as_str()).unwrap_or("?");
            let title = entry
                .get("short_title")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .or_else(|| entry.get("title").and_then(|v| v.as_str()))
                .unwrap_or("Untitled");
            let confidence = entry
                .get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            println!(
                "  {} -- {} (conf {:.0}%) {}",
                slug,
                title,
                confidence * 100.0,
                "[draft]".yellow()
            );
        }
        println!();
        println!("Approve a draft:  agenthandover sops approve <slug>");
    }

    Ok(())
}

pub fn show(slug: &str) -> Result<()> {
    // First try the exported markdown file on disk
    let file_path = sops_dir().join(format!("sop.{}.md", slug));
    if file_path.exists() {
        let content = std::fs::read_to_string(&file_path)?;
        println!("{}", content);
        return Ok(());
    }

    // Fall back to sops-index.json for drafts that haven't been exported yet
    if let Some(index) = read_sops_index() {
        if let Some(sops) = index.get("sops").and_then(|s| s.as_array()) {
            if let Some(entry) = sops.iter().find(|s| {
                s.get("slug").and_then(|v| v.as_str()) == Some(slug)
            }) {
                let status = entry.get("status").and_then(|v| v.as_str()).unwrap_or("unknown");
                let title = entry.get("title").and_then(|v| v.as_str()).unwrap_or("Untitled");
                let sop_id = entry.get("sop_id").and_then(|v| v.as_str()).unwrap_or("?");
                let source = entry.get("source").and_then(|v| v.as_str()).unwrap_or("?");
                let confidence = entry.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let created_at = entry.get("created_at").and_then(|v| v.as_str()).unwrap_or("?");

                println!("{} {}", "SOP:".bold(), title);
                println!("{} {}", "Slug:".bold(), slug);
                println!("{} {}", "Status:".bold(), status.yellow());
                println!("{} {}", "ID:".bold(), sop_id);
                println!("{} {}", "Source:".bold(), source);
                println!("{} {:.0}%", "Confidence:".bold(), confidence * 100.0);
                println!("{} {}", "Created:".bold(), created_at);

                if let Some(tags) = entry.get("tags").and_then(|v| v.as_array()) {
                    let tag_strs: Vec<&str> = tags
                        .iter()
                        .filter_map(|t| t.as_str())
                        .collect();
                    if !tag_strs.is_empty() {
                        println!("{} {}", "Tags:".bold(), tag_strs.join(", "));
                    }
                }

                if status == "draft" {
                    println!();
                    println!(
                        "This SOP is a {}. Approve it with:",
                        "draft".yellow()
                    );
                    println!("  agenthandover sops approve {}", slug);
                }

                return Ok(());
            }
        }
    }

    bail!(
        "SOP '{}' not found.\n  Not on disk at: {}\n  Not in sops-index at: {}",
        slug,
        file_path.display(),
        sops_index_path().display()
    );
}

pub fn dir() -> Result<()> {
    println!("{}", sops_dir().display());
    Ok(())
}

pub fn drafts() -> Result<()> {
    let draft_entries = entries_with_status("draft");

    if draft_entries.is_empty() {
        println!("No draft SOPs awaiting review.");
        println!("Drafts appear when auto_approve is false (the default).");
        return Ok(());
    }

    println!("Draft SOPs awaiting review ({}):", draft_entries.len());
    println!("{}", "-".repeat(70));
    println!(
        "  {:<28}  {:<30}  {}",
        "Slug".bold(),
        "Title".bold(),
        "Confidence".bold()
    );
    println!("{}", "-".repeat(70));

    for entry in &draft_entries {
        let slug = entry.get("slug").and_then(|v| v.as_str()).unwrap_or("?");
        let title = entry
            .get("short_title")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .or_else(|| entry.get("title").and_then(|v| v.as_str()))
            .unwrap_or("Untitled");
        let confidence = entry
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        println!(
            "  {:<28}  {:<30}  {:.0}%",
            slug,
            title,
            confidence * 100.0
        );
    }

    println!();
    println!("Approve a draft:  agenthandover sops approve <slug>");
    println!("View details:     agenthandover sops show <slug>");
    println!("Reject a draft:   agenthandover sops reject <slug>");

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

pub fn promote(slug: &str, to_state: &str) -> Result<()> {
    let valid_states = ["draft", "reviewed", "verified", "agent_ready"];
    if !valid_states.contains(&to_state) {
        bail!(
            "Invalid lifecycle state: '{}'. Must be one of: {}",
            to_state,
            valid_states.join(", ")
        );
    }

    let trigger = serde_json::json!({
        "procedure_slug": slug,
        "to_state": to_state,
        "actor": "human",
        "reason": format!("Promoted via CLI: agenthandover sops promote {} {}", slug, to_state),
        "requested_at": chrono::Utc::now().to_rfc3339(),
    });

    write_trigger(PROMOTE_TRIGGER_FILE, &trigger)?;

    println!(
        "{} Lifecycle promotion queued: '{}' → {}",
        "✓".green(),
        slug.bold(),
        to_state.bold()
    );
    println!("The worker will apply this on its next cycle.");

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
         Check with: agenthandover status"
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
