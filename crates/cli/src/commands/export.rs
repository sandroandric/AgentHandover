//! CLI command for re-exporting SOPs in a specific format.
//!
//! `agenthandover export --format skill-md [--sop <slug>] [--output <dir>]`
//!
//! This triggers the worker to re-export existing SOPs by writing a trigger
//! file that the worker picks up on its next cycle.

use anyhow::Result;
use colored::Colorize;
use agenthandover_common::status::data_dir;
use std::io::Write;

const EXPORT_TRIGGER_FILE: &str = "export-trigger.json";

/// Run the export command.
///
/// Writes a trigger file that the worker picks up on its next poll cycle
/// to re-export SOPs in the specified format.
pub fn run(format: &str, sop_slug: Option<&str>, output_dir: Option<&str>) -> Result<()> {
    let supported_formats = ["skill-md", "generic", "openclaw", "claude-skill"];
    if !supported_formats.contains(&format) {
        anyhow::bail!(
            "Unsupported format: '{}'. Supported: {}",
            format,
            supported_formats.join(", ")
        );
    }

    let trigger = serde_json::json!({
        "format": format,
        "sop_slug": sop_slug,
        "output_dir": output_dir,
        "requested_at": chrono::Utc::now().to_rfc3339(),
    });

    let state_dir = data_dir();
    std::fs::create_dir_all(&state_dir)?;
    let target = state_dir.join(EXPORT_TRIGGER_FILE);
    let tmp = state_dir.join(format!(".{}.tmp", EXPORT_TRIGGER_FILE));

    let json = serde_json::to_string_pretty(&trigger)?;
    let mut file = std::fs::File::create(&tmp)?;
    file.write_all(json.as_bytes())?;
    file.sync_all()?;
    std::fs::rename(&tmp, &target)?;

    println!("{} Export requested", "✓".green());
    println!("  Format: {}", format.bold());
    if let Some(slug) = sop_slug {
        println!("  SOP:    {}", slug);
    } else {
        println!("  SOPs:   all");
    }
    if let Some(dir) = output_dir {
        println!("  Output: {}", dir);
    }
    println!();
    println!("The worker will process this export on its next cycle.");

    Ok(())
}
