//! CLI command for re-exporting SOPs in a specific format.
//!
//! `agenthandover export --format skill-md [--sop <slug>] [--output <dir>]`
//!
//! This asks the worker to re-export existing SOPs through the trigger
//! spool; the worker picks the request up on its next cycle.

use anyhow::Result;
use colored::Colorize;
use crate::trigger::write_trigger;

/// Run the export command.
///
/// Queues a request that the worker picks up on its next poll cycle
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

    write_trigger("export", &trigger)?;

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
