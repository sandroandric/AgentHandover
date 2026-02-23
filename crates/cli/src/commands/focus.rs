//! CLI commands for Focus Recording Mode.
//!
//! `openmimic focus start "title"` — start a focus recording session.
//! `openmimic focus stop` — stop the active focus recording session.

use anyhow::Result;
use chrono::Utc;
use colored::Colorize;
use oc_apprentice_common::focus_session::{
    read_focus_signal, write_focus_signal, FocusSessionSignal, FocusSessionStatus,
};
use oc_apprentice_common::status::data_dir;
use uuid::Uuid;

/// Start a focus recording session.
///
/// Generates a UUID, writes `focus-session.json` with `status: "recording"`.
pub fn start(title: &str) -> Result<()> {
    let state_dir = data_dir();

    // Check for existing active session
    if let Some(existing) = read_focus_signal(&state_dir) {
        if existing.is_recording() {
            println!(
                "{}",
                format!(
                    "A focus session is already recording: \"{}\" ({})",
                    existing.title, existing.session_id
                )
                .yellow()
            );
            println!("Stop it first with: openmimic focus stop");
            return Ok(());
        }
    }

    let session_id = Uuid::new_v4().to_string();
    let signal = FocusSessionSignal {
        session_id: session_id.clone(),
        title: title.to_string(),
        started_at: Utc::now().to_rfc3339(),
        status: FocusSessionStatus::Recording,
    };

    write_focus_signal(&state_dir, &signal)?;

    println!("{} Focus recording started", "●".red());
    println!("  Title:      {}", title.bold());
    println!("  Session ID: {}", session_id.dimmed());
    println!();
    println!("Perform your workflow now. When done, run:");
    println!("  {}", "openmimic focus stop".bold());

    Ok(())
}

/// Stop the active focus recording session.
///
/// Reads the existing signal, sets `status: "stopped"`, and prints the session ID.
pub fn stop() -> Result<()> {
    let state_dir = data_dir();

    let mut signal = match read_focus_signal(&state_dir) {
        Some(s) => s,
        None => {
            println!("{}", "No active focus session found.".dimmed());
            return Ok(());
        }
    };

    if signal.is_stopped() {
        println!(
            "{}",
            format!(
                "Focus session \"{}\" is already stopped.",
                signal.title
            )
            .dimmed()
        );
        return Ok(());
    }

    signal.status = FocusSessionStatus::Stopped;
    write_focus_signal(&state_dir, &signal)?;

    println!("{} Focus recording stopped", "■".green());
    println!("  Title:      {}", signal.title.bold());
    println!("  Session ID: {}", signal.session_id);
    println!();
    println!(
        "The worker will process this session and generate a SOP on its next cycle."
    );

    Ok(())
}
