//! CLI commands for Focus Recording Mode.
//!
//! `agenthandover focus start "title"` — start a focus recording session.
//! `agenthandover focus stop` — stop the active focus recording session.
//! `agenthandover focus finalize` — answer the pending questions.
//! `agenthandover focus skip-questions` — skip them (recorded as unanswered).

use anyhow::Result;
use chrono::Utc;
use colored::Colorize;
use agenthandover_common::focus_session::{
    read_focus_signal, write_focus_signal, FocusSessionSignal, FocusSessionStatus,
};
use agenthandover_common::status::data_dir;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use uuid::Uuid;

/// Focus questions file written by the worker, read/updated by the CLI.
const FOCUS_QUESTIONS_FILE: &str = "focus-questions.json";

/// A single question about a focus recording.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FocusQuestion {
    index: usize,
    question: String,
    category: String,
    context: String,
    default: String,
}

/// The focus-questions.json file format.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FocusQuestionsFile {
    session_id: String,
    slug: String,
    questions: Vec<FocusQuestion>,
    status: String,
    #[serde(default)]
    answers: HashMap<String, String>,
}

/// What `focus start` says when the daemon is not running: the recording
/// would capture nothing.
pub(crate) const NOT_OBSERVING: &str = "AgentHandover is not observing, so a focus recording would capture nothing.\n\
Start it first: click the AgentHandover menu bar icon and choose \"Observe Me\" or \
\"Record a Focus Session\" (or, with the app running, run: agenthandover start daemon).\n\
Then run this command again.";

/// Start a focus recording session.
///
/// Generates a UUID, writes `focus-session.json` with `status: "recording"`.
/// Refuses (exit code 1) when the daemon is not running.
pub fn start(title: &str) -> Result<()> {
    start_in(&data_dir(), title, super::service::is_daemon_running())
}

fn start_in(state_dir: &std::path::Path, title: &str, daemon_running: bool) -> Result<()> {
    if !daemon_running {
        anyhow::bail!(NOT_OBSERVING);
    }

    // Check for existing active session
    if let Some(existing) = read_focus_signal(state_dir) {
        if existing.is_recording() {
            println!(
                "{}",
                format!(
                    "A focus session is already recording: \"{}\" ({})",
                    existing.title, existing.session_id
                )
                .yellow()
            );
            println!("Stop it first with: agenthandover focus stop");
            return Ok(());
        }
    }

    let session_id = Uuid::new_v4().to_string();
    let signal = FocusSessionSignal {
        session_id: session_id.clone(),
        title: title.to_string(),
        started_at: Utc::now().to_rfc3339(),
        status: FocusSessionStatus::Recording,
        stopped_at: None,
    };

    write_focus_signal(state_dir, &signal)?;
    // An earlier "nothing was recorded" outcome (the worker's
    // focus-result.json) no longer applies once a new session records.
    let _ = std::fs::remove_file(state_dir.join("focus-result.json"));

    println!("{} Focus recording started", "●".red());
    println!("  Title:      {}", title.bold());
    println!("  Session ID: {}", session_id.dimmed());
    println!();
    println!("Perform your workflow now. When done, run:");
    println!("  {}", "agenthandover focus stop".bold());

    Ok(())
}

/// Stop the active focus recording session.
///
/// Reads the existing signal, sets `status: "stopped"` and `stopped_at`
/// (the worker times its empty-session wait from it), and prints the
/// session ID.
pub fn stop() -> Result<()> {
    stop_in(&data_dir())
}

fn stop_in(state_dir: &std::path::Path) -> Result<()> {
    let mut signal = match read_focus_signal(state_dir) {
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
    signal.stopped_at = Some(Utc::now().to_rfc3339());
    write_focus_signal(state_dir, &signal)?;

    println!("{} Focus recording stopped", "■".green());
    println!("  Title:      {}", signal.title.bold());
    println!("  Session ID: {}", signal.session_id);
    println!();
    println!(
        "The worker will process this session and draft a Skill on its next cycle."
    );
    println!(
        "If the worker has questions, answer them with: {}",
        "agenthandover focus finalize".bold()
    );

    Ok(())
}

/// Answer pending focus questions interactively; the worker then saves the
/// Skill with the answers (a draft until you approve it).
///
/// Reads `focus-questions.json`, presents each question to the user,
/// collects answers, and writes them back with `status: "answered"`.
/// The worker picks up the answers on its next poll cycle.
pub fn finalize() -> Result<()> {
    let state_dir = data_dir();
    let questions_path = state_dir.join(FOCUS_QUESTIONS_FILE);

    // Read the questions file
    let content = match std::fs::read_to_string(&questions_path) {
        Ok(c) => c,
        Err(_) => {
            println!(
                "{}",
                "No pending focus questions found.".dimmed()
            );
            println!(
                "Questions appear after the worker processes a focus recording."
            );
            println!(
                "If you just stopped a recording, wait a few seconds and try again."
            );
            return Ok(());
        }
    };

    let mut qa_file: FocusQuestionsFile = match serde_json::from_str(&content) {
        Ok(f) => f,
        Err(e) => {
            println!(
                "{}",
                format!("Failed to parse focus-questions.json: {}", e).red()
            );
            return Ok(());
        }
    };

    if qa_file.status == "answered" || qa_file.status == "skipped" {
        println!(
            "{}",
            "Focus questions have already been answered.".dimmed()
        );
        println!("The worker will save the Skill on its next cycle.");
        return Ok(());
    }

    if qa_file.status != "pending" {
        println!(
            "{}",
            format!("Unexpected status: \"{}\"", qa_file.status).yellow()
        );
        return Ok(());
    }

    if qa_file.questions.is_empty() {
        println!("{}", "No questions to answer.".dimmed());
        qa_file.status = "answered".to_string();
        write_questions_file(&questions_path, &qa_file)?;
        return Ok(());
    }

    println!(
        "{} The worker has {} question(s) about your focus recording:",
        "?".cyan().bold(),
        qa_file.questions.len()
    );
    println!("  Workflow: {}", qa_file.slug.bold());
    println!();

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut answers: HashMap<String, String> = HashMap::new();

    for q in &qa_file.questions {
        println!(
            "  {} [{}] {}",
            format!("Q{}.", q.index + 1).bold(),
            q.category.dimmed(),
            q.question
        );
        if !q.context.is_empty() {
            println!("     {}", q.context.dimmed());
        }
        print!(
            "     {} [default: {}]: ",
            ">".green(),
            q.default.dimmed()
        );
        stdout.flush()?;

        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let trimmed = input.trim();

        let answer = if trimmed.is_empty() {
            q.default.clone()
        } else {
            trimmed.to_string()
        };

        answers.insert(q.index.to_string(), answer);
        println!();
    }

    qa_file.answers = answers;
    qa_file.status = "answered".to_string();

    write_questions_file(&questions_path, &qa_file)?;

    println!(
        "{} Answers saved. The worker will add them to the Skill; it stays a draft until you approve it.",
        "✓".green().bold()
    );

    Ok(())
}

/// Skip all pending focus questions: the worker records them as unanswered
/// (not yet known in the Skill) and saves the Skill as a draft.
///
/// Sets `status: "skipped"` in `focus-questions.json` so the worker
/// uses default values for all questions.
pub fn skip_questions() -> Result<()> {
    let state_dir = data_dir();
    let questions_path = state_dir.join(FOCUS_QUESTIONS_FILE);

    let content = match std::fs::read_to_string(&questions_path) {
        Ok(c) => c,
        Err(_) => {
            println!(
                "{}",
                "No pending focus questions found.".dimmed()
            );
            return Ok(());
        }
    };

    let mut qa_file: FocusQuestionsFile = match serde_json::from_str(&content) {
        Ok(f) => f,
        Err(e) => {
            println!(
                "{}",
                format!("Failed to parse focus-questions.json: {}", e).red()
            );
            return Ok(());
        }
    };

    if qa_file.status != "pending" {
        println!(
            "{}",
            format!("Questions are not pending (status: {})", qa_file.status).dimmed()
        );
        return Ok(());
    }

    qa_file.status = "skipped".to_string();
    write_questions_file(&questions_path, &qa_file)?;

    println!(
        "{} Questions skipped. The worker saves the Skill as a draft, with them not yet known.",
        "→".yellow().bold()
    );

    Ok(())
}

/// Atomically write the questions file (tmp + fsync + rename).
fn write_questions_file(
    path: &std::path::Path,
    data: &FocusQuestionsFile,
) -> Result<()> {
    use std::io::Write as IoWrite;

    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("No parent directory for questions file"))?;
    std::fs::create_dir_all(parent)?;

    let tmp = parent.join(format!(".{}.tmp", FOCUS_QUESTIONS_FILE));
    let json = serde_json::to_string_pretty(data)?;
    let mut file = std::fs::File::create(&tmp)?;
    file.write_all(json.as_bytes())?;
    file.sync_all()?;
    std::fs::rename(&tmp, path)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::service::daemon_running_at;

    #[test]
    fn start_without_a_running_daemon_refuses_and_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let err = start_in(dir.path(), "Send weekly report", false).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("not observing"), "{text}");
        assert!(text.contains("Observe Me"), "{text}");
        assert!(text.contains("agenthandover start daemon"), "{text}");
        assert!(read_focus_signal(dir.path()).is_none());
    }

    #[test]
    fn start_with_a_running_daemon_writes_a_recording_signal() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("focus-result.json"), "{}").unwrap();
        start_in(dir.path(), "Send weekly report", true).unwrap();
        assert!(!dir.path().join("focus-result.json").exists(), "an old outcome is cleared");
        let signal = read_focus_signal(dir.path()).unwrap();
        assert!(signal.is_recording());
        assert_eq!(signal.title, "Send weekly report");
    }

    #[test]
    fn stop_writes_stopped_at_and_keeps_the_session() {
        let dir = tempfile::tempdir().unwrap();
        start_in(dir.path(), "Send weekly report", true).unwrap();
        let started = read_focus_signal(dir.path()).unwrap();
        assert_eq!(started.stopped_at, None);

        let before = Utc::now();
        stop_in(dir.path()).unwrap();
        let stopped = read_focus_signal(dir.path()).unwrap();
        assert!(stopped.is_stopped());
        assert_eq!(stopped.session_id, started.session_id);
        assert_eq!(stopped.started_at, started.started_at);
        let at = chrono::DateTime::parse_from_rfc3339(stopped.stopped_at.as_deref().unwrap())
            .unwrap();
        assert!(at >= before - chrono::Duration::seconds(1));

        // A second stop leaves the first stop time alone.
        stop_in(dir.path()).unwrap();
        assert_eq!(read_focus_signal(dir.path()).unwrap().stopped_at, stopped.stopped_at);
    }

    #[test]
    fn no_pid_file_or_a_pid_that_is_not_the_daemon_is_not_running() {
        let dir = tempfile::tempdir().unwrap();
        let pid_file = dir.path().join("daemon.pid");
        assert!(!daemon_running_at(&pid_file));
        // This test process is alive but is not ah-observer.
        std::fs::write(&pid_file, std::process::id().to_string()).unwrap();
        assert!(!daemon_running_at(&pid_file));
        std::fs::write(&pid_file, "not a pid").unwrap();
        assert!(!daemon_running_at(&pid_file));
    }
}
