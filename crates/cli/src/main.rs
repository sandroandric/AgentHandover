use anyhow::Result;
use clap::{Parser, Subcommand};

mod commands;
mod display;
mod paths;

#[derive(Parser)]
#[command(name = "agenthandover", version, about = "AgentHandover CLI — manage the apprentice system")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show daemon and worker status
    Status,
    /// Start services (daemon launched directly, worker via launchd)
    Start {
        /// Which service to start: daemon, worker, or all (default)
        #[arg(default_value = "all")]
        service: String,
    },
    /// Stop services (daemon via SIGTERM, worker via launchd)
    Stop {
        /// Which service to stop: daemon, worker, or all (default)
        #[arg(default_value = "all")]
        service: String,
    },
    /// Restart services
    Restart {
        /// Which service to restart: daemon, worker, or all (default)
        #[arg(default_value = "all")]
        service: String,
    },
    /// View service logs
    Logs {
        /// Which service: daemon or worker
        #[arg(default_value = "daemon")]
        service: String,
        /// Follow the log (like tail -f)
        #[arg(short, long)]
        follow: bool,
        /// Number of lines to show
        #[arg(short = 'n', long, default_value = "50")]
        lines: usize,
    },
    /// Manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// List and view generated Skills
    #[command(alias = "sops")]
    Skills {
        #[command(subcommand)]
        action: SopsAction,
    },
    /// Live-updating status dashboard (refreshes every 2s)
    Watch,
    /// Interactive setup wizard
    Setup {
        /// Check status only, don't modify anything
        #[arg(long)]
        check: bool,
        /// Set up Chrome extension only
        #[arg(long)]
        extension: bool,
        /// Set up VLM only
        #[arg(long)]
        vlm: bool,
    },
    /// Run pre-flight checks
    Doctor,
    /// Uninstall AgentHandover
    Uninstall {
        /// Also remove user data (database, SOPs, config)
        #[arg(long)]
        purge_data: bool,
    },
    /// Focus recording mode — record a single workflow demonstration
    Focus {
        #[command(subcommand)]
        action: FocusAction,
    },
    /// Export SOPs in a specific format
    Export {
        /// Output format
        #[arg(long, default_value = "skill-md")]
        format: String,
        /// Export a specific SOP by slug (default: all)
        #[arg(long)]
        sop: Option<String>,
        /// Output directory (default: workspace/skills)
        #[arg(long)]
        output: Option<String>,
    },
    /// Search activity annotations (full-text search)
    Search {
        /// Search query (e.g. "domain research", "Stripe dashboard")
        query: String,
        /// Filter by date (YYYY-MM-DD)
        #[arg(long)]
        date: Option<String>,
        /// Filter by app name
        #[arg(long)]
        app: Option<String>,
        /// Maximum results to return
        #[arg(short = 'n', long, default_value = "20")]
        limit: usize,
    },
    /// Connect an AI agent (claude-code, codex, openclaw, mcp)
    Connect {
        /// Which agent to connect: claude-code, codex, openclaw, mcp
        agent: String,
    },
    /// Recall what you were doing at a given time
    Recall {
        /// Date to recall (YYYY-MM-DD, default: today)
        #[arg(long)]
        date: Option<String>,
        /// Filter by app name
        #[arg(long)]
        app: Option<String>,
        /// Start time filter (HH:MM)
        #[arg(long)]
        start: Option<String>,
        /// End time filter (HH:MM)
        #[arg(long)]
        end: Option<String>,
    },
}

#[derive(Subcommand)]
enum FocusAction {
    /// Start recording a workflow demonstration
    Start {
        /// A descriptive title for the workflow (e.g. "Expense report filing")
        title: String,
    },
    /// Stop the active focus recording session
    Stop,
    /// Answer pending questions about the focus recording and complete SOP export
    Finalize,
    /// Skip pending questions and export with default answers
    SkipQuestions,
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Show current configuration
    Show,
    /// Open config file in $EDITOR
    Edit,
    /// Print the config file path
    Path,
}

#[derive(Subcommand)]
enum SopsAction {
    /// List all generated SOPs (exported + drafts)
    List,
    /// Show a specific SOP by slug (works for exported and draft SOPs)
    Show { slug: String },
    /// Print the SOPs directory path
    Dir,
    /// List draft SOPs awaiting review
    Drafts,
    /// Approve a draft SOP for export
    Approve {
        /// SOP slug or ID
        slug_or_id: String,
    },
    /// Reject a draft SOP
    Reject {
        /// SOP slug or ID
        slug_or_id: String,
    },
    /// Promote a procedure's lifecycle state (e.g., draft → reviewed → verified → agent_ready)
    Promote {
        /// Procedure slug
        slug: String,
        /// Target lifecycle state: draft, reviewed, verified, agent_ready
        to_state: String,
    },
    /// List failed SOP generations
    Failed,
    /// Retry a failed SOP generation
    Retry {
        /// Failure ID to retry
        failure_id: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Status => commands::status::run(),
        Commands::Start { service } => commands::service::start(&service),
        Commands::Stop { service } => commands::service::stop(&service),
        Commands::Restart { service } => commands::service::restart(&service),
        Commands::Logs {
            service,
            follow,
            lines,
        } => commands::logs::run(&service, follow, lines),
        Commands::Config { action } => match action {
            ConfigAction::Show => commands::config::show(),
            ConfigAction::Edit => commands::config::edit(),
            ConfigAction::Path => commands::config::path(),
        },
        Commands::Skills { action } => match action {
            SopsAction::List => commands::sops::list(),
            SopsAction::Show { slug } => commands::sops::show(&slug),
            SopsAction::Dir => commands::sops::dir(),
            SopsAction::Drafts => commands::sops::drafts(),
            SopsAction::Approve { slug_or_id } => commands::sops::approve(&slug_or_id),
            SopsAction::Reject { slug_or_id } => commands::sops::reject(&slug_or_id),
            SopsAction::Promote { slug, to_state } => commands::sops::promote(&slug, &to_state),
            SopsAction::Failed => commands::sops::failed(),
            SopsAction::Retry { failure_id } => commands::sops::retry(&failure_id),
        },
        Commands::Watch => commands::watch::run(),
        Commands::Setup { check, extension, vlm } => commands::setup::run(check, extension, vlm),
        Commands::Doctor => commands::doctor::run(),
        Commands::Uninstall { purge_data } => commands::uninstall::run(purge_data),
        Commands::Focus { action } => match action {
            FocusAction::Start { title } => commands::focus::start(&title),
            FocusAction::Stop => commands::focus::stop(),
            FocusAction::Finalize => commands::focus::finalize(),
            FocusAction::SkipQuestions => commands::focus::skip_questions(),
        },
        Commands::Connect { agent } => {
            // Delegate to the Python agent_connect script
            let status = std::process::Command::new("agenthandover-connect")
                .arg(&agent)
                .status();
            match status {
                Ok(s) if s.success() => Ok(()),
                Ok(s) => {
                    // Try fallback: call Python directly
                    let venv = "/usr/local/lib/agenthandover/venv/bin/python";
                    let fallback = std::process::Command::new(venv)
                        .args(["-m", "agenthandover_worker.agent_connect", &agent])
                        .status();
                    match fallback {
                        Ok(fs) if fs.success() => Ok(()),
                        _ => anyhow::bail!("agenthandover connect failed (exit {})", s.code().unwrap_or(-1)),
                    }
                }
                Err(_) => {
                    // agenthandover-connect not in PATH, try venv directly
                    let venv = "/usr/local/lib/agenthandover/venv/bin/python";
                    let fallback = std::process::Command::new(venv)
                        .args(["-m", "agenthandover_worker.agent_connect", &agent])
                        .status();
                    match fallback {
                        Ok(s) if s.success() => Ok(()),
                        _ => anyhow::bail!("Could not run agent connect. Is the worker installed?"),
                    }
                }
            }
        }
        Commands::Export { format, sop, output } => {
            commands::export::run(&format, sop.as_deref(), output.as_deref())
        }
        Commands::Search {
            query,
            date,
            app,
            limit,
        } => commands::search::search(&query, date.as_deref(), app.as_deref(), limit),
        Commands::Recall {
            date,
            app,
            start,
            end,
        } => commands::search::recall(
            date.as_deref(),
            app.as_deref(),
            start.as_deref(),
            end.as_deref(),
        ),
    }
}
